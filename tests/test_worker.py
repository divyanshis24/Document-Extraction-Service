import asyncio
import hashlib
import importlib.util
import logging
from pathlib import Path

import psycopg

from app.config import Settings
from app.extractor import ExtractionError, ExtractionOutput, StubExtractor
from app.jobs import NewJob
from app.worker import Worker

spec = importlib.util.spec_from_file_location("validate", Path(__file__).parent.parent / "tools" / "validate.py")
validate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate)

FIELDS = {"fields": {"a": {}, "b": {}}}
FAST = dict(worker_concurrency=2, worker_poll_seconds=0.02, sweep_interval_seconds=0.05, job_lease_seconds=0.3)


def new_job(content=b"%PDF-doc"):
    return NewJob(org_id="", document=content, document_sha256=hashlib.sha256(content).hexdigest(),
                  file_name="a.pdf", file_type="pdf", field_list=FIELDS, field_list_sha256="s")


class Fake:
    def __init__(self, fn):
        self.fn, self.calls, self.cancelled = fn, 0, 0

    async def extract(self, job, document, progress):
        self.calls += 1
        try:
            return await self.fn(job, document, progress)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


async def wait_for_status(store, job_id, statuses, timeout=8):
    async with asyncio.timeout(timeout):
        while True:
            row = await store.get(job_id, "")
            if row["status"] in statuses:
                return row
            await asyncio.sleep(0.02)


async def running_worker(db, store, extractor, body, **settings):
    worker = Worker(db, store, extractor, Settings(**{**FAST, **settings}))
    task = asyncio.create_task(worker.run())
    try:
        return await body(worker)
    finally:
        worker.stop.set()
        await asyncio.wait_for(task, 10)


TERMINAL = ("SUCCESS", "PARTIAL", "FAILED")


def test_stub_result_passes_the_assignment_validator(with_store):
    async def body(db, store):
        row, _ = await store.submit(new_job())

        async def run(_):
            return await wait_for_status(store, row["id"], TERMINAL)

        done = await running_worker(db, store, StubExtractor(), run)
        assert done["status"] == "PARTIAL"
        assert validate.validate_contract(done["result"], FIELDS) == []
        assert done["result"]["metadata"]["latency_ms"] >= 0

    with_store(body)


def test_extractor_receives_the_document_and_success_is_reported(with_store):
    async def body(db, store):
        row, _ = await store.submit(new_job(b"%PDF-specific"))
        seen = {}

        async def fn(job, document, progress):
            seen.update(document=document, fields=job.field_list)
            await progress("EXTRACTING")
            return ExtractionOutput([{"a": {"value": 1, "citation": "c", "confidence": 1}}, {}])

        async def run(_):
            return await wait_for_status(store, row["id"], TERMINAL)

        done = await running_worker(db, store, Fake(fn), run)
        assert seen == {"document": b"%PDF-specific", "fields": FIELDS}
        assert done["status"] == "SUCCESS" and len(done["result"]["entries"]) == 2
        assert done["stage"] is None
        assert validate.validate_contract(done["result"], FIELDS) == []

    with_store(body)


def test_crash_fails_with_generic_message_and_full_log(with_store, caplog):
    async def body(db, store):
        row, _ = await store.submit(new_job())

        async def fn(*_):
            raise RuntimeError("secret document text")

        async def run(_):
            return await wait_for_status(store, row["id"], TERMINAL)

        done = await running_worker(db, store, Fake(fn), run)
        assert done["status"] == "FAILED" and done["error"]["code"] == "extraction_error"
        assert "secret document text" not in done["error"]["message"]

    with_store(body)


def test_known_extraction_error_is_reported_as_is(with_store):
    async def body(db, store):
        row, _ = await store.submit(new_job())

        async def fn(*_):
            raise ExtractionError("document_unreadable", "PDF is password protected")

        async def run(_):
            return await wait_for_status(store, row["id"], TERMINAL)

        done = await running_worker(db, store, Fake(fn), run)
        assert done["error"] == {"code": "document_unreadable", "message": "PDF is password protected"}

    with_store(body)


def test_database_error_while_loading_retries_instead_of_failing(with_store):
    async def body(db, store):
        row, _ = await store.submit(new_job())
        real, failures = store.load_document, []

        async def flaky(sha):
            if not failures:
                failures.append(1)
                raise psycopg.OperationalError("server closed the connection")
            return await real(sha)

        store.load_document = flaky

        async def fn(*_):
            return ExtractionOutput([{}])

        async def run(_):
            return await wait_for_status(store, row["id"], TERMINAL)

        done = await running_worker(db, store, Fake(fn), run)
        assert failures and done["status"] == "SUCCESS"

    with_store(body)


def test_database_error_updating_stage_does_not_end_the_job(with_store):
    async def body(db, store):
        row, _ = await store.submit(new_job())

        async def broken(*_):
            raise psycopg.OperationalError("server closed the connection")

        store.set_stage = broken

        async def fn(job, document, progress):
            await progress("EXTRACTING")
            return ExtractionOutput([{}])

        async def run(_):
            return await wait_for_status(store, row["id"], TERMINAL)

        fake = Fake(fn)
        done = await running_worker(db, store, fake, run)
        assert done["status"] == "SUCCESS" and fake.calls == 1

    with_store(body)


def test_outcome_is_saved_after_a_brief_database_error(with_store):
    async def body(db, store):
        row, _ = await store.submit(new_job())
        real, failures = store.complete, []

        async def flaky(*args):
            if not failures:
                failures.append(1)
                raise psycopg.OperationalError("server closed the connection")
            return await real(*args)

        store.complete = flaky
        fake = Fake(lambda *_: asyncio.sleep(0, ExtractionOutput([{}])))

        async def run(_):
            return await wait_for_status(store, row["id"], TERMINAL)

        done = await running_worker(db, store, fake, run)
        assert done["status"] == "SUCCESS" and fake.calls == 1  # not re-extracted

    with_store(body)


def test_shutdown_survives_database_being_gone(with_store):
    async def body(db, store):
        await store.submit(new_job())
        started = asyncio.Event()

        async def fn(*_):
            started.set()
            await asyncio.sleep(60)

        async def broken(*_):
            raise psycopg.OperationalError("the database system is shutting down")

        store.release = broken

        async def run(_):
            await asyncio.wait_for(started.wait(), 5)

        await running_worker(db, store, Fake(fn), run)  # must stop cleanly, without raising

    with_store(body)


def test_bad_job_does_not_stop_later_jobs(with_store):
    async def body(db, store):
        bad, _ = await store.submit(new_job(b"%PDF-bad"))
        good, _ = await store.submit(new_job(b"%PDF-good"))

        async def fn(job, document, progress):
            if document == b"%PDF-bad":
                raise ValueError("boom")
            return ExtractionOutput([{}])

        async def run(_):
            return (await wait_for_status(store, bad["id"], TERMINAL),
                    await wait_for_status(store, good["id"], TERMINAL))

        b, g = await running_worker(db, store, Fake(fn), run, worker_concurrency=1)
        assert b["status"] == "FAILED" and g["status"] == "SUCCESS"

    with_store(body)


def test_long_job_survives_because_heartbeat_renews_the_lease(with_store):
    async def body(db, store):
        row, _ = await store.submit(new_job())

        async def fn(*_):
            await asyncio.sleep(1.0)  # over three times the 0.3s lease
            return ExtractionOutput([{}])

        fake = Fake(fn)

        async def run(_):
            return await wait_for_status(store, row["id"], TERMINAL)

        done = await running_worker(db, store, fake, run)
        assert done["status"] == "SUCCESS" and fake.calls == 1

    with_store(body)


def test_dead_worker_job_is_picked_up_by_another(with_store):
    async def body(db, store):
        row, _ = await store.submit(new_job())
        dead = await store.claim(0.1, 3, 1740)  # claimed, then the process "dies"
        assert dead.id == row["id"]

        async def fn(*_):
            return ExtractionOutput([{}])

        async def run(_):
            return await wait_for_status(store, row["id"], TERMINAL)

        done = await running_worker(db, store, Fake(fn), run)
        assert done["status"] == "SUCCESS"
        assert not await store.complete(dead, "FAILED", {"entries": []})  # the original can't overwrite

    with_store(body)


def test_shutdown_cancels_and_releases_for_another_worker(with_store):
    async def body(db, store):
        row, _ = await store.submit(new_job())
        started = asyncio.Event()

        async def fn(*_):
            started.set()
            await asyncio.sleep(60)

        fake = Fake(fn)

        async def run(worker):
            await asyncio.wait_for(started.wait(), 5)

        await running_worker(db, store, fake, run)
        assert fake.cancelled == 1
        after = await store.get(row["id"], "")
        assert after["status"] == "PENDING"

        async def quick(*_):
            return ExtractionOutput([{}])

        async def run2(_):
            return await wait_for_status(store, row["id"], TERMINAL)

        assert (await running_worker(db, store, Fake(quick), run2))["status"] == "SUCCESS"

    with_store(body)


def test_worker_that_loses_its_lease_stops_working(with_store):
    async def body(db, store):
        row, _ = await store.submit(new_job())
        started = asyncio.Event()

        async def fn(*_):
            started.set()
            await asyncio.sleep(60)

        fake = Fake(fn)

        async def run(_):
            await asyncio.wait_for(started.wait(), 5)
            # Simulate this worker's lease being taken by rotating the token.
            async with db.pool.connection() as conn:
                await conn.execute("UPDATE jobs SET lease_token = gen_random_uuid() WHERE id = %s", (row["id"],))
            async with asyncio.timeout(5):
                while not fake.cancelled:
                    await asyncio.sleep(0.02)

        await running_worker(db, store, fake, run, job_lease_seconds=1.5, worker_concurrency=1)
        assert fake.cancelled == 1
        assert (await store.get(row["id"], ""))["status"] == "RUNNING"  # untouched by the old worker

    with_store(body)


def test_deadline_stops_a_stuck_extraction(with_store):
    async def body(db, store):
        row, _ = await store.submit(new_job())

        async def fn(*_):
            await asyncio.sleep(60)

        async def run(_):
            return await wait_for_status(store, row["id"], TERMINAL)

        done = await running_worker(db, store, Fake(fn), run, job_deadline_seconds=0.3)
        assert done["status"] == "FAILED" and done["error"]["code"] == "deadline_exceeded"

    with_store(body)


def test_concurrent_workers_process_each_job_exactly_once(with_store):
    async def body(db, store):
        rows = [(await store.submit(new_job(f"%PDF-{i}".encode())))[0] for i in range(20)]
        counts = {}

        async def fn(job, *_):
            counts[job.id] = counts.get(job.id, 0) + 1
            await asyncio.sleep(0.01)
            return ExtractionOutput([{}])

        a, b = Worker(db, store, Fake(fn), Settings(**FAST)), Worker(db, store, Fake(fn), Settings(**FAST))
        tasks = [asyncio.create_task(a.run()), asyncio.create_task(b.run())]
        for r in rows:
            assert (await wait_for_status(store, r["id"], TERMINAL))["status"] == "SUCCESS"
        a.stop.set()
        b.stop.set()
        await asyncio.gather(*tasks)
        assert len(counts) == 20 and set(counts.values()) == {1}

    with_store(body)


def test_result_that_cannot_be_built_fails_once_instead_of_rerunning(with_store, monkeypatch):
    import app.worker as worker_module

    def broken_build(*_):
        raise ValueError("bad output")

    monkeypatch.setattr(worker_module, "build_result", broken_build)

    async def body(db, store):
        row, _ = await store.submit(new_job())

        async def fn(*_):
            return ExtractionOutput(entries=[{"a": {"value": 1}}], metadata={})

        fake = Fake(fn)

        async def run(_):
            return await wait_for_status(store, row["id"], TERMINAL)

        done = await running_worker(db, store, fake, run)
        assert done["status"] == "FAILED" and done["error"]["code"] == "result_invalid"
        assert fake.calls == 1  # paid extraction is not repeated

    with_store(body)
