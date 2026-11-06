import asyncio
import hashlib
import uuid

import psycopg
import pytest

from app.jobs import JobStore, NewJob
from app.migrations import apply_migrations

LEASE, MAX, DEADLINE = 60, 3, 1740


def new_job(content=b"%PDF-doc", schema_hash="s", org=""):
    return NewJob(org_id=org, document=content, document_sha256=hashlib.sha256(content).hexdigest(),
                  file_name="a.pdf", file_type="pdf", field_list={"fields": {"a": {}}}, field_list_sha256=schema_hash)


async def claim(store, lease=LEASE):
    return await store.claim(lease, MAX, DEADLINE)


async def status(store, job_id):
    return (await store.get(job_id, ""))["status"]


def test_migrations_are_safe_to_run_concurrently_and_repeatedly(with_store, pg_url):
    async def body(db, store):
        await asyncio.gather(*[db.migrate() for _ in range(8)])

        async with db.pool.connection() as conn:
            await apply_migrations(conn)
            cur = await conn.execute("SELECT count(*) AS n FROM schema_migrations")
            assert (await cur.fetchone())["n"] == 1

    with_store(body)


def test_simultaneous_duplicates_make_one_job(with_store):
    async def body(db, store):
        results = await asyncio.gather(*[store.submit(new_job()) for _ in range(12)])
        assert len({r[0]["id"] for r in results}) == 1
        assert sum(created for _, created in results) == 1

    with_store(body)


def test_no_job_is_claimed_twice(with_store):
    async def body(db, store):
        for i in range(6):
            await store.submit(new_job(f"%PDF-{i}".encode()))
        claims = await asyncio.gather(*[claim(store) for _ in range(12)])
        got = [c.id for c in claims if c]
        assert len(got) == len(set(got)) == 6

    with_store(body)


def test_oldest_job_is_claimed_first(with_store):
    async def body(db, store):
        first, _ = await store.submit(new_job(b"%PDF-1"))
        await store.submit(new_job(b"%PDF-2"))
        assert (await claim(store)).id == first["id"]

    with_store(body)


def test_live_lease_blocks_claim_and_expired_lease_is_reclaimed(with_store):
    async def body(db, store):
        await store.submit(new_job())
        first = await claim(store, lease=0.2)
        assert await claim(store) is None
        await asyncio.sleep(0.3)
        second = await claim(store)
        assert second.id == first.id and second.attempts == 2
        assert second.lease_token != first.lease_token

    with_store(body)


def test_heartbeat_keeps_the_lease(with_store):
    async def body(db, store):
        await store.submit(new_job())
        job = await claim(store, lease=0.3)
        await asyncio.sleep(0.2)
        assert await store.heartbeat(job, 60)
        await asyncio.sleep(0.2)
        assert await claim(store) is None

    with_store(body)


def test_stale_worker_cannot_write_after_losing_the_job(with_store):
    async def body(db, store):
        await store.submit(new_job())
        stale = await claim(store, lease=0.1)
        await asyncio.sleep(0.2)
        current = await claim(store)
        assert not await store.heartbeat(stale, 60)
        assert not await store.set_stage(stale, "X")
        assert not await store.fail(stale, "c", "m")
        assert not await store.complete(stale, "SUCCESS", {"entries": []})
        assert not await store.release(stale)
        assert await status(store, current.id) == "RUNNING"
        assert await store.complete(current, "SUCCESS", {"entries": ["new"]})
        assert (await store.get(current.id, ""))["result"] == {"entries": ["new"]}

    with_store(body)


@pytest.mark.parametrize("outcome", ["SUCCESS", "PARTIAL", "FAILED"])
def test_terminal_status_cannot_change(with_store, outcome):
    async def body(db, store):
        await store.submit(new_job())
        job = await claim(store)
        if outcome == "FAILED":
            assert await store.fail(job, "c", "m")
        else:
            assert await store.complete(job, outcome, {"entries": []})
        # The same worker writing twice is a no-op.
        assert not await store.complete(job, "SUCCESS", {"entries": ["again"]})
        assert not await store.fail(job, "c2", "m2")
        # Direct SQL, as a bug might do, is stopped by the trigger.
        async with db.pool.connection() as conn:
            with pytest.raises(psycopg.errors.IntegrityConstraintViolation):
                await conn.execute("UPDATE jobs SET status='PENDING', result=NULL, error=NULL WHERE id=%s", (job.id,))
        assert await status(store, job.id) == outcome
        assert await claim(store) is None

    with_store(body)


def test_release_returns_job_without_using_an_attempt(with_store):
    async def body(db, store):
        await store.submit(new_job())
        job = await claim(store)
        assert await store.release(job)
        again = await claim(store)
        assert again.id == job.id and again.attempts == 1

    with_store(body)


def test_job_that_keeps_killing_workers_ends_as_failed(with_store):
    async def body(db, store):
        row, _ = await store.submit(new_job())
        for _ in range(MAX):
            assert await claim(store, lease=0.05)
            await asyncio.sleep(0.1)
        assert await claim(store) is None  # out of attempts
        swept = await store.sweep(MAX, DEADLINE)
        assert [s["error"]["code"] for s in swept] == ["worker_lost"]
        assert await status(store, row["id"]) == "FAILED"

    with_store(body)


def test_sweep_leaves_healthy_jobs_alone(with_store):
    async def body(db, store):
        await store.submit(new_job(b"%PDF-1"))
        await store.submit(new_job(b"%PDF-2"))
        await claim(store)
        assert await store.sweep(MAX, DEADLINE) == []

    with_store(body)


def test_jobs_past_the_deadline_fail_whether_queued_or_running(with_store):
    async def body(db, store):
        queued, _ = await store.submit(new_job(b"%PDF-1"))
        running, _ = await store.submit(new_job(b"%PDF-2"))
        await claim(store)  # claims the older one
        await asyncio.sleep(0.05)
        swept = await store.sweep(MAX, 0.01)
        assert {s["error"]["code"] for s in swept} == {"deadline_exceeded"} and len(swept) == 2
        assert await status(store, queued["id"]) == "FAILED"
        assert await status(store, running["id"]) == "FAILED"

    with_store(body)


def test_tenant_scoping_in_store(with_store):
    async def body(db, store):
        row, _ = await store.submit(new_job(org="acme"))
        assert await store.get(row["id"], "acme")
        assert await store.get(row["id"], "") is None
        assert await store.get(uuid.uuid4(), "acme") is None

    with_store(body)


def test_identical_documents_are_stored_once(with_store):
    async def body(db, store):
        await store.submit(new_job(org="a"))
        await store.submit(new_job(org="b"))
        async with db.pool.connection() as conn:
            cur = await conn.execute("SELECT count(*) AS n FROM documents")
            assert (await cur.fetchone())["n"] == 1

    with_store(body)
