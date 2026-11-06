import asyncio
import hashlib
import json
import logging

import httpx
import pytest

from app.config import Settings
from app.extractor import StubExtractor
from app.jobs import NewJob
from app.webhooks import WebhookDispatcher, _Blocked, check_destination
from app.worker import Worker

URL = "http://127.0.0.1:9/hook?token=SECRET-TOKEN"
LEASE, MAX, DEADLINE = 60, 3, 1740


def settings(**kw):
    return Settings(**{**dict(webhook_backoff_base_seconds=0.0, webhook_max_attempts=3), **kw})


def new_job(content=b"%PDF-doc", url=URL):
    return NewJob(org_id="", document=content, document_sha256=hashlib.sha256(content).hexdigest(), file_name="a.pdf",
                  file_type="pdf", field_list={"fields": {"a": {}}}, field_list_sha256="s", webhook_url=url)


async def finished_job(store, status="SUCCESS", **kw):
    row, _ = await store.submit(new_job(**kw))
    job = await store.claim(LEASE, MAX, DEADLINE)
    if status == "FAILED":
        await store.fail(job, "bad_doc", "Unreadable")
    else:
        await store.complete(job, status, {"entries": [], "metadata": {}})
    return row["id"]


async def state(db):
    async with db.pool.connection() as conn:
        cur = await conn.execute("SELECT attempts, delivered_at, gave_up_at, last_error FROM job_webhooks")
        return await cur.fetchall()


def dispatcher(db, handler, **kw):
    return WebhookDispatcher(db, settings(**kw), transport=httpx.MockTransport(handler))


def test_nothing_is_sent_before_the_job_finishes(with_store):
    async def body(db, store):
        sent = []
        await store.submit(new_job())
        d = dispatcher(db, lambda r: sent.append(r) or httpx.Response(200))
        assert await d.run_once() == 0
        await store.claim(LEASE, MAX, DEADLINE)
        assert await d.run_once() == 0 and not sent
        await d.close()

    with_store(body)


@pytest.mark.parametrize("status,key", [("SUCCESS", "result"), ("PARTIAL", "result"), ("FAILED", "error")])
def test_delivers_final_job_once(with_store, status, key):
    async def body(db, store):
        job_id = await finished_job(store, status)
        sent = []
        d = dispatcher(db, lambda r: sent.append(r) or httpx.Response(204))
        assert await d.run_once() == 1
        assert await d.run_once() == 0  # delivered, so not sent again
        payload = json.loads(sent[0].content)
        assert payload["job_id"] == str(job_id) and payload["status"] == status and key in payload
        assert sent[0].headers["x-job-id"] == str(job_id)
        assert (await state(db))[0]["delivered_at"] is not None
        await d.close()

    with_store(body)


@pytest.mark.parametrize("failure", [httpx.Response(500), httpx.Response(404), httpx.Response(302),
                                     httpx.ConnectError("refused"), httpx.ReadTimeout("slow")], ids=str)
def test_failed_delivery_is_retried_then_succeeds(with_store, failure):
    async def body(db, store):
        await finished_job(store)
        steps = [failure, httpx.Response(200)]

        def handler(request):
            step = steps.pop(0)
            if isinstance(step, Exception):
                raise step
            return step

        d = dispatcher(db, handler)
        await d.run_once()
        assert (await state(db))[0]["delivered_at"] is None
        await d.run_once()
        row = (await state(db))[0]
        assert row["delivered_at"] is not None and row["attempts"] == 2 and row["last_error"] is None
        await d.close()

    with_store(body)


def test_gives_up_after_max_attempts_and_job_is_unchanged(with_store):
    async def body(db, store):
        job_id = await finished_job(store)
        before = await store.get(job_id, "")
        d = dispatcher(db, lambda r: httpx.Response(500))
        for _ in range(3):
            assert await d.run_once() == 1
        assert await d.run_once() == 0
        row = (await state(db))[0]
        assert row["gave_up_at"] is not None and row["attempts"] == 3 and row["last_error"] == "HTTP 500"
        assert await store.get(job_id, "") == before
        await d.close()

    with_store(body)


def test_backoff_delays_the_next_attempt(with_store):
    async def body(db, store):
        await finished_job(store)
        d = dispatcher(db, lambda r: httpx.Response(500), webhook_backoff_base_seconds=30)
        assert await d.run_once() == 1
        assert await d.run_once() == 0  # not due yet
        await d.close()

    with_store(body)


def test_concurrent_dispatchers_send_each_delivery_once(with_store):
    async def body(db, store):
        for i in range(5):
            await finished_job(store, content=f"%PDF-{i}".encode())
        sent = []

        async def handler(request):
            sent.append(request)
            await asyncio.sleep(0.05)
            return httpx.Response(200)

        a, b = dispatcher(db, handler), dispatcher(db, handler)
        await asyncio.gather(a.run_once(), b.run_once(), a.run_once(), b.run_once())
        assert len(sent) == 5
        await a.close()
        await b.close()

    with_store(body)


def test_slow_delivery_is_not_sent_again_while_in_flight(with_store):
    """A receiver slower than the retry interval must not get an overlapping retry."""

    async def body(db, store):
        await finished_job(store)
        sent, release = [], asyncio.Event()

        async def handler(request):
            sent.append(request)
            await release.wait()
            return httpx.Response(200)

        d = dispatcher(db, handler)  # backoff of 0, so any retry would be due at once
        first = asyncio.create_task(d.run_once())
        await asyncio.sleep(0.1)
        assert await d.run_once() == 0
        release.set()
        await first
        assert len(sent) == 1
        await d.close()

    with_store(body)


def test_late_duplicate_with_a_new_webhook_still_gets_notified(with_store):
    async def body(db, store):
        await finished_job(store)
        sent = []
        d = dispatcher(db, lambda r: sent.append(str(r.url)) or httpx.Response(200))
        await d.run_once()
        await store.submit(new_job(url="http://127.0.0.1:9/second"))
        await d.run_once()
        assert sent[-1] == "http://127.0.0.1:9/second" and len(sent) == 2
        await d.close()

    with_store(body)


def test_blocked_destination_stops_immediately(with_store):
    async def body(db, store):
        await finished_job(store, url="http://169.254.169.254/latest/meta-data")
        sent = []
        d = dispatcher(db, lambda r: sent.append(r) or httpx.Response(200))
        await d.run_once()
        row = (await state(db))[0]
        assert not sent and row["gave_up_at"] is not None and row["attempts"] == 1
        await d.close()

    with_store(body)


def test_url_secrets_are_not_logged(with_store, caplog):
    async def body(db, store):
        await finished_job(store)
        d = dispatcher(db, lambda r: httpx.Response(500))
        await d.run_once()
        await d.close()

    with caplog.at_level(logging.INFO):
        with_store(body)
    assert "SECRET-TOKEN" not in caplog.text


@pytest.mark.parametrize("url", ["http://169.254.169.254/", "http://[fe80::1]/", "http://0.0.0.0/", "http://224.0.0.1/",
                                 "http://[::ffff:169.254.169.254]/"])
def test_dangerous_addresses_are_always_refused(url):
    with pytest.raises(_Blocked):
        asyncio.run(check_destination(url, allow_private=True))


@pytest.mark.parametrize("url", ["http://127.0.0.1/", "http://10.0.0.1/", "http://192.168.1.1/", "http://[::1]/"])
def test_private_addresses_follow_the_setting(url):
    asyncio.run(check_destination(url, allow_private=True))
    with pytest.raises(_Blocked):
        asyncio.run(check_destination(url, allow_private=False))


def test_worker_delivers_webhook_end_to_end(with_store):
    async def body(db, store):
        await store.submit(new_job())
        sent = asyncio.Event()

        def handler(request):
            sent.set()
            return httpx.Response(200)

        s = Settings(worker_poll_seconds=0.02, webhook_poll_seconds=0.02, sweep_interval_seconds=1)
        worker = Worker(db, store, StubExtractor(), s, webhooks=WebhookDispatcher(db, s, transport=httpx.MockTransport(handler)))
        task = asyncio.create_task(worker.run())
        try:
            await asyncio.wait_for(sent.wait(), 5)
        finally:
            worker.stop.set()
            await asyncio.wait_for(task, 5)

    with_store(body)


def test_webhook_failure_cannot_stop_jobs_being_processed(with_store):
    async def body(db, store):
        first, _ = await store.submit(new_job(b"%PDF-1"))
        second, _ = await store.submit(new_job(b"%PDF-2"))

        def handler(request):
            raise httpx.ConnectError("receiver is down")

        s = Settings(worker_poll_seconds=0.02, webhook_poll_seconds=0.02, webhook_backoff_base_seconds=0)
        worker = Worker(db, store, StubExtractor(), s, webhooks=WebhookDispatcher(db, s, transport=httpx.MockTransport(handler)))
        task = asyncio.create_task(worker.run())
        try:
            async with asyncio.timeout(5):
                while (await store.get(second["id"], ""))["status"] == "PENDING":
                    await asyncio.sleep(0.02)
                while (await store.get(second["id"], ""))["status"] != "PARTIAL":
                    await asyncio.sleep(0.02)
            assert (await store.get(first["id"], ""))["status"] == "PARTIAL"
        finally:
            worker.stop.set()
            await asyncio.wait_for(task, 5)

    with_store(body)
