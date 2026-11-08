"""Background worker. Run as many copies as needed: `docker compose up --scale worker=N`.

Each copy runs several jobs at once, and the database hands each job to exactly one of
them. Workers hold no state of their own, so adding copies adds throughput.
"""
from __future__ import annotations

import asyncio
import logging
import random
import signal
import time

import psycopg
from psycopg_pool import PoolClosed, PoolTimeout

from app.config import Settings
from app.db import PostgresDatabase
from app.extraction.extractor import RealExtractor
from app.errors import ExtractionError
from app.extractor import Extractor, ExtractionOutput, StubExtractor
from app.jobs import ClaimedJob, JobStore
from app.llm import CooldownGate, LLMClient
from app.logging_config import configure_logging
from app.result import build_result
from app.webhooks import WebhookDispatcher

log = logging.getLogger("app.worker")

_DB_ERRORS = (psycopg.Error, PoolTimeout, PoolClosed)
# Errors that may succeed on a retry. A DataError or IntegrityError fails the same way every
# time, so retrying it or waiting for the lease to expire only reruns paid extraction.
_TRANSIENT_DB_ERRORS = (psycopg.OperationalError, psycopg.InterfaceError, PoolTimeout, PoolClosed)
_LOAD_RETRIES = 3
_FINISH_RETRIES = 5


class Worker:
    def __init__(self, db: PostgresDatabase, store: JobStore, extractor: Extractor, settings: Settings,
                 webhooks: WebhookDispatcher | None = None) -> None:
        self._db, self._store, self._extractor, self._settings = db, store, extractor, settings
        self._webhooks = webhooks or WebhookDispatcher(db, settings)
        self.stop = asyncio.Event()

    async def run(self) -> None:
        await self._migrate()
        if self.stop.is_set():
            return
        log.info("worker started", extra={"concurrency": self._settings.worker_concurrency})
        slots = [self._slot() for _ in range(self._settings.worker_concurrency)]
        try:
            await asyncio.gather(self._sweeper(), self._webhook_loop(), *slots)
        finally:
            await self._webhooks.close()
        log.info("worker stopped")

    async def _migrate(self) -> None:
        while not self.stop.is_set():
            try:
                await self._db.migrate()
                return
            except Exception as exc:  # noqa: BLE001 - keep trying while the database is away
                log.warning("database not ready", extra={"error_type": type(exc).__name__})
                await self._sleep(1.0)

    async def _sleep(self, seconds: float) -> None:
        """Sleep that ends early on shutdown."""
        try:
            await asyncio.wait_for(self.stop.wait(), timeout=seconds)
        except TimeoutError:
            pass

    async def _slot(self) -> None:
        s = self._settings
        while not self.stop.is_set():
            try:
                job = await self._store.claim(s.job_lease_seconds, s.job_max_attempts, s.job_deadline_seconds)
            except _DB_ERRORS as exc:
                log.warning("could not claim a job", extra={"error_type": type(exc).__name__})
                await self._sleep(1.0)
                continue
            if job is None:
                # Jitter stops idle workers from all polling at the same moment.
                await self._sleep(s.worker_poll_seconds * random.uniform(0.75, 1.25))
                continue
            try:
                await self._process(job)
            except Exception as exc:  # noqa: BLE001 - one bad job must never stop this slot
                log.exception("worker error while processing job", extra={"job_id": str(job.id)})
                if not isinstance(exc, _TRANSIENT_DB_ERRORS):
                    # A bug, not an outage: rerunning would fail the same way after paying
                    # for the extraction again. Fail it now. Fencing makes this a no-op if
                    # the job was already finished or taken over.
                    try:
                        await self._store.fail(job, "internal_error", "Internal error while processing the job")
                    except Exception:  # noqa: BLE001 - the lease expiring is the fallback
                        log.warning("could not mark job failed after an internal error", extra={"job_id": str(job.id)})

    async def _sweeper(self) -> None:
        s = self._settings
        while not self.stop.is_set():
            try:
                for row in await self._store.sweep(s.job_max_attempts, s.job_deadline_seconds):
                    log.error("job failed by sweeper",
                              extra={"job_id": str(row["id"]), "error_code": row["error"]["code"]})
            except _DB_ERRORS as exc:
                log.warning("sweep failed", extra={"error_type": type(exc).__name__})
            await self._sleep(s.sweep_interval_seconds)

    async def _webhook_loop(self) -> None:
        while not self.stop.is_set():
            try:
                delivered = await self._webhooks.run_once()
            except _DB_ERRORS as exc:
                log.warning("webhook poll failed", extra={"error_type": type(exc).__name__})
                delivered = 0
            if not delivered:
                await self._sleep(self._settings.webhook_poll_seconds)

    async def _heartbeat(self, job: ClaimedJob, lease_lost: asyncio.Event) -> None:
        s = self._settings
        while True:
            await asyncio.sleep(s.job_lease_seconds / 3)
            try:
                if not await self._store.heartbeat(job, s.job_lease_seconds):
                    lease_lost.set()
                    return
            except _DB_ERRORS as exc:
                # Unknown, not lost. Keep working: fencing rejects the final write if
                # another worker took over in the meantime.
                log.warning("heartbeat failed", extra={"job_id": str(job.id), "error_type": type(exc).__name__})

    async def _load_document(self, job: ClaimedJob) -> bytes:
        # A blip here would otherwise cost an attempt and a full lease timeout.
        for attempt in range(_LOAD_RETRIES):
            try:
                return await self._store.load_document(job.document_sha256)
            except _TRANSIENT_DB_ERRORS as exc:
                if attempt == _LOAD_RETRIES - 1:
                    raise
                log.warning("could not load document, retrying", extra={"job_id": str(job.id), "error_type": type(exc).__name__})
                await asyncio.sleep(1 + attempt)
        raise AssertionError("unreachable")

    async def _extract(self, job: ClaimedJob) -> ExtractionOutput:
        document = await self._load_document(job)

        async def progress(stage: str) -> None:
            # Stage is advisory, so a database blip here must not end the job.
            try:
                await self._store.set_stage(job, stage)
            except _DB_ERRORS as exc:
                log.warning("could not update stage", extra={"job_id": str(job.id), "error_type": type(exc).__name__})

        return await self._extractor.extract(job, document, progress)

    async def _process(self, job: ClaimedJob) -> None:
        job_log = {"job_id": str(job.id), "attempt": job.attempts, "file_type": job.file_type}
        log.info("job started", extra={**job_log, "queued_seconds": round(job.elapsed_seconds, 3)})
        started = time.monotonic()
        lease_lost = asyncio.Event()

        heartbeat = asyncio.create_task(self._heartbeat(job, lease_lost))
        try:
            await self._run_and_finish(job, lease_lost, job_log, started)
        finally:
            # Stops only after the outcome is saved, so a slow save can't lose the lease
            # and cause finished, already paid-for work to be redone.
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

    async def _run_and_finish(self, job: ClaimedJob, lease_lost: asyncio.Event, job_log: dict, started: float) -> None:
        work = asyncio.create_task(self._extract(job))
        stop_wait = asyncio.create_task(self.stop.wait())
        lost_wait = asyncio.create_task(lease_lost.wait())
        try:
            await asyncio.wait({work, stop_wait, lost_wait}, timeout=max(job.remaining_seconds, 0),
                               return_when=asyncio.FIRST_COMPLETED)
        finally:
            stop_wait.cancel()
            lost_wait.cancel()
            if not work.done():
                work.cancel()
            await asyncio.gather(stop_wait, lost_wait, work, return_exceptions=True)

        if not work.cancelled():
            await self._finish_work(job, work, job_log, started)
        elif lease_lost.is_set():
            # Another worker owns this job now, so write nothing.
            log.warning("lease lost, abandoning job", extra=job_log)
        elif self.stop.is_set():
            try:
                released = await self._store.release(job)
                log.info("job released for shutdown", extra={**job_log, "released": released})
            except _DB_ERRORS as exc:
                # The database may be stopping too. The lease expiring has the same effect.
                log.warning("could not release job, its lease will expire",
                            extra={**job_log, "error_type": type(exc).__name__})
        else:
            await self._fail(job, "deadline_exceeded", "Job did not finish before its deadline", job_log)

    async def _finish_work(self, job: ClaimedJob, work: asyncio.Task, job_log: dict, started: float) -> None:
        error = work.exception()
        if isinstance(error, _TRANSIENT_DB_ERRORS):
            # Our infrastructure failed, not the document, so the job must not end as
            # FAILED. Leave it: the lease expires and another attempt picks it up.
            log.warning("database error during job, it will be retried",
                        extra={**job_log, "error_type": type(error).__name__})
            return
        if isinstance(error, ExtractionError):
            await self._fail(job, error.code, error.message, job_log)
            return
        if error is not None:
            # Log the details, but give the client only a generic message: exception
            # text can include document contents or provider responses.
            log.error("extraction crashed", extra=job_log, exc_info=error)
            await self._fail(job, "extraction_error",
                             f"Extraction failed ({type(error).__name__}); see logs for this job id", job_log)
            return

        output = work.result()
        latency_ms = int((job.elapsed_seconds + time.monotonic() - started) * 1000)
        status = "PARTIAL" if output.partial else "SUCCESS"
        try:
            result = build_result(output.entries, list(job.field_list["fields"]), output.metadata, latency_ms)
            written = await self._with_retry(lambda: self._store.complete(job, status, result), job_log)
        except Exception as exc:  # noqa: BLE001 - see below
            if isinstance(exc, _TRANSIENT_DB_ERRORS):
                raise
            # The output can't be turned into a valid, storable result. Rerunning would
            # produce the same output, so fail now rather than pay for it three times.
            log.error("could not build or store the result", extra=job_log, exc_info=exc)
            await self._fail(job, "result_invalid", "The extracted result could not be stored", job_log)
            return
        log.info("job finished" if written else "result discarded, job no longer owned",
                 extra={**job_log, "status": status, "latency_ms": latency_ms,
                        "entries": len(result["entries"]), "cost_usd": result["metadata"]["cost_usd"]})

    async def _fail(self, job: ClaimedJob, code: str, message: str, job_log: dict) -> None:
        written = await self._with_retry(lambda: self._store.fail(job, code, message), job_log)
        log.error("job failed" if written else "failure discarded, job no longer owned",
                  extra={**job_log, "error_code": code})

    async def _with_retry(self, write, job_log: dict) -> bool:
        """Don't lose finished, already paid-for work to a brief database outage."""
        for attempt in range(_FINISH_RETRIES):
            try:
                return await write()
            except _TRANSIENT_DB_ERRORS as exc:
                log.warning("could not save job outcome, retrying",
                            extra={**job_log, "error_type": type(exc).__name__})
                await asyncio.sleep(min(2 ** attempt, 10))
        log.error("giving up saving job outcome; the lease will expire and the job will rerun", extra=job_log)
        return False


async def main() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    db = PostgresDatabase(settings.database_url.get_secret_value(),
                          acquire_timeout_seconds=5.0,
                          max_size=max(settings.worker_concurrency + 2, 5))
    client = LLMClient(settings, CooldownGate(db))
    extractor: Extractor = (StubExtractor(settings.stub_extractor_delay_seconds) if settings.extractor == "stub"
                            else RealExtractor(client, settings))
    if settings.extractor != "stub" and not settings.llm_configured:
        log.warning("LLM_BASE_URL or LLM_API_KEY is not set; jobs will fail with llm_not_configured")
    worker = Worker(db, JobStore(db), extractor, settings)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, worker.stop.set)
    await db.open()
    try:
        await worker.run()
    finally:
        await client.close()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
