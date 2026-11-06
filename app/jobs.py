"""Every SQL statement that touches a job. The queue is the jobs table itself."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from psycopg.types.json import Jsonb

from app.db import PostgresDatabase

_VIEW_COLUMNS = "id, status, stage, created_at, updated_at, result, error"


@dataclass(frozen=True)
class NewJob:
    org_id: str
    document: bytes
    document_sha256: str
    file_name: str
    file_type: str
    field_list: dict
    field_list_sha256: str
    webhook_url: str | None = None


@dataclass(frozen=True)
class ClaimedJob:
    id: uuid.UUID
    lease_token: uuid.UUID
    file_name: str
    file_type: str
    field_list: dict
    document_sha256: str
    attempts: int
    elapsed_seconds: float
    remaining_seconds: float


def job_view(row: dict) -> dict:
    """Shape a row as the contract's JobStatus object."""

    def iso(value: datetime) -> str:
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    view = {
        "job_id": str(row["id"]),
        "status": row["status"],
        "stage": row["stage"],
        "created_at": iso(row["created_at"]),
        "updated_at": iso(row["updated_at"]),
    }
    if row["status"] in ("SUCCESS", "PARTIAL"):
        view["result"] = row["result"]
    if row["status"] == "FAILED":
        view["error"] = row["error"]
    return view


class JobStore:
    def __init__(self, db: PostgresDatabase) -> None:
        self._db = db

    # ----- API side -----

    async def active_count(self) -> int:
        async with self._db.pool.connection() as conn:
            cur = await conn.execute("SELECT count(*) AS n FROM jobs WHERE status IN ('PENDING','RUNNING')")
            return (await cur.fetchone())["n"]

    async def submit(self, new: NewJob) -> tuple[dict, bool]:
        """Return (job row, created). An existing live job is returned instead of a new one."""
        # Loops only when another submission wins a race. The next pass then finds its
        # job, or inserts if that job has since failed.
        for _ in range(5):
            async with self._db.pool.connection() as conn, conn.transaction():
                # Check first, so a duplicate never rewrites up to 50 MB.
                cur = await conn.execute(
                    f"SELECT {_VIEW_COLUMNS} FROM jobs WHERE org_id = %s AND document_sha256 = %s"
                    " AND field_list_sha256 = %s AND status <> 'FAILED'",
                    (new.org_id, new.document_sha256, new.field_list_sha256),
                )
                row = await cur.fetchone()
                created = False
                if row is None:
                    await conn.execute(
                        "INSERT INTO documents (sha256, content, size_bytes) VALUES (%s, %s, %s)"
                        " ON CONFLICT (sha256) DO NOTHING",
                        (new.document_sha256, new.document, len(new.document)),
                    )
                    cur = await conn.execute(
                        "INSERT INTO jobs (id, org_id, document_sha256, file_name, file_type,"
                        " field_list, field_list_sha256) VALUES (%s, %s, %s, %s, %s, %s, %s)"
                        " ON CONFLICT (org_id, document_sha256, field_list_sha256)"
                        " WHERE status <> 'FAILED' DO NOTHING"
                        f" RETURNING {_VIEW_COLUMNS}",
                        (uuid.uuid4(), new.org_id, new.document_sha256, new.file_name, new.file_type,
                         Jsonb(new.field_list), new.field_list_sha256),
                    )
                    row = await cur.fetchone()
                    created = row is not None
                if row is not None:
                    if new.webhook_url:
                        await conn.execute(
                            "INSERT INTO job_webhooks (job_id, url) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                            (row["id"], new.webhook_url),
                        )
                    return row, created
        raise RuntimeError("could not submit job after repeated conflicts")

    async def get(self, job_id: uuid.UUID, org_id: str) -> dict | None:
        # The org is part of the lookup, so another tenant's job is simply not found.
        async with self._db.pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT {_VIEW_COLUMNS} FROM jobs WHERE id = %s AND org_id = %s", (job_id, org_id)
            )
            return await cur.fetchone()

    # ----- Worker side -----

    async def claim(self, lease_seconds: float, max_attempts: int, deadline_seconds: float) -> ClaimedJob | None:
        """Atomically take the oldest runnable job, including any whose worker died."""
        token = uuid.uuid4()
        async with self._db.pool.connection() as conn:
            cur = await conn.execute(
                """
                UPDATE jobs SET status = 'RUNNING', stage = 'STARTING', attempts = attempts + 1,
                    lease_token = %(token)s,
                    lease_expires_at = now() + make_interval(secs => %(lease)s::double precision),
                    updated_at = now()
                WHERE id = (
                    SELECT id FROM jobs
                    WHERE attempts < %(max)s
                      AND (status = 'PENDING' OR (status = 'RUNNING' AND lease_expires_at < now()))
                    ORDER BY created_at, id
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                RETURNING id, file_name, file_type, field_list, document_sha256, attempts,
                    extract(epoch FROM now() - created_at)::double precision AS elapsed,
                    extract(epoch FROM created_at
                        + make_interval(secs => %(deadline)s::double precision) - now())::double precision
                        AS remaining
                """,
                {"token": token, "lease": lease_seconds, "max": max_attempts, "deadline": deadline_seconds},
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return ClaimedJob(
            id=row["id"], lease_token=token, file_name=row["file_name"], file_type=row["file_type"],
            field_list=row["field_list"], document_sha256=row["document_sha256"], attempts=row["attempts"],
            elapsed_seconds=row["elapsed"], remaining_seconds=row["remaining"],
        )

    async def load_document(self, sha256: str) -> bytes:
        async with self._db.pool.connection() as conn:
            cur = await conn.execute("SELECT content FROM documents WHERE sha256 = %s", (sha256,))
            return bytes((await cur.fetchone())["content"])

    # Every write below is fenced on lease_token: a worker that lost its job can't
    # touch it, even if it wakes up after another worker has taken over.

    async def heartbeat(self, job: ClaimedJob, lease_seconds: float) -> bool:
        async with self._db.pool.connection() as conn:
            cur = await conn.execute(
                "UPDATE jobs SET lease_expires_at = now() + make_interval(secs => %s::double precision)"
                " WHERE id = %s AND lease_token = %s AND status = 'RUNNING'",
                (lease_seconds, job.id, job.lease_token),
            )
            return cur.rowcount == 1

    async def set_stage(self, job: ClaimedJob, stage: str) -> bool:
        async with self._db.pool.connection() as conn:
            cur = await conn.execute(
                "UPDATE jobs SET stage = %s, updated_at = now()"
                " WHERE id = %s AND lease_token = %s AND status = 'RUNNING'",
                (stage, job.id, job.lease_token),
            )
            return cur.rowcount == 1

    async def complete(self, job: ClaimedJob, status: str, result: dict) -> bool:
        return await self._finish(job, status, result=result)

    async def fail(self, job: ClaimedJob, code: str, message: str) -> bool:
        return await self._finish(job, "FAILED", error={"code": code, "message": message})

    async def _finish(self, job: ClaimedJob, status: str, result: dict | None = None,
                      error: dict | None = None) -> bool:
        async with self._db.pool.connection() as conn:
            cur = await conn.execute(
                "UPDATE jobs SET status = %s, stage = NULL, result = %s, error = %s, lease_token = NULL,"
                " lease_expires_at = NULL, finished_at = now(), updated_at = now()"
                " WHERE id = %s AND lease_token = %s AND status = 'RUNNING'",
                (status, Jsonb(result) if result is not None else None,
                 Jsonb(error) if error is not None else None, job.id, job.lease_token),
            )
            return cur.rowcount == 1

    async def release(self, job: ClaimedJob) -> bool:
        """Hand a job back on graceful shutdown. A deploy doesn't use up an attempt."""
        async with self._db.pool.connection() as conn:
            cur = await conn.execute(
                "UPDATE jobs SET status = 'PENDING', stage = NULL, lease_token = NULL,"
                " lease_expires_at = NULL, attempts = greatest(attempts - 1, 0), updated_at = now()"
                " WHERE id = %s AND lease_token = %s AND status = 'RUNNING'",
                (job.id, job.lease_token),
            )
            return cur.rowcount == 1

    async def sweep(self, max_attempts: int, deadline_seconds: float) -> list[dict]:
        """Fail jobs that can never finish. Any number of workers can run this at once."""
        async with self._db.pool.connection() as conn:
            cur = await conn.execute(
                """
                UPDATE jobs SET status = 'FAILED', stage = NULL, lease_token = NULL, lease_expires_at = NULL,
                    finished_at = now(), updated_at = now(),
                    error = CASE
                        WHEN created_at + make_interval(secs => %(deadline)s::double precision) < now()
                        THEN jsonb_build_object('code', 'deadline_exceeded',
                                                'message', 'Job did not finish before its deadline')
                        ELSE jsonb_build_object('code', 'worker_lost',
                                                'message', 'Workers repeatedly died while processing this job')
                    END
                WHERE status IN ('PENDING', 'RUNNING') AND (
                    created_at + make_interval(secs => %(deadline)s::double precision) < now()
                    OR (status = 'RUNNING' AND lease_expires_at < now() AND attempts >= %(max)s)
                )
                RETURNING id, error
                """,
                {"deadline": deadline_seconds, "max": max_attempts},
            )
            return await cur.fetchall()
