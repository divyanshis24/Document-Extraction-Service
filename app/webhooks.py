"""Delivers finished jobs to webhook URLs. At-least-once, separate from job status.

A receiver that is down or wrong can never change a job's outcome, because delivery state
lives in its own table and the job is already terminal before delivery starts.
"""
from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from urllib.parse import urlsplit

import httpx

from app.config import Settings
from app.db import PostgresDatabase
from app.jobs import job_view

log = logging.getLogger("app.webhooks")


class _Blocked(Exception):
    pass


async def check_destination(url: str, allow_private: bool) -> None:
    """Refuse addresses a webhook must never reach. Raises _Blocked if refused.

    This checks at send time, but DNS could still change before the request connects.
    Closing that gap needs an egress proxy, which is a deployment-level control.
    """
    parts = urlsplit(url)
    infos = await asyncio.get_running_loop().getaddrinfo(
        parts.hostname, parts.port or (443 if parts.scheme == "https" else 80), type=socket.SOCK_STREAM)
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%")[0])
        mapped = getattr(address, "ipv4_mapped", None)
        address = mapped or address
        # Link-local includes cloud metadata services such as 169.254.169.254.
        if address.is_link_local or address.is_multicast or address.is_unspecified:
            raise _Blocked("destination address is not allowed")
        # Checked before is_reserved, which is also true for the IPv6 loopback address.
        if address.is_private or address.is_loopback:
            if not allow_private:
                raise _Blocked("private destination addresses are not allowed")
        elif address.is_reserved:
            raise _Blocked("destination address is not allowed")


class WebhookDispatcher:
    def __init__(self, db: PostgresDatabase, settings: Settings,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._db, self._settings = db, settings
        # No redirects: following one could lead to an address the check refused.
        self._http = httpx.AsyncClient(transport=transport, follow_redirects=False,
                                       timeout=settings.webhook_timeout_seconds)

    async def close(self) -> None:
        await self._http.aclose()

    async def _claim_due(self, limit: int) -> list[dict]:
        """Hide the row for longer than a request can take, so nothing else sends it
        meanwhile. If this worker crashes, the row becomes due again after that time."""
        visible_after = self._settings.webhook_timeout_seconds + 30
        async with self._db.pool.connection() as conn:
            cur = await conn.execute(
                """
                WITH due AS (
                    SELECT w.job_id, w.url FROM job_webhooks w JOIN jobs j ON j.id = w.job_id
                    WHERE w.delivered_at IS NULL AND w.gave_up_at IS NULL AND w.next_attempt_at <= now()
                      AND j.status IN ('SUCCESS', 'PARTIAL', 'FAILED')
                    ORDER BY w.next_attempt_at
                    LIMIT %(limit)s
                    FOR UPDATE OF w SKIP LOCKED
                ), claimed AS (
                    UPDATE job_webhooks w SET attempts = w.attempts + 1,
                        next_attempt_at = now() + make_interval(secs => %(visible_after)s::double precision)
                    FROM due WHERE w.job_id = due.job_id AND w.url = due.url
                    RETURNING w.job_id, w.url, w.attempts
                )
                SELECT c.job_id, c.url, c.attempts, j.id, j.status, j.stage, j.created_at,
                       j.updated_at, j.result, j.error
                FROM claimed c JOIN jobs j ON j.id = c.job_id
                """,
                {"limit": limit, "visible_after": visible_after},
            )
            return await cur.fetchall()

    async def _mark_delivered(self, row: dict) -> None:
        async with self._db.pool.connection() as conn:
            await conn.execute(
                "UPDATE job_webhooks SET delivered_at = now(), last_error = NULL WHERE job_id = %s AND url = %s",
                (row["job_id"], row["url"]),
            )

    async def _mark_failed(self, row: dict, error: str, give_up: bool) -> None:
        async with self._db.pool.connection() as conn:
            await conn.execute(
                "UPDATE job_webhooks SET last_error = %s,"
                " gave_up_at = CASE WHEN %s THEN now() ELSE NULL END,"
                " next_attempt_at = now() + make_interval(secs => least("
                "     %s::double precision * power(2, attempts - 1), 3600))"
                " WHERE job_id = %s AND url = %s",
                (error, give_up, self._settings.webhook_backoff_base_seconds, row["job_id"], row["url"]),
            )

    async def _deliver(self, row: dict) -> None:
        # The URL can carry a secret token, so log only the job and the host.
        context = {"job_id": str(row["job_id"]), "attempt": row["attempts"],
                   "webhook_host": urlsplit(row["url"]).hostname}
        give_up = row["attempts"] >= self._settings.webhook_max_attempts
        try:
            await check_destination(row["url"], self._settings.webhook_allow_private)
            response = await self._http.post(row["url"], json=job_view(row), headers={
                "X-Job-Id": str(row["job_id"]), "X-Delivery-Attempt": str(row["attempts"])})
        except _Blocked as exc:
            # Retrying won't help, so stop now.
            await self._mark_failed(row, str(exc), give_up=True)
            log.error("webhook refused", extra={**context, "reason": str(exc)})
            return
        except (httpx.HTTPError, OSError, ValueError) as exc:
            await self._mark_failed(row, type(exc).__name__, give_up)
            log.warning("webhook delivery failed", extra={**context, "reason": type(exc).__name__, "gave_up": give_up})
            return
        if 200 <= response.status_code < 300:
            await self._mark_delivered(row)
            log.info("webhook delivered", extra=context)
            return
        await self._mark_failed(row, f"HTTP {response.status_code}", give_up)
        log.warning("webhook delivery failed",
                    extra={**context, "reason": f"HTTP {response.status_code}", "gave_up": give_up})

    async def run_once(self, limit: int = 10) -> int:
        rows = await self._claim_due(limit)
        results = await asyncio.gather(*[self._deliver(r) for r in rows], return_exceptions=True)
        for row, outcome in zip(rows, results):
            if isinstance(outcome, Exception):
                log.warning("webhook bookkeeping failed; it will be retried",
                            extra={"job_id": str(row["job_id"]), "error_type": type(outcome).__name__})
        return len(rows)
