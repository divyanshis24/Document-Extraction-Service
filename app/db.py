"""Postgres connection pool."""
from __future__ import annotations

from typing import Protocol

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.migrations import apply_migrations


class Database(Protocol):
    async def open(self) -> None: ...
    async def close(self) -> None: ...
    async def migrate(self) -> None: ...
    async def ping(self) -> None: ...


class PostgresDatabase:
    def __init__(self, url: str, acquire_timeout_seconds: float = 2.0, max_size: int = 5) -> None:
        self._acquire_timeout = acquire_timeout_seconds
        self._pool = AsyncConnectionPool(
            url,
            min_size=1,
            max_size=max_size,
            open=False,
            kwargs={"row_factory": dict_row},
            timeout=acquire_timeout_seconds,
            # Test each connection on checkout. Without this, the first request
            # after a Postgres restart gets a dead connection from the pool.
            check=AsyncConnectionPool.check_connection,
        )

    @property
    def pool(self) -> AsyncConnectionPool:
        return self._pool

    async def open(self) -> None:
        # wait=False lets the process start while Postgres is down. The pool keeps
        # reconnecting in the background and /health reports the real state.
        await self._pool.open(wait=False)

    async def close(self) -> None:
        await self._pool.close()

    async def migrate(self) -> None:
        async with self._pool.connection() as conn:
            await apply_migrations(conn)

    async def ping(self) -> None:
        async with self._pool.connection(timeout=self._acquire_timeout) as conn:
            await conn.execute("SELECT 1")
