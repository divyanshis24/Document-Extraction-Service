import asyncio
import os

import psycopg
import pytest

from app.db import PostgresDatabase
from app.jobs import JobStore


@pytest.fixture
def pg_url():
    """A real, empty Postgres. Set TEST_DATABASE_URL to run these tests; it gets wiped."""
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("DROP SCHEMA public CASCADE")
        conn.execute("CREATE SCHEMA public")
    return url


@pytest.fixture
def with_store(pg_url):
    """Run an async test body with a migrated store: with_store(lambda db, store: ...)."""

    def run(body):
        async def main():
            db = PostgresDatabase(pg_url, acquire_timeout_seconds=10, max_size=12)
            await db.open()
            try:
                await db.migrate()
                return await body(db, JobStore(db))
            finally:
                await db.close()

        return asyncio.run(main())

    return run
