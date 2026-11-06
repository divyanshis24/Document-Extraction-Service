import asyncio
import logging
import os
import time

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import PostgresDatabase
from app.logging_config import configure_logging
from app.main import create_app

SECRET_DETAIL = "password=hunter2 host=internal-db"


class FakeDb:
    def __init__(self):
        self.error = None
        self.hang = False
        self.opened = False
        self.closed = False

    async def open(self):
        self.opened = True

    async def close(self):
        self.closed = True

    async def migrate(self):
        pass

    async def ping(self):
        if self.hang:
            await asyncio.sleep(30)
        if self.error:
            raise self.error


def settings(**kw):
    defaults = dict(llm_base_url="http://gw/v1", llm_api_key="sk-secret", app_version="abc123",
                    health_db_timeout_seconds=0.2)
    return Settings(**{**defaults, **kw})


@pytest.fixture
def db():
    return FakeDb()


@pytest.fixture
def app_logs():
    """Capture the "app" logger directly. configure_logging() replaces the root
    handlers, which removes pytest's caplog handler and leaves caplog empty."""
    records = []

    class ListHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = ListHandler()
    logger = logging.getLogger("app")
    logger.addHandler(handler)
    yield records
    logger.removeHandler(handler)


@pytest.fixture
def client(db):
    with TestClient(create_app(settings(), db), raise_server_exceptions=False) as c:
        yield c


def test_health_ok_matches_contract(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": "abc123"}


def test_health_503_when_database_fails_without_leaking_details(client, db, app_logs):
    db.error = RuntimeError(SECRET_DETAIL)
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json() == {"status": "unavailable"}
    assert "hunter2" not in r.text
    failures = [rec for rec in app_logs if rec.getMessage() == "health check failed"]
    assert failures, "the failure should be logged"
    assert failures[0].error_type == "RuntimeError"
    assert all("hunter2" not in str(rec.__dict__) for rec in app_logs)


def test_health_503_quickly_when_database_hangs(client, db):
    db.hang = True
    start = time.monotonic()
    r = client.get("/health")
    assert r.status_code == 503
    assert time.monotonic() - start < 2


def test_health_recovers_after_outage(client, db):
    db.error = RuntimeError("down")
    assert client.get("/health").status_code == 503
    db.error = None
    assert client.get("/health").status_code == 200


def test_lifespan_opens_and_closes_database(db):
    with TestClient(create_app(settings(), db)):
        assert db.opened and not db.closed
    assert db.closed


def test_starts_and_warns_without_llm_config(db, app_logs):
    s = settings(llm_base_url="", llm_api_key="")
    assert not s.llm_configured
    with TestClient(create_app(s, db)) as c:
        assert c.get("/health").status_code == 200
    assert any("LLM_BASE_URL" in rec.getMessage() for rec in app_logs)


def test_whitespace_llm_config_counts_as_missing():
    assert not settings(llm_base_url="  ", llm_api_key=" ").llm_configured


def test_llm_config_has_no_built_in_default(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert Settings().llm_base_url == ""
    assert Settings().llm_api_key.get_secret_value() == ""


def test_llm_config_comes_from_environment(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://gateway/v1")
    monkeypatch.setenv("LLM_API_KEY", "k")
    s = Settings()
    assert s.llm_base_url == "http://gateway/v1" and s.llm_configured


def test_secrets_do_not_appear_in_settings_repr():
    s = settings(database_url="postgresql://u:dbpass@h/d")
    assert "sk-secret" not in repr(s) and "dbpass" not in repr(s)


def test_invalid_log_level_does_not_stop_startup(db):
    with TestClient(create_app(settings(log_level="NOPE"), db)) as c:
        assert c.get("/health").status_code == 200


def test_logging_setup_keeps_access_logs_off():
    # Reproduces what uvicorn's --no-access-log does before the app is imported.
    access = logging.getLogger("uvicorn.access")
    saved = access.propagate, access.handlers[:]
    access.handlers.clear()
    access.propagate = False
    try:
        configure_logging("INFO")
        assert access.propagate is False and not access.handlers
    finally:
        access.propagate, access.handlers[:] = saved


def test_unknown_route_uses_error_shape(client):
    r = client.get("/nope")
    assert r.status_code == 404
    assert r.json()["code"] == "not_found"
    assert "message" in r.json()


def test_wrong_method_keeps_allow_header(client):
    r = client.post("/health")
    assert r.status_code == 405
    assert r.json()["code"] == "method_not_allowed"
    assert "GET" in r.headers["allow"]


def test_unhandled_error_is_a_generic_500(db):
    app = create_app(settings(), db)

    @app.get("/boom")
    async def boom():
        raise RuntimeError(SECRET_DETAIL)

    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/boom")
    assert r.status_code == 500
    assert r.json() == {"code": "internal_error", "message": "Internal server error"}


def test_validation_error_returns_400(db):
    app = create_app(settings(), db)

    @app.get("/typed")
    async def typed(n: int):
        return n

    with TestClient(app) as c:
        r = c.get("/typed", params={"n": "x"})
    assert r.status_code == 400
    assert r.json()["code"] == "bad_request"


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")
def test_ping_against_real_postgres():
    async def run():
        d = PostgresDatabase(os.environ["TEST_DATABASE_URL"], acquire_timeout_seconds=5)
        await d.open()
        try:
            await d.ping()
        finally:
            await d.close()

    asyncio.run(run())


def test_ping_fails_cleanly_when_postgres_is_unreachable():
    async def run():
        d = PostgresDatabase("postgresql://u:p@127.0.0.1:1/d", acquire_timeout_seconds=0.3)
        await d.open()
        try:
            with pytest.raises(Exception):
                await d.ping()
        finally:
            await d.close()

    asyncio.run(run())
