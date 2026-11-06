"""HTTP entry point. See contract/openapi.yaml for the interface."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from contextlib import asynccontextmanager
from http import HTTPStatus

import psycopg
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from psycopg_pool import PoolClosed, PoolTimeout
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import ClientDisconnect

from app.config import Settings
from app.db import Database, PostgresDatabase
from app.jobs import JobStore, NewJob, job_view
from app.logging_config import configure_logging
from app.middleware import BodySizeLimitMiddleware, BodyTooLarge
from app.validation import (
    MAX_SCHEMA_BYTES,
    RequestError,
    clean_file_name,
    detect_file_type,
    normalize_org_id,
    parse_field_list,
    validate_webhook_url,
)

log = logging.getLogger("app")

_MULTIPART_OVERHEAD_BYTES = MAX_SCHEMA_BYTES + 64 * 1024
_RETRY_AFTER_SECONDS = "5"


def _error(status: int, code: str, message: str, headers: dict | None = None) -> JSONResponse:
    """Every error uses the contract's {code, message} shape."""
    return JSONResponse(status_code=status, content={"code": code, "message": message}, headers=headers)


async def _migrate_until_done(db: Database, ready: asyncio.Event) -> None:
    while True:
        try:
            await db.migrate()
            ready.set()
            log.info("database schema is ready")
            return
        except Exception as exc:  # noqa: BLE001 - keep trying while the database is away
            log.warning("database migration failed, retrying", extra={"error_type": type(exc).__name__})
            await asyncio.sleep(1)


def create_app(settings: Settings | None = None, database: Database | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level)
    db = database or PostgresDatabase(
        settings.database_url.get_secret_value(),
        acquire_timeout_seconds=settings.health_db_timeout_seconds,
    )
    store = JobStore(db)
    ready = asyncio.Event()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if not settings.llm_configured:
            # Not fatal: the API can still start and report health, and extraction
            # jobs will surface the problem. Never log the values themselves.
            log.warning("LLM_BASE_URL or LLM_API_KEY is not set; extraction will not work")
        await db.open()
        migrator = asyncio.create_task(_migrate_until_done(db, ready))
        log.info("service started", extra={"version": settings.app_version})
        try:
            yield
        finally:
            migrator.cancel()
            await asyncio.gather(migrator, return_exceptions=True)
            await db.close()
            log.info("service stopped")

    app = FastAPI(title="Document Extraction Service", version=settings.app_version, lifespan=lifespan)
    app.add_middleware(BodySizeLimitMiddleware, path="/jobs",
                       max_bytes=settings.max_upload_bytes + _MULTIPART_OVERHEAD_BYTES)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = HTTPStatus(exc.status_code).name.lower()
        return _error(exc.status_code, code, str(exc.detail), getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # The contract specifies 400 for malformed requests; FastAPI defaults to 422.
        return _error(400, "bad_request", str(exc.errors()[:3]))

    @app.exception_handler(RequestError)
    async def _request_error(_: Request, exc: RequestError) -> JSONResponse:
        return _error(exc.status, exc.code, exc.message)

    @app.exception_handler(BodyTooLarge)
    async def _too_large(_: Request, __: BodyTooLarge) -> JSONResponse:
        return _error(413, "file_too_large", "File is larger than the limit")

    @app.exception_handler(ClientDisconnect)
    async def _disconnect(_: Request, __: ClientDisconnect) -> JSONResponse:
        return _error(400, "client_disconnected", "Client disconnected during upload")

    async def _database_unavailable(_: Request, exc: Exception) -> JSONResponse:
        # Log the type only: driver messages can contain connection details.
        log.warning("database unavailable", extra={"error_type": type(exc).__name__})
        return _error(503, "database_unavailable", "Service is temporarily unavailable",
                      {"Retry-After": _RETRY_AFTER_SECONDS})

    for error_type in (psycopg.Error, PoolTimeout, PoolClosed, asyncio.TimeoutError):
        app.add_exception_handler(error_type, _database_unavailable)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error", extra={"path": request.url.path}, exc_info=exc)
        return _error(500, "internal_error", "Internal server error")

    @app.get("/health")
    async def health() -> JSONResponse:
        """200 only when the service can accept jobs: database reachable, schema applied."""

        async def check() -> None:
            await ready.wait()
            await db.ping()

        try:
            await asyncio.wait_for(check(), timeout=settings.health_db_timeout_seconds)
        except Exception as exc:  # noqa: BLE001 - any failure means "not ready"
            log.warning("health check failed", extra={"error_type": type(exc).__name__})
            return JSONResponse(status_code=503, content={"status": "unavailable"})
        return JSONResponse(content={"status": "ok", "version": settings.app_version})

    @app.post("/jobs")
    async def create_job(request: Request) -> JSONResponse:
        org_id = normalize_org_id(request.headers.get("x-organization-id"))

        # Parsed by hand rather than with FastAPI form parameters, which would turn
        # every parsing error, including an oversized body, into the same response.
        async with request.form(max_files=2, max_fields=10, max_part_size=MAX_SCHEMA_BYTES) as form:
            files = form.getlist("file")
            schemas = form.getlist("schema")
            webhooks = form.getlist("webhook_url")
            if len(files) != 1 or not isinstance(files[0], UploadFile):
                raise RequestError(400, "missing_file", "Send exactly one `file` part")
            if len(schemas) != 1:
                raise RequestError(400, "missing_schema", "Send exactly one `schema` part")
            if len(webhooks) > 1 or (webhooks and not isinstance(webhooks[0], str)):
                raise RequestError(400, "invalid_webhook_url", "Send at most one `webhook_url` value")

            # The contract says string, but accept `-F schema=@file.json` as well.
            schema_raw = schemas[0]
            if isinstance(schema_raw, UploadFile):
                schema_raw = await schema_raw.read(MAX_SCHEMA_BYTES + 1)
            field_list, field_list_sha256 = parse_field_list(schema_raw)
            webhook_url = validate_webhook_url(webhooks[0] if webhooks else None)

            upload = files[0]
            file_name = clean_file_name(upload.filename)
            content = await upload.read(settings.max_upload_bytes + 1)

        if len(content) > settings.max_upload_bytes:
            raise RequestError(413, "file_too_large", "File is larger than the limit")
        if not content:
            raise RequestError(400, "empty_file", "The uploaded file is empty")
        file_type = await asyncio.to_thread(detect_file_type, file_name, content)
        document_sha256 = await asyncio.to_thread(lambda: hashlib.sha256(content).hexdigest())

        await asyncio.wait_for(ready.wait(), timeout=settings.health_db_timeout_seconds)
        if await store.active_count() >= settings.max_active_jobs:
            return _error(429, "too_many_jobs", "Too many jobs in the queue", {"Retry-After": "30"})

        row, created = await store.submit(NewJob(
            org_id=org_id, document=content, document_sha256=document_sha256, file_name=file_name,
            file_type=file_type, field_list=field_list, field_list_sha256=field_list_sha256,
            webhook_url=webhook_url,
        ))
        log.info("job accepted" if created else "duplicate submission returned existing job",
                 extra={"job_id": str(row["id"]), "file_type": file_type, "size_bytes": len(content)})
        return JSONResponse(status_code=202, content=job_view(row))

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str, request: Request) -> JSONResponse:
        not_found = _error(404, "not_found", "Job not found")
        try:
            parsed = uuid.UUID(job_id)
            org_id = normalize_org_id(request.headers.get("x-organization-id"))
        except (ValueError, RequestError):
            return not_found  # a malformed id or org can't match any job
        await asyncio.wait_for(ready.wait(), timeout=settings.health_db_timeout_seconds)
        row = await store.get(parsed, org_id)
        return JSONResponse(content=job_view(row)) if row else not_found

    return app


app = create_app()
