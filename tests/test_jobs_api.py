import json
import time
import uuid

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

SCHEMA = json.dumps({"name": "t", "fields": {"a": {"description": "d"}, "b": {"primary": True}}})
PDF = b"%PDF-1.7 body"


@pytest.fixture
def make_client(pg_url):
    clients = []

    def make(**kw):
        s = Settings(database_url=pg_url, llm_base_url="http://gw", llm_api_key="k", **kw)
        c = TestClient(create_app(s), raise_server_exceptions=False)
        c.__enter__()
        clients.append(c)
        return c

    yield make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client):
    return make_client()


def post(c, content=PDF, name="a.pdf", schema=SCHEMA, headers=None, **data):
    return c.post("/jobs", files={"file": (name, content)}, data={"schema": schema, **data}, headers=headers or {})


def test_post_returns_contract_shape_and_get_matches(client):
    r = post(client)
    assert r.status_code == 202
    body = r.json()
    uuid.UUID(body["job_id"])
    assert body["status"] == "PENDING"
    assert body["created_at"].endswith("Z") and body["updated_at"].endswith("Z")
    assert "result" not in body and "error" not in body
    assert client.get(f"/jobs/{body['job_id']}").json() == body


@pytest.mark.parametrize("name,content", [
    ("a.pdf", PDF), ("a.docx", b"PK\x03\x04x"), ("a.xlsx", b"PK\x03\x04x"), ("a.csv", b"a,b\n"), ("A.CSV", b"a,b\n"),
])
def test_all_four_types_are_accepted(client, name, content):
    assert post(client, content, name).status_code == 202


@pytest.mark.parametrize("kwargs,code", [
    ({"schema": "nope"}, "invalid_schema"),
    ({"schema": "{}"}, "invalid_schema"),
    ({"name": "a.png", "content": b"\x89PNG\x00"}, "unsupported_file_type"),
    ({"content": b""}, "empty_file"),
    ({"webhook_url": "ftp://x"}, "invalid_webhook_url"),
])
def test_bad_requests_return_400(client, kwargs, code):
    r = post(client, **kwargs)
    assert r.status_code == 400 and r.json()["code"] == code


def test_missing_parts_return_400(client):
    r = client.post("/jobs", data={"schema": SCHEMA})
    assert r.status_code == 400 and r.json()["code"] == "missing_file"
    r = client.post("/jobs", files={"file": ("a.pdf", PDF)})
    assert r.status_code == 400 and r.json()["code"] == "missing_schema"
    r = client.post("/jobs", data={"file": "just text", "schema": SCHEMA})
    assert r.status_code == 400 and r.json()["code"] == "missing_file"


def test_not_multipart_returns_400(client):
    assert client.post("/jobs", json={"a": 1}).status_code == 400
    r = client.post("/jobs", content=b"garbage", headers={"content-type": "multipart/form-data; boundary=x"})
    assert r.status_code == 400
    r = client.post("/jobs", content=b"garbage", headers={"content-type": "multipart/form-data"})
    assert r.status_code == 400


def test_multiple_files_rejected(client):
    files = [("file", ("a.pdf", PDF)), ("file", ("b.pdf", PDF))]
    assert client.post("/jobs", files=files, data={"schema": SCHEMA}).status_code == 400


def test_schema_may_be_uploaded_as_a_file_part(client):
    r = client.post("/jobs", files={"file": ("a.pdf", PDF), "schema": ("s.json", SCHEMA.encode())})
    assert r.status_code == 202


def test_upload_size_limits(make_client):
    c = make_client(max_upload_bytes=1000)
    assert post(c, PDF + b"x" * (1000 - len(PDF))).status_code == 202
    r = post(c, PDF + b"x" * 1000)
    assert r.status_code == 413 and r.json()["code"] == "file_too_large"
    # Far over the limit: rejected from Content-Length before parsing.
    assert post(c, PDF + b"x" * (3 * 1024 * 1024)).status_code == 413


def test_oversized_chunked_upload_is_stopped(make_client):
    c = make_client(max_upload_bytes=1000)

    def body():
        yield b"--x\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.pdf\"\r\n\r\n"
        for _ in range(40):
            yield b"z" * (64 * 1024)

    r = c.post("/jobs", content=body(), headers={"content-type": "multipart/form-data; boundary=x"})
    assert r.status_code == 413


def test_unknown_and_malformed_job_ids_return_404(client):
    assert client.get(f"/jobs/{uuid.uuid4()}").status_code == 404
    r = client.get("/jobs/not-a-uuid")
    assert r.status_code == 404 and r.json()["code"] == "not_found"


def test_tenant_isolation(client):
    job = post(client, headers={"X-Organization-Id": "acme"}).json()["job_id"]
    assert client.get(f"/jobs/{job}", headers={"X-Organization-Id": "acme"}).status_code == 200
    assert client.get(f"/jobs/{job}", headers={"X-Organization-Id": "other"}).status_code == 404
    assert client.get(f"/jobs/{job}").status_code == 404
    no_org = post(client, content=PDF + b"2").json()["job_id"]
    assert client.get(f"/jobs/{no_org}", headers={"X-Organization-Id": "acme"}).status_code == 404


def test_same_document_in_two_tenants_makes_two_jobs(client):
    a = post(client, headers={"X-Organization-Id": "a"}).json()["job_id"]
    b = post(client, headers={"X-Organization-Id": "b"}).json()["job_id"]
    assert a != b


def test_duplicate_returns_existing_job(client):
    first = post(client).json()["job_id"]
    reformatted = json.dumps(json.loads(SCHEMA), indent=4, sort_keys=True)
    assert post(client, schema=reformatted, name="renamed.pdf").json()["job_id"] == first


def test_different_document_or_schema_makes_new_job(client):
    first = post(client).json()["job_id"]
    assert post(client, content=PDF + b"!").json()["job_id"] != first
    other = json.dumps({"fields": {"z": {}}})
    assert post(client, schema=other).json()["job_id"] != first


def test_failed_job_can_be_resubmitted(client, pg_url):
    first = post(client).json()["job_id"]
    with psycopg.connect(pg_url, autocommit=True) as conn:
        conn.execute("UPDATE jobs SET status='FAILED', error='{\"code\":\"x\",\"message\":\"y\"}' WHERE id=%s", (first,))
    assert post(client).json()["job_id"] != first
    assert client.get(f"/jobs/{first}").json()["status"] == "FAILED"


def test_error_only_on_failed_and_result_only_on_success(client, pg_url):
    a, b = post(client).json()["job_id"], post(client, content=PDF + b"2").json()["job_id"]
    with psycopg.connect(pg_url, autocommit=True) as conn:
        conn.execute("UPDATE jobs SET status='FAILED', error='{\"code\":\"x\",\"message\":\"y\"}' WHERE id=%s", (a,))
        conn.execute("UPDATE jobs SET status='SUCCESS', result='{\"entries\":[]}' WHERE id=%s", (b,))
    failed, done = client.get(f"/jobs/{a}").json(), client.get(f"/jobs/{b}").json()
    assert failed["error"] == {"code": "x", "message": "y"} and "result" not in failed
    assert done["result"] == {"entries": []} and "error" not in done


def test_queue_limit_returns_429_with_retry_after(make_client):
    c = make_client(max_active_jobs=1)
    assert post(c).status_code == 202
    r = post(c, content=PDF + b"2")
    assert r.status_code == 429 and r.headers["retry-after"]


def test_webhook_url_is_stored_for_each_submitter(client, pg_url):
    post(client, webhook_url="https://x.io/1")
    post(client, webhook_url="https://x.io/2")
    post(client, webhook_url="https://x.io/2")
    with psycopg.connect(pg_url) as conn:
        assert conn.execute("SELECT count(*) FROM job_webhooks").fetchone()[0] == 2


def test_database_outage_gives_503_not_a_crash():
    s = Settings(database_url="postgresql://u:secretpw@127.0.0.1:1/d", health_db_timeout_seconds=0.2)
    with TestClient(create_app(s), raise_server_exceptions=False) as c:
        start = time.monotonic()
        r = post(c)
        assert r.status_code == 503 and r.headers["retry-after"]
        assert "secretpw" not in r.text
        assert c.get(f"/jobs/{uuid.uuid4()}").status_code == 503
        assert time.monotonic() - start < 3
