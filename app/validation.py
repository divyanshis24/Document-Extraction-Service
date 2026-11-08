"""Checks on untrusted request input. Each failure maps to a contract error."""
from __future__ import annotations

import codecs
import hashlib
import io
import json
import zipfile
from pathlib import PurePosixPath
from urllib.parse import urlsplit

MAX_SCHEMA_BYTES = 1024 * 1024
MAX_ORG_ID_LENGTH = 255
MAX_WEBHOOK_URL_LENGTH = 2048
_EXTENSIONS = {".pdf": "pdf", ".docx": "docx", ".xlsx": "xlsx", ".xlsm": "xlsx", ".csv": "csv"}


class RequestError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _bad(code: str, message: str) -> RequestError:
    return RequestError(400, code, message)


def normalize_org_id(raw: str | None) -> str:
    """No header and an empty header are the same (empty) tenant."""
    org = (raw or "").strip()
    if len(org) > MAX_ORG_ID_LENGTH or any(ord(c) < 32 for c in org):
        raise _bad("invalid_organization_id", f"X-Organization-Id must be at most {MAX_ORG_ID_LENGTH} printable characters")
    return org


def clean_file_name(raw: str | None) -> str:
    name = PurePosixPath((raw or "").replace("\\", "/")).name
    name = "".join(c for c in name if ord(c) >= 32 and not 0xD800 <= ord(c) <= 0xDFFF)
    return name[-255:] or "document"


def _reject_constant(name: str):
    raise ValueError(f"{name} is not valid JSON")


def _check_strings(node) -> None:
    """Postgres jsonb can't store NUL characters or lone surrogates, so reject them here."""
    if isinstance(node, str):
        if "\x00" in node or any(0xD800 <= ord(c) <= 0xDFFF for c in node):
            raise ValueError("strings must not contain NUL or unpaired surrogate characters")
    elif isinstance(node, dict):
        for key, value in node.items():
            _check_strings(key)
            _check_strings(value)
    elif isinstance(node, list):
        for value in node:
            _check_strings(value)


def parse_field_list(raw: bytes | str) -> tuple[dict, str]:
    """Return (field list, hash). Only the shape is enforced, so unseen field lists work."""
    if isinstance(raw, str):
        raw = raw.encode("utf-8", "surrogatepass")
    if len(raw) > MAX_SCHEMA_BYTES:
        raise _bad("invalid_schema", "schema is larger than 1 MB")
    try:
        schema = json.loads(raw.decode("utf-8-sig"), parse_constant=_reject_constant)
        _check_strings(schema)
    except RecursionError:
        raise _bad("invalid_schema", "schema is too deeply nested") from None
    except ValueError as exc:  # includes JSONDecodeError and UnicodeDecodeError
        raise _bad("invalid_schema", f"schema is not valid JSON: {exc}") from None

    if not isinstance(schema, dict):
        raise _bad("invalid_schema", "schema must be a JSON object")
    fields = schema.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise _bad("invalid_schema", "schema must have a non-empty `fields` object")
    for key, spec in fields.items():
        if not key.strip():
            raise _bad("invalid_schema", "field keys must not be blank")
        if not isinstance(spec, dict):
            raise _bad("invalid_schema", f"field {key!r} must be an object")
    primaries = [k for k, spec in fields.items() if spec.get("primary") is True]
    if len(primaries) > 1:
        raise _bad("invalid_schema", f"schema has more than one primary field: {primaries}")

    # Key order and whitespace don't change the meaning, so they don't change the hash.
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return schema, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _looks_like_text(content: bytes) -> bool:
    sample = content[:8192]
    if sample.startswith((codecs.BOM_UTF8, codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return True
    if b"\x00" in sample:
        return False
    try:
        codecs.getincrementaldecoder("utf-8")().decode(sample, final=False)
        return True
    except UnicodeDecodeError:
        # Probably a legacy encoding such as Windows-1252. Binary has many control bytes.
        control = sum(1 for b in sample if b < 32 and b not in (9, 10, 13))
        return control / len(sample) < 0.02


def detect_file_type(file_name: str, content: bytes) -> str:
    """Trust the content first and the extension second.

    A damaged file with a supported extension is accepted and later ends as FAILED with
    a reason, because refusing a supported type with a 400 scores as a failed document.
    """
    if b"%PDF-" in content[:1024]:
        return "pdf"
    if content[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                names = set(archive.namelist())
            if "word/document.xml" in names:
                return "docx"
            if "xl/workbook.xml" in names:
                return "xlsx"
        except zipfile.BadZipFile:
            pass
    file_type = _EXTENSIONS.get(PurePosixPath(file_name).suffix.lower())
    if file_type:
        return file_type
    if _looks_like_text(content):
        return "csv"
    raise RequestError(400, "unsupported_file_type", "Supported file types are PDF, DOCX, XLSX and CSV")


def validate_webhook_url(raw: str | None) -> str | None:
    url = (raw or "").strip()
    if not url:
        return None
    try:
        parts = urlsplit(url)
        valid = parts.scheme in ("http", "https") and bool(parts.hostname)
    except ValueError:
        valid = False
    if not valid or len(url) > MAX_WEBHOOK_URL_LENGTH or any(ord(c) <= 32 for c in url):
        raise _bad("invalid_webhook_url", "webhook_url must be an absolute http or https URL")
    return url
