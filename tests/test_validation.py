import io
import json
import zipfile

import pytest

from app.validation import (
    RequestError,
    clean_file_name,
    detect_file_type,
    normalize_org_id,
    parse_field_list,
    validate_webhook_url,
)

GOOD = {"name": "x", "fields": {"a": {"name": "A", "description": "d"}}}


def zip_with(name):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, "<x/>")
    return buf.getvalue()


@pytest.mark.parametrize("raw", [
    "not json", "[]", '"str"', "{}", '{"fields": {}}', '{"fields": []}', '{"fields": {"a": "text"}}',
    '{"fields": {" ": {}}}', '{"fields": {"a": NaN}}', '{"fields": {"a": {"description": Infinity}}}',
    '{"fields": {"a": {"primary": true}, "b": {"primary": true}}}',
    '{"fields": {"a": {"description": "x\\u0000y"}}}', '{"fields": {"a": {"description": "\\ud800"}}}',
    b"\xff\xfe\x00bad", "[" * 100_000, " " * (1024 * 1024 + 1),
])
def test_bad_field_lists_are_rejected(raw):
    with pytest.raises(RequestError) as e:
        parse_field_list(raw)
    assert e.value.status == 400 and e.value.code == "invalid_schema"


@pytest.mark.parametrize("schema", [
    {"fields": {"a": {}}},  # no name or description: unseen lists stay accepted
    {"name": "new", "extra": [1], "fields": {"a": {"description": "d", "hint": "h"}}},
    {"fields": {"sum total (%)": {"primary": True}, "b": {"primary": False}}},
    {"fields": {"नाम": {"description": "यूनिकोड"}}},
])
def test_unfamiliar_field_lists_are_accepted(schema):
    assert parse_field_list(json.dumps(schema))[0] == schema


def test_field_list_accepts_byte_order_mark():
    assert parse_field_list(b"\xef\xbb\xbf" + json.dumps(GOOD).encode())[0] == GOOD


def test_hash_ignores_formatting_but_not_content():
    base = parse_field_list(json.dumps(GOOD))[1]
    reordered = '{ "fields": {"a": {"description":"d","name":"A"}},\n "name":"x" }'
    assert parse_field_list(reordered)[1] == base
    assert parse_field_list(json.dumps({**GOOD, "name": "y"}))[1] != base


@pytest.mark.parametrize("name,content,expected", [
    ("a.pdf", b"%PDF-1.7 data", "pdf"),
    ("A.PDF", b"%PDF-1.7 data", "pdf"),
    ("noext", b"%PDF-1.7 data", "pdf"),
    ("wrong.csv", b"\n\n%PDF-1.4", "pdf"),
    ("a.docx", zip_with("word/document.xml"), "docx"),
    ("noext", zip_with("word/document.xml"), "docx"),
    ("a.xlsx", zip_with("xl/workbook.xml"), "xlsx"),
    ("noext", zip_with("xl/workbook.xml"), "xlsx"),
    ("a.xlsm", b"junk", "xlsx"),
    ("a.csv", b"a,b\n1,2\n", "csv"),
    ("noext", b"a,b\n1,2\n", "csv"),
    ("a.txt", "é,ü\n".encode("latin-1"), "csv"),
    ("a.csv", "a,b".encode("utf-16"), "csv"),
    # Damaged files with a supported extension are accepted, then end as FAILED.
    ("corrupt.pdf", b"\x89PNG\x00\x00", "pdf"),
    ("corrupt.xlsx", b"PK\x03\x04broken", "xlsx"),
])
def test_file_type_detection(name, content, expected):
    assert detect_file_type(name, content) == expected


@pytest.mark.parametrize("name,content", [
    ("a.png", b"\x89PNG\r\n\x1a\n\x00\x00\x00"),
    ("a.xls", b"\xd0\xcf\x11\xe0\x00\x00\x00"),
    ("noext", b"\x00\x01\x02\x03"),
    ("a.zip", zip_with("other.txt")),
])
def test_unsupported_types_are_rejected(name, content):
    with pytest.raises(RequestError) as e:
        detect_file_type(name, content)
    assert e.value.code == "unsupported_file_type"


@pytest.mark.parametrize("raw,expected", [
    (None, "document"), ("", "document"), ("../../etc/passwd", "passwd"),
    ("C:\\docs\\a.pdf", "a.pdf"), ("a\x00b\n.pdf", "ab.pdf"),
])
def test_file_names_are_cleaned(raw, expected):
    assert clean_file_name(raw) == expected


def test_org_id_rules():
    assert normalize_org_id(None) == normalize_org_id("") == normalize_org_id("  ") == ""
    assert normalize_org_id(" acme ") == "acme"
    for bad in ("x" * 256, "a\nb"):
        with pytest.raises(RequestError):
            normalize_org_id(bad)


@pytest.mark.parametrize("url", ["ftp://a/b", "not a url", "http://", "javascript:alert(1)", "http://a b/", "http://[x/"])
def test_bad_webhook_urls(url):
    with pytest.raises(RequestError):
        validate_webhook_url(url)


def test_good_webhook_urls():
    assert validate_webhook_url(None) is None and validate_webhook_url("  ") is None
    assert validate_webhook_url("https://x.io/h?a=1") == "https://x.io/h?a=1"
