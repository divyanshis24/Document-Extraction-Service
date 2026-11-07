import importlib.util
from pathlib import Path

import pytest

from app.result import build_result

spec = importlib.util.spec_from_file_location("validate", Path(__file__).parent.parent / "tools" / "validate.py")
validate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate)

KEYS = ["a", "b"]
SCHEMA = {"fields": {k: {} for k in KEYS}}


def check(result):
    """Judge results with the assignment's own validator, not a copy of its rules."""
    assert validate.validate_contract(result, SCHEMA) == []
    return result


def test_missing_fields_become_null():
    r = check(build_result([{"a": {"value": 1, "citation": "[Page 1] 1", "confidence": 0.9}}], KEYS, {}, 5))
    assert r["entries"][0]["b"] == {"value": None, "citation": "", "confidence": 0.0}
    assert r["metadata"]["latency_ms"] == 5


@pytest.mark.parametrize("entries", [[], None, "x", [1, None]])
def test_no_usable_entries_gives_one_null_entry(entries):
    r = check(build_result(entries, KEYS, None, 0))
    assert len(r["entries"]) == 1


def test_extra_fields_are_dropped():
    r = check(build_result([{"a": {"value": 1}, "zzz": {"value": 2}}], KEYS, {}, 0))
    assert "zzz" not in r["entries"][0]


@pytest.mark.parametrize("value,expected", [
    (float("nan"), None), (float("inf"), None), ({"x": 1}, None),
    ("$1,803", 1803), ("1,250.50", 1250.5), ("€900", 900),
    ("10025", "10025"),  # digit strings like ZIP codes stay strings
    ("a\x00b", "ab"), ("\ud800x", "x"), (True, True), (0, 0),
    ([1, {"x": 1}, "$5"], [1, None, 5]),
])
def test_values_are_cleaned(value, expected):
    r = check(build_result([{"a": {"value": value, "citation": "c", "confidence": 1}}], KEYS, {}, 0))
    assert r["entries"][0]["a"]["value"] == expected


@pytest.mark.parametrize("confidence,expected", [
    (1.5, 1.0), (-1, 0.0), ("high", 0.0), (None, 0.0), (True, 0.0), (float("nan"), 0.0), (0.7, 0.7),
])
def test_confidence_is_bounded(confidence, expected):
    r = check(build_result([{"a": {"value": 1, "citation": "c", "confidence": confidence}}], KEYS, {}, 0))
    assert r["entries"][0]["a"]["confidence"] == expected


def test_null_value_has_empty_citation():
    r = check(build_result([{"a": {"value": None, "citation": "made up", "confidence": 1, "page_numbers": [1]}}], KEYS, {}, 0))
    assert r["entries"][0]["a"]["citation"] == "" and "page_numbers" not in r["entries"][0]["a"]


def test_bad_locations_are_filtered():
    raw = {"value": 1, "citation": "c", "confidence": 1, "page_numbers": [0, 2, "x", True], "source_locations": ["S!A1", 3]}
    r = check(build_result([{"a": raw}], KEYS, {}, 0))
    assert r["entries"][0]["a"]["page_numbers"] == [2] and r["entries"][0]["a"]["source_locations"] == ["S!A1"]


def test_bad_metadata_is_replaced_and_extras_kept():
    meta = {"model_calls": -1, "tokens_in": "9", "cost_usd": float("nan"), "models": {"m": {"calls": 1}}}
    r = check(build_result([{}], KEYS, meta, -5))
    assert r["metadata"]["model_calls"] == 0 and r["metadata"]["tokens_in"] == 0
    assert r["metadata"]["cost_usd"] == 0.0 and r["metadata"]["latency_ms"] == 0
    assert r["metadata"]["models"] == {"m": {"calls": 1}}


def test_money_text_without_digits_stays_text_instead_of_crashing():
    from app.result import build_result
    result = build_result([{"a": {"value": "$,", "citation": "x", "confidence": 1}}], ["a"], {}, 1)
    assert result["entries"][0]["a"]["value"] == "$,"
