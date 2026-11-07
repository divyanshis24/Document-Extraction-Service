"""Forces any extractor output into a result that satisfies the output contract."""
from __future__ import annotations

import math
import re

# The validator rejects numbers formatted as money, so convert them.
_MONEY_RE = re.compile(r"^[$€£₹]\s*[+-]?(\d[\d,]*)(\.\d+)?$|^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")


def _clean_scalar(value):
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        text = value.replace("\x00", "").encode("utf-8", "ignore").decode("utf-8")
        if _MONEY_RE.match(text.strip()):
            number = float(re.sub(r"[$€£₹,\s]", "", text))
            return int(number) if number.is_integer() else number
        return text
    return None  # not representable as a scalar, so report it as not found


def _clean_value(value):
    if isinstance(value, list):
        return [_clean_scalar(item) for item in value]
    return _clean_scalar(value)


def _clean_field(raw) -> dict:
    if not isinstance(raw, dict):
        raw = {}
    value = _clean_value(raw.get("value"))
    citation = raw.get("citation")
    citation = citation if isinstance(citation, str) else ""
    citation = citation.replace("\x00", "").encode("utf-8", "ignore").decode("utf-8")
    confidence = raw.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
        confidence = 0.0
    out = {
        "value": value,
        "citation": "" if value is None else citation,  # the contract requires empty for null
        "confidence": min(max(float(confidence), 0.0), 1.0),
    }
    pages = raw.get("page_numbers")
    if isinstance(pages, list):
        pages = [p for p in pages if isinstance(p, int) and not isinstance(p, bool) and p >= 1]
        if pages and value is not None:
            out["page_numbers"] = pages
    locations = raw.get("source_locations")
    if isinstance(locations, list):
        locations = [s for s in locations if isinstance(s, str)]
        if locations and value is not None:
            out["source_locations"] = locations
    return out


def _count(metadata: dict, key: str) -> int:
    value = metadata.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def build_result(entries, field_keys: list[str], metadata: dict | None, latency_ms: int) -> dict:
    """Every entry gets exactly the requested keys, with null for anything missing."""
    metadata = dict(metadata or {})
    rows = [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []
    cost = metadata.get("cost_usd")
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
        cost = 0.0
    return {
        "entries": [{key: _clean_field(row.get(key)) for key in field_keys} for row in rows or [{}]],
        "metadata": {
            **metadata,
            "model_calls": _count(metadata, "model_calls"),
            "tokens_in": _count(metadata, "tokens_in"),
            "tokens_out": _count(metadata, "tokens_out"),
            "cost_usd": cost,
            "latency_ms": max(int(latency_ms), 0),
        },
    }
