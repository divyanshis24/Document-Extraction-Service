#!/usr/bin/env python3
"""
validate.py - check an extraction result against the output contract and,
optionally, score it against a ground-truth file.

Usage:
  python validate.py result.json --schema schema.json
  python validate.py result.json --schema schema.json --truth truth.json
  python validate.py results_dir/ --schema-dir schemas/ [--truth-dir truth/]

Batch mode expects results_dir/<document>.json, schemas/<schema_name>.json and
truth/<document>.json, with a top-level "schema": "<schema_name>" key in each
result telling us which schema it was extracted against (optional in single mode).

Exit code 0 = valid. 1 = contract violation. Standard library only, Python 3.9+.

Contract checks:
  - `entries` is a non-empty list; every entry has EVERY schema field
  - each field is {value, citation, confidence} (+ optional page_numbers, source_locations)
  - value is a scalar or a list of scalars; confidence in [0,1]; citation is a string
  - `metadata` has non-negative model_calls, tokens_in, tokens_out, latency_ms, cost_usd

Scoring (with --truth):
  - rows are matched to truth rows by the schema's primary field when there is one,
    otherwise by position
  - per cell: exact match after normalisation (numbers +-0.5%, dates in several
    formats, currency symbols and thousands separators stripped, case and
    whitespace folded). Lists: order-insensitive F1.
  - null handling: truth null & got null = 1; truth value & got null = 0 (miss);
    truth null & got value = 0 and counted as a hallucination
  - extra rows and missing rows score 0 for every field in that row
Our internal scorer adds an LLM judge for free-text fields and tier weights. This
script is stricter, so a cell it marks correct is correct in our scoring too.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

NUM_RE = re.compile(r"^[+-]?\d+(\.\d+)?$")
# "$1,803", "1,250.00", "€900": a number dressed as money. Plain digit strings (ZIP codes,
# unit numbers, GL codes) are legitimate string values and are left alone.
CURRENCY_STR_RE = re.compile(r"^[$€£₹]\s*[+-]?[\d,]+(\.\d+)?$|^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")
DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%b %d, %Y", "%B %d, %Y",
                "%d %b %Y", "%d %B %Y", "%m/%d/%y", "%Y-%m-%dT%H:%M:%S", "%m-%d-%Y"]
FIELD_KEYS = {"value", "citation", "confidence", "page_numbers", "source_locations"}
META_INTS = ["model_calls", "tokens_in", "tokens_out", "latency_ms"]


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"{path}: not valid JSON ({exc})")


def schema_fields(schema: dict) -> tuple[dict, str | None]:
    fields = schema.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise SystemExit("schema has no `fields` object")
    primaries = [k for k, v in fields.items() if isinstance(v, dict) and v.get("primary") is True]
    if len(primaries) > 1:
        raise SystemExit(f"schema has more than one primary field: {primaries}")
    return fields, (primaries[0] if primaries else None)


def is_scalar(v) -> bool:
    return v is None or isinstance(v, (str, bool, int, float))


# ---------- contract ----------

def validate_contract(result: dict, schema: dict) -> list[str]:
    errors: list[str] = []
    fields, _ = schema_fields(schema)

    entries = result.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append("`entries` must be a non-empty list (one object per row; one entry for a document with no table)")
        entries = []

    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"entries[{i}]: must be an object")
            continue
        for key in fields:
            if key not in entry:
                errors.append(f"entries[{i}].{key}: missing (every entry must carry every schema field; use null)")
        for key, fr in entry.items():
            if key not in fields:
                errors.append(f"entries[{i}].{key}: not in schema")
                continue
            if not isinstance(fr, dict):
                errors.append(f"entries[{i}].{key}: must be an object with value, citation, confidence")
                continue
            for req in ("value", "citation", "confidence"):
                if req not in fr:
                    errors.append(f"entries[{i}].{key}.{req}: missing")
            for extra in fr:
                if extra not in FIELD_KEYS:
                    errors.append(f"entries[{i}].{key}.{extra}: unknown key")
            v = fr.get("value")
            if not (is_scalar(v) or (isinstance(v, list) and all(is_scalar(x) for x in v))):
                errors.append(f"entries[{i}].{key}.value: must be a scalar or a list of scalars")
            if isinstance(v, str) and CURRENCY_STR_RE.match(v.strip()):
                errors.append(f"entries[{i}].{key}.value: currency-formatted string ({v!r}); use a plain JSON number")
            if "citation" in fr and not isinstance(fr["citation"], str):
                errors.append(f"entries[{i}].{key}.citation: must be a string (empty when value is null)")
            c = fr.get("confidence")
            if "confidence" in fr and not (isinstance(c, (int, float)) and not isinstance(c, bool) and 0 <= c <= 1):
                errors.append(f"entries[{i}].{key}.confidence: must be a number in [0,1]")
            pn = fr.get("page_numbers")
            if pn is not None and not (isinstance(pn, list) and all(isinstance(p, int) and p >= 1 for p in pn)):
                errors.append(f"entries[{i}].{key}.page_numbers: must be a list of integers >= 1")
            sl = fr.get("source_locations")
            if sl is not None and not (isinstance(sl, list) and all(isinstance(s, str) for s in sl)):
                errors.append(f"entries[{i}].{key}.source_locations: must be a list of strings")
        if len(errors) > 40:
            errors.append("... (more errors truncated)")
            break

    meta = result.get("metadata")
    if not isinstance(meta, dict):
        errors.append("`metadata` must be an object")
    else:
        for k in META_INTS:
            if not isinstance(meta.get(k), int) or isinstance(meta.get(k), bool) or meta[k] < 0:
                errors.append(f"metadata.{k}: required non-negative integer")
        c = meta.get("cost_usd")
        if not isinstance(c, (int, float)) or isinstance(c, bool) or c < 0:
            errors.append("metadata.cost_usd: required non-negative number")

    for key in result:
        if key not in ("entries", "metadata", "schema", "document"):
            errors.append(f"top-level key `{key}` not allowed")
    return errors


# ---------- normalisation ----------

def norm(v):
    """Normalise a scalar for comparison -> (kind, value)."""
    if v is None:
        return ("null", None)
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return ("null", None)
        return ("num", round(float(v), 2))
    s = str(v).strip()
    if s == "" or s.lower() in {"null", "none", "n/a", "na", "-", "--"}:
        return ("null", None)
    low = s.lower()
    if low in {"true", "yes", "y"}:
        return ("bool", True)
    if low in {"false", "no", "n"}:
        return ("bool", False)
    for fmt in DATE_FORMATS:
        try:
            return ("date", datetime.strptime(s, fmt).date().isoformat())
        except ValueError:
            pass
    t = s.replace(",", "").replace("$", "").replace("€", "").replace("£", "").replace("₹", "").strip()
    neg = t.startswith("(") and t.endswith(")")
    if neg:
        t = t[1:-1]
    pct = t.endswith("%")
    if pct:
        t = t[:-1].strip()
    if NUM_RE.match(t):
        x = float(t)
        if neg:
            x = -x
        return ("num", round(x, 2))
    return ("str", re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", low)).strip())


def scalar_match(a, b) -> float:
    ka, va = norm(a)
    kb, vb = norm(b)
    if ka == "null" and kb == "null":
        return 1.0
    if ka == "null" or kb == "null":
        return 0.0
    if ka == "num" and kb == "num":
        return 1.0 if abs(va - vb) <= max(0.01, abs(vb) * 0.005) else 0.0
    if ka != kb:
        return 1.0 if str(va) == str(vb) else 0.0
    return 1.0 if va == vb else 0.0


def list_f1(pred: list, truth: list) -> float:
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
    used, tp = set(), 0
    for p in pred:
        for i, t in enumerate(truth):
            if i not in used and scalar_match(p, t) == 1.0:
                used.add(i)
                tp += 1
                break
    if tp == 0:
        return 0.0
    prec, rec = tp / len(pred), tp / len(truth)
    return 2 * prec * rec / (prec + rec)


def cell_match(pv, tv) -> float:
    if isinstance(tv, list):
        return list_f1(pv if isinstance(pv, list) else ([] if pv is None else [pv]), tv)
    if isinstance(pv, list):
        pv = pv[0] if len(pv) == 1 else None
    return scalar_match(pv, tv)


# ---------- scoring ----------

def value_of(entry: dict, key: str):
    fr = entry.get(key)
    return fr.get("value") if isinstance(fr, dict) else fr  # truth may be bare {key: value}


def align_rows(pred_entries: list, truth_entries: list, primary: str | None):
    """Return list of (pred_entry|None, truth_entry|None, row_label)."""
    if primary is None:
        n = max(len(pred_entries), len(truth_entries))
        return [(pred_entries[i] if i < len(pred_entries) else None,
                 truth_entries[i] if i < len(truth_entries) else None, f"row{i}") for i in range(n)]
    t_index: dict = {}
    for e in truth_entries:
        t_index.setdefault(norm(value_of(e, primary)), e)
    p_index: dict = {}
    for e in pred_entries:
        p_index.setdefault(norm(value_of(e, primary)), e)
    pairs = []
    for rid, te in t_index.items():
        pairs.append((p_index.get(rid), te, str(value_of(te, primary))))
    for rid, pe in p_index.items():
        if rid not in t_index:
            pairs.append((pe, None, str(value_of(pe, primary))))
    return pairs


def score(result: dict, truth: dict, schema: dict) -> dict:
    fields, primary = schema_fields(schema)
    pred_entries = result.get("entries", [])
    truth_entries = truth.get("entries", truth if isinstance(truth, list) else [truth])
    pairs = align_rows(pred_entries, truth_entries, primary)

    cells: dict[str, list[float]] = {k: [] for k in fields}
    halluc: list[str] = []
    misses: list[str] = []
    extra_rows, missing_rows = [], []

    for pe, te, label in pairs:
        if te is None:
            extra_rows.append(label)
            for k in fields:
                cells[k].append(0.0)
            continue
        if pe is None:
            missing_rows.append(label)
            for k in fields:
                cells[k].append(0.0)
            continue
        for k in fields:
            tv, pv = value_of(te, k), value_of(pe, k)
            s = cell_match(pv, tv)
            cells[k].append(s)
            if s == 0.0:
                if norm(tv)[0] == "null" and norm(pv)[0] != "null" and not isinstance(tv, list):
                    halluc.append(f"{k}[{label}]")
                elif norm(tv)[0] != "null" and norm(pv)[0] == "null":
                    misses.append(f"{k}[{label}]")

    per_field = {k: (sum(v) / len(v) if v else 0.0) for k, v in cells.items()}
    total = sum(per_field.values()) / len(per_field) if per_field else 0.0
    return {"score": round(total, 4), "per_field": {k: round(v, 3) for k, v in per_field.items()},
            "hallucinations": halluc, "misses": misses,
            "extra_rows": extra_rows, "missing_rows": missing_rows,
            "rows": {"predicted": len(pred_entries), "truth": len(truth_entries)}}


# ---------- cli ----------

def run_one(result_path: Path, schema_path: Path, truth_path: Path | None, quiet=False) -> tuple[bool, dict | None]:
    result, schema = load_json(result_path), load_json(schema_path)
    errors = validate_contract(result, schema)
    if errors:
        print(f"INVALID  {result_path.name}")
        for e in errors:
            print(f"   - {e}")
        return False, None
    if not quiet:
        print(f"VALID    {result_path.name}   ({len(result['entries'])} entries)")
    if truth_path is None:
        return True, None
    rep = score(result, load_json(truth_path), schema)
    print(f"SCORE    {result_path.name}: {rep['score']:.3f}   rows {rep['rows']['predicted']}/{rep['rows']['truth']}"
          f"   hallucinations={len(rep['hallucinations'])} misses={len(rep['misses'])}"
          f" extra_rows={len(rep['extra_rows'])} missing_rows={len(rep['missing_rows'])}")
    if not quiet:
        for k, v in sorted(rep["per_field"].items(), key=lambda kv: kv[1]):
            print(f"   {k:32s} {v:.2f}{'' if v == 1.0 else '  <-- check'}")
        if rep["hallucinations"]:
            print(f"   hallucinated: {', '.join(rep['hallucinations'][:10])}")
        if rep["misses"]:
            print(f"   missed: {', '.join(rep['misses'][:10])}")
        if rep["extra_rows"]:
            print(f"   extra rows: {', '.join(rep['extra_rows'][:10])}")
        if rep["missing_rows"]:
            print(f"   missing rows: {', '.join(rep['missing_rows'][:10])}")
    return True, rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="result.json, or a directory of result JSON files")
    ap.add_argument("--schema", help="schema.json (single-file mode)")
    ap.add_argument("--truth", help="ground truth JSON (single-file mode)")
    ap.add_argument("--schema-dir", help="directory of schemas named <schema_name>.json (batch mode)")
    ap.add_argument("--truth-dir", help="directory of truth files named <document>.json (batch mode)")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    target = Path(args.target)
    if target.is_file():
        if not args.schema:
            ap.error("--schema is required for a single file")
        ok, _ = run_one(target, Path(args.schema), Path(args.truth) if args.truth else None, args.quiet)
        return 0 if ok else 1

    if not args.schema_dir:
        ap.error("--schema-dir is required for a directory")
    files = sorted(target.glob("*.json"))
    if not files:
        raise SystemExit(f"no .json files in {target}")
    all_ok, scores = True, []
    for rp in files:
        res = load_json(rp)
        sname = res.get("schema", "")
        sp = Path(args.schema_dir) / f"{sname}.json"
        if not sp.exists():
            print(f"INVALID  {rp.name}: add a top-level \"schema\" key naming a file in {args.schema_dir}")
            all_ok = False
            continue
        tp = None
        if args.truth_dir:
            cand = Path(args.truth_dir) / rp.name
            tp = cand if cand.exists() else None
            if tp is None:
                print(f"NOTRUTH  {rp.name}")
        ok, rep = run_one(rp, sp, tp, quiet=True)
        all_ok &= ok
        if rep:
            scores.append(rep["score"])
    if scores:
        print(f"\nMEAN SCORE over {len(scores)} scored documents: {sum(scores)/len(scores):.3f}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
