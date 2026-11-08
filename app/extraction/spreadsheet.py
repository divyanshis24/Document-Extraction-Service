"""Runs an extraction plan over a sheet using plain code.

A model reads a sample of the sheet once and writes the plan. Code then applies it to every
row, so a 3,000-row sheet costs about the same as a 30-row one and values are copied exactly.
Anything the rules can't express is marked "llm" and handled by a model, row by row.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.extraction.fields import FieldSpec, coerce, to_number
from app.extraction.grounding import TextIndex, value_in
from app.extraction.parsing import Cell, ParsedDocument, Sheet, format_cell


_HEADER_SEARCH_ROWS = 100


class PlanError(ValueError):
    pass


PLAN_RULES = """Rules you may use for each field:
- {"rule":"document","value":<value>,"cell":"A3"}: one value for the whole document, read from that cell. "value" is what the cell prints with only a label such as "Name =" removed: keep every part of a name, code or identifier exactly as written (dates as YYYY-MM-DD).
- {"rule":"column","column":"F","default":<optional>}: the value in that column on the row's first row.
- {"rule":"block_match","match_column":"H","match":"Tax","value_column":"I","default":<optional>}: when one logical row spans several sheet rows, find the row in the block where match_column equals match, and take value_column from it.
- {"rule":"block_sum","value_column":"I","match_column":"H","exclude":["Tax"],"default":<optional>}: sum value_column over rows of the block that have a non-empty match_column, except those listed.
- {"rule":"row_stat","stat":"count_nonzero","columns":["C","D"]}: count of cells in those columns (list every column) on the row's first row that hold a non-zero number. "stat" may also be "sum".
- {"rule":"nearest_above","column":"B","require_empty_columns":["C"]}: the value in that column from the nearest row above that is a header, meaning those columns are empty on it.
- {"rule":"llm"}: use only when the value needs judgement that the rules above can't express.
Set "default" only when the field description says what to use if the value is missing (such as "0 if none"). Otherwise leave it out, so a missing cell stays null."""


@dataclass
class Block:
    start: int
    rows: list[int]


@dataclass
class PlanResult:
    entries: list[dict]
    blocks: list[Block]
    sheet: Sheet
    llm_fields: list[FieldSpec] = field(default_factory=list)
    primary_values: list = field(default_factory=list)


def _column(ref: str) -> str:
    return re.sub(r"[0-9]", "", ref)


def _column_index(letters: str) -> int:
    index = 0
    for char in letters.upper():
        if "A" <= char <= "Z":
            index = index * 26 + ord(char) - 64
    return index


def _row_cells(sheet: Sheet, number: int) -> dict[str, Cell]:
    return {_column(c.ref): c for c in sheet.rows.get(number, [])}


def _citation(sheet: Sheet, cell: Cell) -> dict:
    return {"citation": f"[{sheet.name}!{cell.ref}] {format_cell(cell.value)}", "source_locations": [f"{sheet.name}!{cell.ref}"]}


def find_blocks(sheet: Sheet, plan: dict) -> list[Block]:
    start = plan.get("row_start") or {}
    if not isinstance(start, dict):
        raise PlanError("plan row_start is not an object")
    column = str(start.get("column") or "").upper()
    if not column:
        raise PlanError("plan has no row_start column")
    try:
        pattern = re.compile(str(start.get("regex") or ".+"))
        exclude = re.compile(str(plan["exclude_regex"])) if plan.get("exclude_regex") else None
    except re.error as exc:
        raise PlanError(f"bad regular expression in plan: {exc}") from exc
    required = plan.get("require_columns") or []
    required = [str(c).upper() for c in (required if isinstance(required, list) else [required])]
    header_row = plan.get("header_row") if isinstance(plan.get("header_row"), int) else 0

    # Columns at or left of the identifying ones, where a total or section label would sit.
    label_limit = max(_column_index(c) for c in [column, *required])

    numbers = sorted(n for n in sheet.rows if n > header_row)
    # Every row that looks like a start ends the previous block, even if it is itself
    # filtered out, so a skipped row's lines can't leak into its neighbour. So does a row
    # that can't belong to a block: a section heading in the start column ("Status Summary"),
    # or a total labelled on the left ("Grand Total:"). Without this the last row of a
    # table swallows the grand total and every summary section below it.
    starts: list[int] = []
    breaks: list[int] = []
    for number in numbers:
        cells = _row_cells(sheet, number)
        cell = cells.get(column)
        if cell is not None and pattern.search(format_cell(cell.value)):
            starts.append(number)
        elif cell is not None and format_cell(cell.value).strip():
            breaks.append(number)
        elif exclude and any(_column_index(col) <= label_limit and exclude.search(format_cell(c.value))
                             for col, c in cells.items()):
            breaks.append(number)
    boundaries = sorted(starts + breaks)
    blocks = []
    for number in starts:
        cells = _row_cells(sheet, number)
        text = format_cell(cells[column].value)
        if exclude and exclude.search(text):
            continue
        if any(c not in cells for c in required):
            continue
        end = next((b for b in boundaries if b > number), numbers[-1] + 1)
        blocks.append(Block(start=number, rows=[n for n in numbers if number <= n < end]))
    return blocks


def _apply_rule(rule: dict, block: Block, sheet: Sheet) -> tuple[object, dict, float]:
    """Return (value, citation fields, confidence)."""
    kind = rule.get("rule")
    default = rule.get("default")
    if kind == "column":
        cell = _row_cells(sheet, block.start).get(str(rule.get("column")).upper())
        if cell is None:
            return default, {}, 0.8
        return cell.value, _citation(sheet, cell), 0.95
    if kind == "block_match":
        wanted = str(rule.get("match", "")).strip().lower()
        match_column, value_column = str(rule.get("match_column")).upper(), str(rule.get("value_column")).upper()
        found = []
        for number in block.rows:
            cells = _row_cells(sheet, number)
            label = cells.get(match_column)
            if label is not None and format_cell(label.value).strip().lower() == wanted and value_column in cells:
                found.append(cells[value_column])
        if not found:
            return default, {}, 0.8
        amounts = [to_number(c.value) for c in found]
        if len(found) == 1 or any(a is None for a in amounts):
            return found[0].value, _citation(sheet, found[0]), 0.95
        # The same charge line can appear more than once (a $0 line and the real one, or a
        # mid-month change). The unit's charge for that line is their total.
        total = round(sum(amounts), 6)
        total = int(total) if float(total).is_integer() else total
        refs = [f"{sheet.name}!{c.ref}" for c in found]
        return total, {"citation": f"[{refs[0]}] sum of {len(refs)} '{rule.get('match')}' lines = {total}",
                       "source_locations": refs}, 0.9
    if kind == "block_sum":
        match_column, value_column = str(rule.get("match_column")).upper(), str(rule.get("value_column")).upper()
        excluded = {str(x).strip().lower() for x in rule.get("exclude") or []}
        total, used = 0.0, []
        for number in block.rows:
            cells = _row_cells(sheet, number)
            label, value = cells.get(match_column), cells.get(value_column)
            if label is None or value is None or format_cell(label.value).strip().lower() in excluded:
                continue
            amount = to_number(value.value)
            if amount is not None:
                total += amount
                used.append(value)
        if not used:
            return default, {}, 0.8
        refs = [f"{sheet.name}!{c.ref}" for c in used]
        total = round(total, 6)
        total = int(total) if float(total).is_integer() else total
        return total, {"citation": f"[{refs[0]}] sum of {len(refs)} cells = {total}", "source_locations": refs}, 0.9
    if kind == "row_stat":
        cells = _row_cells(sheet, block.start)
        used = [cells[str(c).upper()] for c in rule.get("columns") or [] if str(c).upper() in cells]
        numbers = [to_number(c.value) for c in used]
        refs = [f"{sheet.name}!{c.ref}" for c in used]
        if rule.get("stat") == "count_nonzero":
            value = sum(1 for n in numbers if n)
        elif rule.get("stat") == "sum":
            value = round(sum(n for n in numbers if n is not None), 6)
        else:
            raise PlanError(f"unknown row_stat {rule.get('stat')!r}")
        return value, {"citation": f"[{sheet.name}!{used[0].ref if used else block.start}] {rule.get('stat')} over {len(used)} cells = {value}",
                       "source_locations": refs}, 0.9
    if kind == "nearest_above":
        column = str(rule.get("column")).upper()
        empty = [str(c).upper() for c in rule.get("require_empty_columns") or []]
        for number in sorted((n for n in sheet.rows if n < block.start), reverse=True):
            cells = _row_cells(sheet, number)
            if column in cells and all(c not in cells for c in empty):
                return cells[column].value, _citation(sheet, cells[column]), 0.85
        return default, {}, 0.5
    raise PlanError(f"unknown rule {kind!r}")


def _find_cell(parsed: ParsedDocument, sheet: Sheet, ref: str) -> tuple[Sheet, Cell] | None:
    """A cell by reference, "A3" or "Params!B2", on the planned sheet first and then the others."""
    ref = ref.strip().replace("$", "")
    if "!" in ref:
        name, ref = ref.rsplit("!", 1)
        name = name.strip("'\"")
        named = next((s for s in parsed.sheets if s.name == name), None)
        cell = named.cell(ref) if named else None
        return (named, cell) if cell else None
    for candidate in [sheet, *(s for s in parsed.sheets if s is not sheet)]:
        cell = candidate.cell(ref)
        if cell is not None:
            return candidate, cell
    return None


def _document_field(parsed: ParsedDocument, sheet: Sheet, rule: dict, spec: FieldSpec) -> dict:
    """A whole-document value from the plan, kept only if the sheet shows it.

    The plan writes the cleaned value itself (a date as YYYY-MM-DD, say), so its value is
    used, but only when the cited cell, or failing that some other cell, contains it.
    """
    value = coerce(rule.get("value"), spec)
    if value is None:
        return {"value": None, "citation": "", "confidence": 0.6}
    located = _find_cell(parsed, sheet, str(rule.get("cell") or ""))
    if located and value_in(value, TextIndex(format_cell(located[1].value))):
        return {"value": value, "confidence": 0.9, **_citation(*located)}
    if spec.computed and located:
        return {"value": value, "confidence": 0.7, **_citation(*located)}
    # The cited cell doesn't hold it. Look for a cell that does in each sheet's header area,
    # where document-wide values sit. Bounded, because this runs on the event loop.
    for candidate in [sheet, *(s for s in parsed.sheets if s is not sheet)]:
        for number in sorted(candidate.rows)[:_HEADER_SEARCH_ROWS]:
            for cell in candidate.rows[number]:
                if value_in(value, TextIndex(format_cell(cell.value))):
                    return {"value": value, "confidence": 0.8, **_citation(candidate, cell)}
    if spec.computed:
        return {"value": value, "confidence": 0.5, "citation": "computed from the sheet"}
    return {"value": None, "citation": "", "confidence": 0.3}  # not in the sheet, so not trusted


def _clean(value):
    return value.strip() if isinstance(value, str) else value


def execute_plan(parsed: ParsedDocument, plan: dict, specs: list[FieldSpec]) -> PlanResult:
    sheet = next((s for s in parsed.sheets if s.name == plan.get("sheet")), None)
    if sheet is None:
        raise PlanError(f"plan names an unknown sheet: {plan.get('sheet')!r}")
    rules = plan.get("fields")
    if not isinstance(rules, dict):
        raise PlanError("plan has no fields")
    # A rule that isn't an object ("llm", ["llm"]) means the plan had no rule for it.
    rules = {k: (v if isinstance(v, dict) else {"rule": "llm"}) for k, v in rules.items()}
    primary = next((s for s in specs if s.primary), None)
    if primary is None or (rules.get(primary.key) or {}).get("rule") != "column":
        raise PlanError("the primary field must use a column rule")

    blocks = find_blocks(sheet, plan)
    if not blocks:
        raise PlanError("plan matched no rows")

    llm_fields = [s for s in specs if (rules.get(s.key) or {"rule": "llm"}).get("rule") == "llm"]
    document_fields = {s.key: _document_field(parsed, sheet, rules[s.key], s)
                       for s in specs if (rules.get(s.key) or {}).get("rule") == "document"}
    entries, primary_values = [], []
    for block in blocks:
        entry = {}
        for spec in specs:
            rule = rules.get(spec.key) or {"rule": "llm"}
            if rule.get("rule") == "llm":
                entry[spec.key] = {"value": None, "citation": "", "confidence": 0.0}
            elif rule.get("rule") == "document":
                entry[spec.key] = dict(document_fields[spec.key])
            else:
                value, cited, confidence = _apply_rule(rule, block, sheet)
                value = coerce(_clean(value), spec)
                entry[spec.key] = {"value": value, "confidence": confidence,
                                   **(cited if value is not None and cited else {"citation": ""})}
        entries.append(entry)
        primary_values.append(entry[primary.key]["value"])

    # Rows without an identifier can't be scored or trusted.
    kept = [(e, b, p) for e, b, p in zip(entries, blocks, primary_values) if p is not None]
    if not kept:
        raise PlanError("plan produced no primary values")
    if len({str(p) for _, _, p in kept}) < len(kept) * 0.9:
        raise PlanError("plan produced mostly duplicate primary values")
    # A few repeats come from a later section listing the same rows again (a recap or an
    # appendix). The first listing is the main table, so keep that one.
    seen: set[str] = set()
    kept = [k for k in kept if not (str(k[2]) in seen or seen.add(str(k[2])))]
    return PlanResult(entries=[k[0] for k in kept], blocks=[k[1] for k in kept], sheet=sheet,
                      llm_fields=llm_fields, primary_values=[k[2] for k in kept])


def render_block(sheet: Sheet, block: Block, max_chars: int = 1500) -> str:
    return "\n".join(sheet.render_row(n) for n in block.rows)[:max_chars]


def sample_rows(sheet: Sheet, head: int = 45, middle: int = 25, tail: int = 12) -> str:
    numbers = sorted(sheet.rows)
    if len(numbers) <= head + middle + tail:
        return "\n".join(sheet.render_row(n) for n in numbers)
    mid_start = (len(numbers) - middle) // 2
    parts = [numbers[:head], numbers[mid_start:mid_start + middle], numbers[-tail:]]
    return "\n...\n".join("\n".join(sheet.render_row(n) for n in part) for part in parts)
