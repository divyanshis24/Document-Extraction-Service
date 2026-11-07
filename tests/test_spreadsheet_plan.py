import pytest

from app.extraction.fields import load_fields
from app.extraction.parsing import Cell, ParsedDocument, Sheet
from app.extraction.spreadsheet import PlanError, execute_plan, find_blocks, sample_rows


def sheet_from(rows, name="S"):
    sheet = Sheet(name=name)
    for number, cells in rows.items():
        sheet.rows[number] = [Cell(ref=f"{col}{number}", value=value) for col, value in cells.items()]
    return sheet


RENT_ROLL = {
    1: {"A": "Willow Creek"},
    2: {"A": "Unit", "B": "Rent", "H": "Code", "I": "Charge"},
    3: {"A": "100-1", "B": 3000, "H": "Rent", "I": 2800},
    4: {"H": "Pet", "I": 25},
    5: {"H": "Storage", "I": 50},
    6: {"G": "Charge Total:", "I": 2875},
    7: {"A": "100-2", "B": 2500, "H": "Storage", "I": 40},
    8: {"G": "Charge Total:", "I": 40},
    9: {"A": "Grand Total", "B": 5500},
}
FIELDS = {"fields": {
    "name": {"description": "Property. Same for every row."},
    "unit": {"description": "Unit id.", "primary": True},
    "market": {"description": "Market rent, as a number."},
    "rent": {"description": "Rent charge, as a number. 0 if none."},
    "other": {"description": "Sum of other charges, as a number."},
    "total": {"description": "Charge total, as a number."},
}}
PLAN = {
    "sheet": "S", "header_row": 2, "row_start": {"column": "A", "regex": r"^\d+-\d+$"},
    "fields": {
        "name": {"rule": "document", "value": "Willow Creek", "cell": "A1"},
        "unit": {"rule": "column", "column": "A"},
        "market": {"rule": "column", "column": "B"},
        "rent": {"rule": "block_match", "match_column": "H", "match": "rent", "value_column": "I", "default": 0},
        "other": {"rule": "block_sum", "match_column": "H", "exclude": ["Rent"], "value_column": "I", "default": 0},
        "total": {"rule": "block_match", "match_column": "G", "match": "Charge Total:", "value_column": "I"},
    },
}


def run(plan=PLAN, rows=RENT_ROLL, fields=FIELDS):
    return execute_plan(ParsedDocument(kind="sheets", sheets=[sheet_from(rows)]), plan, load_fields(fields))


def values(entry):
    return {k: v["value"] for k, v in entry.items()}


def test_blocks_are_folded_into_one_row_each():
    result = run()
    assert values(result.entries[0]) == {"name": "Willow Creek", "unit": "100-1", "market": 3000, "rent": 2800, "other": 75, "total": 2875}
    assert values(result.entries[1]) == {"name": "Willow Creek", "unit": "100-2", "market": 2500, "rent": 0, "other": 40, "total": 40}
    assert len(result.entries) == 2  # the total row doesn't match the row pattern


def test_cited_cells_are_exact():
    entry = run().entries[0]
    assert entry["market"]["citation"] == "[S!B3] 3000" and entry["market"]["source_locations"] == ["S!B3"]
    assert entry["other"]["source_locations"] == ["S!I4", "S!I5"]
    assert entry["name"]["citation"] == "[S!A1] Willow Creek"


def test_missing_value_without_default_is_null_with_no_citation():
    plan = {**PLAN, "fields": {**PLAN["fields"], "market": {"rule": "column", "column": "Z"}}}
    assert run(plan).entries[0]["market"] == {"value": None, "citation": "", "confidence": 0.8}


def test_exclude_and_require_filter_rows():
    rows = {1: {"A": "500", "B": "Header"}, 2: {"A": "510", "B": "Rent", "C": 5}, 3: {"A": "599", "B": "Total", "C": 5}, 4: {"A": "600", "B": "Tax", "C": 1}}
    plan = {"row_start": {"column": "A", "regex": r"^\d+$"}, "require_columns": ["C"], "exclude_regex": "^599"}
    assert [b.start for b in find_blocks(sheet_from(rows), plan)] == [2, 4]


def test_filtered_row_does_not_leak_into_the_previous_block():
    rows = {1: {"A": "1", "C": 5}, 2: {"A": "2"}, 3: {"H": "belongs to row 2"}}
    blocks = find_blocks(sheet_from(rows), {"row_start": {"column": "A"}, "require_columns": ["C"]})
    assert blocks[0].rows == [1]


def test_row_stats_and_nearest_header():
    rows = {1: {"B": "REVENUE"}, 2: {"A": "1", "B": "Rent", "C": 5, "D": 0, "E": 7}, 3: {"B": "EXPENSE"}, 4: {"A": "2", "B": "Tax", "C": 0, "D": 0, "E": 0}}
    fields = {"fields": {"id": {"primary": True}, "section": {}, "months": {}, "total": {}}}
    plan = {"sheet": "S", "row_start": {"column": "A"}, "fields": {
        "id": {"rule": "column", "column": "A"},
        "section": {"rule": "nearest_above", "column": "B", "require_empty_columns": ["C"]},
        "months": {"rule": "row_stat", "stat": "count_nonzero", "columns": ["C", "D", "E"]},
        "total": {"rule": "row_stat", "stat": "sum", "columns": ["C", "D", "E"]}}}
    result = run(plan, rows, fields)
    assert values(result.entries[0]) == {"id": 1 if False else "1", "section": "REVENUE", "months": 2, "total": 12}
    assert values(result.entries[1])["section"] == "EXPENSE" and values(result.entries[1])["months"] == 0


def test_fields_left_out_of_the_plan_go_to_the_model():
    plan = {**PLAN, "fields": {k: v for k, v in PLAN["fields"].items() if k != "other"}}
    result = run(plan)
    assert [s.key for s in result.llm_fields] == ["other"]
    assert result.entries[0]["other"]["value"] is None


@pytest.mark.parametrize("change,message", [
    ({"sheet": "Nope"}, "unknown sheet"),
    ({"fields": None}, "no fields"),
    ({"row_start": {}}, "row_start"),
    ({"row_start": {"column": "A", "regex": "("}}, "regular expression"),
    ({"row_start": {"column": "A", "regex": "^NO MATCH$"}}, "no rows"),
    ({"fields": {**PLAN["fields"], "unit": {"rule": "llm"}}}, "primary"),
    ({"fields": {**PLAN["fields"], "market": {"rule": "made_up"}}}, "unknown rule"),
    ({"fields": {**PLAN["fields"], "unit": {"rule": "column", "column": "Z"}}}, "no primary"),
])
def test_bad_plans_are_rejected_so_the_caller_can_fall_back(change, message):
    with pytest.raises(PlanError, match=message):
        run({**PLAN, **change})


def test_mostly_duplicate_identifiers_are_rejected():
    rows = {i: {"A": "1-1", "B": i} for i in range(1, 6)}
    with pytest.raises(PlanError, match="duplicate"):
        run(rows=rows)


def test_large_sheets_are_sampled_from_head_middle_and_tail():
    sheet = sheet_from({i: {"A": i} for i in range(1, 1001)})
    text = sample_rows(sheet)
    assert "R1 |" in text and "R500 |" in text and "R1000 |" in text and "R200 |" not in text


def test_last_row_does_not_swallow_grand_total_or_summary_sections():
    # The shape that broke rentroll_001: a total labelled on the left with an empty start
    # column, then summary sections with text in the start column.
    rows = {
        1: {"A": "Unit", "B": "Rent", "K": "Deposit"},
        2: {"A": "100-1", "B": 3000, "C": 900, "K": 500},
        3: {"G": "Charge Total:", "I": 3000},
        4: {"B": "Property Total:", "C": 900, "K": 500},
        5: {"A": "Status Summary", "B": 1},
        6: {"A": "Occupied", "B": 1, "K": 999},
    }
    plan = {"row_start": {"column": "A", "regex": r"^\d+-\d+$"}, "require_columns": ["A", "C"], "exclude_regex": "Total"}
    assert [b.rows for b in find_blocks(sheet_from(rows), plan)] == [[2, 3]]  # Charge Total stays, it's in G


def test_a_later_section_repeating_rows_keeps_the_first_listing():
    units = {10 + i: {"A": f"200-{i}", "B": 2000 + i} for i in range(10)}
    rows = {**RENT_ROLL, **units, 30: {"A": "Future Residents"}, 31: {"A": "100-2", "B": 9999, "H": "Rent", "I": 1}}
    result = run(rows=rows)
    assert [values(e)["unit"] for e in result.entries].count("100-2") == 1
    assert next(values(e) for e in result.entries if values(e)["unit"] == "100-2")["market"] == 2500


def test_repeated_charge_lines_are_added_up():
    rows = {**RENT_ROLL, 4: {"H": "Rent", "I": 0}, 5: {"H": "rent", "I": 150}}
    entry = run(rows=rows).entries[0]
    assert entry["rent"]["value"] == 2950 and entry["rent"]["source_locations"] == ["S!I3", "S!I4", "S!I5"]


@pytest.mark.parametrize("change", [
    {"fields": {**PLAN["fields"], "rent": "llm"}},
    {"fields": {**PLAN["fields"], "rent": ["llm"]}},
])
def test_rules_that_are_not_objects_go_to_the_model(change):
    result = run({**PLAN, **change})
    assert "rent" in [s.key for s in result.llm_fields]


def test_row_start_that_is_not_an_object_is_a_plan_error():
    with pytest.raises(PlanError):
        run({**PLAN, "row_start": "A"})


def test_document_value_not_in_the_sheet_is_not_trusted():
    plan = {**PLAN, "fields": {**PLAN["fields"], "name": {"rule": "document", "value": "Moonrise Towers", "cell": "A1"}}}
    assert run(plan).entries[0]["name"] == {"value": None, "citation": "", "confidence": 0.3}


def test_document_value_with_a_wrong_cell_is_cited_where_it_really_is():
    plan = {**PLAN, "fields": {**PLAN["fields"], "name": {"rule": "document", "value": "Willow Creek", "cell": "Z9"}}}
    assert run(plan).entries[0]["name"]["citation"] == "[S!A1] Willow Creek"
