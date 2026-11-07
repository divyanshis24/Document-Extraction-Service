import datetime as dt

import pytest

from app.extraction.fields import coerce, load_fields


def spec(description, key="f", **extra):
    return load_fields({"fields": {key: {"description": description, **extra}}})[0]


@pytest.mark.parametrize("description,key,kind", [
    ("Rentable square footage, as a number.", "f", "number"),
    ("Whole number of units.", "f", "number"),
    ("Count of vacant units.", "f", "number"),
    ("Number of model units.", "f", "number"),
    ("ZIP code as a string.", "f", "text"),
    ("Number of stories. Keep the OM's wording.", "f", "text"),
    ("Start date in YYYY-MM-DD.", "f", "date"),
    ("The month, as YYYY-MM.", "report_date", "text"),
    ("When the lease begins.", "lease_start_date", "date"),
    ("true if a letter of credit is required, false otherwise.", "f", "bool"),
    ("List of the brokers.", "f", "list"),
    ("Name of the tenant.", "f", "text"),
])
def test_kind_is_inferred_from_the_description(description, key, kind):
    assert spec(description, key).kind == kind


def test_list_item_kinds():
    assert spec("List of the start date of each period").item_kind == "date"
    assert spec("List of the monthly rent. Numbers only.").item_kind == "number"
    assert spec("List of the brokers").item_kind == "text"


def test_document_level_and_primary_and_computed():
    assert spec("Property name. Same for every row.").document_level
    assert not spec("Code. Same for every row.", primary=True).document_level
    assert spec("Code", primary=True).primary
    assert spec("Compute it from the dates.").computed
    assert spec("Asking price divided by units.").computed
    assert not spec("Count of residential buildings, as a number.").computed


def test_missing_description_falls_back_to_name_then_key():
    fields = load_fields({"fields": {"a": {"name": "Alpha"}, "b": {}}})
    assert fields[0].description == "Alpha" and fields[1].description == "b"


@pytest.mark.parametrize("value,expected", [
    ("$1,234.50", 1234.5), ("(500)", -500), ("93.75%", 93.75), (12, 12), (1.5, 1.5), ("N/A", None), ("none", None),
    ("", None), (None, None), ("no figure", None), ("about 5", "about 5"), (["7"], 7), ([1, 2], None),
])
def test_coerce_numbers(value, expected):
    assert coerce(value, spec("As a number.")) == expected


@pytest.mark.parametrize("value,expected", [
    ("03/05/1993", "1993-03-05"), ("March 5, 1993", "1993-03-05"), ("March 5th, 1993", "1993-03-05"), ("Mar 5, 1993", "1993-03-05"),
    ("1993-03-05", "1993-03-05"), (dt.datetime(1993, 3, 5), "1993-03-05"), ("not stated", None), (None, None),
    ("the first Monday", "the first Monday"),
])
def test_coerce_dates(value, expected):
    assert coerce(value, spec("In YYYY-MM-DD.")) == expected


def test_coerce_bool_and_text_and_list():
    assert coerce("Yes", spec("true if it applies.")) is True and coerce(False, spec("true if it applies.")) is False
    assert coerce("  keep me", spec("A name.")) == "  keep me" and coerce(5, spec("A name.")) == 5
    s = spec("List of the monthly rent. Numbers only.")
    assert coerce(["$1,000", 2000], s) == [1000, 2000] and coerce(1000, s) == [1000]
    assert coerce([], s) == [] and coerce(None, s) is None


@pytest.mark.parametrize("text,expected", [
    ("$(1,234.00)", -1234), ("$ (1,234)", -1234), ("(5)", -5), ("1,234-", -1234), ("−1,250", -1250),
    ("1.234,56", 1234.56), ("€1.234,56", 1234.56), ("1.234.567", 1234567), ("12,50", 12.5),
    ("1.250", 1.25),  # a lone dot group stays a decimal: in a US document that's a rate
    ("1,250 SF", 1250), ("12 months", 12), ("$4,250/mo", 4250), ("5.26%", 5.26), ("-46.78%", -46.78),
    ("1,2,3", None), ("abc", None),
])
def test_numbers_as_documents_print_them(text, expected):
    from app.extraction.fields import to_number
    assert to_number(text) == expected


def test_list_items_are_typed_by_what_the_list_is_of():
    from app.extraction.fields import load_fields
    specs = {s.key: s.item_kind for s in load_fields({"fields": {
        "starts": {"description": "List of the start date of each rent period, in order."},
        "rents": {"description": "List of the monthly base rent for each period, aligned item by item with the start dates."},
        "terms": {"description": "List of the length of each renewal option in months."},
        "brokers": {"description": "List of the brokerage firms named in the lease."},
    }})}
    assert specs == {"starts": "date", "rents": "number", "terms": "number", "brokers": "text"}


def test_one_of_fields_accept_only_the_listed_options():
    from app.extraction.fields import coerce, load_fields
    spec = load_fields({"fields": {"category": {"description": "One of: Revenue, Expense, Other. Revenue for income lines."}}})[0]
    assert spec.choices == ("Revenue", "Expense", "Other")
    assert [coerce(v, spec) for v in ("expense", "Revenue", "Income")] == ["Expense", "Revenue", None]


@pytest.mark.parametrize("text", ["-- Vacant --", "--", "-", "— Admin —"])
def test_placeholder_cells_mean_no_value(text):
    from app.extraction.fields import coerce, load_fields
    assert coerce(text, load_fields({"fields": {"name": {"description": "Resident name."}}})[0]) is None


def test_a_short_value_drops_the_field_name_repeated_after_it():
    from app.extraction.fields import coerce, load_fields
    spec = load_fields({"fields": {"submarket": {"name": "Submarket", "description": "Submarket the OM names."}}})[0]
    assert coerce("Fairhaven International Airport submarket", spec) == "Fairhaven International Airport"
    assert coerce("Submarket", spec) == "Submarket"
    city = load_fields({"fields": {"city": {"name": "City", "description": "City."}}})[0]
    assert coerce("Oklahoma City", city) == "Oklahoma City"  # part of the name, not a label
    long = "The property sits in a growing area that the broker calls the airport submarket"
    assert coerce(long, spec) == long
