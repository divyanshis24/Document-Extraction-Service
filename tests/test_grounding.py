"""Values must really be in the document. These are the cases the old substring check got wrong."""
import pytest

from app.extraction.grounding import TextIndex, value_in

TEXT = TextIndex("Unit 101 rent $1,750.00 from March 10,2023; made the 15th day of January, 2024; deposit (2,500)")


@pytest.mark.parametrize("value,expected", [
    (1750, True), (1750.0, True), (101, True), (2500, True), (-2500, True), (2023, True), (10, True),
    (7, False), (250, False), (750, False), (9999, False),  # not digit fragments of other numbers
    ("2024-01-15", True), ("2023-03-10", True), ("2024-01-16", False),
    ("Unit 101", True), ("unit  101", True), ("Unit 102", False),
    (None, True), ([1750, 2500], True), ([1750, 9999], False),
])
def test_values_are_found_only_when_really_there(value, expected):
    assert value_in(value, TEXT) is expected


@pytest.mark.parametrize("text", ["01/15/2024", "1/15/24", "January 15, 2024", "Jan 15, 2024", "15 January 2024",
                                  "January 15th, 2024", "15th day of January, 2024", "2024-01-15"])
def test_dates_are_found_in_printed_forms(text):
    assert value_in("2024-01-15", TextIndex(f"Commencement: {text}."))


def test_year_month_matches_a_month_heading():
    assert value_in("2023-02", TextIndex("Feb 2023")) and not value_in("2023-03", TextIndex("Feb 2023"))


def test_first_or_last_day_of_a_month_matches_the_month_heading():
    period = TextIndex("Period = Nov 2024-Oct 2025")
    assert value_in("2024-11-01", period) and value_in("2025-10-31", period)
    assert not value_in("2024-11-15", period)
