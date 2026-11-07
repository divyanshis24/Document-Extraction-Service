"""Checks that a value really appears in the document. A value that can't be found isn't trusted.

Text is compared with case, spacing and punctuation ignored. Numbers are compared as whole
numbers, so 250 is not found inside "1,250" and 7 is not found inside "1,750". Dates are
looked for in the forms documents print them in ("01/15/2024", "January 15, 2024",
"the 15th day of January, 2024"), not only as YYYY-MM-DD.
"""
from __future__ import annotations

import calendar
import re

from app.extraction.fields import to_number

_NUMBER_TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?")
_ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_ISO_MONTH = re.compile(r"^(\d{4})-(\d{2})$")


def norm(text) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _suffix(day: int) -> str:
    return "th" if 11 <= day % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def date_forms(value: str) -> list[str]:
    """The normalised ways a YYYY-MM-DD or YYYY-MM value can be printed."""
    match = _ISO_DATE.match(value)
    if match:
        y, m, d = (int(g) for g in match.groups())
        if not (1 <= m <= 12 and 1 <= d <= 31):
            return [norm(value)]
        full, short = calendar.month_name[m], calendar.month_abbr[m]
        yy = f"{y % 100:02d}"
        forms = [
            value, f"{m}/{d}/{y}", f"{m:02d}/{d:02d}/{y}", f"{m}/{d}/{yy}", f"{m:02d}/{d:02d}/{yy}",
            f"{m}-{d}-{y}", f"{m:02d}-{d:02d}-{y}", f"{y}/{m:02d}/{d:02d}", f"{m}.{d}.{y}",
        ]
        for month in {full, short, "Sept" if m == 9 else short}:
            forms += [f"{month} {d}, {y}", f"{month} {d}{_suffix(d)}, {y}", f"{d} {month} {y}",
                      f"{d}{_suffix(d)} day of {month}, {y}", f"{d} day of {month}, {y}"]
        normalised = [norm(f) for f in forms]
        if d == 1 or d == calendar.monthrange(y, m)[1]:
            # The first or last day of a month is how a period like "Nov 2024" is written
            # as a date, so the month on its own also counts.
            normalised += date_forms(f"{y}-{m:02d}")
        return normalised
    match = _ISO_MONTH.match(value)
    if match:
        y, m = int(match.group(1)), int(match.group(2))
        if not 1 <= m <= 12:
            return [norm(value)]
        return [norm(f) for f in (value, f"{calendar.month_name[m]} {y}", f"{calendar.month_abbr[m]} {y}",
                                  f"{m}/{y}", f"{m:02d}/{y}", f"{m:02d}-{y}")]
    return [norm(value)]


class TextIndex:
    """One piece of document text, prepared for repeated lookups."""

    __slots__ = ("norm", "numbers")

    def __init__(self, text: str) -> None:
        text = str(text)
        self.norm = norm(text)
        self.numbers: set[float] = set()
        for token in _NUMBER_TOKEN.findall(text):
            token = token.rstrip(",")
            whole = to_number(token)
            # A token that isn't one number, like "March 10,2023" or "1,2", is several. A
            # valid grouping like "1,250" is one number only, so 250 is not found inside it.
            pieces = [token] if whole is not None else token.split(",")
            for piece in pieces:
                number = to_number(piece) if piece else None
                if number is not None:
                    # Sign is ignored: documents print negatives as -5, (5) or 5 under a
                    # "decrease" heading, so the check is whether the figure is there at all.
                    self.numbers.add(round(abs(float(number)), 6))


def value_in(value, index: TextIndex) -> bool:
    """Whether a value literally appears in the text, in any common formatting."""
    if value is None or isinstance(value, bool):
        return True
    if isinstance(value, list):
        return all(value_in(item, index) for item in value)
    if isinstance(value, (int, float)):
        return round(abs(float(value)), 6) in index.numbers
    text = str(value)
    if _ISO_DATE.match(text) or _ISO_MONTH.match(text):
        return any(form in index.norm for form in date_forms(text))
    wanted = norm(text)
    return wanted in index.norm
