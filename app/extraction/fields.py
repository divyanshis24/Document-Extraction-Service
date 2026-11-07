"""Reads a field list and works out what each field expects, using only its description.

Nothing here knows any particular field list, so lists it has never seen work the same way.
Two layers decide a field's type:
- Explicit wording ("YYYY-MM-DD", "as a number", "List of", "Same for every row") always
  decides, because the field list's author said so.
- Anything else is decided by a model reading the whole field list once (see
  refine_with_model), with broad keyword rules as the fallback when no model answer is
  available. Keywords alone only understand the phrasing they were written for.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, replace

_DOC_LEVEL = re.compile(r"same for every|same value|every row|whole document|entire document", re.I)
_COMPUTED = re.compile(r"\b(comput\w*|calculat\w*|sum of|divided by|rounded|times 12|derive\w*)\b", re.I)
# "The nearest section header above the line": a value that holds until the next header,
# so a row on a continuation page inherits it even though the header isn't on that page.
_CARRIES_DOWN = re.compile(r"\b(header|heading|section|group|category)\b[^.]*\babove\b", re.I)
_CHOICES = re.compile(r"\bone of\s*:?\s*(.+?)(?:\.\s|\.$|;|$)", re.I)
_LIST = re.compile(r"\blists? of\b", re.I)
_BOOL = re.compile(r"^\s*(true if|whether\b.*\btrue\b)", re.I)
_NUMBER = re.compile(r"\b(as a (percentage )?number|whole number|numbers only|four-digit (number|year)|as a number)\b", re.I)
_NOT_NUMBER = re.compile(r"\b(wording|as a string|short phrase|in words|verbatim|as stated|as the lease states|describe\w*)\b", re.I)
# Fallback cues, used only when neither explicit wording nor a model answer decides.
_NUMBER_HINT = re.compile(r"(\bin (us )?dollars\b|\(\$\)|\$|\busd\b|\bamount\b|\bsquare f(ee|oo)t\w*|\bsq\.? ?ft\b|\bhow many\b|"
                          r"\bcount\b|\bnumber of\b|\bpercent\w*|\(%\)|\brate\b|\bratio\b|\bprice\b|\bcost\b|\brent\b|"
                          r"\bfee\b|\bbalance\b|\bdeposit\b|\bnoi\b|\bincome\b|\bexpense\b|\byear (built|renovated)\b|"
                          r"\bacres?\b|\bunits\b|\bin months\b|\bin years\b)", re.I)
_DATE_HINT = re.compile(r"\b(date|dated|when|expir\w*|commence\w*|as of|as-of|start|end|begin\w*)\b", re.I)
_DOC_LEVEL_HINT = re.compile(r"\b(report header|page header|top of the (report|page|document)|applies to (all|every)|"
                             r"repeated (on|for|in) (each|every|all)|for the whole (report|statement|document))\b", re.I)
_LIST_HINT = re.compile(r"\b(as a list|as an array|list them|one item per)\b", re.I)
KINDS = ("text", "number", "date", "bool", "list")
# A cell that marks absence rather than holding a value: "-", "--", "-- Vacant --".
_PLACEHOLDER = re.compile(r"[-\u2013\u2014]+|[-\u2013\u2014]{1,3}\s*[A-Za-z][A-Za-z ]*?\s*[-\u2013\u2014]{1,3}")
_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y",
                 "%d %B %Y", "%d %b %Y", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S")


@dataclass(frozen=True)
class FieldSpec:
    key: str
    name: str
    description: str
    primary: bool
    kind: str  # text, number, date, bool or list
    item_kind: str  # for lists: text, number or date
    document_level: bool
    computed: bool
    carries_down: bool = False
    choices: tuple[str, ...] = ()
    kind_explicit: bool = False  # the description states the type, so nothing overrides it
    level_explicit: bool = False  # the description says it is the same for every row  # "One of: Revenue, Expense, Other": a judgement, not a copied value

    def choice(self, value):
        """The listed option a value names, spelled as listed, or None."""
        wanted = str(value).strip().lower()
        return next((c for c in self.choices if c.lower() == wanted), None)

    def prompt_line(self) -> str:
        hint = {"number": "JSON number", "date": "YYYY-MM-DD string", "bool": "true or false",
                "list": f"JSON array of {self.item_kind} values", "text": "string, or a number if the column is numeric"}[self.kind]
        return f'- "{self.key}" ({self.name}; {hint}): {self.description}'


def _is_date(key: str, description: str) -> bool:
    if "YYYY-MM-DD" in description:
        return True
    if re.search(r"YYYY-MM\b", description):
        return False
    return bool(re.search(r"(^|_)date(s)?($|_)", key.lower()))


_LIST_NOUN = re.compile(r"\blists? of (?:the |all |each )?(.+?)(?=\b(?:for|of|in|named|that|which|aligned|from)\b|[,.(;:]|$)", re.I)
_NUMBER_NOUN = re.compile(r"\b(rents?|amounts?|prices?|costs?|fees?|months?|years?|days?|rates?|counts?|numbers?|"
                          r"percent\w*|square f\w*|rsf|sf|dollars?|payments?|values?)\b", re.I)


def _choices(description: str) -> tuple[str, ...]:
    match = _CHOICES.search(description)
    if not match:
        return ()
    options = [o.strip(" '\"") for o in re.split(r",|\bor\b", match.group(1))]
    options = [o for o in options if o and len(o) <= 40]
    return tuple(options) if 2 <= len(options) <= 20 else ()


def _list_item_kind(description: str) -> str:
    """What one item of a list is, judged by the noun the list is of.

    "List of the monthly base rent for each period, aligned with the start dates" is a list
    of numbers, even though it mentions dates, so the whole description can't decide it.
    """
    if re.search(r"\bnumbers? only\b", description, re.I):
        return "number"
    match = _LIST_NOUN.search(description)
    noun = match.group(1) if match else ""
    if re.search(r"\bdates?\b", noun, re.I):
        return "date"
    if _NUMBER_NOUN.search(noun) or re.search(r"\bin (months|years|days)\b|\bper rentable\b", description, re.I):
        return "number"
    return "text"


def _explicit_kind(key: str, description: str) -> str | None:
    """The type the description states outright, or None if it doesn't say."""
    if _LIST.search(description):
        return "list"
    if _BOOL.search(description):
        return "bool"
    if re.search(r"YYYY-MM\b(?!-)", description):
        return "text"  # a month written as YYYY-MM, which isn't a full date
    if _is_date(key, description):
        return "date"
    if _NOT_NUMBER.search(description):
        return "text" if not _NUMBER.search(description) else None
    if _NUMBER.search(description) or re.match(r"^(count|number) of\b", description, re.I):
        return "number"
    if _choices(description):
        return "text"
    return None


def _guessed_kind(key: str, description: str) -> str:
    """Fallback when nothing explicit and no model answer: broad cues from wording and key."""
    # Examples ("e.g. 'Rent Revenue'") and quoted names say what the value looks like, not
    # what it is, so they are not cues. A "%" anywhere is, even in a quoted column name.
    plain = re.sub(r"\be\.g\.[^.;)]*|'[^']*'|\"[^\"]*\"", " ", description)
    words = f"{key.replace('_', ' ')} {plain}"
    if _LIST_HINT.search(plain):
        return "list"
    if re.search(r"^(is|has|can|allows?)_", key) or re.search(r"\(true/false\)|\byes or no\b", plain, re.I):
        return "bool"
    if "%" in description or re.search(r"\b(length|duration|number of (months|years|days)|in (whole )?(months|years|days))\b", words, re.I):
        return "number"
    not_a_number = re.search(r"\b(name|address|description|what|which|who)\b|\bhow\b(?! many| much)", plain, re.I)
    if _DATE_HINT.search(words) and not _NUMBER_HINT.search(plain):
        return "date"
    if _NUMBER_HINT.search(words) and not not_a_number:
        return "number"
    return "text"


def load_fields(field_list: dict) -> list[FieldSpec]:
    specs = []
    for key, raw in field_list["fields"].items():
        description = str(raw.get("description") or raw.get("name") or key)
        primary = raw.get("primary") is True
        explicit = _explicit_kind(key, description)
        kind = explicit or _guessed_kind(key, description)
        level_explicit = bool(_DOC_LEVEL.search(description)) and not primary
        specs.append(FieldSpec(
            key=key, name=str(raw.get("name") or key), description=description, primary=primary,
            kind=kind, item_kind=_list_item_kind(description) if kind == "list" else "text",
            document_level=(level_explicit or bool(_DOC_LEVEL_HINT.search(description))) and not primary,
            computed=bool(_COMPUTED.search(description)),
            carries_down=bool(_CARRIES_DOWN.search(description)) and not primary,
            choices=_choices(description) if kind == "text" else (),
            kind_explicit=explicit is not None, level_explicit=level_explicit,
        ))
    return specs


def refine_with_model(specs: list[FieldSpec], answer) -> list[FieldSpec]:
    """Apply a model's reading of the field list wherever the description isn't explicit.

    The answer maps field key to {"type", "item_type", "whole_document"}. Anything missing,
    malformed or outside the allowed values is ignored, so a bad answer changes nothing.
    """
    if not isinstance(answer, dict):
        return specs
    refined = []
    for spec in specs:
        got = answer.get(spec.key)
        if not isinstance(got, dict):
            refined.append(spec)
            continue
        kind, item_kind, level = spec.kind, spec.item_kind, spec.document_level
        if not spec.kind_explicit and got.get("type") in KINDS:
            kind = got["type"]
            if kind == "list":
                item = got.get("item_type")
                item_kind = item if item in ("text", "number", "date") else _list_item_kind(spec.description)
        if not spec.level_explicit and not spec.primary and isinstance(got.get("whole_document"), bool):
            level = got["whole_document"]
        refined.append(replace(spec, kind=kind, item_kind=item_kind, document_level=level,
                               choices=spec.choices if kind == "text" else ()))
    return refined


def to_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip().replace("\u2212", "-").replace("\u2013", "-")  # Unicode minus and en dash
    # A trailing unit ("1,250 SF", "12 months", "$4,250/mo") says what the number measures.
    text = re.sub(r"(?<=[\d.)%])\s*(?:/\s*)?[A-Za-z][A-Za-z. /]*$", "", text).strip()
    # Accounting negatives: "(1,234)", "$(1,234.00)", "($ 1,234)", and a trailing "1,234-".
    negative = bool(re.fullmatch(r"[$€£₹]?\s*\(\s*[$€£₹]?\s*[\d.,]+\s*%?\s*\)", text)) or text.endswith("-")
    cleaned = re.sub(r"[$€£₹%\s()]", "", text).rstrip("-")
    if cleaned.startswith("-"):
        negative, cleaned = True, cleaned[1:]
    cleaned = cleaned.lstrip("+")
    if re.fullmatch(r"\d{1,3}(\.\d{3})+,\d+|\d{1,3}(\.\d{3}){2,}|\d+,\d{1,2}", cleaned):
        # European format: dots group thousands and a comma marks decimals ("1.234,56").
        # A lone "1.250" stays 1.25: in a US document that is a rate, not one thousand.
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif not re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?|\d+(\.\d+)?|\.\d+", cleaned):
        return None
    number = float(cleaned.replace(",", ""))
    number = -number if negative else number
    return int(number) if number.is_integer() else number


def to_date(value):
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if not isinstance(value, str):
        return None
    text = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", value.strip())
    for fmt in _DATE_FORMATS:
        try:
            return dt.datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    return None


def _coerce_scalar(value, kind: str):
    if isinstance(value, str) and (value.strip().lower() in ("", "null", "none", "n/a", "na", "not stated", "not found")
                                   or _PLACEHOLDER.fullmatch(value.strip())):
        return None
    if value is None or kind == "text":
        if isinstance(value, (dt.date, dt.datetime)):
            return to_date(value)
        return value
    if kind == "number":
        number = to_number(value)
        return number if number is not None else (None if isinstance(value, str) and not re.search(r"\d", value) else value)
    if kind == "date":
        parsed = to_date(value)
        return parsed if parsed is not None else value  # kept for review rather than silently lost
    if kind == "bool":
        if isinstance(value, str):
            return {"true": True, "yes": True, "false": False, "no": False}.get(value.strip().lower(), value)
        return value
    return value


def coerce(value, spec: FieldSpec):
    """Put a value in the contract's format. Never invents a value that isn't there."""
    if spec.choices and value is not None and not isinstance(value, list):
        # A value outside the listed options isn't one of the answers the field allows.
        return spec.choice(value)
    if spec.kind == "list":
        if value is None:
            return None
        items = value if isinstance(value, list) else [value]
        return [_coerce_scalar(item, spec.item_kind) for item in items]
    if isinstance(value, list):
        value = value[0] if len(value) == 1 else None
    value = _coerce_scalar(value, spec.kind)
    if spec.kind == "text" and isinstance(value, str):
        value = _without_own_name(value, spec.name)
    return value


def _without_own_name(value: str, name: str) -> str:
    """'Fairhaven Airport submarket' for a field named Submarket is 'Fairhaven Airport'.

    Only for short values such as names, where the trailing word is a label rather than
    part of a sentence, only when it is written in lowercase (so "Oklahoma City" keeps its
    "City", which is part of the name), and only when something is left.
    """
    words = value.split()
    label = name.strip().lower()
    if not label or len(words) > 6 or not value.endswith(" " + label):
        return value
    trimmed = value[: -len(label)].rstrip(" ,-")
    return trimmed if trimmed else value
