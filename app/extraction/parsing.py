"""Turns any supported file into text the models can read and code can check.

Everything here is CPU-bound and blocking, so callers run it in a thread. Otherwise a large
workbook would starve the event loop and the worker's heartbeat with it.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import re
import zipfile
from dataclasses import dataclass, field

import docx
import openpyxl
import pymupdf
from docx.document import Document as DocxDocument
from openpyxl.utils import column_index_from_string, get_column_letter

from app.errors import ExtractionError

_MIN_TEXT_CHARS = 40  # below this a page has no usable text layer
_IMAGE_PAGE_MAX_CHARS = 400  # a mostly-image page with this little text gets read as an image
_IMAGE_PAGE_COVERAGE = 0.5
_RENDER_LONG_SIDE_PX = 1800
_MAX_UNZIPPED_BYTES = 300 * 1024 * 1024  # refuse zip bombs before opening them
_MAX_SHEET_CELLS = 2_000_000
_DOCX_SECTION_CHARS = 4000
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"


def unreadable(message: str) -> ExtractionError:
    return ExtractionError("document_unreadable", message)


@dataclass
class Page:
    """One PDF page, or one section of a DOCX. Image-only pages have text filled in later."""

    number: int
    label: str  # how citations refer to this unit, e.g. "Page 3"
    text: str = ""
    needs_vision: bool = False
    image_jpeg: bytes | None = None
    unreadable: bool = False


@dataclass
class Cell:
    ref: str  # e.g. "F4"
    value: object


@dataclass
class Sheet:
    name: str
    rows: dict[int, list[Cell]] = field(default_factory=dict)  # 1-based row number -> visible cells

    def cell(self, ref: str) -> Cell | None:
        match = re.fullmatch(r"([A-Z]+)(\d+)", ref.strip().upper())
        if not match:
            return None
        for cell in self.rows.get(int(match.group(2)), []):
            if cell.ref == ref.strip().upper():
                return cell
        return None

    def render_row(self, number: int) -> str:
        return f"R{number} | " + " | ".join(f"{re.sub(r'[0-9]', '', c.ref)}={format_cell(c.value)}" for c in self.rows[number])


@dataclass
class ParsedDocument:
    kind: str  # "pages" (PDF, DOCX) or "sheets" (XLSX, CSV)
    pages: list[Page] = field(default_factory=list)
    sheets: list[Sheet] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def format_cell(value: object) -> str:
    if isinstance(value, dt.datetime):
        return value.date().isoformat() if value.time() == dt.time(0) else value.isoformat(sep=" ")
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else repr(round(value, 6))
    return str(value).replace("\n", " ").strip()


def parse_document(content: bytes, file_type: str) -> ParsedDocument:
    try:
        if file_type == "pdf":
            return _parse_pdf(content)
        if file_type == "docx":
            return _parse_docx(content)
        if file_type == "xlsx":
            return _parse_xlsx(content)
        return _parse_csv(content)
    except ExtractionError:
        raise
    except Exception as exc:  # noqa: BLE001 - parsers raise many types on damaged files
        raise unreadable(f"Could not read this {file_type.upper()} file ({type(exc).__name__})") from exc


# ----- PDF -----

def _layout_lines(page: pymupdf.Page) -> str:
    """Rebuild lines from word positions, in visual reading order.

    The file's own text order can put a table's columns out of sequence, which would pair
    numbers with the wrong headers. Spacing is kept so columns stay visibly aligned.
    """
    words = page.get_text("words")
    if not words:
        return ""
    words.sort(key=lambda w: (w[1] + w[3]) / 2)
    lines: list[list[tuple]] = []
    current: list[tuple] = []
    current_y = None
    for word in words:
        y = (word[1] + word[3]) / 2
        height = max(word[3] - word[1], 1)
        if current_y is None or abs(y - current_y) <= height * 0.55:
            current.append(word)
            current_y = y if current_y is None else (current_y * (len(current) - 1) + y) / len(current)
        else:
            lines.append(current)
            current, current_y = [word], y
    lines.append(current)

    char_width = max(sum((w[2] - w[0]) / max(len(w[4]), 1) for w in words) / len(words), 1)
    out = []
    for line in lines:
        line.sort(key=lambda w: w[0])
        text, previous_end = "", None
        for word in line:
            if previous_end is not None:
                # Measured from the previous word's end: a normal gap is one space, and a
                # wide gap, which means a column break, keeps some visible width.
                gap = (word[0] - previous_end) / char_width
                text += " " * (1 if gap < 2.5 else min(int(gap), 8))
            text += word[4]
            previous_end = word[2]
        out.append(text)
    return "\n".join(out)


def _image_coverage(page: pymupdf.Page) -> float:
    area = abs(page.rect) or 1
    covered = 0.0
    for info in page.get_image_info():
        covered += abs(pymupdf.Rect(info["bbox"]) & page.rect)
    return min(covered / area, 1.0)


def _render_jpeg(page: pymupdf.Page) -> bytes:
    long_side = max(page.rect.width, page.rect.height) or 1
    zoom = min(_RENDER_LONG_SIDE_PX / long_side, 3.0)
    return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY, alpha=False).tobytes("jpeg", jpg_quality=75)


def _parse_pdf(content: bytes) -> ParsedDocument:
    with pymupdf.open(stream=content, filetype="pdf") as pdf:
        if pdf.needs_pass:
            raise unreadable("This PDF is password protected")
        if pdf.page_count == 0:
            raise unreadable("This PDF has no pages")
        parsed = ParsedDocument(kind="pages")
        for index, page in enumerate(pdf, start=1):
            entry = Page(number=index, label=f"Page {index}")
            try:
                entry.text = _layout_lines(page)
                chars = len(entry.text.strip())
                if chars < _MIN_TEXT_CHARS or (chars < _IMAGE_PAGE_MAX_CHARS and _image_coverage(page) >= _IMAGE_PAGE_COVERAGE):
                    entry.needs_vision = True
                    entry.image_jpeg = _render_jpeg(page)
            except Exception as exc:  # noqa: BLE001 - one bad page must not lose the others
                entry.unreadable = True
                parsed.warnings.append(f"Page {index} could not be read ({type(exc).__name__})")
            parsed.pages.append(entry)
    return parsed


# ----- DOCX -----

def _check_zip(content: bytes) -> None:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        if sum(info.file_size for info in archive.infolist()) > _MAX_UNZIPPED_BYTES:
            raise unreadable("This file expands to an unreasonable size")


def _paragraph_text(element) -> str:
    """All visible text of a paragraph, read from the XML.

    python-docx's Paragraph.text skips tracked insertions (w:ins) and content controls.
    Deleted text is stored as w:delText, so it stays out.
    """
    parts: list[str] = []

    def walk(node) -> None:
        for child in node:
            tag = child.tag
            if tag == _MC_FALLBACK:
                continue  # a second copy of the same text box, for older Word versions
            if tag == f"{_W}t":
                parts.append(child.text or "")
            elif tag in (f"{_W}tab", f"{_W}br", f"{_W}cr"):
                parts.append(" ")
            elif tag == f"{_W}txbxContent":
                # Text boxes are separate pieces of text, even inside one paragraph.
                parts.append(" ")
                walk(child)
                parts.append(" ")
            else:
                walk(child)

    walk(element)
    return " ".join("".join(parts).split())


def _xml_blocks(container):
    """Paragraph and table-row lines in reading order, descending into content controls
    (w:sdt), custom XML and similar wrappers that hide text from a plain body walk."""
    for element in container.iterchildren():
        tag = element.tag
        if tag == f"{_W}p":
            text = _paragraph_text(element)
            if text:
                yield text
        elif tag == f"{_W}tbl":
            for row in element.iterchildren(f"{_W}tr"):
                # One w:tc per grid cell, so a horizontally merged cell appears once. Empty
                # cells stay as "" so values keep their column positions.
                cells = [" ".join(_xml_blocks(cell)) for cell in row.iterchildren(f"{_W}tc")]
                while cells and not cells[-1]:
                    cells.pop()
                if any(cells):
                    yield " | ".join(cells)
        elif tag in (f"{_W}sectPr", f"{_W}tblPr", f"{_W}tcPr", f"{_W}trPr", _MC_FALLBACK):
            continue
        else:
            yield from _xml_blocks(element)


def _docx_blocks(document: DocxDocument):
    # Headers and footers often hold the property, parties or date. Each distinct one once.
    seen: set[str] = set()
    for section in document.sections:
        for part in (section.header, section.footer):
            try:
                if part.is_linked_to_previous:
                    continue
                text = " / ".join(_xml_blocks(part._element))
            except Exception:  # noqa: BLE001 - a broken header must not lose the body
                continue
            key = re.sub(r"\d+", "#", text)  # the same header with a different page number
            if text and key not in seen:
                seen.add(key)
                yield f"[Header/footer] {text}"
    yield from _xml_blocks(document.element.body)


def _parse_docx(content: bytes) -> ParsedDocument:
    _check_zip(content)
    document = docx.Document(io.BytesIO(content))
    parsed = ParsedDocument(kind="pages")
    # DOCX has no fixed pages, so citations point at numbered sections instead.
    buffer: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal buffer, size
        if buffer:
            number = len(parsed.pages) + 1
            parsed.pages.append(Page(number=number, label=f"Section {number}", text="\n".join(buffer)))
        buffer, size = [], 0

    for block in _docx_blocks(document):
        buffer.append(block)
        size += len(block)
        if size >= _DOCX_SECTION_CHARS:
            flush()
    flush()
    if not parsed.pages:
        raise unreadable("This DOCX has no text")
    return parsed


# ----- Spreadsheets -----

_SUM_RE = re.compile(r"^=\s*SUM\(\s*([A-Z]+)(\d+)\s*:\s*([A-Z]+)(\d+)\s*\)\s*$", re.I)


def _resolve_formula(formula: str, value_at) -> object:
    """Files written by some tools hold formulas with no saved result. SUM ranges are common."""
    match = _SUM_RE.match(formula)
    if not match or match.group(1).upper() != match.group(3).upper():
        return None
    column = column_index_from_string(match.group(1).upper())
    total, found = 0.0, False
    for row in range(int(match.group(2)), int(match.group(4)) + 1):
        value = value_at(row, column)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total += value
            found = True
    return round(total, 6) if found else None


def _parse_xlsx(content: bytes) -> ParsedDocument:
    _check_zip(content)
    computed = openpyxl.load_workbook(io.BytesIO(content), data_only=True)
    formulas = openpyxl.load_workbook(io.BytesIO(content), data_only=False)
    parsed = ParsedDocument(kind="sheets")
    for ws in computed.worksheets:
        if ws.sheet_state != "visible":
            parsed.warnings.append(f"Skipped hidden sheet {ws.title!r}")
            continue
        # Only cells stored in the file. ws.max_row counts cells that merely carry a style,
        # so one formatted cell at row 1,048,576 would look like a billion-cell sheet, and
        # iterating ws.iter_rows() would create every one of those cells in memory.
        stored = ws._cells  # noqa: SLF001 - openpyxl's public API has no sparse iterator
        if len(stored) > _MAX_SHEET_CELLS:
            parsed.warnings.append(f"Skipped sheet {ws.title!r}: too large to process")
            continue
        raw_cells = formulas[ws.title]._cells  # noqa: SLF001

        def value_at(row: int, col: int, _raw=raw_cells):
            cell = _raw.get((row, col))
            return cell.value if cell is not None else None
        hidden_cols = {k for k, d in ws.column_dimensions.items() if d.hidden}
        # A hidden group is stored on its first column with a min/max range.
        for key, dim in ws.column_dimensions.items():
            if dim.hidden and dim.min and dim.max:
                hidden_cols.update(get_column_letter(i) for i in range(dim.min, dim.max + 1))
        hidden_rows = {k for k, d in ws.row_dimensions.items() if d.hidden}

        merged: dict[str, object] = {}
        for rng in ws.merged_cells.ranges:
            # Spreading a value across a merged range matters only when it spans rows, such
            # as a section label. A title merged across columns just adds noise.
            if rng.min_row != rng.max_row:
                top_left = ws.cell(rng.min_row, rng.min_col).value
                for row in range(rng.min_row, rng.max_row + 1):
                    for col in range(rng.min_col, rng.max_col + 1):
                        merged[f"{get_column_letter(col)}{row}"] = top_left

        sheet = Sheet(name=ws.title)
        positions = sorted(set(stored) | {(r, c) for r, c in (_split_ref(ref) for ref in merged)})
        for number, col in positions:
            if number in hidden_rows or get_column_letter(col) in hidden_cols:
                continue
            cell = stored.get((number, col))
            coordinate = f"{get_column_letter(col)}{number}"
            value = cell.value if cell is not None else None
            if value is None:
                formula = value_at(number, col)
                if isinstance(formula, str) and formula.startswith("="):
                    value = _resolve_formula(formula, value_at)
                elif coordinate in merged:
                    value = merged[coordinate]
            elif cell is not None and _is_percent(cell, value):
                # Stored as a fraction but printed as a percentage. The contract wants
                # percentages as printed, so 0.0526 shown as 5.26% becomes 5.26.
                value = round(value * 100, 10)
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            sheet.rows.setdefault(number, []).append(Cell(ref=coordinate, value=value))
        if sheet.rows:
            parsed.sheets.append(sheet)
    if not parsed.sheets:
        raise unreadable("This spreadsheet has no visible data")
    return parsed


def _split_ref(ref: str) -> tuple[int, int]:
    match = re.fullmatch(r"([A-Z]+)(\d+)", ref)
    return int(match.group(2)), column_index_from_string(match.group(1))


def _is_percent(cell, value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and "%" in str(getattr(cell, "number_format", "") or ""))


def _decode_text(content: bytes) -> str:
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        return content.decode("utf-16")
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("cp1252", errors="replace")


def _parse_csv(content: bytes) -> ParsedDocument:
    text = _decode_text(content)
    if "\x00" in text:
        raise unreadable("This CSV contains binary data")
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    csv.field_size_limit(10 * 1024 * 1024)
    sheet = Sheet(name="Sheet1")
    for number, row in enumerate(csv.reader(io.StringIO(text), dialect), start=1):
        cells = [Cell(ref=f"{get_column_letter(i)}{number}", value=_csv_value(v.strip()))
                 for i, v in enumerate(row, start=1) if v.strip()]
        if cells:
            sheet.rows[number] = cells
    if not sheet.rows:
        raise unreadable("This CSV has no data")
    return ParsedDocument(kind="sheets", sheets=[sheet])


def _csv_value(text: str) -> object:
    # Plain decimals only. Codes such as "00501" or "510-500" must stay text.
    if re.fullmatch(r"-?(0|[1-9]\d*)(\.\d+)?", text):
        return float(text) if "." in text else int(text)
    return text
