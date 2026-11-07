import datetime as dt
import io

import docx
import openpyxl
import pymupdf
import pytest

from app.errors import ExtractionError
from app.extraction.parsing import format_cell, parse_document


def xlsx(build):
    wb = openpyxl.Workbook()
    build(wb)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def pdf(build):
    doc = pymupdf.open()
    build(doc)
    return doc.tobytes()


def test_csv_keeps_codes_as_text_and_numbers_as_numbers():
    p = parse_document(b"Code,Zip,Rent,Pct\nPR - P1,00501,5135,89.47\n", "csv")
    s = p.sheets[0]
    assert s.cell("A2").value == "PR - P1" and s.cell("B2").value == "00501"
    assert s.cell("C2").value == 5135 and s.cell("D2").value == 89.47


@pytest.mark.parametrize("content", [
    "a,b\nx,1\n".encode("utf-8-sig"), "a,b\nx,1\n".encode("utf-16"), "a,b\nx,1\n".encode("cp1252"),
    b"a;b\nx;1\n", b"a\tb\nx\t1\n",
])
def test_csv_encodings_and_delimiters(content):
    assert parse_document(content, "csv").sheets[0].cell("B2").value == 1


def test_csv_with_no_data_is_unreadable():
    with pytest.raises(ExtractionError) as e:
        parse_document(b" , ,\n,,\n", "csv")
    assert e.value.code == "document_unreadable"


def test_xlsx_skips_hidden_columns_rows_and_sheets():
    def build(wb):
        ws = wb.active
        ws.title = "Main"
        ws.append(["keep", "hidden col", "keep2"])
        ws.append(["hidden row", "x", "y"])
        ws.append(["third", "x", "z"])
        ws.column_dimensions["B"].hidden = True
        ws.row_dimensions[2].hidden = True
        wb.create_sheet("Secret").append(["secret"])
        wb["Secret"].sheet_state = "hidden"

    p = parse_document(xlsx(build), "xlsx")
    assert [s.name for s in p.sheets] == ["Main"]
    s = p.sheets[0]
    assert s.cell("A1").value == "keep" and s.cell("B1") is None and s.cell("A2") is None and s.cell("C3").value == "z"
    assert any("Secret" in w for w in p.warnings)


def test_xlsx_formula_with_no_saved_result_is_calculated():
    def build(wb):
        ws = wb.active
        ws["A1"], ws["A2"], ws["A3"], ws["A4"] = 10, 2.5, "=SUM(A1:A2)", "=A1*2"

    s = parse_document(xlsx(build), "xlsx").sheets[0]
    assert s.cell("A3").value == 12.5
    assert s.cell("A4") is None  # unknown formula: left empty, never guessed


def test_xlsx_value_merged_down_rows_is_spread_but_across_columns_is_not():
    def build(wb):
        ws = wb.active
        ws["A1"] = "Section"
        ws.merge_cells("A1:A3")
        ws["C1"] = "Title"
        ws.merge_cells("C1:E1")
        ws["B3"] = 1

    s = parse_document(xlsx(build), "xlsx").sheets[0]
    assert s.cell("A3").value == "Section" and s.cell("C1").value == "Title" and s.cell("D1") is None


def test_xlsx_with_nothing_visible_is_unreadable():
    with pytest.raises(ExtractionError):
        parse_document(xlsx(lambda wb: None), "xlsx")


@pytest.mark.parametrize("kind,content", [
    ("xlsx", b"PK\x03\x04broken"), ("xlsx", b"not a zip"), ("docx", b"PK\x03\x04broken"), ("pdf", b"%PDF-1.7 broken"), ("pdf", b"junk"),
])
def test_damaged_files_are_reported_as_unreadable(kind, content):
    with pytest.raises(ExtractionError) as e:
        parse_document(content, kind)
    assert e.value.code == "document_unreadable"


def test_docx_keeps_paragraphs_and_tables_in_order():
    d = docx.Document()
    d.add_paragraph("Before the table")
    t = d.add_table(rows=1, cols=3)
    t.rows[0].cells[0].text, t.rows[0].cells[2].text = "Merged", "Rent 100"
    t.rows[0].cells[0].merge(t.rows[0].cells[1])
    d.add_paragraph("After the table")
    buf = io.BytesIO()
    d.save(buf)
    p = parse_document(buf.getvalue(), "docx")
    text = p.pages[0].text
    assert text.index("Before") < text.index("Merged | Rent 100") < text.index("After")
    assert p.pages[0].label == "Section 1"


def test_long_docx_is_split_into_numbered_sections():
    d = docx.Document()
    for i in range(30):
        d.add_paragraph(f"Paragraph {i} " + "x" * 400)
    buf = io.BytesIO()
    d.save(buf)
    p = parse_document(buf.getvalue(), "docx")
    assert len(p.pages) > 1 and p.pages[1].label == "Section 2"
    assert all(f"Paragraph {i} " in " ".join(pg.text for pg in p.pages) for i in range(30))


def test_empty_docx_is_unreadable():
    buf = io.BytesIO()
    docx.Document().save(buf)
    with pytest.raises(ExtractionError):
        parse_document(buf.getvalue(), "docx")


def test_pdf_reads_columns_in_visual_order_even_if_stored_out_of_order():
    def build(doc):
        page = doc.new_page()
        # Written in the wrong order on purpose: this is what broke plain text extraction.
        page.insert_text((400, 100), "THIRD")
        page.insert_text((60, 100), "FIRST")
        page.insert_text((230, 100), "SECOND")
        page.insert_text((60, 140), "next line " + "words " * 12)

    text = parse_document(pdf(build), "pdf").pages[0].text
    first_line = text.splitlines()[0]
    assert first_line.index("FIRST") < first_line.index("SECOND") < first_line.index("THIRD")
    assert "next line" in text.splitlines()[1]


def test_pdf_page_without_text_is_sent_to_vision_and_text_page_is_not():
    def build(doc):
        doc.new_page().insert_text((60, 100), "This page has a real text layer " * 3)
        blank = doc.new_page()
        blank.draw_rect(pymupdf.Rect(50, 50, 300, 300), fill=(0, 0, 0))

    pages = parse_document(pdf(build), "pdf").pages
    assert not pages[0].needs_vision and pages[0].image_jpeg is None
    assert pages[1].needs_vision and pages[1].image_jpeg[:2] == b"\xff\xd8"


def test_password_protected_pdf_says_so():
    doc = pymupdf.open()
    doc.new_page().insert_text((60, 100), "secret")
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="own")
    with pytest.raises(ExtractionError) as e:
        parse_document(data, "pdf")
    assert "password" in e.value.message


@pytest.mark.parametrize("value,expected", [
    (dt.datetime(2024, 1, 2), "2024-01-02"), (dt.date(2024, 1, 2), "2024-01-02"), (5.0, "5"), (5.5, "5.5"), (" a\nb ", "a b"),
])
def test_cell_formatting(value, expected):
    assert format_cell(value) == expected


def _docx_bytes(build):
    import docx
    d = docx.Document()
    build(d)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def test_docx_reads_content_controls_tracked_insertions_nested_tables_and_headers():
    from docx.oxml import parse_xml
    from docx.oxml.ns import nsdecls

    def build(d):
        d.sections[0].header.paragraphs[0].text = "Property: Harbor Point Plaza"
        d.add_paragraph("Plain paragraph")
        body = d.element.body
        body.insert(1, parse_xml(f'<w:sdt {nsdecls("w")}><w:sdtContent><w:p><w:r><w:t>Tenant: Acme LLC</w:t></w:r></w:p></w:sdtContent></w:sdt>'))
        body.insert(2, parse_xml(f'<w:p {nsdecls("w")}><w:r><w:t xml:space="preserve">Base Rent: </w:t></w:r>'
                                 f'<w:ins w:id="1" w:author="x"><w:r><w:t>$5,000</w:t></w:r></w:ins>'
                                 f'<w:del w:id="2" w:author="x"><w:r><w:delText>$4,000</w:delText></w:r></w:del></w:p>'))
        t = d.add_table(rows=1, cols=2)
        t.rows[0].cells[0].text = "Outer"
        t.rows[0].cells[1].add_table(rows=1, cols=1).rows[0].cells[0].text = "Deposit $9,000"

    text = parse_document(_docx_bytes(build), "docx").pages[0].text
    for expected in ("[Header/footer] Property: Harbor Point Plaza", "Tenant: Acme LLC", "Base Rent: $5,000", "Deposit $9,000"):
        assert expected in text
    assert "$4,000" not in text  # deleted text stays out


def test_docx_table_keeps_column_positions_and_equal_values():
    def build(d):
        t = d.add_table(rows=3, cols=4)
        for r, row in enumerate([["Period", "Rent", "Deposit", "Free"], ["Year 1", "$2,500", "$2,500", ""], ["Year 2", "", "$2,600", "1"]]):
            for c, value in enumerate(row):
                t.rows[r].cells[c].text = value

    lines = parse_document(_docx_bytes(build), "docx").pages[0].text.splitlines()
    assert "Year 1 | $2,500 | $2,500" in lines and "Year 2 |  | $2,600 | 1" in lines


def _xlsx_bytes(build):
    wb = openpyxl.Workbook()
    build(wb.active)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_one_styled_empty_cell_far_away_does_not_fail_the_workbook():
    from openpyxl.styles import Font

    def build(ws):
        ws.append(["Unit", "Rent"])
        ws.append(["A1", 1000])
        ws["C1048576"].font = Font(bold=True)

    parsed = parse_document(_xlsx_bytes(build), "xlsx")
    assert sorted(parsed.sheets[0].rows) == [1, 2]


def test_percent_formatted_cells_are_read_as_printed():
    def build(ws):
        ws["A1"], ws["B1"] = "Occupancy", 0.0526
        ws["B1"].number_format = "0.00%"

    cells = {c.ref: c.value for c in parse_document(_xlsx_bytes(build), "xlsx").sheets[0].rows[1]}
    assert cells["B1"] == pytest.approx(5.26)
