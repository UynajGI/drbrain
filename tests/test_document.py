"""Tests for src/drbrain/services/document.py.

Covers inspect() dispatch, plus inspect_docx / inspect_pptx / inspect_xlsx
with mocked python-docx / python-pptx / openpyxl so tests run without the
optional ``office`` extra installed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from drbrain.services.document import inspect, inspect_docx, inspect_pptx, inspect_xlsx

# ── dispatch: inspect() ────────────────────────────────────────────────────


class TestInspectDispatch:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            inspect(Path("/nonexistent/file.docx"))

    def test_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not a file"):
            inspect(tmp_path)

    def test_unsupported_format_raises(self, tmp_path):
        f = tmp_path / "test.xyz"
        f.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported file format"):
            inspect(f)

    def test_unsupported_explicit_format_raises(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported file format"):
            inspect(f, fmt="unknown")

    def test_auto_detect_dispatches_to_docx(self, tmp_path):
        f = tmp_path / "test.docx"
        f.write_text("dummy")
        with patch("drbrain.services.document.inspect_docx", return_value="OK") as m:
            result = inspect(f)
        assert result == "OK"
        m.assert_called_once_with(f)

    def test_auto_detect_dispatches_to_pptx(self, tmp_path):
        f = tmp_path / "test.pptx"
        f.write_text("dummy")
        with patch("drbrain.services.document.inspect_pptx", return_value="OK") as m:
            result = inspect(f)
        assert result == "OK"
        m.assert_called_once_with(f)

    def test_auto_detect_dispatches_to_xlsx(self, tmp_path):
        f = tmp_path / "test.xlsx"
        f.write_text("dummy")
        with patch("drbrain.services.document.inspect_xlsx", return_value="OK") as m:
            result = inspect(f)
        assert result == "OK"
        m.assert_called_once_with(f)

    def test_explicit_format_overrides_extension(self, tmp_path):
        f = tmp_path / "test.dat"
        f.write_text("dummy")
        with patch("drbrain.services.document.inspect_docx", return_value="OK") as m:
            result = inspect(f, fmt="docx")
        assert result == "OK"
        m.assert_called_once()

    def test_extension_case_insensitive(self, tmp_path):
        f = tmp_path / "test.DOCX"
        f.write_text("dummy")
        with patch("drbrain.services.document.inspect_docx", return_value="OK") as m:
            inspect(f)
        m.assert_called_once_with(f)


# ── inspect_docx ───────────────────────────────────────────────────────────


def _install_fake_docx(monkeypatch, document_return_value):
    """Install a fake ``docx`` package whose Document() returns the given mock."""
    fake = types.ModuleType("docx")
    oxml_mod = types.ModuleType("docx.oxml")
    ns_mod = types.ModuleType("docx.oxml.ns")

    def qn(tag: str) -> str:
        return tag

    ns_mod.qn = qn
    oxml_mod.ns = ns_mod
    fake.oxml = oxml_mod

    text_mod = types.ModuleType("docx.text")
    para_mod = types.ModuleType("docx.text.paragraph")

    class Paragraph:
        def __init__(self, element, doc):
            self.text = getattr(element, "_text", "")
            self.runs = []
            self.style = element._style

    para_mod.Paragraph = Paragraph
    text_mod.paragraph = para_mod
    fake.text = text_mod

    tbl_mod = types.ModuleType("docx.table")
    # Table is a callable MagicMock so callers can configure its return value;
    # by default returns a table-shaped mock.
    default_table = MagicMock()
    default_table.rows = []
    default_table.columns = []
    default_table.style = None
    tbl_mod.Table = MagicMock(return_value=default_table)
    fake.table = tbl_mod

    # Document is a MagicMock so return_value is configurable
    fake.Document = MagicMock(return_value=document_return_value)

    monkeypatch.setitem(sys.modules, "docx", fake)
    monkeypatch.setitem(sys.modules, "docx.oxml", oxml_mod)
    monkeypatch.setitem(sys.modules, "docx.oxml.ns", ns_mod)
    monkeypatch.setitem(sys.modules, "docx.text", text_mod)
    monkeypatch.setitem(sys.modules, "docx.text.paragraph", para_mod)
    monkeypatch.setitem(sys.modules, "docx.table", tbl_mod)
    return fake


def _empty_body_doc():
    body = MagicMock()
    body.iterchildren.return_value = []
    root = MagicMock()
    root.body = body
    return MagicMock(sections=[], element=root)


class TestInspectDocx:
    def test_raises_import_error_when_docx_missing(self, tmp_path):
        f = tmp_path / "x.docx"
        f.write_text("dummy")
        with patch.dict(sys.modules, {"docx": None}):
            with pytest.raises(ImportError, match="drbrain\\[office\\]"):
                inspect_docx(f)

    def test_empty_document_report(self, tmp_path, monkeypatch):
        f = tmp_path / "empty.docx"
        f.write_text("dummy")
        _install_fake_docx(monkeypatch, _empty_body_doc())
        result = inspect_docx(f)
        assert "=== DOCX:" in result
        assert "Summary" in result
        assert "Paragraphs: 0" in result

    def test_report_includes_filename(self, tmp_path, monkeypatch):
        f = tmp_path / "weird-name.docx"
        f.write_text("dummy")
        _install_fake_docx(monkeypatch, _empty_body_doc())
        result = inspect_docx(f)
        assert "weird-name.docx" in result

    def test_section_orientation_portrait(self, tmp_path, monkeypatch):
        f = tmp_path / "s.docx"
        f.write_text("dummy")
        section = MagicMock()
        section.page_width.inches = 8.5
        section.page_height.inches = 11.0
        body = MagicMock()
        body.iterchildren.return_value = []
        root = MagicMock()
        root.body = body
        doc = MagicMock(sections=[section], element=root)
        _install_fake_docx(monkeypatch, doc)
        result = inspect_docx(f)
        assert "Section 1" in result
        assert "portrait" in result

    def test_section_orientation_landscape(self, tmp_path, monkeypatch):
        f = tmp_path / "s.docx"
        f.write_text("dummy")
        section = MagicMock()
        section.page_width.inches = 11.0
        section.page_height.inches = 8.5
        body = MagicMock()
        body.iterchildren.return_value = []
        root = MagicMock()
        root.body = body
        doc = MagicMock(sections=[section], element=root)
        _install_fake_docx(monkeypatch, doc)
        result = inspect_docx(f)
        assert "landscape" in result

    def test_table_rendered(self, tmp_path, monkeypatch):
        f = tmp_path / "t.docx"
        f.write_text("dummy")
        tbl_el = MagicMock()
        tbl_el.tag = "w:tbl"
        header_cell = MagicMock()
        header_cell.text = "Col1"
        data_cell = MagicMock()
        data_cell.text = "data"
        row0 = MagicMock()
        row0.cells = [header_cell]
        row1 = MagicMock()
        row1.cells = [data_cell]
        style_obj = MagicMock()
        style_obj.name = "TableStyle"
        fake_table = MagicMock()
        fake_table.rows = [row0, row1]
        fake_table.columns = [MagicMock()]
        fake_table.style = style_obj
        body = MagicMock()
        body.iterchildren.return_value = [tbl_el]
        root = MagicMock()
        root.body = body
        doc = MagicMock(sections=[], element=root)
        fake = _install_fake_docx(monkeypatch, doc)
        fake.table.Table.return_value = fake_table
        result = inspect_docx(f)
        assert "[Table" in result
        assert "TableStyle" in result
        assert "Header:" in result

    def test_heading_paragraph_rendered(self, tmp_path, monkeypatch):
        f = tmp_path / "h.docx"
        f.write_text("dummy")
        heading_el = MagicMock()
        heading_el.tag = "w:p"
        heading_el._text = "My Heading"
        style = MagicMock()
        style.name = "Heading 1"
        heading_el._style = style
        body = MagicMock()
        body.iterchildren.return_value = [heading_el]
        root = MagicMock()
        root.body = body
        doc = MagicMock(sections=[], element=root)
        _install_fake_docx(monkeypatch, doc)
        # Paragraph instances come from our fake module's Paragraph class
        # but the inspector imports via `from docx.text.paragraph import Paragraph`
        # — the fake Paragraph reads element._text/_style, so this works.
        result = inspect_docx(f)
        assert "Heading 1" in result
        assert "My Heading" in result


# ── inspect_pptx ───────────────────────────────────────────────────────────


def _install_fake_pptx(monkeypatch, presentation_return_value):
    fake = types.ModuleType("pptx")
    enum_mod = types.ModuleType("pptx.enum")
    shapes_mod = types.ModuleType("pptx.enum.shapes")

    class MSO_SHAPE_TYPE:  # noqa: N801
        PICTURE = "PICTURE"

    shapes_mod.MSO_SHAPE_TYPE = MSO_SHAPE_TYPE
    enum_mod.shapes = shapes_mod
    fake.enum = enum_mod
    fake.Presentation = MagicMock(return_value=presentation_return_value)
    monkeypatch.setitem(sys.modules, "pptx", fake)
    monkeypatch.setitem(sys.modules, "pptx.enum", enum_mod)
    monkeypatch.setitem(sys.modules, "pptx.enum.shapes", shapes_mod)
    return fake


class TestInspectPptx:
    def test_raises_import_error_when_pptx_missing(self, tmp_path):
        f = tmp_path / "x.pptx"
        f.write_text("dummy")
        with patch.dict(sys.modules, {"pptx": None}):
            with pytest.raises(ImportError, match="drbrain\\[office\\]"):
                inspect_pptx(f)

    def test_empty_presentation_report(self, tmp_path, monkeypatch):
        f = tmp_path / "empty.pptx"
        f.write_text("dummy")
        prs = MagicMock(slide_width=914400 * 10, slide_height=914400 * 7.5, slides=[])
        _install_fake_pptx(monkeypatch, prs)
        result = inspect_pptx(f)
        assert "=== PPTX:" in result
        assert "Summary" in result
        assert "Slides: 0" in result

    def test_filename_present(self, tmp_path, monkeypatch):
        f = tmp_path / "deck.pptx"
        f.write_text("dummy")
        prs = MagicMock(slide_width=914400 * 10, slide_height=914400 * 7.5, slides=[])
        _install_fake_pptx(monkeypatch, prs)
        result = inspect_pptx(f)
        assert "deck.pptx" in result

    def test_text_shape_rendered(self, tmp_path, monkeypatch):
        f = tmp_path / "t.pptx"
        f.write_text("dummy")
        para = MagicMock()
        para.text = "Hello world"
        para.runs = []
        tf = MagicMock()
        tf.paragraphs = [para]
        shape = MagicMock()
        shape.shape_type = "TEXT"
        shape.left = 914400
        shape.top = 914400
        shape.width = 914400 * 4
        shape.height = 914400 * 1
        shape.has_table = False
        shape.has_text_frame = True
        shape.text_frame = tf
        shape.placeholder_format = None
        slide = MagicMock()
        slide.slide_layout.name = "Title"
        slide.shapes = [shape]
        prs = MagicMock(slide_width=914400 * 10, slide_height=914400 * 7.5, slides=[slide])
        _install_fake_pptx(monkeypatch, prs)
        result = inspect_pptx(f)
        assert "Slide 1/1" in result
        assert "Hello world" in result

    def test_overflow_warning(self, tmp_path, monkeypatch):
        f = tmp_path / "o.pptx"
        f.write_text("dummy")
        shape = MagicMock()
        shape.shape_type = "TEXT"
        shape.left = 914400 * 9
        shape.top = 914400
        shape.width = 914400 * 5
        shape.height = 914400
        shape.has_table = False
        shape.has_text_frame = False
        slide = MagicMock()
        slide.slide_layout.name = "Blank"
        slide.shapes = [shape]
        prs = MagicMock(slide_width=914400 * 10, slide_height=914400 * 7.5, slides=[slide])
        _install_fake_pptx(monkeypatch, prs)
        result = inspect_pptx(f)
        assert "Overflow" in result


# ── inspect_xlsx ───────────────────────────────────────────────────────────


def _install_fake_openpyxl(monkeypatch, wb):
    fake = types.ModuleType("openpyxl")
    fake.load_workbook = MagicMock(return_value=wb)
    monkeypatch.setitem(sys.modules, "openpyxl", fake)
    return fake


class TestInspectXlsx:
    def test_raises_import_error_when_openpyxl_missing(self, tmp_path):
        f = tmp_path / "x.xlsx"
        f.write_text("dummy")
        with patch.dict(sys.modules, {"openpyxl": None}):
            with pytest.raises(ImportError, match="drbrain\\[office\\]"):
                inspect_xlsx(f)

    def test_empty_workbook_report(self, tmp_path, monkeypatch):
        f = tmp_path / "empty.xlsx"
        f.write_text("dummy")
        wb = MagicMock()
        wb.sheetnames = []
        wb.active = None
        wb.close = MagicMock()
        _install_fake_openpyxl(monkeypatch, wb)
        result = inspect_xlsx(f)
        assert "=== XLSX:" in result
        assert "Worksheets: 0" in result
        assert "Summary" in result

    def test_worksheet_dimensions_rendered(self, tmp_path, monkeypatch):
        f = tmp_path / "data.xlsx"
        f.write_text("dummy")
        ws = MagicMock()
        ws.dimensions = "A1:B5"
        ws.max_row = 5
        ws.max_column = 2
        ws.freeze_panes = None
        ws.auto_filter = MagicMock(ref=None)
        ws.merged_cells = MagicMock(ranges=[])
        ws._charts = []
        ws.cell = MagicMock(side_effect=lambda row, column: MagicMock(value=f"r{row}c{column}"))
        wb = MagicMock()
        wb.sheetnames = ["Data"]
        wb.active = ws
        wb.__getitem__ = MagicMock(return_value=ws)
        wb.close = MagicMock()
        _install_fake_openpyxl(monkeypatch, wb)
        result = inspect_xlsx(f)
        assert 'Sheet "Data"' in result
        assert "Range: A1:B5" in result
        assert "Header (row 1)" in result
        assert "Worksheets: 1" in result

    def test_workbook_close_called(self, tmp_path, monkeypatch):
        f = tmp_path / "close.xlsx"
        f.write_text("dummy")
        wb = MagicMock()
        wb.sheetnames = []
        wb.close = MagicMock()
        _install_fake_openpyxl(monkeypatch, wb)
        inspect_xlsx(f)
        wb.close.assert_called_once()

    def test_freeze_panes_and_filter_rendered(self, tmp_path, monkeypatch):
        f = tmp_path / "fp.xlsx"
        f.write_text("dummy")
        ws = MagicMock()
        ws.dimensions = "A1:C10"
        ws.max_row = 10
        ws.max_column = 3
        ws.freeze_panes = "A2"
        ws.auto_filter = MagicMock(ref="A1:C1")
        ws.merged_cells = MagicMock(ranges=[])
        ws._charts = []
        ws.cell = MagicMock(return_value=MagicMock(value="x"))
        wb = MagicMock()
        wb.sheetnames = ["S"]
        wb.active = ws
        wb.__getitem__ = MagicMock(return_value=ws)
        wb.close = MagicMock()
        _install_fake_openpyxl(monkeypatch, wb)
        result = inspect_xlsx(f)
        assert "Frozen panes: A2" in result
        assert "Auto-filter: A1:C1" in result
