"""Tests for parser helper functions: normalize, extract_pdf, fallback, arXiv API, sections."""

import tempfile
import unittest.mock
from pathlib import Path

from drbrain.parser.mineru_parser import (
    MinerUParser,
    _extract_arxiv_from_filename,
    _fetch_arxiv_metadata,
    extract_pdf,
    filter_sections,
    normalize_arxiv,
    normalize_doi,
)

# -- normalize_doi --


def test_normalize_doi_strips_url():
    """normalize_doi strips https://doi.org/ prefix."""
    assert normalize_doi("https://doi.org/10.1234/abc") == "10.1234/abc"


def test_normalize_doi_strips_doi_prefix():
    """normalize_doi strips 'doi:' prefix."""
    assert normalize_doi("doi: 10.1234/abc") == "10.1234/abc"


def test_normalize_doi_lowercases():
    """normalize_doi lowercases the result."""
    assert normalize_doi("  10.1234/ABC  ") == "10.1234/abc"


def test_normalize_doi_passthrough():
    """normalize_doi returns bare DOI unchanged."""
    assert normalize_doi("10.1234/abc") == "10.1234/abc"


# -- normalize_arxiv --


def test_normalize_arxiv_strips_version():
    """normalize_arxiv strips vN suffix."""
    assert normalize_arxiv("2401.12345v1") == "2401.12345"


def test_normalize_arxiv_bare_id():
    """normalize_arxiv returns bare ID unchanged."""
    assert normalize_arxiv("2401.12345") == "2401.12345"


def test_normalize_arxiv_with_context():
    """normalize_arxiv extracts ID from longer string."""
    assert normalize_arxiv("arxiv:2401.12345v2") == "2401.12345"


# -- _extract_arxiv_from_filename --


def test_extract_arxiv_from_filename_with_version():
    """Extracts arXiv ID from filename with version suffix."""
    assert _extract_arxiv_from_filename(Path("2602.00617v1.pdf")) == "2602.00617"


def test_extract_arxiv_from_filename_without_version():
    """Extracts arXiv ID from filename without version (pattern requires vN, so this returns None)."""
    # The regex requires v\d* before .pdf, so no-version filenames don't match
    assert _extract_arxiv_from_filename(Path("2401.12345.pdf")) is None


def test_extract_arxiv_from_filename_no_match():
    """Returns None for non-arXiv filename."""
    assert _extract_arxiv_from_filename(Path("some_paper.pdf")) is None


# -- _fetch_arxiv_metadata --


def test_fetch_arxiv_metadata_success():
    """_fetch_arxiv_metadata returns title and year from arXiv API."""
    mock_xml = """<?xml version="1.0"?>
<feed>
  <entry>
    <title>Feed Title</title>
    <title>Real Paper Title</title>
    <published>2024-03-15T00:00:00Z</published>
  </entry>
</feed>"""

    with unittest.mock.patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = unittest.mock.Mock()
        mock_resp.read.return_value = mock_xml.encode("utf-8")
        mock_resp.__enter__ = unittest.mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = unittest.mock.Mock(return_value=False)
        mock_urlopen.return_value = mock_resp

        title, year = _fetch_arxiv_metadata("2401.12345")
        assert title == "Real Paper Title"
        assert year == 2024


def test_fetch_arxiv_metadata_error_returns_none():
    """_fetch_arxiv_metadata returns None on network error."""
    with unittest.mock.patch("urllib.request.urlopen", side_effect=Exception("network")):
        title, year = _fetch_arxiv_metadata("bad_id")
        assert title is None
        assert year is None


def test_fetch_arxiv_metadata_single_title():
    """Uses single title when only one is present."""
    mock_xml = """<?xml version="1.0"?>
<feed>
  <entry>
    <title>Only Title</title>
    <published>2023-01-01T00:00:00Z</published>
  </entry>
</feed>"""

    with unittest.mock.patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = unittest.mock.Mock()
        mock_resp.read.return_value = mock_xml.encode("utf-8")
        mock_resp.__enter__ = unittest.mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = unittest.mock.Mock(return_value=False)
        mock_urlopen.return_value = mock_resp

        title, year = _fetch_arxiv_metadata("2301.00001")
        assert title == "Only Title"


# -- filter_sections --


def test_filter_sections_markdown_headings():
    """Extracts sections from markdown headings."""
    raw = "# Introduction\n\nContent of intro.\n\n# Conclusion\n\nFinal thoughts."
    blocks = filter_sections(raw)
    assert len(blocks) == 2
    assert "Content of intro." in blocks[0]
    assert "Final thoughts." in blocks[1]


def test_filter_sections_inline_markers():
    """Extracts sections from inline markers."""
    raw = "Introduction.—This is the intro content.\n\nConclusion.—Final remarks."
    blocks = filter_sections(raw)
    assert len(blocks) == 2
    assert "This is the intro content." in blocks[0]
    assert "Final remarks." in blocks[1]


def test_filter_sections_fallback_no_sections():
    """Returns all text when no target sections detected."""
    raw = "Just some random text without any section headers."
    blocks = filter_sections(raw)
    assert len(blocks) == 1
    assert "random text" in blocks[0]


def test_filter_sections_excludes_thinking_header():
    """Fallback mode strips 'Thinking...' prefix."""
    raw = "Thinking... paper.pdf\nSome text content."
    blocks = filter_sections(raw)
    assert len(blocks) == 1
    assert "Thinking" not in blocks[0]


def test_filter_sections_empty_input():
    """Returns empty list for empty input."""
    assert filter_sections("") == []


# -- extract_pdf convenience --


def test_extract_pdf_from_config():
    """extract_pdf creates parser from config and extracts."""
    cfg = {
        "mineru": {
            "token": "test-token",
            "model": "vlm",
            "is_ocr": False,
            "enable_formula": True,
            "enable_table": True,
        }
    }

    with tempfile.TemporaryDirectory() as td:
        pdf_path = Path(td) / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 dummy")

        with unittest.mock.patch.object(MinerUParser, "extract") as mock_extract:
            from drbrain.parser.mineru_parser import ParsedPaper

            mock_extract.return_value = ParsedPaper(title="Test", year=2024)

            result = extract_pdf(pdf_path, cfg)
            assert result.title == "Test"
            mock_extract.assert_called_once()


# -- PyMuPDF fallback --


def test_fallback_pymupdf():
    """_fallback_pymupdf extracts markdown from PDF via PyMuPDF."""
    with tempfile.TemporaryDirectory() as td:
        pdf_path = Path(td) / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 dummy")

        mock_page = unittest.mock.Mock()
        mock_page.get_text.return_value = "# Extracted markdown content."

        mock_doc = unittest.mock.Mock()
        mock_doc.__iter__ = unittest.mock.Mock(return_value=iter([mock_page]))
        mock_doc.close = unittest.mock.Mock()

        with unittest.mock.patch("fitz.open", return_value=mock_doc):
            parser = MinerUParser()
            result = parser._fallback_pymupdf(pdf_path)
            assert "Extracted markdown content." in result


def test_fallback_pymupdf_empty_markdown_uses_text():
    """_fallback_pymupdf falls back to plain text when markdown is empty."""
    with tempfile.TemporaryDirectory() as td:
        pdf_path = Path(td) / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 dummy")

        mock_page = unittest.mock.Mock()
        mock_page.get_text.side_effect = ["", "plain text content"]

        mock_doc = unittest.mock.Mock()
        mock_doc.close = unittest.mock.Mock()
        mock_doc.__iter__ = unittest.mock.MagicMock(side_effect=lambda: iter([mock_page]))

        with unittest.mock.patch("fitz.open", return_value=mock_doc):
            parser = MinerUParser()
            result = parser._fallback_pymupdf(pdf_path)
            assert "plain text content" in result


def test_mineru_cli_not_found_uses_fallback():
    """When mineru CLI not found, returns None from _try_mineru_open_api."""
    parser = MinerUParser()
    with unittest.mock.patch("drbrain.parser.mineru_parser._find_cli", return_value=None):
        result = parser._try_mineru_open_api(Path("/tmp/test.pdf"))
        assert result == (None, None)


def test_parser_full_extract_flow_with_fallback():
    """extract() falls back to PyMuPDF when mineru CLI fails."""
    with tempfile.TemporaryDirectory() as td:
        pdf_path = Path(td) / "2401.00001v1.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 dummy")

        def mock_fallback(path):
            return "# Test Title\n\nIntroduction.—Test content."

        with (
            unittest.mock.patch(
                "drbrain.parser.mineru_parser._find_cli", return_value="mineru-open-api"
            ),
            unittest.mock.patch("subprocess.run", side_effect=FileNotFoundError("not found")),
            unittest.mock.patch.object(
                MinerUParser, "_fallback_pymupdf", side_effect=mock_fallback
            ),
            unittest.mock.patch(
                "drbrain.parser.mineru_parser._fetch_arxiv_metadata", return_value=(None, None)
            ),
            unittest.mock.patch.object(MinerUParser, "_count_pages", return_value=1),
        ):
            parser = MinerUParser(max_retries=1, retry_delay=0.01)
            result = parser.extract(pdf_path)
            assert result.title == "Test Title"
            assert result.arxiv == "2401.00001"  # From filename
            assert len(result.text_blocks) >= 1
