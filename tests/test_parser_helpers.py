"""Tests for parser helper functions: normalize, extract_pdf, fallback, arXiv API, sections."""

import subprocess
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


# -- _extract_title (instance method on MinerUParser) --


def test_extract_title_from_heading():
    """Extracts title from first # heading in markdown."""
    parser = MinerUParser()
    title = parser._extract_title("# My Title\n\nSome content.", "/path/to/paper.pdf")
    assert title == "My Title"


def test_extract_title_ignores_subheading():
    """Only uses h1 (# ), not h2 (## ) or deeper."""
    parser = MinerUParser()
    title = parser._extract_title("## Subheading\n\n# Real Title\n\nContent.", "/f/paper.pdf")
    assert title == "Real Title"


def test_extract_title_fallback_to_filename():
    """Uses filename stem when no heading found."""
    parser = MinerUParser()
    title = parser._extract_title("No heading here.\nJust content.", "/path/to/my-paper.pdf")
    assert title == "my-paper"


def test_extract_title_empty_markdown():
    """Uses filename stem when markdown is empty."""
    parser = MinerUParser()
    title = parser._extract_title("", "/path/to/research-paper.pdf")
    assert title == "research-paper"


def test_extract_title_strips_whitespace():
    """Strips whitespace from extracted heading."""
    parser = MinerUParser()
    title = parser._extract_title("#   Padded Title   \n\nContent.", "/f/paper.pdf")
    assert title == "Padded Title"


# -- _extract_year (instance method on MinerUParser) --


def test_extract_year_from_text():
    """Extracts year from text in first 20 lines."""
    parser = MinerUParser()
    year = parser._extract_year("Published in 2024 by Nature.")
    assert year == 2024


def test_extract_year_no_year_returns_none():
    """Returns None when no year found."""
    parser = MinerUParser()
    year = parser._extract_year("No year in this text.")
    assert year is None


def test_extract_year_ignores_out_of_range():
    """Ignores years outside 1900-2999 range."""
    parser = MinerUParser()
    year = parser._extract_year("Ancient text from year 1800.")
    assert year is None


def test_extract_year_returns_first_valid():
    """Returns first valid year found in first 20 lines."""
    parser = MinerUParser()
    year = parser._extract_year("Published 2023. Revised 2024.")
    assert year == 2023


def test_extract_year_handles_early_1900s():
    """Extracts years from 1900 onwards."""
    parser = MinerUParser()
    year = parser._extract_year("Published 1954.")
    assert year == 1954


# -- _extract_ids (instance method on MinerUParser) --


def test_extract_ids_both_doi_and_arxiv():
    """Extracts both DOI and arXiv from separate lines."""
    parser = MinerUParser()
    doi, arxiv = parser._extract_ids("DOI: 10.1234/example\narXiv: 2401.12345v1")
    assert doi == "10.1234/example"
    assert arxiv == "2401.12345"


def test_extract_ids_only_doi():
    """Extracts only DOI when no arXiv present."""
    parser = MinerUParser()
    doi, arxiv = parser._extract_ids("See https://doi.org/10.5678/foo.bar for details.")
    assert doi == "10.5678/foo.bar"
    assert arxiv is None


def test_extract_ids_only_arxiv():
    """Extracts only arXiv when no DOI present."""
    parser = MinerUParser()
    doi, arxiv = parser._extract_ids("Preprint available at arxiv: 2602.00617v1")
    assert doi is None
    assert arxiv == "2602.00617"


def test_extract_ids_no_matches():
    """Returns (None, None) when no identifiers found."""
    parser = MinerUParser()
    doi, arxiv = parser._extract_ids("No identifiers in this text.")
    assert doi is None
    assert arxiv is None


def test_extract_ids_arxiv_case_insensitive():
    """arXiv prefix detection is case-insensitive (ArXiv, arXiv, etc.)."""
    parser = MinerUParser()
    doi, arxiv = parser._extract_ids("See ArXiv:2401.12345 for preprint.")
    assert doi is None
    assert arxiv == "2401.12345"


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


# -- _extract_mineru_only fallback path --


def test_extract_mineru_only_fallback_path():
    """_extract_mineru_only falls back to PyMuPDF when _try_mineru_open_api returns (None, None)."""
    parser = MinerUParser()
    with (
        unittest.mock.patch.object(parser, "_try_mineru_open_api", return_value=(None, None)),
        unittest.mock.patch.object(parser, "_fallback_pymupdf", return_value="Fallback markdown"),
    ):
        raw_md, img_dir, managed_tmp = parser._extract_mineru_only(Path("/fake/test.pdf"))
        assert raw_md == "Fallback markdown"
        assert img_dir is None
        assert managed_tmp is None


# -- _try_mineru_open_api all retries exhausted --


def test_try_mineru_open_api_all_retries_exhausted():
    """Returns (None, None) when all retry attempts fail with non-zero returncode."""
    parser = MinerUParser(token="t", max_retries=2, retry_delay=0.001)
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "out"
        out_dir.mkdir()

        with (
            unittest.mock.patch(
                "drbrain.parser.mineru_parser._find_cli", return_value="/usr/bin/mineru-open-api"
            ),
            unittest.mock.patch(
                "subprocess.run", return_value=unittest.mock.Mock(returncode=1, stderr="fail")
            ),
        ):
            result_dir, managed_tmp = parser._try_mineru_open_api(
                Path("/fake/paper.pdf"), out_dir=out_dir
            )
            assert result_dir is None
            assert managed_tmp is None


def test_try_mineru_open_api_all_timeouts_exhausted():
    """Returns (None, None) on repeated TimeoutExpired errors."""
    parser = MinerUParser(token="t", max_retries=2, retry_delay=0.001)
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "out"
        out_dir.mkdir()

        with (
            unittest.mock.patch(
                "drbrain.parser.mineru_parser._find_cli", return_value="/usr/bin/mineru-open-api"
            ),
            unittest.mock.patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="test", timeout=600),
            ),
        ):
            result_dir, managed_tmp = parser._try_mineru_open_api(
                Path("/fake/paper.pdf"), out_dir=out_dir
            )
            assert result_dir is None
            assert managed_tmp is None


# -- _extract_single with arXiv metadata enrichment --


def test_extract_single_with_arxiv_enrichment():
    """_extract_single enriches title/year from arXiv API when arxiv ID is present."""
    parser = MinerUParser()
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "out"
        out_dir.mkdir()
        (out_dir / "images").mkdir()
        (out_dir / "output.md").write_text(
            "# Original Title\n\narXiv: 2401.12345\n\nIntroduction.—Content."
        )

        with (
            unittest.mock.patch.object(
                parser, "_try_mineru_open_api", return_value=(out_dir, None)
            ),
            unittest.mock.patch(
                "drbrain.parser.mineru_parser._fetch_arxiv_metadata",
                return_value=("Better Title from arXiv", 2024),
            ),
            unittest.mock.patch(
                "drbrain.extractor.openalex.search_authors_by_work", return_value=[]
            ),
        ):
            result = parser._extract_single(Path("/fake/paper.pdf"))
            assert result.title == "Better Title from arXiv"
            assert result.year == 2024
            assert result.arxiv == "2401.12345"


# -- _merge_images (instance method on MinerUParser) --


def test_merge_images_copies_files():
    """Copies image files from multiple directories into a single destination."""
    parser = MinerUParser()
    with tempfile.TemporaryDirectory() as td:
        # Create two source image dirs
        src1 = Path(td) / "src1"
        src1.mkdir()
        (src1 / "img1.png").write_text("png content")
        (src1 / "img2.jpg").write_text("jpg content")

        src2 = Path(td) / "src2"
        src2.mkdir()
        (src2 / "img3.png").write_text("another png")

        dest = Path(td) / "merged_images"
        result = parser._merge_images([src1, src2], dest)

        assert result == dest
        assert (dest / "img1.png").exists()
        assert (dest / "img2.jpg").exists()
        assert (dest / "img3.png").exists()
        assert (dest / "img1.png").read_text() == "png content"


def test_merge_images_skips_none_and_missing():
    """Skips None dirs and non-existent dirs gracefully."""
    parser = MinerUParser()
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        src.mkdir()
        (src / "img.png").write_text("valid")

        nonexistent = Path(td) / "nonexistent"

        dest = Path(td) / "merged"
        result = parser._merge_images([None, src, nonexistent], dest)

        assert result == dest
        assert (dest / "img.png").exists()


def test_merge_images_all_empty_returns_none():
    """Returns None when no images were found across any dir."""
    parser = MinerUParser()
    with tempfile.TemporaryDirectory() as td:
        empty1 = Path(td) / "empty1"
        empty1.mkdir()

        dest = Path(td) / "merged"
        result = parser._merge_images([empty1, None], dest)
        assert result is None


def test_merge_markdown_single():
    """Returns single markdown unchanged."""
    parser = MinerUParser()
    result = parser._merge_markdown(["# Title\n\nContent."])
    assert result == "# Title\n\nContent."


def test_merge_markdown_multiple_strips_duplicate_title():
    """Strips duplicate #-heading title from subsequent chunks."""
    parser = MinerUParser()
    result = parser._merge_markdown(
        ["# Paper Title\n\nAbstract content.", "# Paper Title\n\nMethods section."]
    )
    assert "Abstract content." in result
    assert "Methods section." in result
    assert "---" in result
    # Title should appear only once in the merged result
    assert result.count("# Paper Title") == 1


def test_merge_markdown_empty_subsequent_skipped():
    """Empty stripped subsequent chunk is not appended."""
    parser = MinerUParser()
    result = parser._merge_markdown(["# Title\n\nContent.", "# Title"])
    assert result == "# Title\n\nContent."


# -- _read_output_md (instance method on MinerUParser) --


def test_read_output_md_finds_and_reads():
    """Reads first .md file from output directory."""
    parser = MinerUParser()
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        (out_dir / "result.md").write_text("# Hello")
        result = parser._read_output_md(out_dir)
        assert result == "# Hello"


def test_read_output_md_no_md_files():
    """Returns empty string when no .md files found."""
    parser = MinerUParser()
    with tempfile.TemporaryDirectory() as td:
        result = parser._read_output_md(Path(td))
        assert result == ""


# -- _extract_mineru_only with mineru API success --


def test_extract_mineru_only_success_path():
    """When _try_mineru_open_api succeeds, reads output and returns image dir."""
    parser = MinerUParser()
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "out"
        out_dir.mkdir()
        img_dir = out_dir / "images"
        img_dir.mkdir()
        (out_dir / "output.md").write_text("# Extracted content")

        with (
            unittest.mock.patch.object(
                parser, "_try_mineru_open_api", return_value=(out_dir, None)
            ),
        ):
            raw_md, result_img_dir, managed_tmp = parser._extract_mineru_only(
                Path("/fake/test.pdf"), out_dir=out_dir
            )
            assert raw_md == "# Extracted content"
            assert result_img_dir == img_dir
            assert managed_tmp is None


# -- _try_mineru_open_api with token-based CLI args --


def test_try_mineru_open_api_token_based_cmd():
    """Builds CLI command with token and model options when token is set."""
    parser = MinerUParser(
        token="test-token", model="vlm", is_ocr=True, enable_formula=False, enable_table=False
    )
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "out"
        out_dir.mkdir()
        img_dir = out_dir / "images"
        img_dir.mkdir()
        (out_dir / "output.md").write_text("## Content")

        with (
            unittest.mock.patch(
                "drbrain.parser.mineru_parser._find_cli", return_value="/usr/bin/mineru-open-api"
            ),
            unittest.mock.patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value.returncode = 0

            result_dir, managed_tmp = parser._try_mineru_open_api(
                Path("/fake/paper.pdf"), out_dir=out_dir
            )
            assert result_dir == out_dir
            assert managed_tmp is None
            # Verify token-based command was constructed
            call_args = mock_run.call_args[0][0]
            assert "--token" in call_args
            assert "test-token" in call_args
            assert "--model" in call_args
            assert "--ocr" in call_args
            assert "--formula=false" in call_args
            assert "--table=false" in call_args


def test_try_mineru_open_api_no_token_uses_simple_cmd():
    """Builds simple CLI command when no token is set."""
    parser = MinerUParser(token="")
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "out"
        out_dir.mkdir()
        img_dir = out_dir / "images"
        img_dir.mkdir()

        with (
            unittest.mock.patch(
                "drbrain.parser.mineru_parser._find_cli", return_value="/usr/bin/mineru-open-api"
            ),
            unittest.mock.patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value.returncode = 0

            parser._try_mineru_open_api(Path("/fake/paper.pdf"), out_dir=out_dir)
            call_args = mock_run.call_args[0][0]
            assert "--token" not in call_args


def test_try_mineru_open_api_retry_on_nonzero_returncode():
    """Retries on non-zero returncode, eventually succeeds."""
    parser = MinerUParser(token="t", max_retries=2, retry_delay=0.001)
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "out"
        out_dir.mkdir()
        img_dir = out_dir / "images"
        img_dir.mkdir()
        (out_dir / "output.md").write_text("ok")

        with (
            unittest.mock.patch(
                "drbrain.parser.mineru_parser._find_cli", return_value="/usr/bin/mineru-open-api"
            ),
            unittest.mock.patch("subprocess.run") as mock_run,
        ):
            # First attempt fails (non-zero), second succeeds
            mock_run.side_effect = [
                unittest.mock.Mock(returncode=1, stderr="error msg"),
                unittest.mock.Mock(returncode=0),
            ]

            result_dir, managed_tmp = parser._try_mineru_open_api(
                Path("/fake/paper.pdf"), out_dir=out_dir
            )
            assert result_dir == out_dir
            assert mock_run.call_count == 2


def test_try_mineru_open_api_retry_on_missing_images():
    """Retries when images directory is missing, eventually succeeds."""
    parser = MinerUParser(token="t", max_retries=3, retry_delay=0.001)
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "out"
        out_dir.mkdir()

        with (
            unittest.mock.patch(
                "drbrain.parser.mineru_parser._find_cli", return_value="/usr/bin/mineru-open-api"
            ),
            unittest.mock.patch("subprocess.run") as mock_run,
        ):
            # First attempt: success but no images dir
            # Second attempt: success AND images dir exists
            call_count = [0]

            def side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 2:
                    (out_dir / "images").mkdir()
                return unittest.mock.Mock(returncode=0)

            mock_run.side_effect = side_effect

            result_dir, managed_tmp = parser._try_mineru_open_api(
                Path("/fake/paper.pdf"), out_dir=out_dir
            )
            assert result_dir == out_dir
            assert mock_run.call_count == 2


def test_try_mineru_open_api_timeout_retry():
    """Retries on TimeoutExpired, eventually succeeds."""
    parser = MinerUParser(token="t", max_retries=2, retry_delay=0.001)
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "out"
        out_dir.mkdir()
        img_dir = out_dir / "images"
        img_dir.mkdir()
        (out_dir / "output.md").write_text("ok")

        with (
            unittest.mock.patch(
                "drbrain.parser.mineru_parser._find_cli", return_value="/usr/bin/mineru-open-api"
            ),
            unittest.mock.patch("subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                subprocess.TimeoutExpired(cmd="test", timeout=600),
                unittest.mock.Mock(returncode=0),
            ]

            result_dir, managed_tmp = parser._try_mineru_open_api(
                Path("/fake/paper.pdf"), out_dir=out_dir
            )
            assert result_dir == out_dir
            assert mock_run.call_count == 2


def test_try_mineru_open_api_cli_not_found_returns_none():
    """Returns (None, None) when mineru CLI is not on PATH."""
    parser = MinerUParser()
    with unittest.mock.patch("drbrain.parser.mineru_parser._find_cli", return_value=None):
        result = parser._try_mineru_open_api(Path("/fake/test.pdf"))
        assert result == (None, None)


# -- filter_sections edge cases --


def test_filter_sections_fallback_truncates_to_max_chars():
    """Fallback mode truncates content to MAX_CHARS when too long."""
    long_text = "A" * 15000
    blocks = filter_sections(long_text)
    assert len(blocks) == 1
    assert len(blocks[0]) == 12000  # MAX_CHARS


def test_filter_sections_skips_non_target_headings():
    """Only includes blocks from target sections (Abstract, Introduction, etc.)."""
    raw = "# Acknowledgements\n\nThanks to many.\n\n# Introduction\n\nThis is the intro."
    blocks = filter_sections(raw)
    assert len(blocks) == 1
    assert "This is the intro." in blocks[0]
    assert "Thanks to many." not in blocks[0]


# -- _extract_single with mineru API path --


def test_extract_single_with_mineru_success():
    """_extract_single uses mineru output when API succeeds."""
    parser = MinerUParser()
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "out"
        out_dir.mkdir()
        (out_dir / "images").mkdir()
        (out_dir / "output.md").write_text(
            "# Paper Title\n\nDOI: 10.1234/test\nPublished 2024.\n\nIntroduction.—Main content."
        )

        with (
            unittest.mock.patch.object(
                parser, "_try_mineru_open_api", return_value=(out_dir, None)
            ),
            unittest.mock.patch(
                "drbrain.parser.mineru_parser._fetch_arxiv_metadata", return_value=(None, None)
            ),
            unittest.mock.patch(
                "drbrain.extractor.openalex.search_authors_by_work", return_value=[]
            ),
        ):
            result = parser._extract_single(Path("/fake/paper.pdf"))
            assert result.title == "Paper Title"
            assert result.year == 2024
            assert result.doi == "10.1234/test"
            assert len(result.text_blocks) >= 1
