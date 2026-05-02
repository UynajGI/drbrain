"""Tests for MinerU PDF parser integration."""

from pathlib import Path

from drbrain.parser.mineru_parser import (
    MAX_CHARS,
    MinerUParser,
    _extract_arxiv_from_filename,
    filter_sections,
    normalize_arxiv,
)


def test_mineru_parser_no_token():
    """MinerUParser initializes with empty token."""
    parser = MinerUParser(token="", model="vlm")
    assert parser.token == ""
    assert parser.model == "vlm"
    assert parser.enable_formula is True
    assert parser.enable_table is True


def test_mineru_parser_with_token():
    """MinerUParser stores token and config."""
    parser = MinerUParser(token="abc-123", model="pipeline", enable_formula=False)
    assert parser.token == "abc-123"
    assert parser.enable_formula is False


def test_extract_arxiv_from_filename():
    """arXiv ID is extracted from typical PDF filenames."""
    assert _extract_arxiv_from_filename(Path("2602.00617v1.pdf")) == "2602.00617"
    assert _extract_arxiv_from_filename(Path("2409.12932v3.pdf")) == "2409.12932"
    assert _extract_arxiv_from_filename(Path("paper.pdf")) is None


def test_normalize_arxiv_strips_version():
    """Version suffix is stripped from arXiv IDs."""
    assert normalize_arxiv("2602.00617v1") == "2602.00617"
    assert normalize_arxiv("2409.12932v3") == "2409.12932"


def test_filter_sections_keeps_target_headings():
    """filter_sections keeps Abstract, Introduction, Related Work, Conclusion, Limitations."""
    md = """# Abstract
This paper proposes a new method.
# Introduction
Background and motivation.
# Method
Technical details here.
# Conclusion
We find that X works well.
# Appendix
Extra tables.
"""
    blocks = filter_sections(md)
    assert len(blocks) >= 3
    assert any("method" in b.lower() for b in blocks)


def test_filter_sections_discards_non_target():
    """Sections like Appendix, References are filtered out."""
    md = """# Appendix A
Supplementary data.
# References
[1] Author et al.
# Introduction
Real content here.
"""
    blocks = filter_sections(md)
    assert len(blocks) == 1
    assert "content" in blocks[0].lower()


def test_filter_sections_truncates_to_max_chars():
    """Each block is truncated to MAX_CHARS."""
    long_text = "x" * 20000
    md = f"# Introduction\n{long_text}"
    blocks = filter_sections(md)
    assert len(blocks) == 1
    assert len(blocks[0]) <= MAX_CHARS


def test_filter_sections_no_headings_returns_all_text():
    """When no markdown headings exist, all text is returned as one block."""
    raw_text = "This is raw PDF text with no headings."
    blocks = filter_sections(raw_text)
    assert len(blocks) == 1
    assert blocks[0] == raw_text
