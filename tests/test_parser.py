"""Tests for MinerU PDF parser integration."""
from brbrain.parser.mineru_parser import MinerUParser, filter_sections, MAX_CHARS

def test_mineru_parser_flash_mode():
    """MinerUParser initializes in flash mode with no token."""
    parser = MinerUParser(token="", model="vlm")
    assert parser.mode == "flash"
    assert parser.model == "vlm"

def test_mineru_parser_token_mode():
    """MinerUParser initializes in token mode with a token."""
    parser = MinerUParser(token="abc-123", model="pipeline")
    assert parser.mode == "token"
    assert parser.token == "abc-123"

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
