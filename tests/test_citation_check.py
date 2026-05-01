"""Tests for citation extraction and library matching."""

from drbrain.extractor.citation_check import extract_citations, match_citations


def test_extract_citations_narrative():
    """Extract narrative citations like 'Author (Year)'."""
    text = "Smith (2023) proposed a novel method."
    citations = extract_citations(text)
    assert len(citations) == 1
    assert citations[0].author == "Smith"
    assert citations[0].year == "2023"


def test_extract_citations_parenthetical():
    """Extract parenthetical citations like '(Author, Year)'."""
    text = "The approach is well-established (Jones, 2022)."
    citations = extract_citations(text)
    assert len(citations) >= 1
    assert any(c.author == "Jones" and c.year == "2022" for c in citations)


def test_extract_citations_et_al():
    """Extract 'et al.' citations."""
    text = "Lee et al. (2024) demonstrated similar results."
    citations = extract_citations(text)
    assert len(citations) >= 1
    assert any(c.author == "Lee" and c.year == "2024" for c in citations)


def test_extract_citations_multiple():
    """Extract multiple citations from text."""
    text = "Wang (2021) and (Chen, 2020; Zhang, 2019) built the foundation."
    citations = extract_citations(text)
    authors = {c.author for c in citations}
    assert "Wang" in authors


def test_extract_citations_no_citations():
    """Text with no citations returns empty list."""
    text = "This text contains no academic citations whatsoever."
    assert extract_citations(text) == []


def test_extract_citations_strips_duplicates():
    """Duplicate author+year pairs are deduplicated."""
    text = "Smith (2023) and also Smith (2023) in another context."
    citations = extract_citations(text)
    smith_citations = [c for c in citations if c.author == "Smith" and c.year == "2023"]
    assert len(smith_citations) == 1


def test_match_citations_empty():
    """Empty citation list returns empty results."""
    assert match_citations([], None) == []
