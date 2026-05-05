"""Tests for export formatters."""

from drbrain.storage.export import (
    batch_export,
    meta_to_bibtex,
    meta_to_markdown,
    meta_to_ris,
)

SAMPLE_META = {
    "local_id": "p1a2b3c4",
    "title": "Deep Learning for Graphs",
    "year": 2024,
    "doi": "10.1234/example",
    "arxiv": "2401.00001",
    "first_author_lastname": "Smith",
    "authors": "J. Smith and A. Jones",
    "journal": "Journal of AI Research",
    "volume": "42",
    "pages": "100-120",
    "paper_type": "paper",
}


def test_meta_to_bibtex():
    """BibTeX output contains citation key and required fields."""
    result = meta_to_bibtex(SAMPLE_META)
    assert "Smith2024Deep" in result
    assert "Deep Learning for Graphs" in result
    assert "2024" in result
    assert "@article{" in result or "@misc{" in result


def test_meta_to_bibtex_minimal():
    """BibTeX handles minimal metadata gracefully."""
    minimal = {"local_id": "px", "title": "Test", "year": 2025}
    result = meta_to_bibtex(minimal)
    assert "{" in result and "}" in result


def test_meta_to_ris():
    """RIS output contains required tags."""
    result = meta_to_ris(SAMPLE_META)
    assert "TY  - JOUR" in result
    assert "TI  - Deep Learning for Graphs" in result
    assert "PY  - 2024" in result
    assert "ER  -" in result


def test_meta_to_ris_minimal():
    """RIS handles minimal metadata."""
    minimal = {"local_id": "px", "title": "Test", "year": 2025}
    result = meta_to_ris(minimal)
    assert "TY  - JOUR" in result
    assert "TI  - Test" in result


def test_meta_to_markdown():
    """Markdown output includes title, authors, year."""
    result = meta_to_markdown(SAMPLE_META)
    assert "Deep Learning for Graphs" in result
    assert "(2024)" in result
    assert "J. Smith" in result


def test_meta_to_markdown_no_authors():
    """Markdown without authors shows 'Unknown'."""
    minimal = {"local_id": "px", "title": "Test", "year": 2025}
    result = meta_to_markdown(minimal)
    assert "**Test** (2025)" in result


def test_batch_export():
    """Batch export joins entries with newlines."""
    metas = [SAMPLE_META, SAMPLE_META]
    result = batch_export(metas, "bib")
    entries = result.strip().split("\n\n")
    assert len(entries) == 2


def test_batch_export_unknown_format():
    """Unknown format returns empty."""
    assert batch_export([SAMPLE_META], "docx") == ""


def test_bibtex_includes_volume_pages():
    """BibTeX output must include volume and pages when present."""
    result = meta_to_bibtex(SAMPLE_META)
    assert "volume" in result.lower()
    assert "pages" in result.lower()


def test_ris_includes_volume_pages():
    """RIS output must include VL (volume) and SP/EP (pages) when present."""
    result = meta_to_ris(SAMPLE_META)
    assert "VL" in result
    assert "SP" in result
