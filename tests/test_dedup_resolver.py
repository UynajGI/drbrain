"""Tests for dedup/resolver.py: triple-ID resolution and title utilities."""
import tempfile
from pathlib import Path

from drbrain.dedup.resolver import (
    normalize_doi, normalize_arxiv, title_key, title_hash,
    DedupEngine, PaperIDs,
)
from drbrain.storage.database import Database


def _make_db() -> Database:
    td = tempfile.mkdtemp()
    return Database(Path(td) / "test.db")


# -- normalize_doi --

def test_normalize_doi_url():
    assert normalize_doi("https://doi.org/10.1234/abc") == "10.1234/abc"


def test_normalize_doi_prefix():
    assert normalize_doi("DOI: 10.5678/xyz") == "10.5678/xyz"


# -- normalize_arxiv --

def test_normalize_arxiv_version():
    assert normalize_arxiv("2401.12345v2") == "2401.12345"


def test_normalize_arxiv_no_match():
    # Returns raw when no pattern matches
    assert normalize_arxiv("not-an-arxiv-id") == "not-an-arxiv-id"


# -- title_key --

def test_title_key_removes_articles():
    assert title_key("The quick brown fox") == "quick brown fox"


def test_title_key_removes_punctuation():
    assert title_key("Hello, World!") == "hello world"


def test_title_key_normalizes_whitespace():
    assert title_key("A   title   with  spaces") == "title with spaces"


# -- title_hash --

def test_title_hash_deterministic():
    """Same title produces same hash."""
    h1 = title_hash("The Transformer Paper")
    h2 = title_hash("The Transformer Paper")
    assert h1 == h2


def test_title_hash_different_titles():
    """Different titles produce different hashes."""
    assert title_hash("Paper A") != title_hash("Paper B")


def test_title_hash_short():
    """Hash is 12 characters."""
    assert len(title_hash("Test")) == 12


# -- DedupEngine --

def test_resolve_by_doi():
    """DOI match has highest priority."""
    db = _make_db()
    db.insert_paper("p1", "Test", 2024, "uploaded")
    db.insert_paper_ids("p1", doi="10.1234/test")
    db.commit()

    engine = DedupEngine(db)
    result = engine.resolve(PaperIDs(doi="10.1234/test"))
    assert result == "p1"
    db.close()


def test_resolve_by_arxiv():
    """arXiv match used when no DOI."""
    db = _make_db()
    db.insert_paper("p1", "Test", 2024, "uploaded")
    db.insert_paper_ids("p1", arxiv="2401.12345")
    db.commit()

    engine = DedupEngine(db)
    result = engine.resolve(PaperIDs(arxiv="2401.12345"))
    assert result == "p1"
    db.close()


def test_resolve_by_s2_id():
    """S2 ID match used when no DOI/arXiv."""
    db = _make_db()
    db.insert_paper("p1", "Test", 2024, "uploaded")
    db.insert_paper_ids("p1", s2_id="abc123")
    db.commit()

    engine = DedupEngine(db)
    result = engine.resolve(PaperIDs(s2_id="abc123"))
    assert result == "p1"
    db.close()


def test_resolve_by_openalex():
    """OpenAlex ID match used as last ID-based lookup."""
    db = _make_db()
    db.insert_paper("p1", "Test", 2024, "uploaded")
    db.insert_paper_ids("p1", openalex_id="W123456")
    db.commit()

    engine = DedupEngine(db)
    result = engine.resolve(PaperIDs(openalex_id="W123456"))
    assert result == "p1"
    db.close()


def test_resolve_by_title_year():
    """Title+year fuzzy match as final fallback."""
    db = _make_db()
    db.insert_paper("p1", "Exact Title Match", 2024, "uploaded")
    db.commit()

    engine = DedupEngine(db)
    result = engine.resolve(PaperIDs(), title="Exact Title Match", year=2024)
    assert result == "p1"
    db.close()


def test_resolve_returns_none_no_match():
    """resolve returns None when no match found."""
    db = _make_db()
    engine = DedupEngine(db)
    result = engine.resolve(
        PaperIDs(doi="10.9999/nope"),
        title="Unknown Paper",
        year=2099,
    )
    assert result is None
    db.close()


def test_resolve_returns_none_empty_ids():
    """resolve returns None with empty PaperIDs and no title."""
    db = _make_db()
    engine = DedupEngine(db)
    result = engine.resolve(PaperIDs())
    assert result is None
    db.close()


def test_resolve_priority_doi_over_arxiv():
    """When both DOI and arXiv are set, DOI is checked first."""
    db = _make_db()
    db.insert_paper("p1", "DOI Paper", 2024, "uploaded")
    db.insert_paper_ids("p1", doi="10.1234/doi", arxiv="2401.00001")
    db.insert_paper("p2", "Arxiv Paper", 2024, "uploaded")
    db.insert_paper_ids("p2", arxiv="2401.00001")
    db.commit()

    engine = DedupEngine(db)
    # DOI should resolve to p1, not p2
    result = engine.resolve(PaperIDs(doi="10.1234/doi", arxiv="2401.00001"))
    assert result == "p1"
    db.close()
