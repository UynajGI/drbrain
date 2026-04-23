"""Tests for citation.py: expand_citations, match_to_local, parse_s2_response."""
import tempfile
import unittest.mock
from pathlib import Path

from brbrain.extractor.citation import (
    match_to_local, parse_s2_response, expand_citations,
    fetch_s2_paper, search_s2,
)
from brbrain.storage.database import Database
from brbrain.report.generator import RefEntry


def _make_db_with_paper(local_id: str, title: str, year: int, doi: str = None, arxiv: str = None, s2_id: str = None) -> Database:
    """Create a temp DB with one paper."""
    td = tempfile.mkdtemp()
    db = Database(Path(td) / "test.db")
    db.insert_paper(local_id, title, year, "uploaded")
    db.insert_paper_ids(local_id, doi=doi, arxiv=arxiv, s2_id=s2_id)
    db.commit()
    return db


# -- parse_s2_response --

def test_parse_s2_response_minimal():
    """parse_s2_response handles minimal response."""
    data = {"paperId": "abc123", "title": "Test", "year": 2024}
    parsed = parse_s2_response(data)
    assert parsed["s2_id"] == "abc123"
    assert parsed["title"] == "Test"
    assert parsed["year"] == 2024
    assert parsed["citation_count"] == 0


def test_parse_s2_response_with_ids():
    """parse_s2_response extracts all external IDs."""
    data = {
        "paperId": "abc",
        "title": "Test",
        "year": 2024,
        "externalIds": {
            "DOI": "10.1234/test",
            "ArXiv": "2401.12345",
            "OpenAlex": "W123",
        },
        "citationCount": 42,
    }
    parsed = parse_s2_response(data)
    assert parsed["doi"] == "10.1234/test"
    assert parsed["arxiv"] == "2401.12345"
    assert parsed["openalex_id"] == "W123"
    assert parsed["citation_count"] == 42


def test_parse_s2_response_null_ext_ids():
    """parse_s2_response handles null externalIds."""
    data = {"paperId": "abc", "title": "Test", "year": 2024, "externalIds": None}
    parsed = parse_s2_response(data)
    assert parsed["doi"] is None


# -- match_to_local --

def test_match_to_local_by_doi():
    """match_to_local finds paper by DOI."""
    db = _make_db_with_paper("p1", "Test Paper", 2024, doi="10.1234/test")
    ref = {"title": "Test Paper", "year": 2024, "doi": "10.1234/test"}
    entry = match_to_local(db, ref)
    assert entry.in_graph is True
    assert entry.local_id == "p1"
    db.close()


def test_match_to_local_by_arxiv():
    """match_to_local finds paper by arXiv ID."""
    db = _make_db_with_paper("p1", "Test Paper", 2024, arxiv="2401.12345")
    ref = {"title": "Test Paper", "year": 2024, "arxiv": "2401.12345"}
    entry = match_to_local(db, ref)
    assert entry.in_graph is True
    assert entry.local_id == "p1"
    db.close()


def test_match_to_local_by_s2_id():
    """match_to_local finds paper by S2 ID."""
    db = _make_db_with_paper("p1", "Test Paper", 2024, s2_id="abc123")
    ref = {"title": "Test Paper", "year": 2024, "s2_id": "abc123"}
    entry = match_to_local(db, ref)
    assert entry.in_graph is True
    assert entry.local_id == "p1"
    db.close()


def test_match_to_local_by_title_year():
    """match_to_local finds paper by exact title+year."""
    db = _make_db_with_paper("p1", "Exact Title", 2024)
    ref = {"title": "Exact Title", "year": 2024}
    entry = match_to_local(db, ref)
    assert entry.in_graph is True
    assert entry.local_id == "p1"
    db.close()


def test_match_to_local_not_found():
    """match_to_local returns in_graph=False when no match."""
    db = _make_db_with_paper("p1", "Different Paper", 2024)
    ref = {"title": "Unknown Paper", "year": 2025, "doi": "10.9999/nope"}
    entry = match_to_local(db, ref)
    assert entry.in_graph is False
    assert entry.local_id is None
    db.close()


# -- expand_citations --

def test_expand_citations_creates_placeholder_neighbors():
    """expand_citations creates placeholder nodes for references not in graph."""
    s2_data = {
        "paperId": "s2_abc",
        "title": "Seed Paper",
        "year": 2024,
        "externalIds": {"DOI": "10.1234/seed"},
        "citationCount": 5,
        "references": [
            {"paperId": "ref1", "title": "Ref Paper 1", "year": 2020, "externalIds": None, "citationCount": 0},
        ],
        "citations": [],
    }

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        db = Database(str(db_path))
        db.insert_paper("p1", "Seed Paper", 2024, "uploaded")
        db.insert_paper_ids("p1", doi="10.1234/seed", s2_id="s2_abc")
        db.commit()

        cfg = {"api": {"s2_rate_limit": 100}}

        with unittest.mock.patch("brbrain.extractor.citation.fetch_s2_paper", return_value=s2_data):
            refs, cits = expand_citations(db, "p1", cfg)

        # Reference should be created as placeholder
        papers = db.get_all_papers()
        placeholder_titles = {p["title"] for p in papers if p["status"] == "placeholder"}
        assert "Ref Paper 1" in placeholder_titles

        # Edge should exist
        edges = db.conn.execute("SELECT src_id, dst_id, relation FROM edges WHERE relation='cites'").fetchall()
        assert len(edges) >= 1

        db.close()


def test_expand_citations_matches_existing_neighbor():
    """expand_citations marks existing papers as in_graph."""
    s2_data = {
        "paperId": "s2_abc",
        "title": "Seed Paper",
        "year": 2024,
        "externalIds": {"DOI": "10.1234/seed"},
        "citationCount": 0,
        "references": [
            {"paperId": "ref1", "title": "Existing Ref", "year": 2020,
             "externalIds": {"DOI": "10.5678/existing"}, "citationCount": 0},
        ],
        "citations": [],
    }

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        db = Database(str(db_path))
        db.insert_paper("p1", "Seed Paper", 2024, "uploaded")
        db.insert_paper_ids("p1", doi="10.1234/seed", s2_id="s2_abc")
        db.insert_paper("p2", "Existing Ref", 2020, "uploaded")
        db.insert_paper_ids("p2", doi="10.5678/existing")
        db.commit()

        cfg = {"api": {"s2_rate_limit": 100}}

        with unittest.mock.patch("brbrain.extractor.citation.fetch_s2_paper", return_value=s2_data):
            refs, cits = expand_citations(db, "p1", cfg)

        assert len(refs) == 1
        assert refs[0].in_graph is True
        assert refs[0].local_id == "p2"
        db.close()


def test_expand_citations_returns_empty_when_no_paper():
    """expand_citations returns empty when paper not found."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        refs, cits = expand_citations(db, "nonexistent", {})
        assert refs == []
        assert cits == []
        db.close()


def test_expand_citations_returns_empty_when_no_s2_id():
    """expand_citations returns empty when paper has no S2 ID and search fails."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "No ID Paper", 2024, "uploaded")
        db.commit()

        with unittest.mock.patch("brbrain.extractor.citation.search_s2", return_value=[]), \
             unittest.mock.patch("brbrain.extractor.citation._expand_with_openalex", return_value=([], [])):
            refs, cits = expand_citations(db, "p1", {})

        assert refs == []
        assert cits == []
        db.close()


def test_expand_citations_handles_citations_direction():
    """Citations create edges from citing paper TO seed paper (cited_by)."""
    s2_data = {
        "paperId": "s2_seed",
        "title": "Seed Paper",
        "year": 2024,
        "externalIds": None,
        "citationCount": 0,
        "references": [],
        "citations": [
            {"paperId": "cit1", "title": "Citing Paper", "year": 2025, "externalIds": None, "citationCount": 0},
        ],
    }

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        db = Database(str(db_path))
        db.insert_paper("p1", "Seed Paper", 2024, "uploaded")
        db.insert_paper_ids("p1", s2_id="s2_seed")
        db.commit()

        cfg = {"api": {"s2_rate_limit": 100}}

        with unittest.mock.patch("brbrain.extractor.citation.fetch_s2_paper", return_value=s2_data):
            refs, cits = expand_citations(db, "p1", cfg)

        # cited_by edge: placeholder -> seed
        edges = db.conn.execute(
            "SELECT src_id, dst_id, relation FROM edges WHERE relation='cited_by'"
        ).fetchall()
        assert len(edges) >= 1
        # dst should be the seed paper
        assert any(e[1] == "p1" for e in edges)

        db.close()


# -- original non-retry functions --

def test_fetch_s2_paper_success():
    """fetch_s2_paper returns parsed JSON on success."""
    mock_resp = unittest.mock.Mock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = unittest.mock.Mock()
    mock_resp.json.return_value = {"paperId": "abc"}

    with unittest.mock.patch("requests.get", return_value=mock_resp):
        result = fetch_s2_paper("abc")
        assert result is not None
        assert result["paperId"] == "abc"


def test_fetch_s2_paper_error():
    """fetch_s2_paper returns None on error."""
    with unittest.mock.patch("requests.get", side_effect=Exception("fail")):
        result = fetch_s2_paper("abc")
        assert result is None


def test_search_s2_success():
    """search_s2 returns data list on success."""
    mock_resp = unittest.mock.Mock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = unittest.mock.Mock()
    mock_resp.json.return_value = {"data": [{"paperId": "a"}, {"paperId": "b"}]}

    with unittest.mock.patch("requests.get", return_value=mock_resp):
        results = search_s2("test query")
        assert len(results) == 2


def test_search_s2_error():
    """search_s2 returns empty list on error."""
    with unittest.mock.patch("requests.get", side_effect=Exception("fail")):
        results = search_s2("test")
        assert results == []


def test_search_s2_uses_api_key():
    """search_s2 includes x-api-key header when provided."""
    captured = {}

    def mock_get(*args, **kwargs):
        captured["headers"] = kwargs.get("headers", {})
        resp = unittest.mock.Mock()
        resp.status_code = 200
        resp.raise_for_status = unittest.mock.Mock()
        resp.json.return_value = {"data": []}
        return resp

    with unittest.mock.patch("requests.get", side_effect=mock_get):
        search_s2("test", api_key="key-123")
        assert captured.get("headers", {}).get("x-api-key") == "key-123"
