"""Tests for Semantic Scholar citation API."""
from brbrain.extractor.citation import parse_s2_response, match_to_local, expand_citations
import tempfile
from pathlib import Path
from unittest import mock
from brbrain.storage.database import Database

def test_parse_s2_response():
    """parse_s2_response extracts title, year, ids from S2 JSON."""
    s2_data = {
        "title": "Attention Is All You Need",
        "year": 2017,
        "externalIds": {"DOI": "10.1234/test", "ArXiv": "1706.03762"},
        "paperId": "abc123",
    }
    result = parse_s2_response(s2_data)
    assert result["title"] == "Attention Is All You Need"
    assert result["year"] == 2017
    assert result["doi"] == "10.1234/test"
    assert result["arxiv"] == "1706.03762"

def test_match_to_local_finds_existing():
    """match_to_local finds existing paper by DOI."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test Paper", 2024, "uploaded")
        db.insert_paper_ids("p1", doi="10.1234/test")
        db.commit()

        ref = {"doi": "10.1234/test", "title": "Test Paper", "year": 2024}
        entry = match_to_local(db, ref)
        assert entry.in_graph is True
        assert entry.local_id == "p1"
        db.close()

def test_match_to_local_creates_placeholder():
    """match_to_local creates placeholder for unknown paper."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Existing", 2024, "uploaded")
        db.commit()

        ref = {"doi": "10.9999/new", "title": "New Paper", "year": 2025}
        entry = match_to_local(db, ref)
        assert entry.in_graph is False
        assert entry.local_id is None
        db.close()


def test_expand_backfills_doi_from_s2():
    """expand_citations backfills DOI when paper has s2_id but no DOI."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test Paper", 2024, "uploaded")
        db.insert_paper_ids("p1", s2_id="abc123")  # no DOI
        db.commit()

        # S2 returns data with a DOI
        s2_data = {
            "title": "Test Paper",
            "year": 2024,
            "paperId": "abc123",
            "externalIds": {"DOI": "10.1234/backfilled", "ArXiv": "2401.00001"},
            "authors": [],
            "citationCount": 0,
            "references": [],
            "citations": [],
        }

        cfg = {"api": {"s2_rate_limit": 100}}
        with mock.patch("brbrain.extractor.citation.fetch_s2_paper", return_value=s2_data):
            refs, cits = expand_citations(db, "p1", cfg)

        doi = db.conn.execute(
            "SELECT doi FROM paper_ids WHERE local_id = 'p1'"
        ).fetchone()[0]
        assert doi == "10.1234/backfilled"
        db.close()


def test_expand_does_not_overwrite_existing_doi():
    """expand_citations does not overwrite an existing DOI."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test Paper", 2024, "uploaded")
        db.insert_paper_ids("p1", s2_id="abc123", doi="10.1234/existing")
        db.commit()

        s2_data = {
            "title": "Test Paper",
            "year": 2024,
            "paperId": "abc123",
            "externalIds": {"DOI": "10.9999/different"},
            "authors": [],
            "citationCount": 0,
            "references": [],
            "citations": [],
        }

        cfg = {"api": {"s2_rate_limit": 100}}
        with mock.patch("brbrain.extractor.citation.fetch_s2_paper", return_value=s2_data):
            expand_citations(db, "p1", cfg)

        doi = db.conn.execute(
            "SELECT doi FROM paper_ids WHERE local_id = 'p1'"
        ).fetchone()[0]
        assert doi == "10.1234/existing"
        db.close()


def test_expand_backfills_arxiv_if_missing():
    """expand_citations backfills arXiv ID when missing."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test Paper", 2024, "uploaded")
        db.insert_paper_ids("p1", s2_id="abc123")  # no arXiv
        db.commit()

        s2_data = {
            "title": "Test Paper",
            "year": 2024,
            "paperId": "abc123",
            "externalIds": {"DOI": "10.1234/test", "ArXiv": "2401.99999"},
            "authors": [],
            "citationCount": 0,
            "references": [],
            "citations": [],
        }

        cfg = {"api": {"s2_rate_limit": 100}}
        with mock.patch("brbrain.extractor.citation.fetch_s2_paper", return_value=s2_data):
            expand_citations(db, "p1", cfg)

        arxiv = db.conn.execute(
            "SELECT arxiv FROM paper_ids WHERE local_id = 'p1'"
        ).fetchone()[0]
        assert arxiv == "2401.99999"
        db.close()
