"""Tests for Semantic Scholar citation API."""
from brbrain.extractor.citation import parse_s2_response, match_to_local
import tempfile
from pathlib import Path
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
