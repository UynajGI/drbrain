"""Tests for delete paper functionality."""

import tempfile
from pathlib import Path

from drbrain.storage.database import Database


def test_delete_paper_removes_concepts_and_edges():
    """delete_paper removes paper, concepts, and edges."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test Paper", 2024, "uploaded")
        db.insert_paper_ids("p1", doi="10.1234/test")
        db.insert_concept("p1", "Method", "transformer", 0.95, year=2024)
        db.insert_edge("p1", "p2", "cites", "p1")
        db.commit()

        counts = db.delete_paper("p1")
        assert counts["concepts"] == 1
        assert counts["arguments"] == 0
        assert counts["edges"] == 1

        # Verify paper is gone
        assert db.get_paper("p1") is None

        # Verify concept is gone
        concepts = db.conn.execute("SELECT COUNT(*) FROM concepts WHERE local_id = 'p1'").fetchone()
        assert concepts[0] == 0

        # Verify edge is gone
        edges = db.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE src_id = 'p1' OR dst_id = 'p1'"
        ).fetchone()
        assert edges[0] == 0

        # Verify paper_ids is gone
        ids = db.conn.execute("SELECT COUNT(*) FROM paper_ids WHERE local_id = 'p1'").fetchone()
        assert ids[0] == 0

        db.close()


def test_delete_paper_removes_arguments():
    """delete_paper removes arguments associated with paper."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test Paper", 2024, "uploaded")
        db.insert_argument("p1", "X works well", "supports", "Y", "Method", confidence=0.9)
        db.commit()

        counts = db.delete_paper("p1")
        assert counts["arguments"] == 1

        args = db.conn.execute(
            "SELECT COUNT(*) FROM arguments WHERE source_paper = 'p1'"
        ).fetchone()
        assert args[0] == 0
        db.close()


def test_delete_paper_removes_queue_items():
    """delete_paper removes confidence queue items for that paper."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test Paper", 2024, "uploaded")
        db.insert_queue_item("p1", "concept", '{"label": "xyz"}', 0.4)
        db.commit()

        counts = db.delete_paper("p1")
        assert counts["queue_items"] == 1

        items = db.conn.execute(
            "SELECT COUNT(*) FROM confidence_queue WHERE source_paper = 'p1'"
        ).fetchone()
        assert items[0] == 0
        db.close()


def test_delete_paper_nonexistent():
    """delete_paper with nonexistent paper returns zero counts."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        counts = db.delete_paper("nonexistent")
        assert counts["concepts"] == 0
        assert counts["edges"] == 0
        db.close()


def test_delete_paper_does_not_affect_others():
    """delete_paper only removes the target paper, not others."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Paper A", 2024, "uploaded")
        db.insert_paper("p2", "Paper B", 2025, "uploaded")
        db.insert_concept("p2", "Method", "other", 0.9, year=2025)
        db.commit()

        db.delete_paper("p1")

        # p2 should still exist with its concept
        p2 = db.get_paper("p2")
        assert p2 is not None
        assert p2["title"] == "Paper B"

        concepts = db.conn.execute("SELECT COUNT(*) FROM concepts WHERE local_id = 'p2'").fetchone()
        assert concepts[0] == 1
        db.close()
