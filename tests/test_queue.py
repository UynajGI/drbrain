"""Tests for confidence queue routing and resolution."""
import tempfile
from pathlib import Path
import json
from brbrain.storage.database import Database
from brbrain.extractor.queue import route_item, check_consensus, resolve_accept, resolve_reject

def test_route_item_below_threshold():
    """Items below weak_threshold go to queue."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        result = route_item(db, "p1", "concept", {"label": "weird thing", "type": "Method"}, 0.4, weak_threshold=0.7)
        assert result["action"] == "queued"
        assert result["queue_id"] is not None
        db.close()

def test_route_item_above_threshold_direct_ingest():
    """Items above auto_accept go directly to ingest."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        result = route_item(db, "p1", "concept", {"label": "clear thing", "type": "Method"}, 0.95, weak_threshold=0.7, auto_accept=0.9)
        assert result["action"] == "accepted"
        db.close()

def test_route_item_weak_marker():
    """Items between thresholds get weak marker."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        result = route_item(db, "p1", "concept", {"label": "ok thing"}, 0.8, weak_threshold=0.7, auto_accept=0.9)
        assert result["action"] == "weak"
        db.close()

def test_check_consensus_auto_promotes():
    """Concept appearing in 3+ papers with conf > 0.8 auto-promotes matching queue items."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "A", 2020, "uploaded")
        db.insert_paper("p2", "B", 2021, "uploaded")
        db.insert_paper("p3", "C", 2022, "uploaded")
        db.insert_concept("p1", "Method", "transformer", 0.95, year=2020)
        db.insert_concept("p2", "Method", "transformer", 0.90, year=2021)
        db.insert_concept("p3", "Method", "transformer", 0.85, year=2022)
        db.commit()

        qid = db.insert_queue_item("p4", "concept", json.dumps({"label": "transformer", "type": "Method"}), 0.6)
        db.commit()

        is_consensus = check_consensus(db, "transformer")
        assert is_consensus is True

        resolve_accept(db, qid)
        row = db.conn.execute("SELECT status FROM confidence_queue WHERE queue_id = ?", (qid,)).fetchone()
        assert row[0] == "accepted"
        db.close()

def test_resolve_reject():
    """resolve_reject sets status to rejected."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        qid = db.insert_queue_item("p1", "concept", '{"label": "bad"}', 0.3)
        resolve_reject(db, qid)
        row = db.conn.execute("SELECT status FROM confidence_queue WHERE queue_id = ?", (qid,)).fetchone()
        assert row[0] == "rejected"
        db.close()


def test_route_item_at_weak_threshold():
    """Items exactly at weak_threshold get weak marker, not queued."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        result = route_item(db, "p1", "concept", {"label": "borderline", "type": "Method"}, 0.7, weak_threshold=0.7, auto_accept=0.9)
        assert result["action"] == "weak"
        db.close()


def test_route_item_at_auto_accept_threshold():
    """Items exactly at auto_accept get accepted."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        result = route_item(db, "p1", "concept", {"label": "exact threshold"}, 0.9, weak_threshold=0.7, auto_accept=0.9)
        assert result["action"] == "accepted"
        db.close()


def test_route_item_returns_queue_id():
    """route_item returns a queue_id when queued."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        result = route_item(db, "p1", "concept", {"label": "test concept", "type": "Problem"}, 0.5, weak_threshold=0.7, auto_accept=0.9)
        assert result["action"] == "queued"
        assert result["queue_id"] is not None
        db.close()


def test_check_consensus_false_insufficient_papers():
    """check_consensus returns False when concept appears in fewer than 3 papers."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "A", 2020, "uploaded")
        db.insert_concept("p1", "Method", "transformer", 0.95, year=2020)
        db.commit()

        is_consensus = check_consensus(db, "transformer")
        assert is_consensus is False
        db.close()


def test_check_consensus_false_no_concept():
    """check_consensus returns False for nonexistent concept."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        is_consensus = check_consensus(db, "nonexistent")
        assert is_consensus is False
        db.close()


def test_resolve_accept_nonexistent_queue_id():
    """resolve_accept with nonexistent queue_id is safe."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        # Should not raise
        resolve_accept(db, "nonexistent")
        db.close()
