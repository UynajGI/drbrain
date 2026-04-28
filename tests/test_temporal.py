"""Tests for v1.1 database schema: arguments, confidence_queue, temporal fields."""
import tempfile
from pathlib import Path
from drbrain.storage.database import Database

def test_arguments_table_exists():
    """arguments table is created on Database init."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        row = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='arguments'"
        ).fetchone()
        assert row is not None
        db.close()

def test_confidence_queue_table_exists():
    """confidence_queue table is created on Database init."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        row = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='confidence_queue'"
        ).fetchone()
        assert row is not None
        db.close()

def test_insert_argument():
    """insert_argument stores argument and returns arg_id."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test", 2024, "uploaded")
        db.commit()
        arg_id = db.insert_argument(
            source_paper="p1",
            claim="Self-attention replaces RNN",
            claim_type="proposes",
            target_label="Transformer",
            target_type="Method",
            evidence_type="empirical",
            evidence_detail="WMT14 EN-DE BLEU +2.0",
            confidence=0.95,
        )
        assert arg_id is not None
        row = db.conn.execute("SELECT claim, claim_type FROM arguments WHERE arg_id = ?", (arg_id,)).fetchone()
        assert row[0] == "Self-attention replaces RNN"
        assert row[1] == "proposes"
        db.close()

def test_insert_queue_item():
    """insert_queue_item stores pending item and returns queue_id."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        qid = db.insert_queue_item(
            source_paper="p1",
            item_type="concept",
            item_data='{"label": "neuro-symbolic reasoning", "type": "Method"}',
            confidence=0.52,
        )
        assert qid is not None
        row = db.conn.execute("SELECT status FROM confidence_queue WHERE queue_id = ?", (qid,)).fetchone()
        assert row[0] == "pending"
        db.close()

def test_resolve_queue_item_accept():
    """accept_queue_item sets status to 'accepted'."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        qid = db.insert_queue_item("p1", "concept", '{"label": "test"}', 0.5)
        db.accept_queue_item(qid)
        row = db.conn.execute("SELECT status FROM confidence_queue WHERE queue_id = ?", (qid,)).fetchone()
        assert row[0] == "accepted"
        db.close()

def test_resolve_queue_item_reject():
    """reject_queue_item sets status to 'rejected'."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        qid = db.insert_queue_item("p1", "concept", '{"label": "test"}', 0.5)
        db.reject_queue_item(qid)
        row = db.conn.execute("SELECT status FROM confidence_queue WHERE queue_id = ?", (qid,)).fetchone()
        assert row[0] == "rejected"
        db.close()

def test_get_queue_pending():
    """get_queue_pending returns only pending items."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        q1 = db.insert_queue_item("p1", "concept", '{"label": "a"}', 0.5)
        q2 = db.insert_queue_item("p1", "concept", '{"label": "b"}', 0.4)
        db.accept_queue_item(q1)
        pending = db.get_queue_pending()
        assert len(pending) == 1
        assert pending[0]["queue_id"] == q2
        db.close()

def test_concepts_have_temporal_fields():
    """concepts table has first_seen and last_seen columns."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test", 2020, "uploaded")
        db.commit()
        db.insert_concept("p1", "Method", "transformer", 0.95, year=2020)
        cid = db.conn.execute("SELECT concept_id FROM concepts").fetchone()[0]
        row = db.conn.execute(
            "SELECT first_seen, last_seen FROM concepts WHERE concept_id = ?", (cid,)
        ).fetchone()
        assert row[0] == 2020
        assert row[1] == 2020
        db.close()

def test_get_concept_evolution():
    """get_concept_evolution returns year-by-year stats for a concept label."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Paper A", 2017, "uploaded")
        db.insert_paper("p2", "Paper B", 2020, "uploaded")
        db.insert_paper("p3", "Paper C", 2023, "uploaded")
        db.insert_concept("p1", "Method", "transformer", 0.95, year=2017)
        db.insert_concept("p2", "Method", "transformer", 0.90, year=2020)
        db.insert_concept("p3", "Method", "transformer", 0.75, year=2023)
        db.commit()
        evolution = db.get_concept_evolution("transformer")
        assert len(evolution) == 3
        assert evolution[0]["year"] == 2017
        assert evolution[0]["count"] == 1
        assert evolution[1]["year"] == 2020
        assert evolution[1]["count"] == 1
        db.close()

def test_detect_evolution_signals():
    """detect_evolution_signals returns signal type for a concept."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "A", 2024, "uploaded")
        db.insert_paper("p2", "B", 2025, "uploaded")
        db.insert_concept("p1", "Method", "new_thing", 0.9, year=2024)
        db.insert_concept("p2", "Method", "new_thing", 0.85, year=2025)
        db.commit()
        signals = db.detect_evolution_signals()
        assert any(s["label"] == "new_thing" for s in signals)
        db.close()
