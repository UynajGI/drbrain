"""Tests for v1.1 database schema: arguments, confidence_queue, temporal fields."""


def test_arguments_table_exists(tmp_db):
    """arguments table is created on Database init."""
    row = tmp_db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='arguments'"
    ).fetchone()
    assert row is not None


def test_confidence_queue_table_exists(tmp_db):
    """confidence_queue table is created on Database init."""
    row = tmp_db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='confidence_queue'"
    ).fetchone()
    assert row is not None


def test_insert_argument(tmp_db):
    """insert_argument stores argument and returns arg_id."""
    tmp_db.insert_paper("p1", "Test", 2024, "uploaded")
    tmp_db.commit()
    arg_id = tmp_db.insert_argument(
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
    row = tmp_db.conn.execute(
        "SELECT claim, claim_type FROM arguments WHERE arg_id = ?", (arg_id,)
    ).fetchone()
    assert row[0] == "Self-attention replaces RNN"
    assert row[1] == "proposes"


def test_insert_queue_item(tmp_db):
    """insert_queue_item stores pending item and returns queue_id."""
    qid = tmp_db.insert_queue_item(
        source_paper="p1",
        item_type="concept",
        item_data='{"label": "neuro-symbolic reasoning", "type": "Method"}',
        confidence=0.52,
    )
    assert qid is not None
    row = tmp_db.conn.execute(
        "SELECT status FROM confidence_queue WHERE queue_id = ?", (qid,)
    ).fetchone()
    assert row[0] == "pending"


def test_resolve_queue_item_accept(tmp_db):
    """accept_queue_item sets status to 'accepted'."""
    qid = tmp_db.insert_queue_item("p1", "concept", '{"label": "test"}', 0.5)
    tmp_db.accept_queue_item(qid)
    row = tmp_db.conn.execute(
        "SELECT status FROM confidence_queue WHERE queue_id = ?", (qid,)
    ).fetchone()
    assert row[0] == "accepted"


def test_resolve_queue_item_reject(tmp_db):
    """reject_queue_item sets status to 'rejected'."""
    qid = tmp_db.insert_queue_item("p1", "concept", '{"label": "test"}', 0.5)
    tmp_db.reject_queue_item(qid)
    row = tmp_db.conn.execute(
        "SELECT status FROM confidence_queue WHERE queue_id = ?", (qid,)
    ).fetchone()
    assert row[0] == "rejected"


def test_get_queue_pending(tmp_db):
    """get_queue_pending returns only pending items."""
    q1 = tmp_db.insert_queue_item("p1", "concept", '{"label": "a"}', 0.5)
    q2 = tmp_db.insert_queue_item("p1", "concept", '{"label": "b"}', 0.4)
    tmp_db.accept_queue_item(q1)
    pending = tmp_db.get_queue_pending()
    assert len(pending) == 1
    assert pending[0]["queue_id"] == q2


def test_concepts_have_temporal_fields(tmp_db):
    """concepts table has first_seen and last_seen columns."""
    tmp_db.insert_paper("p1", "Test", 2020, "uploaded")
    tmp_db.commit()
    tmp_db.insert_concept("p1", "Method", "transformer", 0.95, year=2020)
    cid = tmp_db.conn.execute("SELECT concept_id FROM concepts").fetchone()[0]
    row = tmp_db.conn.execute(
        "SELECT first_seen, last_seen FROM concepts WHERE concept_id = ?", (cid,)
    ).fetchone()
    assert row[0] == 2020
    assert row[1] == 2020


def test_get_concept_evolution(tmp_db):
    """get_concept_evolution returns year-by-year stats for a concept label."""
    tmp_db.insert_paper("p1", "Paper A", 2017, "uploaded")
    tmp_db.insert_paper("p2", "Paper B", 2020, "uploaded")
    tmp_db.insert_paper("p3", "Paper C", 2023, "uploaded")
    tmp_db.insert_concept("p1", "Method", "transformer", 0.95, year=2017)
    tmp_db.insert_concept("p2", "Method", "transformer", 0.90, year=2020)
    tmp_db.insert_concept("p3", "Method", "transformer", 0.75, year=2023)
    tmp_db.commit()
    evolution = tmp_db.get_concept_evolution("transformer")
    assert len(evolution) == 3
    assert evolution[0]["year"] == 2017
    assert evolution[0]["count"] == 1
    assert evolution[1]["year"] == 2020
    assert evolution[1]["count"] == 1


def test_detect_evolution_signals(tmp_db):
    """detect_evolution_signals returns signal type for a concept."""
    tmp_db.insert_paper("p1", "A", 2024, "uploaded")
    tmp_db.insert_paper("p2", "B", 2025, "uploaded")
    tmp_db.insert_concept("p1", "Method", "new_thing", 0.9, year=2024)
    tmp_db.insert_concept("p2", "Method", "new_thing", 0.85, year=2025)
    tmp_db.commit()
    signals = tmp_db.detect_evolution_signals()
    assert any(s["label"] == "new_thing" for s in signals)
