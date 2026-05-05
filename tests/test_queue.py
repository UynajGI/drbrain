"""Tests for confidence queue routing and resolution."""

import json

from drbrain.extractor.queue import (
    check_consensus,
    resolve_accept,
    resolve_all,
    resolve_reject,
    route_item,
)


def test_route_item_below_threshold(tmp_db):
    """Items below weak_threshold go to queue."""
    result = route_item(
        tmp_db, "p1", "concept", {"label": "weird thing", "type": "Method"}, 0.4, weak_threshold=0.7
    )
    assert result["action"] == "queued"
    assert result["queue_id"] is not None


def test_route_item_above_threshold_direct_ingest(tmp_db):
    """Items above auto_accept go directly to ingest."""
    result = route_item(
        tmp_db,
        "p1",
        "concept",
        {"label": "clear thing", "type": "Method"},
        0.95,
        weak_threshold=0.7,
        auto_accept=0.9,
    )
    assert result["action"] == "accepted"


def test_route_item_weak_marker(tmp_db):
    """Items between thresholds get weak marker."""
    result = route_item(
        tmp_db, "p1", "concept", {"label": "ok thing"}, 0.8, weak_threshold=0.7, auto_accept=0.9
    )
    assert result["action"] == "weak"


def test_check_consensus_auto_promotes(tmp_db):
    """Concept appearing in 3+ papers with conf > 0.8 auto-promotes matching queue items."""
    tmp_db.insert_paper("p1", "A", 2020, "uploaded")
    tmp_db.insert_paper("p2", "B", 2021, "uploaded")
    tmp_db.insert_paper("p3", "C", 2022, "uploaded")
    tmp_db.insert_concept("p1", "Method", "transformer", 0.95, year=2020)
    tmp_db.insert_concept("p2", "Method", "transformer", 0.90, year=2021)
    tmp_db.insert_concept("p3", "Method", "transformer", 0.85, year=2022)
    tmp_db.commit()

    qid = tmp_db.insert_queue_item(
        "p4", "concept", json.dumps({"label": "transformer", "type": "Method"}), 0.6
    )
    tmp_db.commit()

    is_consensus = check_consensus(tmp_db, "transformer")
    assert is_consensus is True

    resolve_accept(tmp_db, qid)
    row = tmp_db.conn.execute(
        "SELECT status FROM confidence_queue WHERE queue_id = ?", (qid,)
    ).fetchone()
    assert row[0] == "accepted"


def test_resolve_reject(tmp_db):
    """resolve_reject sets status to rejected."""
    qid = tmp_db.insert_queue_item("p1", "concept", '{"label": "bad"}', 0.3)
    resolve_reject(tmp_db, qid)
    row = tmp_db.conn.execute(
        "SELECT status FROM confidence_queue WHERE queue_id = ?", (qid,)
    ).fetchone()
    assert row[0] == "rejected"


def test_route_item_at_weak_threshold(tmp_db):
    """Items exactly at weak_threshold get weak marker, not queued."""
    result = route_item(
        tmp_db,
        "p1",
        "concept",
        {"label": "borderline", "type": "Method"},
        0.7,
        weak_threshold=0.7,
        auto_accept=0.9,
    )
    assert result["action"] == "weak"


def test_route_item_at_auto_accept_threshold(tmp_db):
    """Items exactly at auto_accept get accepted."""
    result = route_item(
        tmp_db,
        "p1",
        "concept",
        {"label": "exact threshold"},
        0.9,
        weak_threshold=0.7,
        auto_accept=0.9,
    )
    assert result["action"] == "accepted"


def test_route_item_returns_queue_id(tmp_db):
    """route_item returns a queue_id when queued."""
    result = route_item(
        tmp_db,
        "p1",
        "concept",
        {"label": "test concept", "type": "Problem"},
        0.5,
        weak_threshold=0.7,
        auto_accept=0.9,
    )
    assert result["action"] == "queued"
    assert result["queue_id"] is not None


def test_check_consensus_false_insufficient_papers(tmp_db):
    """check_consensus returns False when concept appears in fewer than 3 papers."""
    tmp_db.insert_paper("p1", "A", 2020, "uploaded")
    tmp_db.insert_concept("p1", "Method", "transformer", 0.95, year=2020)
    tmp_db.commit()

    is_consensus = check_consensus(tmp_db, "transformer")
    assert is_consensus is False


def test_check_consensus_false_no_concept(tmp_db):
    """check_consensus returns False for nonexistent concept."""
    is_consensus = check_consensus(tmp_db, "nonexistent")
    assert is_consensus is False


def test_resolve_accept_nonexistent_queue_id(tmp_db):
    """resolve_accept with nonexistent queue_id is safe."""
    # Should not raise
    resolve_accept(tmp_db, "nonexistent")


def test_resolve_all_accept_all(tmp_db):
    """resolve_all with no filters accepts all pending items."""
    tmp_db.insert_queue_item("p1", "concept", '{"label": "a"}', 0.3)
    tmp_db.insert_queue_item("p2", "concept", '{"label": "b"}', 0.5)
    tmp_db.insert_queue_item("p3", "concept", '{"label": "c"}', 0.2)
    tmp_db.commit()

    result = resolve_all(tmp_db, "accept")
    assert result["count"] == 3

    pending = tmp_db.get_queue_pending()
    assert len(pending) == 0


def test_resolve_all_with_type_filter(tmp_db):
    """resolve_all only processes items matching the type filter."""
    tmp_db.insert_queue_item("p1", "concept", '{"label": "a"}', 0.3)
    tmp_db.insert_queue_item("p2", "alias", '{"label": "b"}', 0.5)
    tmp_db.insert_queue_item("p3", "concept", '{"label": "c"}', 0.2)
    tmp_db.commit()

    result = resolve_all(tmp_db, "reject", type_filter="alias")
    assert result["count"] == 1

    pending = tmp_db.get_queue_pending()
    assert len(pending) == 2


def test_resolve_all_with_max_conf(tmp_db):
    """resolve_all only processes items with confidence <= max_conf."""
    tmp_db.insert_queue_item("p1", "concept", '{"label": "a"}', 0.3)
    tmp_db.insert_queue_item("p2", "concept", '{"label": "b"}', 0.5)
    tmp_db.insert_queue_item("p3", "concept", '{"label": "c"}', 0.7)
    tmp_db.commit()

    result = resolve_all(tmp_db, "reject", max_conf=0.5)
    assert result["count"] == 2

    pending = tmp_db.get_queue_pending()
    assert len(pending) == 1


def test_resolve_all_empty_queue(tmp_db):
    """resolve_all returns count 0 when queue is empty."""
    result = resolve_all(tmp_db, "accept")
    assert result["count"] == 0
