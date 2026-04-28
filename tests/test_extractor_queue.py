"""Tests for extractor queue: consensus, route_item edge cases, resolve_accept."""
import asyncio
import tempfile
from pathlib import Path
from unittest import mock

from drbrain.extractor.queue import route_item, check_consensus, resolve_accept
from drbrain.storage.database import Database


def _make_db() -> Database:
    td = tempfile.mkdtemp()
    return Database(Path(td) / "test.db")


def test_route_item_accepted():
    """High confidence items are auto-accepted."""
    db = _make_db()
    result = route_item(db, "p1", "concept", {"label": "X"}, confidence=0.95)
    assert result["action"] == "accepted"
    assert result["queue_id"] is None
    db.close()


def test_route_item_weak():
    """Medium confidence items marked as weak."""
    db = _make_db()
    result = route_item(db, "p1", "concept", {"label": "X"}, confidence=0.8)
    assert result["action"] == "weak"
    assert result["queue_id"] is None
    db.close()


def test_route_item_queued():
    """Low confidence items are queued."""
    db = _make_db()
    result = route_item(db, "p1", "concept", {"label": "X"}, confidence=0.5)
    assert result["action"] == "queued"
    assert result["queue_id"] is not None
    db.close()


def test_check_consensus_true():
    """Consensus returns True when N+ papers agree with high confidence."""
    db = _make_db()
    for i in range(3):
        db.insert_paper(f"p{i}", f"Paper {i}", 2024, "uploaded")
        db.insert_concept(f"p{i}", "Problem", "Transformer", 0.95, year=2024)
    db.commit()

    assert check_consensus(db, "Transformer", min_papers=3, min_confidence=0.8) is True
    db.close()


def test_check_consensus_false_insufficient_papers():
    """Consensus returns False when too few papers."""
    db = _make_db()
    db.insert_paper("p1", "Paper", 2024, "uploaded")
    db.insert_concept("p1", "Problem", "Transformer", 0.95, year=2024)
    db.commit()

    assert check_consensus(db, "Transformer", min_papers=3) is False
    db.close()


def test_check_consensus_false_no_concept():
    """Consensus returns False when concept doesn't exist."""
    db = _make_db()
    assert check_consensus(db, "Nonexistent") is False
    db.close()


def test_resolve_accept_no_consensus_cascade():
    """Accept without consensus only marks the single item."""
    db = _make_db()
    db.insert_paper("p1", "A", 2024, "uploaded")
    qid = db.insert_queue_item("p1", "concept", '{"label": "RareConcept"}', 0.5)
    db.commit()

    resolve_accept(db, qid)

    status = db.conn.execute(
        "SELECT status FROM confidence_queue WHERE queue_id = ?", (qid,)
    ).fetchone()[0]
    assert status == "accepted"
    db.close()


def test_resolve_accept_nonexistent_queue_id():
    """resolve_accept silently returns for nonexistent queue_id."""
    db = _make_db()
    resolve_accept(db, 9999)  # Should not raise
    db.close()


# -- concept.py extract_concepts --

def test_extract_concepts_returns_none_on_llm_failure():
    """extract_concepts returns None when LLM extraction fails."""
    from drbrain.extractor.concept import extract_concepts

    async def run():
        with mock.patch("drbrain.extractor.concept.acall_with_fallback", return_value=None):
            result = await extract_concepts("some text", [{"provider": "test"}])
            assert result is None

    asyncio.run(run())


def test_extract_concepts_returns_data_on_success():
    """extract_concepts returns ExtractedConcepts on success."""
    from drbrain.extractor.concept import extract_concepts

    async def run():
        mock_data = {
            "problems": [{"label": "X", "confidence": 0.9}],
            "methods": [], "conclusions": [], "debates": [], "gaps": [],
            "actors": [], "relations": [], "arguments": [],
        }
        with mock.patch("drbrain.extractor.concept.acall_with_fallback", return_value=mock_data):
            result = await extract_concepts("some text", [{"provider": "test"}])
            assert result is not None
            assert len(result.problems) == 1
            assert result.problems[0]["label"] == "X"

    asyncio.run(run())
