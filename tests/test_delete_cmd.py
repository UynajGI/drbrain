"""Tests for delete paper functionality."""

import json
import tempfile
from pathlib import Path

import numpy as np

from drbrain.concept_graph.concept_extract import ensure_cache_table
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
        db.save_embedding("unrelated", [1.0], 1)
        revision = db.get_embedding_revision()
        counts = db.delete_paper("nonexistent")
        assert counts["concepts"] == 0
        assert counts["edges"] == 0
        embeddings = db.load_embeddings()
        assert list(embeddings) == ["unrelated"]
        np.testing.assert_array_equal(embeddings["unrelated"], np.array([1.0], dtype=np.float32))
        assert db.get_embedding_revision() == revision
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


def test_delete_paper_removes_paper_node_edges_asserted_by_other_papers():
    """Deleting a paper cannot leave a graph edge pointing at its local ID."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Paper A", 2024, "uploaded")
        db.insert_paper("p2", "Paper B", 2025, "uploaded")
        db.insert_edge("p1", "external-node", "cites", "p2")
        db.insert_edge("external-node", "p1", "cites", "p2")
        db.commit()

        counts = db.delete_paper("p1")

        assert counts["edges"] == 2
        assert (
            db.conn.execute(
                "SELECT COUNT(*) FROM edges WHERE src_id = ? OR dst_id = ?", ("p1", "p1")
            ).fetchone()[0]
            == 0
        )
        assert db.get_paper("p2") is not None
        db.close()


def test_delete_paper_clears_lazy_pipeline_caches():
    """Deleting a paper must not let a re-ingest reuse stale derived rows."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test Paper", 2024, "uploaded")
        ensure_cache_table(db)
        db.conn.execute(
            "INSERT INTO paper_concepts_cache (local_id, concepts_json) VALUES (?, ?)",
            ("p1", '["stale"]'),
        )
        db.conn.execute(
            """
            CREATE TABLE kg_l1_attempted (
                local_id TEXT PRIMARY KEY,
                attempted_at TEXT NOT NULL,
                extractor TEXT NOT NULL,
                concept_count INTEGER NOT NULL
            )
            """
        )
        db.conn.execute(
            "INSERT INTO kg_l1_attempted VALUES (?, ?, ?, ?)",
            ("p1", "now", "heuristic", 0),
        )
        db.commit()

        counts = db.delete_paper("p1")

        assert counts["paper_concepts_cache"] == 1
        assert counts["kg_l1_attempted"] == 1
        assert (
            db.conn.execute(
                "SELECT COUNT(*) FROM paper_concepts_cache WHERE local_id = ?", ("p1",)
            ).fetchone()[0]
            == 0
        )
        assert (
            db.conn.execute(
                "SELECT COUNT(*) FROM kg_l1_attempted WHERE local_id = ?", ("p1",)
            ).fetchone()[0]
            == 0
        )
        db.close()


def test_delete_paper_preserves_audit_references_as_evidence_tombstones():
    """Historical answers/claims keep resolvable, redacted evidence IDs."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Paper A", 2024, "uploaded")
        db.insert_paper("p2", "Paper B", 2025, "uploaded")

        answer_id = db.record_answer(
            "Which paper supports this?",
            "Paper A",
            evidence_ids=["p1:n1", "p2:n2"],
            provenance="retriever",
        )
        # Enrich the sparse answer-time evidence so deletion proves that the
        # source payload is cleared while the audit identity is retained.
        db.record_evidence("p1", "n1", snippet="private source text", value="42")
        db.record_evidence("p2", "n2", snippet="surviving source text")
        claim_id = db.record_claim(
            "support",
            "Paper A",
            evidence_node_ids="p1:n1,p2:n2",
        )
        db.record_claim_evidence(claim_id, ["p1:n1", "p2:n2"])
        db.commit()

        counts = db.delete_paper("p1")

        assert counts["evidence"] == 1
        answer = db.conn.execute(
            "SELECT evidence_ids FROM answer_records WHERE answer_id = ?", (answer_id,)
        ).fetchone()
        assert answer is not None
        assert json.loads(answer[0]) == ["p1:n1", "p2:n2"]

        tombstone = db.conn.execute(
            "SELECT paper_id, node_id, snippet, value, provenance "
            "FROM evidence WHERE evidence_id = ?",
            ("p1:n1",),
        ).fetchone()
        assert tombstone == ("", "", "", "", "PAPER_DELETED:p1")

        # The claim and its FK-backed relation remain auditable, while evidence
        # from an unrelated surviving paper remains untouched.
        assert (
            db.conn.execute(
                "SELECT evidence_node_ids FROM claims WHERE claim_id = ?", (claim_id,)
            ).fetchone()[0]
            == "p1:n1,p2:n2"
        )
        assert (
            db.conn.execute(
                "SELECT COUNT(*) FROM claim_evidence WHERE claim_id = ? AND evidence_id = ?",
                (claim_id, "p1:n1"),
            ).fetchone()[0]
            == 1
        )
        assert (
            db.conn.execute(
                "SELECT snippet FROM evidence WHERE evidence_id = ?", ("p2:n2",)
            ).fetchone()[0]
            == "surviving source text"
        )
        db.close()
