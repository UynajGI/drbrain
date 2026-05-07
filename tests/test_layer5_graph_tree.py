"""Tests for Layer 5: Graph engine tree integration."""

import tempfile
from pathlib import Path

from drbrain.graph.engine import GraphEngine
from drbrain.storage.database import Database


def _setup_graph_with_sections(db: Database):
    """Create papers, concepts with node_ids, and edges for testing."""
    db.conn.execute(
        "INSERT INTO papers (local_id, title) VALUES (?, ?)",
        ("paper-a", "Paper A"),
    )
    db.conn.execute(
        "INSERT INTO papers (local_id, title) VALUES (?, ?)",
        ("paper-b", "Paper B"),
    )

    # Concepts with node_ids linking to tree sections
    concepts = [
        ("paper-a", "Method", "Self-Attention", 1.0, 2020, "Methods", "node-methods"),
        ("paper-a", "Method", "Transformer", 1.0, 2020, "Methods", "node-methods"),
        ("paper-a", "Problem", "Long Sequence", 0.9, 2020, "Introduction", "node-intro"),
        ("paper-b", "Method", "SparseAttention", 1.0, 2021, "Proposed Method", "node-pm"),
        ("paper-b", "Problem", "Quadratic Cost", 1.0, 2021, "Introduction", "node-intro-b"),
    ]
    for c in concepts:
        db.insert_concept(*c)

    # Edges
    db.insert_edge("1", "3", "addresses", "paper-a")  # Self-Attention → Long Sequence
    db.insert_edge("4", "5", "addresses", "paper-b")  # SparseAttention → Quadratic Cost
    db.insert_edge("4", "1", "extends", "paper-b")  # SparseAttention → Self-Attention

    db.commit()

    engine = GraphEngine()
    engine.load_from_db(db)
    return engine


# ── Concept-to-node lookup ──────────────────────────────────────────────────


def test_get_concepts_by_node():
    """get_concepts_by_node returns concepts linked to a specific tree node."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        engine = _setup_graph_with_sections(db)

        # node-methods has 2 concepts (Self-Attention, Transformer)
        results = engine.get_concepts_by_node(db.conn, "node-methods")
        assert len(results) == 2
        labels = {r["label"] for r in results}
        assert "Self-Attention" in labels
        assert "Transformer" in labels

        # node-intro has 1 concept
        results = engine.get_concepts_by_node(db.conn, "node-intro")
        assert len(results) == 1
        assert results[0]["label"] == "Long Sequence"

        # Unknown node → empty
        results = engine.get_concepts_by_node(db.conn, "nonexistent")
        assert results == []


def test_get_section_context():
    """get_section_context returns tree context for a concept."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        engine = _setup_graph_with_sections(db)

        ctx = engine.get_section_context(db.conn, "Self-Attention")
        assert ctx is not None
        assert ctx["node_id"] == "node-methods"
        assert ctx["section"] == "Methods"
        assert ctx["paper_id"] == "paper-a"

        # Unknown concept → None
        ctx = engine.get_section_context(db.conn, "Nonexistent")
        assert ctx is None


def test_get_section_contexts_batch():
    """get_section_contexts_batch returns contexts for multiple concepts."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        engine = _setup_graph_with_sections(db)

        contexts = engine.get_section_contexts_batch(
            db.conn, ["Self-Attention", "SparseAttention", "Nonexistent"]
        )
        assert len(contexts) == 2  # only 2 found
        nodes = {c["node_id"] for c in contexts.values()}
        assert "node-methods" in nodes
        assert "node-pm" in nodes


# ── Traversal with sections ─────────────────────────────────────────────────


def test_traverse_with_sections():
    """traverse results can be enriched with section provenance."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        engine = _setup_graph_with_sections(db)

        # Traverse from Self-Attention
        steps = engine.traverse_with_sections(db.conn, "Self-Attention", max_hops=2)
        assert len(steps) >= 1

        # Each step has source/destination labels
        for step in steps:
            assert "src" in step
            assert "dst" in step
            if step.get("relation"):
                assert "src" in step  # label
                assert "dst" in step  # label
                if step.get("src_section"):  # present when node has tree context
                    assert "src_node_id" in step


# ── Closure with section provenance ─────────────────────────────────────────


def test_closure_with_sections():
    """closure enriches inferred edges with section provenance."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        engine = _setup_graph_with_sections(db)

        # Run closure with section awareness
        new_edges, meta = engine.closure_with_sections(db.conn)

        assert isinstance(new_edges, list)
        assert isinstance(meta, dict)

        # Each inferred edge should carry section info if available
        for edge in new_edges:
            assert "src_id" in edge
            assert "dst_id" in edge
            assert "relation" in edge
            assert "source_paper" in edge
            # Section info present when sources have node_ids
            if "src_section" in edge:
                assert isinstance(edge["src_section"], str)
            if "dst_section" in edge:
                assert isinstance(edge["dst_section"], str)
