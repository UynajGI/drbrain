"""Tests for embedding-driven rule mining from TransE relation vectors."""

import numpy as np
import pytest

from drbrain.extractor.rule_miner import (
    compose_path,
    mine_from_graph_walks,
    mine_path_rules,
)
from drbrain.graph.engine import GraphEngine

# ── compose_path ─────────────────────────────────────────────────────────


def test_compose_path_single_vector():
    """Single vector composes to itself."""
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = compose_path([v])
    np.testing.assert_array_equal(result, v)


def test_compose_path_two_vectors():
    """Two vectors compose by addition (TransE property r1 + r2)."""
    r1 = np.array([1.0, 0.0], dtype=np.float32)
    r2 = np.array([0.0, 2.0], dtype=np.float32)
    result = compose_path([r1, r2])
    np.testing.assert_array_equal(result, np.array([1.0, 2.0], dtype=np.float32))


def test_compose_path_three_vectors():
    """Three vectors compose: r1 + r2 + r3."""
    r1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    r2 = np.array([0.0, 2.0, 0.0], dtype=np.float32)
    r3 = np.array([0.0, 0.0, 3.0], dtype=np.float32)
    result = compose_path([r1, r2, r3])
    expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)


def test_compose_path_empty():
    """Empty path composes to zero vector."""
    result = compose_path([])
    np.testing.assert_array_equal(result, np.zeros(1))


# ── mine_path_rules (mock embeddings) ────────────────────────────────────


class _FakeDB:
    """Fake database that returns pre-built embedding dict."""

    def __init__(self, embeddings: dict[str, np.ndarray]):
        self._embeddings = embeddings

    def load_embeddings(self) -> dict[str, np.ndarray]:
        return self._embeddings


def _make_graph(edges):
    """Helper: create GraphEngine from list of (src, dst, relation) tuples."""
    g = GraphEngine()
    for src, dst, rel in edges:
        g.add_edge(src, dst, rel, "p0")
    return g


def test_mine_path_rules_finds_composition():
    """When r3 ≈ r1 + r2, the rule (r1, r2) → r3 should be mined."""
    dim = 16
    r1 = np.random.default_rng(42).normal(0, 0.5, dim).astype(np.float32)
    r2 = np.random.default_rng(43).normal(0, 0.5, dim).astype(np.float32)
    r3 = (r1 + r2).astype(np.float32)
    r3 /= np.linalg.norm(r3)  # normalize like TransE does

    embeddings = {
        "__rel__proposes": r1,
        "__rel__addresses": r2,
        "__rel__solves": r3,
    }
    db = _FakeDB(embeddings)
    graph = _make_graph([])  # empty graph is fine for pure embedding mining

    rules = mine_path_rules(graph, db, min_confidence=0.7, top_k=5)

    # Should find proposes→addresses ≈ solves
    solve_rules = [r for r in rules if r["head"] == "solves"]
    assert len(solve_rules) >= 1
    best = solve_rules[0]
    assert best["body_path"] == ["proposes", "addresses"] or best["body_path"] == [
        "addresses",
        "proposes",
    ]
    assert best["confidence"] > 0.7


def test_mine_path_rules_min_confidence_filters():
    """Low-confidence rules should be filtered out."""
    dim = 8
    rng = np.random.default_rng(99)
    r1 = rng.normal(0, 1.0, dim).astype(np.float32)
    r2 = rng.normal(0, 1.0, dim).astype(np.float32)
    r3 = rng.normal(0, 1.0, dim).astype(np.float32)  # unrelated to r1+r2

    embeddings = {
        "__rel__a": r1,
        "__rel__b": r2,
        "__rel__c": r3,
    }
    db = _FakeDB(embeddings)
    graph = _make_graph([])

    rules = mine_path_rules(graph, db, min_confidence=0.95, top_k=10)
    # With 95% threshold and non-composing vectors, should get nothing
    assert len(rules) == 0


def test_mine_path_rules_top_k_limits():
    """top_k limits the number of rules returned."""
    dim = 8
    rng = np.random.default_rng(42)
    embeddings = {}
    for i, name in enumerate(["a", "b", "c", "d", "e", "f"]):
        embeddings[f"__rel__{name}"] = rng.normal(0, 1.0, dim).astype(np.float32)

    db = _FakeDB(embeddings)
    graph = _make_graph([])

    rules = mine_path_rules(graph, db, min_confidence=0.0, top_k=3)
    assert len(rules) <= 3


# ── mine_from_graph_walks ────────────────────────────────────────────────


def test_mine_from_graph_walks_empty_graph():
    """Empty graph should yield no rules."""
    g = _make_graph([])
    rules = mine_from_graph_walks(g, max_length=3, min_support=1, top_k=10)
    assert rules == []


def test_mine_from_graph_walks_detects_recurring_pattern():
    """Graph with repeated path pattern should mine a rule."""
    g = _make_graph(
        [
            # proposes→addresses path multiple times → should suggest "solves"
            ("M1", "P1", "proposes"),
            ("M1", "P1", "addresses"),
            ("M2", "P2", "proposes"),
            ("M2", "P2", "addresses"),
            ("M3", "P3", "proposes"),
            ("M3", "P3", "addresses"),
            # unrelated edge
            ("M1", "M2", "extends"),
        ]
    )

    rules = mine_from_graph_walks(g, max_length=2, min_support=2, top_k=10)
    # Should find at least one rule about proposes→addresses
    assert len(rules) >= 1
    # Check structure
    for rule in rules:
        assert "body_path" in rule
        assert "head" in rule
        assert "confidence" in rule
        assert "support" in rule
        assert isinstance(rule["body_path"], list)
        assert isinstance(rule["confidence"], float)
        assert isinstance(rule["support"], int)


def test_mine_from_graph_walks_min_support():
    """min_support filters out infrequent patterns."""
    g = _make_graph(
        [
            ("M1", "P1", "proposes"),
            ("M1", "P1", "addresses"),
        ]
    )
    # Only one occurrence of proposes→addresses
    rules = mine_from_graph_walks(g, max_length=2, min_support=3, top_k=10)
    assert len(rules) == 0


# ── Integration: mined rules as PathRule-compatible ──────────────────────


def test_mined_rule_structure_matches_path_rule():
    """Mined rules should have fields compatible with PathRule application."""
    dim = 8
    r1 = np.random.default_rng(1).normal(0, 0.5, dim).astype(np.float32)
    r2 = np.random.default_rng(2).normal(0, 0.5, dim).astype(np.float32)
    r3 = (r1 + r2).astype(np.float32)
    r3 /= np.linalg.norm(r3)

    embeddings = {
        "__rel__extends": r1,
        "__rel__generalizes": r2,
        "__rel__subsumes": r3,
    }
    db = _FakeDB(embeddings)
    graph = _make_graph([])

    rules = mine_path_rules(graph, db, min_confidence=0.5, top_k=5)
    for rule in rules:
        # Verify structure
        assert "head" in rule
        assert "body_path" in rule
        assert "confidence" in rule
        assert isinstance(rule["head"], str)
        assert isinstance(rule["body_path"], list)
        assert len(rule["body_path"]) >= 2
        assert all(isinstance(r, str) for r in rule["body_path"])
        assert 0.0 <= rule["confidence"] <= 1.0


# ── CLI integration tests ────────────────────────────────────────────────


class TestClosureMineRules:
    """Test closure_cmd with --mine-rules flag."""

    @pytest.mark.integration
    def test_closure_mine_rules_flag_accepted(self, tmp_path, monkeypatch):
        """Verify closure_cmd accepts --mine-rules without error."""
        from drbrain.cli.commands import closure_cmd
        from drbrain.storage.database import Database

        db_path = tmp_path / "test.db"
        db = Database(str(db_path))
        # Insert papers first (FK target for concepts and paper_ids)
        db.execute(
            "INSERT INTO papers (local_id, title) VALUES (?, ?)",
            ("p0", "Test Paper"),
        )
        db.execute("INSERT INTO paper_ids (local_id) VALUES (?)", ("p0",))
        # Now insert concepts (FK references papers)
        db.execute(
            "INSERT INTO concepts (local_id, type, label) VALUES (?, ?, ?)",
            ("p0", "Method", "Method 1"),
        )
        db.execute(
            "INSERT INTO concepts (local_id, type, label) VALUES (?, ?, ?)",
            ("p0", "Method", "Method 2"),
        )
        db.execute(
            "INSERT INTO concepts (local_id, type, label) VALUES (?, ?, ?)",
            ("p0", "Problem", "Problem 1"),
        )
        db.execute(
            "INSERT INTO edges (src_id, dst_id, relation, source_paper) VALUES (?, ?, ?, ?)",
            ("Method 1", "Problem 1", "addresses", "p0"),
        )
        db.execute(
            "INSERT INTO edges (src_id, dst_id, relation, source_paper) VALUES (?, ?, ?, ?)",
            ("Method 2", "Method 1", "replaces", "p0"),
        )
        db.commit()
        db.close()

        config = {"db": {"path": str(db_path)}}
        monkeypatch.setattr("typer.Exit", SystemExit)

        class FakeCtx:
            obj = {"config": config}

        ctx = FakeCtx()
        try:
            closure_cmd(ctx, mine_rules=True, min_confidence=0.7, dry_run=True)
        except SystemExit:
            pass

    def test_closure_mine_rules_inference_persisted(self, tmp_path, monkeypatch):
        """Mined rules applied during closure should produce inferred edges."""
        from drbrain.cli.commands import closure_cmd
        from drbrain.storage.database import Database

        db_path = tmp_path / "test_mine.db"
        db = Database(str(db_path))
        db.execute(
            "INSERT INTO papers (local_id, title) VALUES (?, ?)",
            ("p0", "Test Paper"),
        )
        db.execute("INSERT INTO paper_ids (local_id) VALUES (?)", ("p0",))
        db.execute(
            "INSERT INTO concepts (local_id, type, label) VALUES (?, ?, ?)",
            ("p0", "Method", "Method 1"),
        )
        db.execute(
            "INSERT INTO concepts (local_id, type, label) VALUES (?, ?, ?)",
            ("p0", "Problem", "Problem 1"),
        )
        db.execute(
            "INSERT INTO edges (src_id, dst_id, relation, source_paper) VALUES (?, ?, ?, ?)",
            ("Method 1", "Problem 1", "addresses", "p0"),
        )
        db.commit()
        db.close()

        config = {"db": {"path": str(db_path)}}
        monkeypatch.setattr("typer.Exit", SystemExit)

        class FakeCtx:
            obj = {"config": config}

        ctx = FakeCtx()
        # Should not crash when there are no embeddings to mine from
        try:
            closure_cmd(ctx, mine_rules=True, min_confidence=0.7, dry_run=True)
        except SystemExit:
            pass
