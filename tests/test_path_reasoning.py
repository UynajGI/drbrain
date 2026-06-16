"""Tests for multi-hop path reasoning rules."""

from drbrain.graph.engine import GraphEngine
from drbrain.graph.path_reasoning import (
    PathRule,
    apply_path_rules,
    get_builtin_rules,
)


def _make_graph(edges):
    """Helper: create GraphEngine from list of (src, dst, relation, paper) tuples."""
    g = GraphEngine()
    for src, dst, rel, paper in edges:
        g.add_edge(src, dst, rel, paper)
    return g


# -- method_supersedes_problem --
# Method_A addresses Problem_X, Method_B replaces Method_A
# => Method_B addresses Problem_X


def test_method_supersedes_problem():
    """Method_B replaces Method_A which addresses Problem_X => Method_B addresses Problem_X."""
    g = _make_graph(
        [
            ("M1", "Problem_X", "addresses", "p1"),
            ("M2", "M1", "replaces", "p2"),
        ]
    )
    inferred = apply_path_rules(g)
    supersedes = [e for e in inferred if e.get("relation") == "supersedes_address"]
    assert len(supersedes) == 1
    assert supersedes[0]["src"] == "M2"
    assert supersedes[0]["dst"] == "Problem_X"


def test_method_supersedes_problem_no_replace():
    """Without replaces, no supersedes inference."""
    g = _make_graph(
        [
            ("M1", "Problem_X", "addresses", "p1"),
            ("M2", "M1", "extends", "p2"),
        ]
    )
    inferred = apply_path_rules(g)
    assert not any(e.get("relation") == "supersedes_address" for e in inferred)


# -- challenge_chain --
# Method_A challenges Conclusion_X, Method_B extends Method_A
# => Method_B challenges Conclusion_X


def test_challenge_chain():
    """Method_B extends Method_A which challenges C => Method_B challenges C."""
    g = _make_graph(
        [
            ("M1", "Conclusion_X", "challenges", "p1"),
            ("M2", "M1", "extends", "p2"),
        ]
    )
    inferred = apply_path_rules(g)
    chain = [e for e in inferred if e.get("relation") == "challenge_chain"]
    assert len(chain) == 1
    assert chain[0]["src"] == "M2"
    assert chain[0]["dst"] == "Conclusion_X"


def test_challenge_chain_no_extends():
    """Without extends, no challenge chain."""
    g = _make_graph(
        [
            ("M1", "Conclusion_X", "challenges", "p1"),
            ("M2", "M1", "replaces", "p2"),
        ]
    )
    inferred = apply_path_rules(g)
    assert not any(e.get("relation") == "challenge_chain" for e in inferred)


# -- gap_inheritance --
# Paper_A leaves_open Gap_X, Paper_B extends Paper_A
# => Gap_X relevant to Paper_B (via gap_inheritance relation)


def test_gap_inheritance():
    """Paper_B extends Paper_A which leaves_open Gap_X => Gap_X relevant to Paper_B."""
    g = _make_graph(
        [
            ("p1", "Gap_X", "leaves_open", "p1"),
            ("p2", "p1", "extends", "p2"),
        ]
    )
    inferred = apply_path_rules(g)
    gap = [e for e in inferred if e.get("relation") == "gap_inheritance"]
    assert len(gap) == 1
    assert gap[0]["src"] == "p2"
    assert gap[0]["dst"] == "Gap_X"


# -- indirect_support --
# Method_A solves Problem_X, Method_B extends Method_A
# => Method_B solves Problem_X


def test_indirect_support():
    """Method_B extends Method_A which solves Problem_X => Method_B solves Problem_X."""
    g = _make_graph(
        [
            ("M1", "Problem_X", "solves", "p1"),
            ("M2", "M1", "extends", "p2"),
        ]
    )
    inferred = apply_path_rules(g)
    support = [e for e in inferred if e.get("relation") == "indirect_support"]
    assert len(support) == 1
    assert support[0]["src"] == "M2"
    assert support[0]["dst"] == "Problem_X"


def test_indirect_support_no_extends():
    """Without extends, no indirect support."""
    g = _make_graph(
        [
            ("M1", "Problem_X", "solves", "p1"),
            ("M2", "M1", "replaces", "p2"),
        ]
    )
    inferred = apply_path_rules(g)
    assert not any(e.get("relation") == "indirect_support" for e in inferred)


# -- empty graph --


def test_apply_path_rules_empty():
    """Empty graph returns no inferences."""
    g = GraphEngine()
    assert apply_path_rules(g) == []


# -- no false positives --


def test_no_false_positives_unrelated_edges():
    """Unrelated edges don't trigger path rules."""
    g = _make_graph(
        [
            ("A", "B", "cites", "p1"),
            ("C", "D", "addresses", "p1"),
        ]
    )
    inferred = apply_path_rules(g)
    assert len(inferred) == 0


# -- builtin rules --


def test_get_builtin_rules():
    """get_builtin_rules returns non-empty list of PathRule."""
    rules = get_builtin_rules()
    assert len(rules) >= 4
    assert all(isinstance(r, PathRule) for r in rules)


# -- closure integration --


def test_closure_includes_path_rules():
    """closure() includes path rule inferences."""
    g = _make_graph(
        [
            ("M1", "Problem_X", "addresses", "p1"),
            ("M2", "M1", "replaces", "p2"),
        ]
    )
    inferred = g.closure()
    assert any(e["relation"] == "supersedes_address" for e in inferred)


# -- duck-typing guard --


def test_apply_path_rules_non_engine_returns_empty():
    """apply_path_rules returns [] for objects without .graph attribute (duck typing)."""
    assert apply_path_rules("not a graph") == []
    assert apply_path_rules(42) == []
    assert apply_path_rules(None) == []


class _FakeGraph:
    """Minimal duck-typed graph: has .graph but is NOT a GraphEngine."""

    def __init__(self):
        import networkx as nx

        self.graph = nx.MultiDiGraph()
