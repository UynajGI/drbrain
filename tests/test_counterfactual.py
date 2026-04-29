"""Tests for counterfactual queries: 'what if X didn't exist?'"""

from drbrain.extractor.counterfactual import (
    CounterfactualImpact,
    find_critical_nodes,
    run_counterfactual,
)
from drbrain.graph.engine import GraphEngine


def _make_graph(edges):
    """Helper: create GraphEngine from (src, dst, relation, paper) tuples."""
    g = GraphEngine()
    for src, dst, rel, paper in edges:
        g.add_edge(src, dst, rel, paper)
    return g


def test_counterfactual_remove_leaf():
    """Removing a leaf node (no closure rules triggered) has no inference loss."""
    g = _make_graph(
        [
            ("A", "B", "cites", "p1"),
        ]
    )
    impact = run_counterfactual(g, "A")
    assert impact.removed_edges == 1  # direct edge removed
    assert impact.lost_inferences == set()  # no rules triggered by 'cites'


def test_counterfactual_remove_hub():
    """Removing a hub node breaks its direct edges."""
    g = _make_graph(
        [
            ("A", "B", "extends", "p1"),
            ("B", "C", "addresses", "p1"),
            ("B", "D", "proposes", "p1"),
        ]
    )
    impact = run_counterfactual(g, "B")
    assert impact.removed_edges == 3  # all 3 edges touch B


def test_counterfactual_challenges_removal():
    """Removing a challenger node eliminates the creates_debate inference."""
    g = _make_graph(
        [
            ("P1", "Conclusion_X", "challenges", "p1"),
            ("P2", "Conclusion_X", "supports", "p2"),
        ]
    )
    # With both: creates_debate exists
    full_closure = g.closure()
    debate_count = sum(1 for e in full_closure if e["relation"] == "creates_debate")
    assert debate_count >= 1

    # Without P1: no debate
    impact = run_counterfactual(g, "P1")
    # The debate inference is lost
    assert "creates_debate" in impact.lost_inferences


def test_counterfactual_empty_graph():
    """Counterfactual on empty graph returns zero impact."""
    g = GraphEngine()
    impact = run_counterfactual(g, "X")
    assert impact.removed_edges == 0
    assert impact.affected_concepts == 0


def test_counterfactual_nonexistent_node():
    """Removing a node that doesn't exist returns zero impact."""
    g = _make_graph(
        [
            ("A", "B", "cites", "p1"),
        ]
    )
    impact = run_counterfactual(g, "Z")
    assert impact.removed_edges == 0
    assert impact.affected_concepts == 0


def test_counterfactual_summary():
    """CounterfactualImpact has a readable summary."""
    impact = CounterfactualImpact(
        removed_node="X",
        removed_edges=3,
        affected_concepts=2,
        lost_inferences={"creates_debate", "gap_addressed"},
        affected_nodes={"A", "B"},
    )
    summary = impact.summary()
    assert "X" in summary
    assert "3" in summary


def test_find_critical_nodes():
    """find_critical_nodes returns nodes sorted by counterfactual impact."""
    g = _make_graph(
        [
            ("A", "B", "extends", "p1"),
            ("B", "C", "addresses", "p1"),
            ("B", "D", "proposes", "p1"),
            ("E", "F", "cites", "p2"),
        ]
    )
    critical = find_critical_nodes(g)
    assert len(critical) > 0
    # B should be more critical than E or F
    b_score = next((c["impact"] for c in critical if c["node"] == "B"), 0)
    e_score = next((c["impact"] for c in critical if c["node"] == "E"), 0)
    assert b_score >= e_score


def test_find_critical_nodes_empty():
    """find_critical_nodes returns empty list for empty graph."""
    g = GraphEngine()
    assert find_critical_nodes(g) == []
