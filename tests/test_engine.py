"""Tests for graph engine closure with section-aware confidence."""

from drbrain.graph.engine import GraphEngine


def _make_graph(edges):
    g = GraphEngine()
    for src, dst, rel, paper in edges:
        g.add_edge(src, dst, rel, paper)
    return g


def test_closure_section_aware_decay():
    """Inferred edges get section-aware confidence when section_map provided."""
    g = _make_graph(
        [
            ("P1", "Conclusion_Z", "supports", "p1"),
            ("P2", "Conclusion_Z", "challenges", "p2"),
        ]
    )
    section_map = {"P1": "Methods", "P2": "Discussion"}
    inferred = g.closure(section_map=section_map)
    # creates_debate should be inferred
    debate_edges = [e for e in inferred if e["relation"] == "creates_debate"]
    assert len(debate_edges) >= 1
    # Each inferred edge should have a confidence field
    for edge in debate_edges:
        assert "confidence" in edge
        assert 0 < edge["confidence"] <= 1.0


def test_closure_backward_compatible():
    """Without section_map, closure works as before (no confidence field)."""
    g = _make_graph(
        [
            ("P1", "Conclusion_Z", "supports", "p1"),
            ("P2", "Conclusion_Z", "challenges", "p2"),
        ]
    )
    inferred = g.closure()
    debate_edges = [e for e in inferred if e["relation"] == "creates_debate"]
    assert len(debate_edges) >= 1
    # Without section_map, no confidence field on inferred edges
    for edge in debate_edges:
        assert "confidence" not in edge
