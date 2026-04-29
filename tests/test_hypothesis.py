"""Tests for hypothesis generation from graph patterns."""

from drbrain.extractor.hypothesis import (
    Hypothesis,
    generate_hypotheses,
    score_hypothesis,
)
from drbrain.graph.engine import GraphEngine


def _make_graph(edges):
    g = GraphEngine()
    for src, dst, rel, paper in edges:
        g.add_edge(src, dst, rel, paper)
    return g


def test_generate_hypotheses_from_unaddressed_gap():
    """Unaddressed Gap generates a 'method could address' hypothesis."""
    g = _make_graph(
        [
            ("G1", "Gap_X", "leaves_open", "p1"),
        ]
    )
    hyps = generate_hypotheses(g)
    gap_hyps = [h for h in hyps if "Gap_X" in h.description]
    assert len(gap_hyps) >= 1


def test_generate_hypotheses_from_debate():
    """Debate zone generates a 'resolution needed' hypothesis."""
    g = _make_graph(
        [
            ("P1", "Conclusion_Z", "supports", "p1"),
            ("P2", "Conclusion_Z", "challenges", "p2"),
        ]
    )
    hyps = generate_hypotheses(g)
    debate_hyps = [h for h in hyps if "Conclusion_Z" in h.description]
    assert len(debate_hyps) >= 1


def test_generate_hypotheses_empty():
    """Empty graph returns no hypotheses."""
    g = GraphEngine()
    assert generate_hypotheses(g) == []


def test_score_hypothesis_high():
    """Hypothesis with high base confidence scores high."""
    hyp = Hypothesis(
        description="Method M could address Gap G",
        type="gap_filling",
        base_confidence=0.9,
        evidence=["e1", "e2"],
    )
    score = score_hypothesis(hyp)
    assert score > 0.5


def test_score_hypothesis_low():
    """Hypothesis with low base confidence scores low."""
    hyp = Hypothesis(
        description="Speculative claim",
        type="speculative",
        base_confidence=0.1,
        evidence=[],
    )
    score = score_hypothesis(hyp)
    assert score < 0.5


def test_score_hypothesis_with_evidence():
    """Evidence boosts score."""
    hyp_a = Hypothesis(
        description="Claim A",
        type="gap_filling",
        base_confidence=0.5,
        evidence=["e1", "e2", "e3"],
    )
    hyp_b = Hypothesis(
        description="Claim B",
        type="gap_filling",
        base_confidence=0.5,
        evidence=[],
    )
    assert score_hypothesis(hyp_a) > score_hypothesis(hyp_b)


def test_hypothesis_to_dict():
    """Hypothesis serializes to dict."""
    hyp = Hypothesis(
        description="Test",
        type="gap_filling",
        base_confidence=0.7,
        evidence=["e1"],
    )
    d = hyp.to_dict()
    assert d["description"] == "Test"
    assert d["type"] == "gap_filling"
    assert "score" in d
