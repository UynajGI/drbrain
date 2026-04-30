"""Tests for hypothesis generation from graph patterns."""

from drbrain.extractor.hypothesis import (
    Hypothesis,
    detect_section_contradictions,
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


# -- Section-aware hypothesis generation --


def test_generate_hypotheses_with_section_map():
    """Section map adds provenance to evidence strings."""
    g = _make_graph(
        [
            ("G1", "Gap_X", "leaves_open", "p1"),
            ("M1", "Other_Gap", "addresses", "p2"),
        ]
    )
    section_map = {"M1": "Methods"}
    hyps = generate_hypotheses(g, section_map=section_map)
    gap_hyps = [h for h in hyps if "Gap_X" in h.description]
    assert len(gap_hyps) >= 1
    # Evidence should mention Methods section
    evidence_text = " ".join(gap_hyps[0].evidence)
    assert "Methods" in evidence_text


def test_generate_hypotheses_debate_with_sections():
    """Debate hypotheses include section info when available."""
    g = _make_graph(
        [
            ("P1", "Conclusion_Z", "supports", "p1"),
            ("P2", "Conclusion_Z", "challenges", "p2"),
        ]
    )
    section_map = {"P1": "Results", "P2": "Discussion"}
    hyps = generate_hypotheses(g, section_map=section_map)
    debate_hyps = [h for h in hyps if "Conclusion_Z" in h.description]
    assert len(debate_hyps) >= 1
    evidence_text = " ".join(debate_hyps[0].evidence)
    assert "Results" in evidence_text or "Discussion" in evidence_text


def test_generate_hypotheses_without_section_map():
    """Without section_map, evidence strings have no section info."""
    g = _make_graph(
        [
            ("G1", "Gap_X", "leaves_open", "p1"),
            ("M1", "Other_Gap", "addresses", "p2"),
        ]
    )
    hyps = generate_hypotheses(g)
    gap_hyps = [h for h in hyps if "Gap_X" in h.description]
    assert len(gap_hyps) >= 1
    evidence_text = " ".join(gap_hyps[0].evidence)
    assert "section" not in evidence_text


# -- Section contradiction detection --


def test_detect_contradictions_found():
    """Supports and challenges from different sections are detected."""
    g = _make_graph(
        [
            ("P1", "Conclusion_Z", "supports", "p1"),
            ("P2", "Conclusion_Z", "challenges", "p2"),
        ]
    )
    section_map = {"P1": "Results", "P2": "Discussion"}
    contradictions = detect_section_contradictions(g, section_map)
    assert len(contradictions) >= 1
    assert contradictions[0]["conclusion"] == "Conclusion_Z"
    assert "Results" in contradictions[0]["supporting_sections"]
    assert "Discussion" in contradictions[0]["challenging_sections"]


def test_detect_contradictions_same_section_ignored():
    """Supports and challenges from the same section are not reported."""
    g = _make_graph(
        [
            ("P1", "Conclusion_Z", "supports", "p1"),
            ("P2", "Conclusion_Z", "challenges", "p2"),
        ]
    )
    section_map = {"P1": "Discussion", "P2": "Discussion"}
    contradictions = detect_section_contradictions(g, section_map)
    assert len(contradictions) == 0


def test_detect_contradictions_empty():
    """No contradictions returns empty list."""
    g = _make_graph(
        [
            ("P1", "Conclusion_Z", "supports", "p1"),
        ]
    )
    contradictions = detect_section_contradictions(g, {"P1": "Results"})
    assert contradictions == []
