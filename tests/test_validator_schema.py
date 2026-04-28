"""Tests for RBox transitive and asymmetric constraint enforcement."""

from drbrain.validator.schema import (
    detect_asymmetric_violations,
    enforce_transitive,
)

# -- enforce_transitive --


def test_enforce_transitive_simple_chain():
    """A extends B, B extends C => infer A extends C."""
    edges = [
        {"src": "A", "dst": "B", "relation": "extends", "source_paper": "p1"},
        {"src": "B", "dst": "C", "relation": "extends", "source_paper": "p2"},
    ]
    inferred = enforce_transitive(edges)
    assert len(inferred) == 1
    assert inferred[0]["src"] == "A"
    assert inferred[0]["dst"] == "C"
    assert inferred[0]["relation"] == "extends"
    assert "via" in inferred[0]


def test_enforce_transitive_long_chain():
    """A extends B, B extends C, C extends D => infer A extends C, A extends D, B extends D."""
    edges = [
        {"src": "A", "dst": "B", "relation": "extends", "source_paper": "p1"},
        {"src": "B", "dst": "C", "relation": "extends", "source_paper": "p1"},
        {"src": "C", "dst": "D", "relation": "extends", "source_paper": "p1"},
    ]
    inferred = enforce_transitive(edges)
    pairs = {(e["src"], e["dst"]) for e in inferred}
    assert ("A", "C") in pairs
    assert ("A", "D") in pairs
    assert ("B", "D") in pairs
    assert len(inferred) == 3


def test_enforce_transitive_skip_existing():
    """Don't infer A extends C if it already exists."""
    edges = [
        {"src": "A", "dst": "B", "relation": "extends", "source_paper": "p1"},
        {"src": "B", "dst": "C", "relation": "extends", "source_paper": "p2"},
        {"src": "A", "dst": "C", "relation": "extends", "source_paper": "p3"},
    ]
    inferred = enforce_transitive(edges)
    assert len(inferred) == 0


def test_enforce_transitive_only_extends():
    """Only extends is transitive, other relations ignored."""
    edges = [
        {"src": "A", "dst": "B", "relation": "cites", "source_paper": "p1"},
        {"src": "B", "dst": "C", "relation": "cites", "source_paper": "p1"},
    ]
    inferred = enforce_transitive(edges)
    assert len(inferred) == 0


def test_enforce_transitive_empty():
    """Empty edge list returns no inferences."""
    assert enforce_transitive([]) == []


def test_enforce_transitive_single_edge():
    """Single edge cannot form a chain."""
    edges = [
        {"src": "A", "dst": "B", "relation": "extends", "source_paper": "p1"},
    ]
    assert enforce_transitive(edges) == []


def test_enforce_transitive_cycle():
    """A extends B, B extends C, C extends A => infer all pairs but no duplicates."""
    edges = [
        {"src": "A", "dst": "B", "relation": "extends", "source_paper": "p1"},
        {"src": "B", "dst": "C", "relation": "extends", "source_paper": "p2"},
        {"src": "C", "dst": "A", "relation": "extends", "source_paper": "p3"},
    ]
    inferred = enforce_transitive(edges)
    pairs = {(e["src"], e["dst"]) for e in inferred}
    assert ("A", "C") in pairs
    assert ("B", "A") in pairs
    assert ("C", "B") in pairs


# -- detect_asymmetric_violations --


def test_detect_asymmetric_violations_clean():
    """No violations when asymmetric relations are one-way."""
    edges = [
        {"src": "A", "dst": "B", "relation": "extends", "source_paper": "p1"},
        {"src": "C", "dst": "D", "relation": "replaces", "source_paper": "p1"},
    ]
    violations = detect_asymmetric_violations(edges)
    assert len(violations) == 0


def test_detect_asymmetric_violations_extends():
    """A extends B and B extends A is a violation."""
    edges = [
        {"src": "A", "dst": "B", "relation": "extends", "source_paper": "p1"},
        {"src": "B", "dst": "A", "relation": "extends", "source_paper": "p2"},
    ]
    violations = detect_asymmetric_violations(edges)
    assert len(violations) == 1
    assert violations[0]["src"] == "B"
    assert violations[0]["dst"] == "A"
    assert violations[0]["relation"] == "extends"


def test_detect_asymmetric_violations_challenges():
    """A challenges B and B challenges A is a violation."""
    edges = [
        {"src": "A", "dst": "B", "relation": "challenges", "source_paper": "p1"},
        {"src": "B", "dst": "A", "relation": "challenges", "source_paper": "p2"},
    ]
    violations = detect_asymmetric_violations(edges)
    assert len(violations) == 1


def test_detect_asymmetric_violations_supports():
    """A supports B and B supports A is a violation."""
    edges = [
        {"src": "A", "dst": "B", "relation": "supports", "source_paper": "p1"},
        {"src": "B", "dst": "A", "relation": "supports", "source_paper": "p2"},
    ]
    violations = detect_asymmetric_violations(edges)
    assert len(violations) == 1


def test_detect_asymmetric_non_asymmetric_relation():
    """cites is not asymmetric, so A cites B and B cites A is not a violation."""
    edges = [
        {"src": "A", "dst": "B", "relation": "cites", "source_paper": "p1"},
        {"src": "B", "dst": "A", "relation": "cites", "source_paper": "p2"},
    ]
    violations = detect_asymmetric_violations(edges)
    assert len(violations) == 0


def test_detect_asymmetric_empty():
    """Empty edge list returns no violations."""
    assert detect_asymmetric_violations([]) == []
