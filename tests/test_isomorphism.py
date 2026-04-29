"""Tests for cross-domain isomorphism detection."""

import pytest

from drbrain.extractor.isomorphism import (
    IsomorphicMapping,
    find_isomorphic_patterns,
    find_similar_problems,
)
from drbrain.graph.engine import GraphEngine


def _make_graph(edges):
    """Helper: create GraphEngine from (src, dst, relation, paper) tuples."""
    g = GraphEngine()
    for src, dst, rel, paper in edges:
        g.add_edge(src, dst, rel, paper)
    return g


def test_find_similar_problems_shared_structure():
    """Two Problems with similar incoming relation patterns are similar."""
    g = _make_graph(
        [
            ("M1", "P1", "addresses", "p1"),
            ("M2", "P1", "addresses", "p1"),
            ("M3", "P2", "addresses", "p2"),
            ("M4", "P2", "addresses", "p2"),
        ]
    )
    similar = find_similar_problems(g, "P1")
    # P2 shares the same pattern (2 addressing methods)
    assert any("P2" in s for s in similar)


def test_find_similar_problems_no_match():
    """Problem with unique pattern returns empty."""
    g = _make_graph(
        [
            ("M1", "P1", "addresses", "p1"),
            ("M2", "P2", "challenges", "p1"),
        ]
    )
    similar = find_similar_problems(g, "P1")
    assert len(similar) == 0


def test_find_isomorphic_patterns():
    """find_isomorphic_patterns returns mappings for structurally similar subgraphs."""
    g = _make_graph(
        [
            ("A1", "Problem_X", "addresses", "p1"),
            ("A2", "Problem_X", "addresses", "p1"),
            ("B1", "Problem_Y", "addresses", "p2"),
            ("B2", "Problem_Y", "addresses", "p2"),
        ]
    )
    patterns = find_isomorphic_patterns(g)
    assert len(patterns) >= 1
    mapping = patterns[0]
    assert isinstance(mapping, IsomorphicMapping)


def test_find_isomorphic_patterns_empty():
    """Empty graph returns no patterns."""
    g = GraphEngine()
    assert find_isomorphic_patterns(g) == []


def test_isomorphic_mapping_fields():
    """IsomorphicMapping stores source, target, and shared structure."""
    m = IsomorphicMapping(
        source_domain="Domain_A",
        target_domain="Domain_B",
        shared_structure="2 methods address 1 problem",
        confidence=0.7,
    )
    assert m.source_domain == "Domain_A"
    assert m.confidence == pytest.approx(0.7)


def test_find_similar_problems_different_patterns():
    """Problems with different relation counts are not similar."""
    g = _make_graph(
        [
            ("M1", "P1", "addresses", "p1"),
            ("M2", "P1", "addresses", "p1"),
            ("M3", "P1", "addresses", "p1"),
            ("M4", "P2", "addresses", "p2"),
        ]
    )
    similar = find_similar_problems(g, "P1")
    # P2 has only 1 addressing method vs P1's 3
    assert all("P2" not in s for s in similar)
