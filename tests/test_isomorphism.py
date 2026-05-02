"""Tests for cross-domain isomorphism detection."""

import pytest

from drbrain.extractor.isomorphism import (
    IsomorphicMapping,
    _relation_signature,
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


def test_find_isomorphic_patterns_empty_graph():
    """Empty graph returns empty list."""
    g = GraphEngine()
    assert find_isomorphic_patterns(g) == []


def test_find_isomorphic_patterns_single_node():
    """Graph with one node (no edges) returns empty list."""
    g = GraphEngine()
    g.graph.add_node("OnlyNode")
    patterns = find_isomorphic_patterns(g)
    assert patterns == []


def test_find_isomorphic_patterns_unique_signatures():
    """Nodes with unique relation signatures produce no isomorphisms."""
    g = _make_graph(
        [
            ("M1", "P1", "addresses", "p1"),
            ("M2", "P1", "challenges", "p1"),
        ]
    )
    patterns = find_isomorphic_patterns(g)
    # M1 has out:addresses (unique), M2 has out:challenges (unique),
    # P1 has in:addresses + in:challenges (unique). No pairs.
    assert patterns == []


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


# -- Section-aware signature --


def test_relation_signature_without_section():
    """Default signature has no section info."""
    g = _make_graph(
        [
            ("M1", "P1", "addresses", "p1"),
            ("M1", "P1", "supports", "p1"),
        ]
    )
    sig = _relation_signature(g, "P1")
    assert "in:addresses" in sig
    assert "in:supports" in sig
    assert "@" not in str(sig)


def test_relation_signature_with_section():
    """Section-aware signature includes section dimension."""
    g = _make_graph(
        [
            ("M1", "P1", "addresses", "p1"),
            ("M2", "P1", "supports", "p1"),
        ]
    )
    section_map = {"M1": "Methods", "M2": "Results"}
    sig = _relation_signature(g, "P1", section_map=section_map)
    assert "in:addresses@Methods" in sig
    assert "in:supports@Results" in sig


def test_relation_signature_section_unknown():
    """Unknown section in map doesn't add section suffix."""
    g = _make_graph(
        [
            ("M1", "P1", "addresses", "p1"),
        ]
    )
    section_map = {"M1": ""}
    sig = _relation_signature(g, "P1", section_map=section_map)
    assert "in:addresses" in sig
    assert "@" not in str(sig)
