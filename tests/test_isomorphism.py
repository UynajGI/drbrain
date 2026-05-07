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


def test_relation_signature_outgoing_with_section():
    """Outgoing edges include section info from target nodes."""
    g = _make_graph(
        [
            ("M1", "P1", "addresses", "p1"),
            ("M1", "P2", "extends", "p1"),
        ]
    )
    section_map = {"P1": "Introduction", "P2": "Methods"}
    sig = _relation_signature(g, "M1", section_map=section_map)
    assert "out:addresses@Introduction" in sig
    assert "out:extends@Methods" in sig


def test_relation_signature_outgoing_without_section_map():
    """Outgoing edges without section_map have no section suffix."""
    g = _make_graph(
        [
            ("M1", "P1", "addresses", "p1"),
        ]
    )
    sig = _relation_signature(g, "M1")
    assert sig == {"out:addresses": 1}


def test_relation_signature_mixed_in_out():
    """Node with both incoming and outgoing edges gets both in: and out: keys."""
    g = _make_graph(
        [
            ("M1", "Node_X", "addresses", "p1"),
            ("Node_X", "P1", "extends", "p1"),
        ]
    )
    section_map = {"M1": "Methods", "P1": "Results"}
    sig = _relation_signature(g, "Node_X", section_map=section_map)
    assert "in:addresses@Methods" in sig
    assert "out:extends@Results" in sig


# -- Confidence scoring (no longer hardcoded 0.6) --


def test_isomorphism_confidence_not_hardcoded():
    """Different pairs should get different confidence scores."""
    g = _make_graph(
        [
            ("M1", "Problem_X", "addresses", "p1"),
            ("M2", "Problem_X", "addresses", "p1"),
            ("M3", "Problem_Y", "addresses", "p2"),
            ("M4", "Problem_Y", "addresses", "p2"),
            ("M5", "Problem_Z", "supports", "p3"),
        ]
    )
    mappings = find_isomorphic_patterns(g)
    # Problem_X and Problem_Y should form one pair
    # M1, M2, M3, M4 share the same sig -> multiple pairs among them
    assert len(mappings) >= 2
    confidences = [m.confidence for m in mappings]
    # Should not all be the same value
    assert len(set(round(c, 4) for c in confidences)) > 1


def test_isomorphism_high_label_similarity():
    """Concepts with similar labels should get higher confidence."""
    g = GraphEngine()
    g.graph.add_node("graph_neural_network")
    g.graph.add_node("graph_neural_networks")
    g.graph.add_node("Problem_X")
    g.graph.add_node("Problem_Y")
    g.add_edge("graph_neural_network", "Problem_X", "solves", "p1")
    g.add_edge("graph_neural_networks", "Problem_Y", "solves", "p1")

    mappings = find_isomorphic_patterns(g)
    assert len(mappings) >= 1
    # _label_similarity("graph_neural_network", "graph_neural_networks") is high
    assert mappings[0].confidence > 0.5


def test_isomorphism_low_label_similarity():
    """Concepts with very different labels should get lower confidence."""
    g = GraphEngine()
    g.graph.add_node("transformer")
    g.graph.add_node("random_forest")
    g.graph.add_node("Problem_A")
    g.graph.add_node("Problem_B")
    g.add_edge("transformer", "Problem_A", "solves", "p1")
    g.add_edge("random_forest", "Problem_B", "solves", "p1")

    mappings = find_isomorphic_patterns(g)
    assert len(mappings) >= 1
    # transformer vs random_forest should have very low label similarity
    # Jaccard=1.0 (same sig), label_sim ~ 0 → confidence ~ 0.7
    assert mappings[0].confidence < 0.8


# -- CLI command --


def test_isomorphism_cmd_empty_graph():
    """CLI handles empty graph gracefully."""
    import tempfile
    from pathlib import Path

    from drbrain.cli.commands import isomorphism_cmd
    from drbrain.storage.database import Database

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        db = Database(str(db_path))
        db.close()

        cfg = {
            "db": {"path": str(db_path)},
            "dirs": {"papers": td, "reports": td},
            "llm": {"models": []},
        }
        ctx = type("Ctx", (), {"obj": {"config": cfg}})()
        isomorphism_cmd(ctx, concept=None, min_confidence=0.5, json_output=True)

        # Should not raise


def test_isomorphism_cmd_with_data():
    """CLI finds isomorphic patterns with real graph data."""
    import tempfile
    from pathlib import Path

    from drbrain.cli.commands import isomorphism_cmd
    from drbrain.storage.database import Database

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        db = Database(str(db_path))
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Test', 2026, 'extracted')"
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence) "
            "VALUES ('p1', 'Method', 'GNN_v1', 0.9)"
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence) "
            "VALUES ('p1', 'Method', 'GNN_v2', 0.9)"
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence) "
            "VALUES ('p1', 'Method', 'CNN_v1', 0.9)"
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence) "
            "VALUES ('p1', 'Problem', 'Scalability', 0.9)"
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence) "
            "VALUES ('p1', 'Problem', 'Accuracy', 0.9)"
        )
        db.conn.execute(
            "INSERT INTO edges (src_id, dst_id, relation, source_paper) "
            "VALUES ('GNN_v1', 'Scalability', 'solves', 'p1')"
        )
        db.conn.execute(
            "INSERT INTO edges (src_id, dst_id, relation, source_paper) "
            "VALUES ('GNN_v2', 'Scalability', 'solves', 'p1')"
        )
        db.conn.execute(
            "INSERT INTO edges (src_id, dst_id, relation, source_paper) "
            "VALUES ('CNN_v1', 'Accuracy', 'solves', 'p1')"
        )
        db.commit()
        db.close()

        cfg = {
            "db": {"path": str(db_path)},
            "dirs": {"papers": td, "reports": td},
            "llm": {"models": []},
        }
        ctx = type("Ctx", (), {"obj": {"config": cfg}})()
        isomorphism_cmd(ctx, concept=None, min_confidence=0.0, json_output=True)


# ── RAPTOR-enriched isomorphism ──────────────────────────────────────────────


def test_enrich_isomorphisms_with_raptor_context():
    """isomorphism mappings enriched with RAPTOR cross-section summaries."""
    import json
    import tempfile
    from pathlib import Path

    from drbrain.extractor.isomorphism import (
        IsomorphicMapping,
        enrich_isomorphisms_with_raptor,
    )
    from drbrain.storage.database import Database

    mappings = [
        IsomorphicMapping(
            source_domain="GNN_v1",
            target_domain="GNN_v2",
            shared_structure="out:solves: 1",
            confidence=0.95,
        ),
        IsomorphicMapping(
            source_domain="CNN_v1",
            target_domain="GNN_v1",
            shared_structure="out:solves: 1",
            confidence=0.80,
        ),
    ]

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        db = Database(str(db_path))
        # Insert paper and concepts
        db.conn.execute("INSERT INTO papers (local_id, title) VALUES ('p1', 'Graph Methods')")
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence) "
            "VALUES ('p1', 'Method', 'GNN_v1', 0.9)"
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence) "
            "VALUES ('p1', 'Method', 'GNN_v2', 0.9)"
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence) "
            "VALUES ('p1', 'Method', 'CNN_v1', 0.9)"
        )
        # Insert RAPTOR summaries for the paper
        db.conn.execute(
            "INSERT INTO tree_summaries (node_id, paper_id, summary_text, source_node_ids, tree_layer) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                "raptor_p1_L1_abc",
                "p1",
                "Graph neural network architectures for scalable learning.",
                json.dumps(["n1", "n2"]),
                1,
            ),
        )
        db.conn.execute(
            "INSERT INTO tree_summaries (node_id, paper_id, summary_text, source_node_ids, tree_layer) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                "raptor_p1_L1_def",
                "p1",
                "Convolutional methods for image classification tasks.",
                json.dumps(["n3", "n4"]),
                1,
            ),
        )
        db.conn.commit()

        enriched = enrich_isomorphisms_with_raptor(mappings, db)

        assert len(enriched) == 2

        # First mapping: GNN_v1 ↔ GNN_v2 (same paper p1, same GNN domain)
        m0 = enriched[0]
        assert m0.source_domain == "GNN_v1"
        assert isinstance(m0.raptor_source_context, list)
        assert len(m0.raptor_source_context) >= 1
        assert "graph neural network" in m0.raptor_source_context[0]["summary_text"].lower()

        # Second mapping: CNN_v1 ↔ GNN_v1
        m1 = enriched[1]
        assert isinstance(m1.raptor_target_context, list)
        assert len(m1.raptor_target_context) >= 1


def test_enrich_isomorphisms_without_raptor_data():
    """Enrichment returns mappings unchanged when no RAPTOR data exists."""
    import tempfile
    from pathlib import Path

    from drbrain.extractor.isomorphism import (
        IsomorphicMapping,
        enrich_isomorphisms_with_raptor,
    )
    from drbrain.storage.database import Database

    mappings = [
        IsomorphicMapping(
            source_domain="X",
            target_domain="Y",
            shared_structure="out:solves: 1",
            confidence=0.90,
        ),
    ]

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        db = Database(str(db_path))
        # No tree_summaries data

        enriched = enrich_isomorphisms_with_raptor(mappings, db)
        assert len(enriched) == 1
        assert enriched[0].raptor_source_context == []
        assert enriched[0].raptor_target_context == []
        assert enriched[0].confidence == 0.90  # unchanged
