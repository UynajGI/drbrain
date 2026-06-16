"""Unit tests for ClosureMixin.ground_rules() — transitive rule grounding.

Previously only covered by a single integration test. These unit tests
isolate the logic using in-memory GraphEngine instances.
"""

from drbrain.graph.engine import GraphEngine


class TestGroundRules:
    """ground_rules() finds transitive paths and grounds them as triples."""

    def test_transitive_extends(self):
        """A→B (extends), B→C (extends) ⇒ A→C (extends) with min confidence."""
        g = GraphEngine()
        g.add_edge("A", "B", "extends", "p1", weight=0.9)
        g.add_edge("B", "C", "extends", "p2", weight=0.8)

        grounded = g.ground_rules()

        assert len(grounded) == 1
        triple = grounded[0]
        assert triple["src"] == "A"
        assert triple["dst"] == "C"
        assert triple["relation"] == "extends"
        assert triple["confidence"] == 0.8  # min(0.9, 0.8)
        assert triple["via"] == ["B"]
        assert triple["source"] == "rule_grounding"

    def test_min_confidence_filter(self):
        """Triples below min_confidence are excluded."""
        g = GraphEngine()
        g.add_edge("A", "B", "extends", "p1", weight=0.3)
        g.add_edge("B", "C", "extends", "p2", weight=0.4)

        grounded = g.ground_rules(min_confidence=0.5)
        assert len(grounded) == 0  # min(0.3, 0.4) = 0.3 < 0.5

        grounded_low = g.ground_rules(min_confidence=0.2)
        assert len(grounded_low) == 1

    def test_empty_graph(self):
        """Empty graph returns no grounded triples."""
        g = GraphEngine()
        assert g.ground_rules() == []

    def test_non_transitive_relation_ignored(self):
        """'cites' is not in transitive_relations, so no grounding."""
        g = GraphEngine()
        g.add_edge("A", "B", "cites", "p1", weight=0.9)
        g.add_edge("B", "C", "cites", "p2", weight=0.9)

        assert g.ground_rules() == []

    def test_mixed_relations_only_matches_same_type(self):
        """A→B extends, B→C contains ⇒ no grounding (different relation types)."""
        g = GraphEngine()
        g.add_edge("A", "B", "extends", "p1", weight=0.9)
        g.add_edge("B", "C", "contains", "p2", weight=0.9)

        assert g.ground_rules() == []

    def test_dedup_same_triple(self):
        """Multiple paths to same (src, dst, rel) produce one grounded triple."""
        g = GraphEngine()
        g.add_edge("A", "B", "extends", "p1", weight=0.9)
        g.add_edge("B", "C", "extends", "p2", weight=0.8)
        # Second path A→D→C
        g.add_edge("A", "D", "extends", "p1", weight=0.7)
        g.add_edge("D", "C", "extends", "p3", weight=0.6)

        grounded = g.ground_rules()
        # A→C appears twice (via B and via D) but should be deduped
        ac_triples = [t for t in grounded if t["src"] == "A" and t["dst"] == "C"]
        assert len(ac_triples) == 1
        # Confidence should be the max of the two paths' mins: max(0.8, 0.6) = 0.8
        assert ac_triples[0]["confidence"] == 0.8

    def test_all_transitive_relations(self):
        """extends, contains, proposes, addresses are all grounded."""
        g = GraphEngine()
        for rel in ["extends", "contains", "proposes", "addresses"]:
            g.add_edge(f"A_{rel}", f"B_{rel}", rel, "p1", weight=0.9)
            g.add_edge(f"B_{rel}", f"C_{rel}", rel, "p2", weight=0.9)

        grounded = g.ground_rules()
        relations_found = {t["relation"] for t in grounded}
        assert relations_found == {"extends", "contains", "proposes", "addresses"}
