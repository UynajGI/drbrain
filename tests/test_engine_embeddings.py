"""Tests for GraphEngine embedding methods (learn_embeddings, entity_embedding,
predict_link, similar_entities) and closure hybrid mode with persistent embeddings."""

from __future__ import annotations

import numpy as np

from drbrain.graph.engine import GraphEngine
from drbrain.graph.query_embeddings import RELATION_PREFIX


class TestLearnEmbeddings:
    """GraphEngine.learn_embeddings() tests."""

    def test_basic(self):
        """learn_embeddings trains TransE and caches it."""
        g = GraphEngine()
        g.add_edge("A", "B", "cites", "p1")
        g.add_edge("B", "C", "cites", "p1")
        g.add_edge("A", "C", "cites", "p1")

        g.learn_embeddings(dim=8, epochs=50, lr=0.1)

        emb = g.entity_embedding("A")
        assert emb is not None
        assert isinstance(emb, np.ndarray)
        assert len(emb) == 8

    def test_persists_to_db(self, tmp_db):
        """learn_embeddings with db stores vectors in embeddings table."""
        g = GraphEngine()
        g.add_edge("A", "B", "cites", "p1")
        g.add_edge("B", "C", "cites", "p1")

        g.learn_embeddings(dim=8, epochs=50, lr=0.1, db=tmp_db)

        # Check DB has embeddings
        rows = tmp_db.conn.execute("SELECT entity, dim FROM embeddings").fetchall()
        assert len(rows) > 0
        # All entities (A, B, C) should be stored
        entity_names = {r[0] for r in rows if not r[0].startswith(RELATION_PREFIX)}
        assert "A" in entity_names
        assert "B" in entity_names
        assert "C" in entity_names
        # Relations should be stored with prefix
        rel_names = {r[0] for r in rows if r[0].startswith(RELATION_PREFIX)}
        assert len(rel_names) >= 1

    def test_empty_graph(self):
        """learn_embeddings on empty graph does not crash."""
        g = GraphEngine()
        g.learn_embeddings(dim=4, epochs=10)

        assert g.entity_embedding("A") is None

    def test_incremental_preserves_existing(self):
        """Re-training with existing embeddings preserves similarity."""
        g = GraphEngine()
        g.add_edge("A", "B", "cites", "p1")
        g.add_edge("B", "C", "cites", "p1")

        g.learn_embeddings(dim=8, epochs=50, lr=0.1)
        vec_a_before = g.entity_embedding("A").copy()

        # Add new node and retrain
        g.add_edge("C", "D", "cites", "p1")
        g.learn_embeddings(dim=8, epochs=30, lr=0.05)

        vec_a_after = g.entity_embedding("A")
        assert vec_a_after is not None

        # A's embedding should not have changed drastically
        diff = np.linalg.norm(vec_a_before - vec_a_after)
        assert diff < 2.0

        # New node should have embedding
        assert g.entity_embedding("D") is not None
        assert len(g.entity_embedding("D")) == 8


class TestEntityEmbedding:
    """GraphEngine.entity_embedding() tests."""

    def test_from_cache(self):
        """entity_embedding returns vector after learn_embeddings."""
        g = GraphEngine()
        g.add_edge("A", "B", "cites", "p1")

        g.learn_embeddings(dim=8, epochs=50, lr=0.1)

        emb = g.entity_embedding("A")
        assert emb is not None
        assert isinstance(emb, np.ndarray)
        assert emb.dtype == np.float32

    def test_from_db(self, tmp_db):
        """entity_embedding loads from DB when cache is empty."""
        # First, train and persist
        g1 = GraphEngine()
        g1.add_edge("A", "B", "cites", "p1")
        g1.learn_embeddings(dim=8, epochs=50, lr=0.1, db=tmp_db)

        # Second engine with no training - should load from DB
        g2 = GraphEngine()
        g2.add_edge("A", "B", "cites", "p1")

        emb = g2.entity_embedding("A", db=tmp_db)
        assert emb is not None
        assert isinstance(emb, np.ndarray)
        assert len(emb) == 8

    def test_unknown_entity(self):
        """entity_embedding returns None for unknown label."""
        g = GraphEngine()
        g.add_edge("A", "B", "cites", "p1")
        g.learn_embeddings(dim=8, epochs=50, lr=0.1)

        assert g.entity_embedding("Z") is None

    def test_unknown_entity_no_db(self):
        """entity_embedding returns None for unknown label without DB."""
        g = GraphEngine()
        assert g.entity_embedding("Z") is None


class TestPredictLink:
    """GraphEngine.predict_link() tests."""

    def test_basic(self):
        """predict_link returns tail entities ranked by TransE score."""
        g = GraphEngine()
        g.add_edge("A", "B", "cites", "p1")
        g.add_edge("A", "C", "cites", "p1")

        g.learn_embeddings(dim=8, epochs=100, lr=0.1)

        results = g.predict_link("A", "cites", top_k=2)
        assert len(results) == 2
        # B or C should be top predictions
        assert results[0][0] in ("B", "C")

    def test_loads_from_db(self, tmp_db):
        """predict_link works when embeddings loaded from DB."""
        g1 = GraphEngine()
        g1.add_edge("A", "B", "cites", "p1")
        g1.add_edge("A", "C", "cites", "p1")
        g1.learn_embeddings(dim=8, epochs=100, lr=0.1, db=tmp_db)

        g2 = GraphEngine()
        g2.add_edge("A", "B", "cites", "p1")
        g2.add_edge("A", "C", "cites", "p1")

        results = g2.predict_link("A", "cites", top_k=2, db=tmp_db)
        assert len(results) == 2
        assert results[0][0] in ("B", "C")

    def test_no_embeddings(self):
        """predict_link without embeddings returns empty list."""
        g = GraphEngine()
        g.add_edge("A", "B", "cites", "p1")

        results = g.predict_link("A", "cites", top_k=5)
        assert results == []

    def test_unknown_head(self):
        """predict_link with unknown head returns empty list."""
        g = GraphEngine()
        g.add_edge("A", "B", "cites", "p1")
        g.learn_embeddings(dim=8, epochs=50, lr=0.1)

        results = g.predict_link("Z", "cites", top_k=5)
        assert results == []

    def test_unknown_relation(self):
        """predict_link with unknown relation returns empty list."""
        g = GraphEngine()
        g.add_edge("A", "B", "cites", "p1")
        g.learn_embeddings(dim=8, epochs=50, lr=0.1)

        results = g.predict_link("A", "unknown_rel", top_k=5)
        assert results == []


class TestSimilarEntities:
    """GraphEngine.similar_entities() tests."""

    def test_basic(self):
        """similar_entities finds entities with similar structure."""
        g = GraphEngine()
        g.add_edge("A", "B", "cites", "p1")
        g.add_edge("A", "C", "cites", "p1")
        # D has identical outgoing edge pattern to A (cites B and C)
        g.add_edge("D", "B", "cites", "p2")
        g.add_edge("D", "C", "cites", "p2")

        g.learn_embeddings(dim=8, epochs=200, lr=0.1)

        sims = g.similar_entities("A", top_k=2)
        assert len(sims) >= 1
        # A and D share identical citation structure
        assert "D" in [s[0] for s in sims]

    def test_loads_from_db(self, tmp_db):
        """similar_entities works when embeddings loaded from DB."""
        g1 = GraphEngine()
        g1.add_edge("A", "B", "cites", "p1")
        g1.add_edge("A", "C", "cites", "p1")
        g1.add_edge("D", "B", "cites", "p2")
        g1.add_edge("D", "C", "cites", "p2")
        g1.learn_embeddings(dim=8, epochs=200, lr=0.1, db=tmp_db)

        g2 = GraphEngine()
        g2.add_edge("A", "B", "cites", "p1")
        g2.add_edge("A", "C", "cites", "p1")
        g2.add_edge("D", "B", "cites", "p2")
        g2.add_edge("D", "C", "cites", "p2")

        sims = g2.similar_entities("A", top_k=2, db=tmp_db)
        assert len(sims) >= 1
        assert "D" in [s[0] for s in sims]

    def test_no_embeddings(self):
        """similar_entities without embeddings returns empty list."""
        g = GraphEngine()
        g.add_edge("A", "B", "cites", "p1")

        sims = g.similar_entities("A", top_k=5)
        assert sims == []

    def test_unknown_entity(self):
        """similar_entities with unknown label returns empty list."""
        g = GraphEngine()
        g.add_edge("A", "B", "cites", "p1")
        g.learn_embeddings(dim=8, epochs=50, lr=0.1)

        sims = g.similar_entities("Z", top_k=5)
        assert sims == []


class TestClosureHybridRefactor:
    """closure(mode='hybrid') uses persistent embeddings instead of retraining."""

    def test_hybrid_uses_cached_embeddings(self):
        """closure hybrid mode reuses cached TransE when available."""
        g = GraphEngine()
        g.add_edge("M1", "P1", "addresses", "p1")
        g.add_edge("P1", "G1", "leaves_open", "p1")
        g.add_edge("M2", "G1", "addresses", "p1")

        # Train embeddings first
        g.learn_embeddings(dim=8, epochs=50, lr=0.1)

        # Run closure in hybrid mode - should use cached embeddings
        inferred = g.closure(mode="hybrid")

        # Should still produce inferences
        assert len(inferred) > 0

        # Edges should have embedding_score from hybrid mode
        for edge in inferred:
            assert "embedding_score" in edge
            assert "confidence" in edge

    def test_hybrid_falls_back_to_inline_training(self):
        """closure hybrid mode falls back to inline training if no cache."""
        g = GraphEngine()
        g.add_edge("M1", "P1", "addresses", "p1")
        g.add_edge("P1", "G1", "leaves_open", "p1")
        g.add_edge("M2", "G1", "addresses", "p1")

        # No learn_embeddings call - hybrid mode should train inline
        inferred = g.closure(mode="hybrid")

        assert len(inferred) > 0
        for edge in inferred:
            assert "embedding_score" in edge

    def test_symbolic_mode_no_embedding_score(self):
        """closure symbolic mode does not add embedding_score."""
        g = GraphEngine()
        g.add_edge("M1", "P1", "addresses", "p1")
        g.add_edge("P1", "G1", "leaves_open", "p1")
        g.add_edge("M2", "G1", "addresses", "p1")

        inferred = g.closure(mode="symbolic")

        assert len(inferred) > 0
        for edge in inferred:
            assert "embedding_score" not in edge
