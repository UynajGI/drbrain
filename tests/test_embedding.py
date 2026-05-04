"""Tests for TransE graph embeddings."""
import numpy as np
from drbrain.graph.engine import GraphEngine
from drbrain.graph.embedding import TransE


def test_transe_training_converges():
    g = GraphEngine()
    g.add_edge("A", "B", "cites", "p1")
    g.add_edge("B", "C", "cites", "p1")
    g.add_edge("A", "C", "cites", "p1")

    t = TransE(dim=8, epochs=50, lr=0.1)
    t.train(g.graph)

    e = t.entity_embedding("A")
    assert e is not None
    assert len(e) == 8


def test_transe_predict_link():
    g = GraphEngine()
    g.add_edge("A", "B", "cites", "p1")
    g.add_edge("A", "C", "cites", "p1")

    t = TransE(dim=8, epochs=100, lr=0.1)
    t.train(g.graph)

    results = t.predict_link("A", "cites", top_k=2)
    assert len(results) == 2
    # B or C should be top predictions
    assert results[0][0] in ("B", "C")


def test_transe_similar_entities():
    g = GraphEngine()
    g.add_edge("A", "B", "cites", "p1")
    g.add_edge("A", "C", "cites", "p1")
    # D has identical outgoing edge pattern to A (cites B and C)
    g.add_edge("D", "B", "cites", "p2")
    g.add_edge("D", "C", "cites", "p2")

    t = TransE(dim=8, epochs=200, lr=0.1)
    t.train(g.graph)

    sims = t.similar_entities("A", top_k=2)
    assert len(sims) >= 1
    # A and D share identical citation structure
    assert "D" in [s[0] for s in sims]


def test_transe_empty_graph():
    g = GraphEngine()
    t = TransE(dim=4, epochs=10)
    t.train(g.graph)
    assert t.entity_embedding("A") is None


def test_transe_score():
    g = GraphEngine()
    g.add_edge("A", "B", "cites", "p1")

    t = TransE(dim=8, epochs=100, lr=0.1)
    t.train(g.graph)

    score = t.score("A", "cites", "B")
    assert isinstance(score, float)
    assert score >= 0


def test_transe_incremental_training():
    """Incremental training preserves existing embeddings."""
    g = GraphEngine()
    g.add_edge("A", "B", "cites", "p1")
    g.add_edge("B", "C", "cites", "p1")

    # First training
    t1 = TransE(dim=8, epochs=50, lr=0.1)
    t1.train(g.graph)
    vec_a_before = t1.entity_embedding("A").copy()

    # Add new node
    g.add_edge("C", "D", "cites", "p1")

    # Incremental training
    t2 = TransE(dim=8, epochs=30, lr=0.05)
    init_ents = {e: t1.entity_embedding(e) for e in t1.entities}
    t2.train(g.graph, init_entities=init_ents)

    vec_a_after = t2.entity_embedding("A")
    # A's embedding should not have changed drastically
    diff = np.linalg.norm(vec_a_before - vec_a_after)
    assert diff < 2.0  # should be similar, not totally different

    # New node should have an embedding
    assert t2.entity_embedding("D") is not None
    assert len(t2.entity_embedding("D")) == 8
