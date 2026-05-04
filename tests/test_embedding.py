"""Tests for TransE graph embeddings."""
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
