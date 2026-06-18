"""Extra coverage for graph/engine_embeddings.py mixin.

Covers invalidate_embeddings, _ensure_embeddings (with and without db),
learn_embeddings with tiny graph, and post-training predict_link /
similar_entities.

Note: GraphEngine.__init__ does not initialize ``_transE``; tests must set
``g._transE = None`` before exercising mixin methods that read it.
"""

from __future__ import annotations

from unittest import mock

import numpy as np

from drbrain.graph.engine import GraphEngine
from drbrain.graph.query_embeddings import RELATION_PREFIX


def _fresh_engine() -> GraphEngine:
    """GraphEngine with _transE explicitly cleared."""
    g = GraphEngine()
    g._transE = None
    return g


def _tiny_graph() -> GraphEngine:
    """3 nodes, 2 edges, cache cleared."""
    g = _fresh_engine()
    g.add_edge("A", "B", "cites", "p1")
    g.add_edge("B", "C", "cites", "p1")
    return g


# -- invalidate_embeddings --


def test_invalidate_clears_cache():
    """After learn_embeddings, invalidate_embeddings sets _transE to None."""
    g = _tiny_graph()
    g.learn_embeddings(dim=4, epochs=10)
    assert g._transE is not None

    g.invalidate_embeddings()
    assert g._transE is None


def test_invalidate_then_predict_returns_empty():
    """predict_link after invalidate returns [] (no embeddings)."""
    g = _tiny_graph()
    g.learn_embeddings(dim=4, epochs=10)
    g.invalidate_embeddings()

    assert g.predict_link("A", "cites") == []
    assert g.similar_entities("A") == []


def test_invalidate_idempotent_on_empty():
    """Invalidating when cache is already None is a no-op."""
    g = _fresh_engine()
    g.invalidate_embeddings()
    assert g._transE is None


# -- _ensure_embeddings --


def test_ensure_embeddings_skips_when_cache_present():
    """_ensure_embeddings returns early when _transE already set."""
    g = _tiny_graph()
    g.learn_embeddings(dim=4, epochs=10)
    existing = g._transE

    db = mock.MagicMock()
    g._ensure_embeddings(db)
    db.load_embeddings.assert_not_called()
    assert g._transE is existing


def test_ensure_embeddings_loads_from_db():
    """_ensure_embeddings populates cache from db.load_embeddings()."""
    g = _fresh_engine()

    vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    db = mock.MagicMock()
    db.load_embeddings.return_value = {
        "A": vec,
        RELATION_PREFIX + "cites": vec,
    }

    g._ensure_embeddings(db)
    assert g._transE is not None
    assert "A" in g._transE.entities
    assert "cites" in g._transE.relations
    db.load_embeddings.assert_called_once()


def test_ensure_embeddings_no_db_no_op():
    """_ensure_embeddings without db and empty cache leaves _transE None."""
    g = _fresh_engine()
    g._ensure_embeddings(None)
    assert g._transE is None


def test_ensure_embeddings_empty_db_result():
    """_ensure_embeddings with empty load_embeddings result leaves cache empty."""
    g = _fresh_engine()
    db = mock.MagicMock()
    db.load_embeddings.return_value = {}

    g._ensure_embeddings(db)
    assert g._transE is None


# -- learn_embeddings tiny graph --


def test_learn_embeddings_tiny_graph():
    """learn_embeddings with dim=4 epochs=10 trains without error."""
    g = _tiny_graph()
    g.learn_embeddings(dim=4, epochs=10, lr=0.1)

    emb = g.entity_embedding("A")
    assert emb is not None
    assert emb.shape == (4,)


def test_learn_embeddings_persists_relations_to_db():
    """learn_embeddings writes both entities and relations to db."""
    g = _tiny_graph()
    db = mock.MagicMock()

    g.learn_embeddings(dim=4, epochs=10, db=db)

    saved_keys = [c.args[0] for c in db.save_embedding.call_args_list]
    assert "A" in saved_keys
    assert any(k.startswith(RELATION_PREFIX) for k in saved_keys)
    db.commit.assert_called_once()


def test_learn_embeddings_warm_start_from_db():
    """learn_embeddings uses db.load_embeddings() as warm-start init."""
    g = _tiny_graph()
    warm_vec = np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float32)
    db = mock.MagicMock()
    db.load_embeddings.return_value = {"A": warm_vec}

    g.learn_embeddings(dim=4, epochs=10, db=db)
    assert g._transE is not None
    assert "A" in g._transE.entities


# -- predict_link / similar_entities --


def test_predict_link_after_training():
    """predict_link returns ranked candidates after training."""
    g = _tiny_graph()
    g.learn_embeddings(dim=4, epochs=10)

    preds = g.predict_link("A", "cites", top_k=2)
    assert isinstance(preds, list)
    assert len(preds) <= 2
    if preds:
        assert isinstance(preds[0], tuple)
        assert isinstance(preds[0][1], float)


def test_similar_entities_after_training():
    """similar_entities returns ranked candidates after training."""
    g = _tiny_graph()
    g.learn_embeddings(dim=4, epochs=10)

    sims = g.similar_entities("A", top_k=2)
    assert isinstance(sims, list)
    assert len(sims) <= 2


def test_predict_link_no_embeddings_returns_empty():
    """predict_link without trained embeddings returns []."""
    g = _fresh_engine()
    assert g.predict_link("A", "cites") == []


def test_similar_entities_no_embeddings_returns_empty():
    """similar_entities without trained embeddings returns []."""
    g = _fresh_engine()
    assert g.similar_entities("A") == []
