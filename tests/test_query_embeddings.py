"""Tests for embedding-based complex query operators (T1)."""

from __future__ import annotations

import numpy as np

from drbrain.graph.query_embeddings import (
    _cosine_sim,
    intersect,
    negate,
    project,
    query_embed,
    union,
)

_ENTITIES = {
    "A": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "B": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "C": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    "D": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    "E": np.array([0.5, 0.5, 0.0], dtype=np.float32),
}

_RELATIONS = {
    "links_to": np.array([-1.0, 1.0, 0.0], dtype=np.float32),
    "opposes": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
}


def _normalize(d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Normalize all vectors in a dict."""
    out = {}
    for k, v in d.items():
        norm = np.linalg.norm(v)
        out[k] = v / norm if norm > 0 else v
    return out


# ── operator unit tests ──────────────────────────────────────────────


def test_cosine_sim_identical():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert abs(_cosine_sim(a, a) - 1.0) < 1e-6


def test_cosine_sim_orthogonal():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(_cosine_sim(a, b)) < 1e-6


def test_cosine_sim_opposite():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([-1.0, 0.0], dtype=np.float32)
    assert abs(_cosine_sim(a, b) + 1.0) < 1e-6


def test_project_nearest():
    """A + links_to ≈ B: (1,0,0) + (-1,1,0) = (0,1,0) = B.

    After normalization, B and E may tie — B must be in top results.
    """
    ents = _normalize(_ENTITIES)
    rels = _normalize(_RELATIONS)
    results = project(ents["A"], rels["links_to"], ents, top_k=3)
    top_ids = {r[0] for r in results}
    assert "B" in top_ids


def test_project_returns_top_k():
    ents = _normalize(_ENTITIES)
    rels = _normalize(_RELATIONS)
    results = project(ents["A"], rels["links_to"], ents, top_k=2)
    assert len(results) == 2


def test_intersect_centroid():
    """Centroid of A(1,0,0) + E(0.5,0.5,0) ≈ (0.75,0.25,0) closest to A."""
    ents = _normalize(_ENTITIES)
    results = intersect([ents["A"], ents["E"]], ents, top_k=3)
    # A and E should be at the top (closest to their own centroid)
    top_ids = {r[0] for r in results}
    assert "A" in top_ids or "E" in top_ids


def test_intersect_single():
    """Single vector intersect = nearest neighbors to that vector."""
    ents = _normalize(_ENTITIES)
    results = intersect([ents["A"]], ents, top_k=3)
    assert results[0][0] == "A"


def test_union_merge_dedup():
    """Union of overlapping sets keeps max score."""
    set_a = [("X", 0.9), ("Y", 0.5), ("Z", 0.3)]
    set_b = [("X", 0.7), ("W", 0.8), ("Z", 0.6)]
    results = union([set_a, set_b], top_k=4)
    ids = {r[0]: r[1] for r in results}
    assert ids["X"] == 0.9  # higher score from set_a
    assert ids["Z"] == 0.6  # higher score from set_b
    assert "Y" in ids
    assert "W" in ids


def test_union_empty():
    results = union([], top_k=10)
    assert results == []


def test_negate_returns_dissimilar():
    """A (1,0,0) negate should return entities furthest from A like D (-1,0,0)."""
    ents = _normalize(_ENTITIES)
    results = negate(ents["A"], ents, top_k=3)
    # D is opposite of A
    top_ids = [r[0] for r in results]
    assert "D" in top_ids


def test_negate_score_range():
    """Negate scores should be between 0 and 1."""
    ents = _normalize(_ENTITIES)
    results = negate(ents["A"], ents, top_k=10)
    for _, score in results:
        assert 0.0 <= score <= 1.0


# ── query_embed DSL tests ────────────────────────────────────────────


class _FakeDB:
    """Minimal db stand-in that provides load_embeddings()."""

    def __init__(self, entities, relations):
        self._data = dict(entities)
        for rel, vec in relations.items():
            self._data[f"__rel__{rel}"] = vec

    def load_embeddings(self):
        return dict(self._data)


def test_query_embed_project():
    """Project query: find entities reachable from A via links_to."""
    ents = _normalize(_ENTITIES)
    rels = _normalize(_RELATIONS)
    db = _FakeDB(ents, rels)

    query = {"type": "project", "entity": "A", "relation": "links_to"}
    results = query_embed(db, query, top_k=3)
    assert len(results) > 0
    labels = {r["label"] for r in results}
    assert "B" in labels  # B is the target of A + links_to in TransE space
    assert "score" in results[0]


def test_query_embed_intersect():
    """Intersect query: centroid of (A, E)."""
    ents = _normalize(_ENTITIES)
    rels = _normalize(_RELATIONS)
    db = _FakeDB(ents, rels)

    query = {"type": "intersect", "entities": ["A", "E"]}
    results = query_embed(db, query, top_k=3)
    assert len(results) > 0
    labels = {r["label"] for r in results}
    assert "A" in labels or "E" in labels


def test_query_embed_union():
    """Union query: merge results from two project sub-queries."""
    ents = _normalize(_ENTITIES)
    rels = _normalize(_RELATIONS)
    db = _FakeDB(ents, rels)

    query = {
        "type": "union",
        "queries": [
            {"type": "project", "entity": "A", "relation": "links_to"},
            {"type": "project", "entity": "B", "relation": "links_to"},
        ],
    }
    results = query_embed(db, query, top_k=10)
    assert len(results) > 0


def test_query_embed_negate():
    """Negate query: entities far from A."""
    ents = _normalize(_ENTITIES)
    rels = _normalize(_RELATIONS)
    db = _FakeDB(ents, rels)

    query = {"type": "negate", "query": {"type": "project", "entity": "A", "relation": "links_to"}}
    results = query_embed(db, query, top_k=3)
    assert len(results) > 0
    # D should appear since it's opposite of A
    labels = {r["label"] for r in results}
    assert "D" in labels


def test_query_embed_nested():
    """Intersect(project(A, r1), project(B, r2))."""
    ents = _normalize(_ENTITIES)
    rels = _normalize(_RELATIONS)
    db = _FakeDB(ents, rels)

    query = {
        "type": "intersect",
        "queries": [
            {"type": "project", "entity": "A", "relation": "links_to"},
            {"type": "project", "entity": "B", "relation": "links_to"},
        ],
    }
    results = query_embed(db, query, top_k=10)
    assert len(results) > 0


def test_query_embed_missing_entity():
    """Graceful handling of missing entity label."""
    db = _FakeDB({}, {})
    query = {"type": "project", "entity": "NoSuchEntity", "relation": "links_to"}
    results = query_embed(db, query, top_k=10)
    assert results == []


def test_query_embed_missing_relation():
    """Graceful handling of missing relation in project."""
    ents = _normalize(_ENTITIES)
    db = _FakeDB(ents, {})
    query = {"type": "project", "entity": "A", "relation": "no_such_rel"}
    results = query_embed(db, query, top_k=10)
    assert results == []


def test_query_embed_unknown_type():
    """Unknown query type returns empty."""
    db = _FakeDB({}, {})
    query = {"type": "bogus"}
    results = query_embed(db, query, top_k=10)
    assert results == []


def test_query_embed_project_result_structure():
    """Verify result dict structure."""
    ents = _normalize(_ENTITIES)
    rels = _normalize(_RELATIONS)
    db = _FakeDB(ents, rels)

    query = {"type": "project", "entity": "A", "relation": "links_to"}
    results = query_embed(db, query, top_k=1)
    assert len(results) == 1
    assert "label" in results[0]
    assert "score" in results[0]
    assert isinstance(results[0]["label"], str)
    assert isinstance(results[0]["score"], float)
