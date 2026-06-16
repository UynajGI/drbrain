"""Embedding-based complex query operators (T1).

Supports TransE-style vector operations:
  - project:  h + r ≈ t  (which tail entities are reachable?)
  - intersect: ∧ (centroid of entity vectors)
  - union:     ∨ (merge nearest-neighbor sets)
  - negate:    ¬ (entities far from the negated concept)
"""

from __future__ import annotations

import numpy as np

RELATION_PREFIX = "__rel__"


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def _load_embeddings_for_query(
    db, cached: tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None = None
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load embeddings from DB and separate into entities and relations.

    If *cached* is provided (non-None), returns it directly without hitting
    the DB.  This allows callers to reuse an already-loaded embedding cache.
    """
    if cached is not None:
        return cached
    raw = db.load_embeddings()
    entities: dict[str, np.ndarray] = {}
    relations: dict[str, np.ndarray] = {}
    for key, vec in raw.items():
        if key.startswith(RELATION_PREFIX):
            relations[key[len(RELATION_PREFIX) :]] = vec
        else:
            entities[key] = vec
    return entities, relations


# ── core operators ────────────────────────────────────────────────────


def project(
    entity_vec: np.ndarray,
    relation_vec: np.ndarray,
    entity_vectors: dict[str, np.ndarray],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Entities reachable via relation: e + r ≈ t (TransE scoring).

    Args:
        entity_vec: Source entity vector (head).
        relation_vec: Relation vector.
        entity_vectors: All candidate entity vectors.
        top_k: Number of results to return.

    Returns:
        Sorted list of (label, cosine_similarity) pairs.
    """
    target = entity_vec + relation_vec
    scores = [(eid, _cosine_sim(vec, target)) for eid, vec in entity_vectors.items()]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def intersect(
    vecs: list[np.ndarray],
    entity_vectors: dict[str, np.ndarray],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Centroid of entity vectors (∧): find entities closest to the mean.

    Args:
        vecs: List of entity vectors to intersect.
        entity_vectors: All candidate entity vectors.
        top_k: Number of results to return.

    Returns:
        Sorted list of (label, cosine_similarity) pairs.
    """
    if not vecs:
        return []
    centroid = np.mean(vecs, axis=0)
    scores = [(eid, _cosine_sim(vec, centroid)) for eid, vec in entity_vectors.items()]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def union(
    candidate_sets: list[list[tuple[str, float]]],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Merge nearest-neighbor sets from all branches (∨).

    For entities appearing in multiple sets, keeps the highest score.

    Args:
        candidate_sets: List of scored result lists to merge.
        top_k: Number of results to return.

    Returns:
        Sorted list of (label, max_score) pairs.
    """
    combined: dict[str, float] = {}
    for candidates in candidate_sets:
        for eid, score in candidates:
            if eid not in combined or score > combined[eid]:
                combined[eid] = score
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]


def negate(
    entity_vec: np.ndarray,
    entity_vectors: dict[str, np.ndarray],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Entities most dissimilar to the given vector (¬).

    Returns entities with the lowest cosine similarity to the input,
    with scores transformed to 1.0 - similarity for intuitive ranking.

    Args:
        entity_vec: The vector to negate.
        entity_vectors: All candidate entity vectors.
        top_k: Number of results to return.

    Returns:
        Sorted list of (label, dissimilarity_score) pairs.
    """
    scored = [(_cosine_sim(vec, entity_vec), eid) for eid, vec in entity_vectors.items()]
    scored.sort(key=lambda x: x[0])  # ascending = most dissimilar first
    return [(eid, (1.0 - sim) / 2.0) for sim, eid in scored[:top_k]]


# ── DSL query engine ──────────────────────────────────────────────────


def query_embed(
    db,
    query: dict,
    top_k: int = 10,
    _cached_embeddings: tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None = None,
) -> list[dict]:
    """Execute embedding-based complex query over TransE embeddings.

    Query DSL:
        {"type": "project",  "entity": "label", "relation": "rel"}  → e + r ≈ t
        {"type": "intersect", "entities": [...], "queries": [...]}  → ∧
        {"type": "union",     "queries": [...]}                      → ∨
        {"type": "negate",    "query": {...}}                        → ¬

    Args:
        db: Database instance with load_embeddings() method.
        query: DSL query dict.
        top_k: Number of results to return.
        _cached_embeddings: Optional (entities, relations) tuple to
            avoid re-loading from DB.  GraphEngine passes its cached
            TransE entities/relations when available.

    Returns:
        List of [{"label": ..., "score": ...}, ...] sorted by score descending.
    """
    entities, relations = _load_embeddings_for_query(db, cached=_cached_embeddings)

    return _evaluate(query, entities, relations, top_k)


def _evaluate(
    query: dict,
    entities: dict[str, np.ndarray],
    relations: dict[str, np.ndarray],
    top_k: int,
) -> list[dict]:
    """Recursive query evaluation."""
    qtype = query.get("type", "")

    if qtype == "project":
        label = query.get("entity", "")
        rel_name = query.get("relation", "")
        e_vec = entities.get(label)
        r_vec = relations.get(rel_name)
        if e_vec is None or r_vec is None:
            return []
        results = project(e_vec, r_vec, entities, top_k=top_k)
        return _to_dicts(results)

    elif qtype == "intersect":
        # Support both inline entity labels and nested sub-queries
        vecs: list[np.ndarray] = []
        for label in query.get("entities", []):
            v = entities.get(label)
            if v is not None:
                vecs.append(v)
        for sub in query.get("queries", []):
            sub_results = _evaluate(sub, entities, relations, top_k)
            if sub_results:
                # Use the top result's vector as the anchor for this sub-query
                top_label = sub_results[0]["label"]
                top_vec = entities.get(top_label)
                if top_vec is not None:
                    vecs.append(top_vec)
        if not vecs:
            return []
        results = intersect(vecs, entities, top_k=top_k)
        return _to_dicts(results)

    elif qtype == "union":
        all_candidates: list[list[tuple[str, float]]] = []
        for sub in query.get("queries", []):
            sub_results = _evaluate(sub, entities, relations, top_k)
            all_candidates.append([(r["label"], r["score"]) for r in sub_results])
        results = union(all_candidates, top_k=top_k)
        return _to_dicts(results)

    elif qtype == "negate":
        sub = query.get("query", {})
        sub_results = _evaluate(sub, entities, relations, top_k)
        if not sub_results:
            return []
        # Negate the top result of the sub-query
        top_label = sub_results[0]["label"]
        top_vec = entities.get(top_label)
        if top_vec is None:
            return []
        results = negate(top_vec, entities, top_k=top_k)
        return _to_dicts(results)

    else:
        return []


def _to_dicts(results: list[tuple[str, float]]) -> list[dict]:
    """Convert (label, score) tuples to dicts."""
    return [{"label": label, "score": score} for label, score in results]
