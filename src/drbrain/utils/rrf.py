"""Canonical Reciprocal Rank Fusion core (review R-I7 convergence).

There is exactly ONE RRF constant and ONE opaque-key RRF implementation in
the codebase — this module. The typed fusers layer payload bookkeeping on
top of the same math and constant:

- :func:`drbrain.query.fusion.reciprocal_rank_fusion` — ``SearchHit`` lists
  (CLI search paths); per-source weights and rank rewriting.
- :func:`drbrain.rag.fusion._rrf_fuse` — ``NodeWithScore`` legs (LlamaIndex
  fusion retriever); metadata/contribution annotation.
- :func:`drbrain.rag.sql_retrie._fuse` — plain ``(key, score)`` tuple legs
  (SQL-native path); delegates here outright.

All three share :data:`DEFAULT_K` (Cormack et al. 2009) and the same score
definition: ``rrf(item) = Σ_legs weight_leg / (k + rank_leg)`` with 1-based
ranks derived from list position.
"""

from __future__ import annotations

DEFAULT_K = 60


def rrf_fuse_scores(
    ranked_lists: list[list[tuple[str, float]]],
    k: float = DEFAULT_K,
) -> list[tuple[str, float]]:
    """Fuse ranked ``(key, score)`` lists via RRF; opaque keys, no weights.

    The key may be a paper id, ``paper_id:node_id``, a claim id — the fusion
    never inspects it. Rank is the 1-based position in each list; a key's
    fused score is ``Σ 1 / (k + rank)`` over the legs that ranked it. Keys
    absent from a leg simply contribute nothing from that leg.

    Returns ``[(key, fused_score)]`` sorted by descending score (stable on
    accumulation order for ties, matching the historical SQL-path behavior).
    """
    scores: dict[str, float] = {}
    for leg in ranked_lists:
        for rank, (key, _s) in enumerate(leg, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
