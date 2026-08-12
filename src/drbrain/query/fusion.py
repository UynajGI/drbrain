"""Reciprocal Rank Fusion (RRF).

DEPRECATED (T9, 终态清理): the LlamaIndex RAG layer
(``drbrain.rag`` — BM25Retriever / FusionRetriever / CrossEncoderReranker)
replaces this module for CLI-facing retrieval. The file is RETAINED because
concept-asset code still depends on it (see the module docstring for the
specific dependency) — no new call sites should be added. Design doc:
``docs/llamaindex-integration-design.md`` §1 替换清单.

Pure algorithm — knows nothing about BM25 or embeddings. Takes any number of
``SearchHit`` ranked lists (each already sorted by descending score, each hit
carrying a ``paper_id``) and produces a single merged ranked list via RRF::

    rrf_score(paper) = Σ_lists  1 / (k + rank_in_list)

Default ``k=60`` per Cormack et al. (2009), "Reciprocal Rank Fusion Outperforms
Condorcet and Individual Rank Learning Methods".

A paper appearing in multiple lists is merged into one hit: its RRF scores
sum, its sources are deduplicated into ``metadata["sources"]``, and each
source's original rank/score is recorded in ``metadata["contributions"]`` for
debugging. Distinct from the private ``_rrf_score`` in ``tree_retrieval.py``,
which drops all metadata and only returns ``(id, score)`` tuples.
"""

from __future__ import annotations

from drbrain.query.types import SearchHit

DEFAULT_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchHit]],
    k: int = DEFAULT_K,
    weights: dict[str, float] | None = None,
) -> list[SearchHit]:
    """Fuse multiple ranked ``SearchHit`` lists via RRF.

    Args:
        ranked_lists: Each inner list must be sorted by descending ``score``;
            rank is derived from position (1-based). Lists may be empty.
        k: RRF damping constant. Larger ``k`` smooths the advantage of
            top-ranked items. Classic default is 60.
        weights: Optional per-source multipliers, keyed by ``SearchHit.source``
            (e.g. ``{"bm25": 0.3, "embedding": 0.7}``). ``None`` (default)
            gives every source equal weight 1.0 — the canonical RRF. Use this
            to upweight a retriever that empirically delivers higher relevance.
            Unknown sources default to weight 1.0.

    Returns:
        Merged hits sorted by descending RRF score, with ``rank`` rewritten
        (1-based) and ``source="fused"``. Empty input yields ``[]``.
    """
    # Per-paper accumulator: {paper_id: {rrf: float, sources: [str], payload:
    # {src: row}, contributions: {src: {rank, score, weight}}}}
    acc: dict[str, dict] = {}

    for lst in ranked_lists:
        for position, hit in enumerate(lst, start=1):
            pid = hit.paper_id
            if not pid:
                continue
            entry = acc.setdefault(
                pid,
                {"rrf": 0.0, "sources": [], "payload": {}, "contributions": {}},
            )
            src = hit.source or "unknown"
            w = (weights or {}).get(src, 1.0)
            entry["rrf"] += w * (1.0 / (k + position))

            if src not in entry["sources"]:
                entry["sources"].append(src)
            # Keep the first-seen row per source; if a source lists the same
            # paper twice (shouldn't happen post-normalization), the earlier
            # higher-rank entry wins.
            if src not in entry["payload"]:
                entry["payload"][src] = hit.payload
            entry["contributions"][src] = {"rank": position, "score": hit.score, "weight": w}

    if not acc:
        return []

    fused: list[SearchHit] = []
    for pid, entry in acc.items():
        fused.append(
            SearchHit(
                paper_id=pid,
                score=entry["rrf"],
                source="fused",
                payload=entry["payload"],
                metadata={
                    "sources": entry["sources"],
                    "contributions": entry["contributions"],
                },
            )
        )

    fused.sort(key=lambda h: h.score, reverse=True)
    for i, hit in enumerate(fused, start=1):
        hit.rank = i
    return fused


def _ranking_order(fused: list[SearchHit]) -> list[str]:
    """Extract the paper_id order from a fused result list."""
    return [h.paper_id for h in fused]


def kendall_tau(a: list[str], b: list[str]) -> float:
    """Kendall's tau-b rank correlation between two paper_id orderings.

    Returns 1.0 (identical order) to -1.0 (perfect inversion). Used by the
    parameter-sweep tool to measure how much a changed ``k`` or weighting
    perturbs the fused ranking. Handles lists with different element sets by
    restricting comparison to their intersection.
    """
    common = [x for x in a if x in b]
    n = len(common)
    if n < 2:
        return 1.0
    pos_a = {x: i for i, x in enumerate(a)}
    pos_b = {x: i for i, x in enumerate(b)}
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = pos_a[common[i]] - pos_a[common[j]]
            db = pos_b[common[i]] - pos_b[common[j]]
            if da * db > 0:
                concordant += 1
            elif da * db < 0:
                discordant += 1
    return (
        (concordant - discordant) / (concordant + discordant) if (concordant + discordant) else 1.0
    )


def compare_fusion_params(
    ranked_lists: list[list[SearchHit]],
    k_values: list[int],
    weights_options: dict[str, dict[str, float]] | None = None,
    baseline_k: int = DEFAULT_K,
    baseline_weights: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Sweep RRF parameters and report ranking perturbation vs a baseline.

    For tuning offline: feed real BM25 + embedding ranked lists, vary ``k`` and
    per-source weights, and see how much each setting moves the fused order
    relative to the baseline (kendall_tau = 1.0 means identical ranking).

    Args:
        ranked_lists: The retriever outputs to fuse (same input as
            ``reciprocal_rank_fusion``).
        k_values: ``k`` constants to sweep.
        weights_options: Named weighting configs, e.g.
            ``{"bm25_heavy": {"bm25": 0.7, "embedding": 0.3}, ...}``.
        baseline_k / baseline_weights: The reference fusion; everything is
            compared against this. Defaults to canonical RRF (k=60, equal).

    Returns:
        ``{setting_name: {"k": int, "weights": dict|None, "kendall_tau": float,
        "top10": [paper_ids]}}``. A tau near 1.0 = minimal change; low tau =
        the parameter materially reshapes the ranking (investigate whether that
        helps or hurts on a labeled set).
    """
    baseline = reciprocal_rank_fusion(ranked_lists, k=baseline_k, weights=baseline_weights)
    baseline_order = _ranking_order(baseline)

    results: dict[str, dict] = {}

    # Sweep k (equal-weight)
    for k in k_values:
        fused = reciprocal_rank_fusion(ranked_lists, k=k)
        order = _ranking_order(fused)
        results[f"k={k}"] = {
            "k": k,
            "weights": None,
            "kendall_tau": round(kendall_tau(baseline_order, order), 4),
            "top10": order[:10],
        }

    # Sweep named weighting configs (at baseline_k)
    if weights_options:
        for name, w in weights_options.items():
            fused = reciprocal_rank_fusion(ranked_lists, k=baseline_k, weights=w)
            order = _ranking_order(fused)
            results[f"w={name}"] = {
                "k": baseline_k,
                "weights": w,
                "kendall_tau": round(kendall_tau(baseline_order, order), 4),
                "top10": order[:10],
            }

    return results


__all__ = ["reciprocal_rank_fusion", "compare_fusion_params", "kendall_tau", "DEFAULT_K"]
