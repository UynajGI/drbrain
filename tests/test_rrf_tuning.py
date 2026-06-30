"""Tests for weighted RRF and the parameter-sweep tuning tool.

Covers:
  - per-source weights change the fused score and can reorder results
  - unknown sources default to weight 1.0 (no KeyError)
  - weights=None reproduces canonical equal-weight RRF
  - kendall_tau correctness on known orderings
  - compare_fusion_params sweeps k and weighting, reports tau vs baseline
"""

from __future__ import annotations

from drbrain.query.fusion import (
    compare_fusion_params,
    kendall_tau,
    reciprocal_rank_fusion,
)
from drbrain.query.types import SearchHit


def _hit(pid: str, score: float, source: str, rank: int = 0) -> SearchHit:
    return SearchHit(paper_id=pid, score=score, rank=rank, source=source, payload={})


# ── weighted RRF ─────────────────────────────────────────────────────────────


def test_weights_change_fused_score():
    """Upweighting a source increases its contribution to the shared paper."""
    bm25 = [_hit("p1", 2.0, "bm25", rank=1)]
    embed = [_hit("p1", 0.9, "embedding", rank=1)]

    equal = reciprocal_rank_fusion([bm25, embed], k=60, weights=None)
    bm25_heavy = reciprocal_rank_fusion([bm25, embed], k=60, weights={"bm25": 2.0})

    # equal: 1/61 + 1/61. bm25_heavy: 2/61 + 1/61. Score must rise.
    assert bm25_heavy[0].score > equal[0].score
    # Weight recorded in contributions for debugging.
    assert equal[0].metadata["contributions"]["bm25"]["weight"] == 1.0
    assert bm25_heavy[0].metadata["contributions"]["bm25"]["weight"] == 2.0


def test_weights_can_reorder_results():
    """A heavy weight can promote a source-only paper above a shared one."""
    # p_shared appears in both lists; p_bm25_only appears only in BM25 at rank 1.
    bm25 = [_hit("p_bm25_only", 3.0, "bm25", rank=1), _hit("p_shared", 2.0, "bm25", rank=3)]
    embed = [_hit("p_shared", 0.9, "embedding", rank=1)]

    # Equal weight:
    #   p_shared   = 1/63 (bm25 r3) + 1/61 (embed r1) ≈ 0.01587 + 0.01639 = 0.03226
    #   p_bm25_only = 1/61 (bm25 r1) ≈ 0.01639
    #   → p_shared wins.
    equal = reciprocal_rank_fusion([bm25, embed], k=60)
    assert equal[0].paper_id == "p_shared"

    # Heavily downweight embedding: p_shared loses almost all its embedding
    # contribution, leaving 1/63 ≈ 0.01587, while p_bm25_only keeps 1/61 ≈ 0.01639.
    #   → p_bm25_only wins.
    downweighted = reciprocal_rank_fusion([bm25, embed], k=60, weights={"embedding": 0.01})
    assert downweighted[0].paper_id == "p_bm25_only"


def test_unknown_source_defaults_to_weight_one():
    """A source not in the weights dict gets weight 1.0 (no KeyError)."""
    hits = [_hit("p1", 1.0, "mystery_source", rank=1)]
    fused = reciprocal_rank_fusion([hits], k=60, weights={"bm25": 0.5})
    assert len(fused) == 1
    assert fused[0].metadata["contributions"]["mystery_source"]["weight"] == 1.0


def test_weights_none_equals_explicit_equal():
    """weights=None and weights={all:1.0} produce identical scores."""
    bm25 = [_hit("p1", 2.0, "bm25", rank=1)]
    embed = [_hit("p1", 0.9, "embedding", rank=1)]

    a = reciprocal_rank_fusion([bm25, embed], k=60, weights=None)
    b = reciprocal_rank_fusion([bm25, embed], k=60, weights={"bm25": 1.0, "embedding": 1.0})
    assert a[0].score == b[0].score


# ── kendall_tau ──────────────────────────────────────────────────────────────


def test_kendall_tau_identical_orders():
    assert kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_kendall_tau_reversed_orders():
    assert kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0


def test_kendall_tau_partial_overlap():
    """Only common elements are compared; extra elements ignored."""
    assert kendall_tau(["a", "b", "c", "d"], ["a", "b", "c"]) == 1.0


# ── parameter sweep tool ─────────────────────────────────────────────────────


def test_compare_fusion_params_sweeps_k():
    """The sweep returns one entry per k value, each with a tau and top10."""
    bm25 = [_hit("p1", 3.0, "bm25", rank=1), _hit("p2", 2.0, "bm25", rank=2)]
    embed = [_hit("p2", 0.9, "embedding", rank=1), _hit("p3", 0.8, "embedding", rank=2)]

    results = compare_fusion_params([bm25, embed], k_values=[10, 60, 100])

    assert set(results.keys()) == {"k=10", "k=60", "k=100"}
    for entry in results.values():
        assert "kendall_tau" in entry
        assert "top10" in entry
        assert isinstance(entry["top10"], list)
    # Baseline k=60 must have tau 1.0 against itself.
    assert results["k=60"]["kendall_tau"] == 1.0


def test_compare_fusion_params_sweeps_weights():
    """Named weighting configs appear as 'w=<name>' entries."""
    bm25 = [_hit("p1", 3.0, "bm25", rank=1)]
    embed = [_hit("p1", 0.9, "embedding", rank=1)]

    results = compare_fusion_params(
        [bm25, embed],
        k_values=[60],
        weights_options={"embed_heavy": {"bm25": 0.3, "embedding": 0.7}},
    )
    assert "w=embed_heavy" in results
    assert results["w=embed_heavy"]["weights"] == {"bm25": 0.3, "embedding": 0.7}
