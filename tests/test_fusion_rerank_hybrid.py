"""Tests for RRF fusion, reranker fallback, and hybrid retrieval.

Covers the acceptance criteria:
  - RRF correctly fuses two ranked lists (k=60)
  - Duplicate paper_ids merge (scores add, sources dedupe)
  - Fused payloads preserve per-source rows
  - Reranker degrades to no-op when deps are unavailable
  - hybrid_search output fields are stable; pure-BM25 and empty-DB paths
    do not crash.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from drbrain.query.fusion import reciprocal_rank_fusion
from drbrain.query.hybrid_retrieval import hybrid_search
from drbrain.query.rerank import CrossEncoderReranker, NoopReranker, get_reranker
from drbrain.query.types import SearchHit

# ── fixtures ─────────────────────────────────────────────────────────────────


def _hit(pid: str, score: float, source: str, rank: int = 0, **payload) -> SearchHit:
    """Build a SearchHit with a flat payload for compact test setup."""
    return SearchHit(
        paper_id=pid,
        score=score,
        rank=rank,
        source=source,
        payload={"label": pid, "text": payload.get("text", ""), **payload},
    )


# ── RRF fusion ───────────────────────────────────────────────────────────────


def test_rrf_fuses_two_lists():
    """Known-input RRF: hand-computed scores with k=60."""
    # list1: p1(rank1)=1/61, p2(rank2)=1/62
    # list2: p1(rank1) again but here placed at rank3 → 1/63
    list1 = [
        _hit("p1", 2.5, "bm25", rank=1),
        _hit("p2", 1.0, "bm25", rank=2),
    ]
    list2 = [
        _hit("px", 0.9, "embedding", rank=1),
        _hit("py", 0.8, "embedding", rank=2),
        _hit("p1", 0.7, "embedding", rank=3),
    ]

    fused = reciprocal_rank_fusion([list1, list2], k=60)

    by_pid = {h.paper_id: h for h in fused}
    # p1 appears in both lists: 1/61 + 1/63
    assert pytest.approx(by_pid["p1"].score, rel=1e-6) == (1 / 61 + 1 / 63)
    # p2 only in list1 rank2: 1/62
    assert pytest.approx(by_pid["p2"].score, rel=1e-6) == 1 / 62
    # Ordering: p1 > p2
    assert fused[0].paper_id == "p1"
    # Ranks are rewritten 1-based in output
    assert [h.rank for h in fused] == list(range(1, len(fused) + 1))


def test_rrf_merges_duplicate_paper():
    """Same paper_id across lists merges: score sums, sources dedupe."""
    list1 = [_hit("p1", 2.0, "bm25", rank=1)]
    list2 = [_hit("p1", 0.9, "embedding", rank=1)]

    fused = reciprocal_rank_fusion([list1, list2], k=60)

    assert len(fused) == 1
    hit = fused[0]
    assert hit.paper_id == "p1"
    assert hit.score == pytest.approx(1 / 61 + 1 / 61)
    assert hit.source == "fused"
    # Both sources recorded, no duplicates.
    assert hit.metadata["sources"] == ["bm25", "embedding"]
    # Contributions preserve original rank/score per source, plus the applied
    # weight (1.0 under default equal-weight RRF).
    assert hit.metadata["contributions"]["bm25"] == {"rank": 1, "score": 2.0, "weight": 1.0}
    assert hit.metadata["contributions"]["embedding"] == {"rank": 1, "score": 0.9, "weight": 1.0}


def test_rrf_preserves_payloads():
    """Fused payload keeps each source's original row dict."""
    list1 = [_hit("p1", 2.0, "bm25", rank=1, text="bm25 text")]
    list2 = [_hit("p1", 0.9, "embedding", rank=1, node_id="0007")]

    fused = reciprocal_rank_fusion([list1, list2], k=60)
    hit = fused[0]

    assert "bm25" in hit.payload
    assert "embedding" in hit.payload
    assert hit.payload["bm25"]["text"] == "bm25 text"
    assert hit.payload["embedding"]["node_id"] == "0007"


def test_rrf_empty_input():
    """Empty list-of-lists and all-empty lists both yield []."""
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []


# ── Reranker ─────────────────────────────────────────────────────────────────


def test_noop_reranker_returns_topn():
    """NoopReranker preserves input order and truncates to top_n."""
    reranker = NoopReranker()
    hits = [_hit(f"p{i}", float(5 - i), "bm25", rank=i) for i in range(1, 5)]

    out = reranker.rerank("query", hits, top_n=2)

    assert [h.paper_id for h in out] == ["p1", "p2"]


def test_crossencoder_fallback_on_missing_deps(monkeypatch):
    """CrossEncoderReranker must not raise when sentence_transformers is absent.

    We simulate a missing dependency by making the ``sentence_transformers``
    import inside ``_ensure_model`` raise ``ImportError``.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("simulated missing dep")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    reranker = CrossEncoderReranker("BAAI/bge-reranker-base")
    hits = [
        _hit("p1", 0.9, "bm25", rank=1, text="alpha"),
        _hit("p2", 0.8, "bm25", rank=2, text="beta"),
    ]

    # Must not raise; returns input order truncated.
    out = reranker.rerank("query", hits, top_n=1)
    assert len(out) == 1
    assert out[0].paper_id == "p1"
    # Sticky failure: subsequent calls stay degraded without re-importing.
    out2 = reranker.rerank("query", hits, top_n=5)
    assert [h.paper_id for h in out2] == ["p1", "p2"]


def test_get_reranker_factory():
    """Factory returns the right concrete class and is always usable."""
    assert isinstance(get_reranker("none"), NoopReranker)
    auto = get_reranker("auto")
    assert isinstance(auto, CrossEncoderReranker)


# ── hybrid_search ────────────────────────────────────────────────────────────


def test_hybrid_search_pure_bm25(tmp_db):
    """embed_cfg=None → pure BM25 path; output fields stable, no crash."""
    tmp_db.insert_paper("p1", "Transformer Paper", 2024, "extracted")
    tmp_db.insert_paper("p2", "Convolution Paper", 2023, "extracted")
    tmp_db.insert_concept("p1", "Method", "transformer architecture", 0.9, year=2024)
    tmp_db.insert_concept("p2", "Method", "convolutional network", 0.8, year=2023)
    tmp_db.commit()

    hits = hybrid_search("transformer", tmp_db, Path("/nonexistent.db"), embed_cfg=None, top_k=5)

    assert len(hits) >= 1
    for hit in hits:
        assert set(["paper_id", "score", "rank", "source"]).issubset(hit.to_dict().keys())
        assert hit.source == "fused"  # only bm25 leg ran, but fusion still tags fused


def test_hybrid_search_empty_db(tmp_db):
    """Empty database returns empty list without crashing."""
    hits = hybrid_search("anything", tmp_db, Path("/nonexistent.db"), embed_cfg=None)
    assert hits == []


def test_hybrid_search_field_stability(tmp_db):
    """Output hits always carry paper_id/score/rank/source, ranks 1-based."""
    tmp_db.insert_paper("p1", "Alpha", 2024, "extracted")
    tmp_db.insert_concept("p1", "Method", "alpha method", 0.9, year=2024)
    tmp_db.commit()

    hits = hybrid_search("alpha", tmp_db, Path("/nonexistent.db"), embed_cfg=None, top_k=3)

    assert hits, "expected at least one hit"
    d = hits[0].to_dict()
    for field in ("paper_id", "score", "rank", "source", "payload", "metadata"):
        assert field in d
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
