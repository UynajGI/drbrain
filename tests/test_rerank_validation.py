"""Validation tests for the rerank layer with a mock cross-encoder.

These tests verify the rerank *code path* works correctly when a cross-encoder
is available — without requiring ``sentence_transformers``/``torch`` to be
installed (they are heavy optional deps). A fake ``CrossEncoder`` class is
injected via ``importlib``, exposing the same ``.predict(pairs)`` contract the
real ``BAAI/bge-reranker-base`` uses.

What is verified here (real evidence):
  - ``_ensure_model`` returns True and stores the model when the import succeeds
  - ``rerank`` actually reorders hits by the model's predicted scores
  - ``rerank`` truncates to ``top_n``
  - unscoreable hits (no text) are preserved at the tail in input order
  - a mid-flight ``predict`` failure degrades to input order (not a crash)

What is NOT verified here (requires the real dependency):
  - actual relevance quality of ``BAAI/bge-reranker-base`` — see
    ``docs/rerank_validation.md`` for the one-command procedure to validate
    with the real model once ``sentence-transformers`` is installed.
"""

from __future__ import annotations

import sys
import types

import pytest

from drbrain.query.rerank import CrossEncoderReranker
from drbrain.query.types import SearchHit


class _FakeCrossEncoder:
    """Stand-in for sentence_transformers.CrossEncoder.

    ``predict`` returns a deterministic score per (query, doc) pair so tests
    can assert exact ordering. Score = 1.0 when the doc contains the query's
    keyword, else a small value based on doc length.
    """

    def __init__(self, model_name: str = "fake") -> None:
        self.model_name = model_name
        self.predict_calls = 0

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.predict_calls += 1
        scores: list[float] = []
        for query, doc in pairs:
            keyword = query.split()[-1]  # last word of the query is the signal
            scores.append(1.0 if keyword.lower() in doc.lower() else 0.1)
        return scores


@pytest.fixture
def fake_crossencoder_module(monkeypatch):
    """Inject a fake ``sentence_transformers`` module with CrossEncoder.

    Yields the fake module so individual tests can inspect/patch it.
    """
    fake = types.ModuleType("sentence_transformers")
    fake.CrossEncoder = _FakeCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    return fake


def _hit(pid: str, text: str, score: float = 0.5, source: str = "bm25") -> SearchHit:
    return SearchHit(
        paper_id=pid,
        score=score,
        rank=0,
        source=source,
        payload={"label": pid, "text": text},
    )


# ── happy path: real reorder driven by model scores ──────────────────────────


def test_rerank_reorders_by_model_score(fake_crossencoder_module):
    """When the model prefers p3 over p1/p2, rerank surfaces p3 first."""
    reranker = CrossEncoderReranker("fake-model")
    hits = [
        _hit("p1", "irrelevant content about cooking"),
        _hit("p2", "more cooking recipes"),
        _hit("p3", "the key method is described here"),  # matches query keyword
    ]

    out = reranker.rerank("find the key", hits, top_n=3)

    # p3 contains "key" → score 1.0; others → 0.1. So p3 must be first.
    assert out[0].paper_id == "p3"
    # ranks rewritten 1-based
    assert [h.rank for h in out] == [1, 2, 3]
    # model was actually invoked
    assert reranker._model is not None
    assert reranker._model.predict_calls == 1  # type: ignore[attr-defined]


def test_rerank_truncates_to_topn(fake_crossencoder_module):
    """rerank returns at most top_n hits, keeping the highest-scoring."""
    reranker = CrossEncoderReranker("fake-model")
    hits = [
        _hit("p1", "alpha keyword match"),
        _hit("p2", "no signal here"),
        _hit("p3", "another keyword match"),
        _hit("p4", "nothing relevant"),
    ]

    out = reranker.rerank("the keyword", hits, top_n=2)

    assert len(out) == 2
    # Both keyword-matching docs (p1, p3) survive; the non-matching ones drop.
    assert {h.paper_id for h in out} == {"p1", "p3"}


# ── robustness: unscoreable hits preserved at tail ───────────────────────────


def test_rerank_preserves_unscoreable_at_tail(fake_crossencoder_module):
    """Hits with empty text can't be scored; they keep input order at the end."""
    reranker = CrossEncoderReranker("fake-model")
    hits = [
        _hit("p1", "has keyword signal"),
        SearchHit(paper_id="p2", score=0.5, source="embedding", payload={}),  # no text
        _hit("p3", "also has keyword"),
    ]

    out = reranker.rerank("query keyword", hits, top_n=3)

    # Scored hits (p1, p3) rank first by model score; unscoreable p2 goes last.
    assert [h.paper_id for h in out] == ["p1", "p3", "p2"]


# ── robustness: mid-flight predict failure degrades gracefully ────────────────


def test_rerank_predict_failure_falls_back(monkeypatch):
    """If predict() raises after a successful load, return input order, no crash."""
    fake = types.ModuleType("sentence_transformers")

    class _BoomEncoder:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def predict(self, _pairs):  # noqa: ANN001
            raise RuntimeError("simulated inference failure")

    fake.CrossEncoder = _BoomEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    reranker = CrossEncoderReranker("boom-model")
    hits = [_hit("p1", "text"), _hit("p2", "more text")]

    out = reranker.rerank("q", hits, top_n=2)

    # Must not raise; returns input order truncated.
    assert [h.paper_id for h in out] == ["p1", "p2"]
