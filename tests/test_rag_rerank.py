"""T8 tests: reranking (CrossEncoderReranker / RerankPostprocessor) + dedup.

Covers :mod:`drbrain.rag.rerank`:

* :class:`CrossEncoderReranker` — lazy construction (never loads the model),
  single load attempt, degrade-on-failure (``available == False`` + warning),
  device normalization, score passthrough
* :class:`RerankPostprocessor` — re-score + re-order, top_k truncation, every
  Noop degrade path (no reranker / unavailable / no query / rerank error /
  score mismatch)
* :class:`DeduplicatePostprocessor` — keep-first by node_id, content-hash
  fallback
* :func:`build_reranker` — config factory (model name, device mapping, None
  when the model is unset)
* postprocessor-chain assembly in ``build_query_engine`` (rerank=true → the 3
  chain postprocessors in order; rerank=false → pre-T8 behavior; fusion top_k
  bumped to rerank_top_k) and full offline chain execution through
  ``RetrieverQueryEngine._apply_node_postprocessors``
* rank-comparison statistics used by ``scripts/rerank_ab.py``

All unit tests are offline (mocked reranker / monkeypatched CrossEncoder, no
model download, no GPU). One live integration test loads a tiny real
cross-encoder and is skipped when none is cached and the network is
unavailable.
"""

import importlib.util
import logging
import os

import pytest

from drbrain.config import Config, LlamaIndexConfig
from drbrain.rag.engine import SimilarityCutoffPostprocessor, build_query_engine
from drbrain.rag.rerank import (
    CrossEncoderReranker,
    DeduplicatePostprocessor,
    RerankPostprocessor,
    build_reranker,
    kendall_tau,
    mean_rank_displacement,
    top_k_overlap,
)

_HAS_LLAMA_INDEX = importlib.util.find_spec("llama_index") is not None

if _HAS_LLAMA_INDEX:
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

pytestmark = pytest.mark.skipif(not _HAS_LLAMA_INDEX, reason="llama_index not installed")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _cfg(
    rerank: bool = True,
    cutoff: float | None = 0.7,
    rerank_top_k: int = 20,
    model: str = "Qwen/Qwen3-Reranker-0.6B",
) -> Config:
    return Config(
        llamaindex=LlamaIndexConfig(
            enabled=True,
            rerank=rerank,
            similarity_cutoff=cutoff,
            rerank_top_k=rerank_top_k,
            rerank_model=model,
        )
    )


def _node(node_id: str = "n1", paper_id: str = "p1", title: str = "Title 1") -> TextNode:
    return TextNode(
        text=f"section body of {node_id}",
        id_=f"{paper_id}:{node_id}",
        metadata={"paper_id": paper_id, "node_id": node_id, "title": title},
    )


def _fused_node(
    node_id: str = "n1",
    paper_id: str = "p1",
    rrf: float = 0.03,
    vector_score: float = 0.9,
) -> NodeWithScore:
    """A node as the T4 FusionRetriever would annotate it (RRF + contributions)."""
    node = _node(node_id, paper_id)
    node.metadata["source"] = "vector"
    node.metadata["contributions"] = {
        "vector": {"rank": 1, "score": vector_score, "weight": 1.0},
    }
    return NodeWithScore(node=node, score=rrf)


def _qb(query: str = "test query") -> QueryBundle:
    return QueryBundle(query)


class _FakeReranker:
    """Deterministic reranker stub (implements the ``available``/``rerank`` contract)."""

    def __init__(self, scores=None, available: bool = True, error: Exception | None = None):
        self._scores = list(scores) if scores is not None else None
        self._available = available
        self._error = error
        self.calls: list[tuple[str, list[str]]] = []

    @property
    def available(self) -> bool:
        return self._available

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        if self._error is not None:
            raise self._error
        if self._scores is None:
            return [0.0] * len(passages)
        return list(self._scores)[: len(passages)]


class _FakeRetriever(BaseRetriever):
    def __init__(self, nodes=None) -> None:
        super().__init__()
        self._nodes = list(nodes or [])

    def _retrieve(self, query_bundle):
        return list(self._nodes)


def _nws(nid: str, score: float = 0.5) -> NodeWithScore:
    return NodeWithScore(node=_node(nid), score=score)


# ── CrossEncoderReranker (lazy load / degrade) ──────────────────────────────


def test_cross_encoder_lazy_construct_never_loads(monkeypatch):
    class _CE:
        def __init__(self, *args, **kwargs):
            raise AssertionError("model must not load at construction time")

    monkeypatch.setattr("sentence_transformers.CrossEncoder", _CE)
    r = CrossEncoderReranker("some/model")  # must not raise
    assert r._model is None


def test_cross_encoder_load_failure_degrades(monkeypatch, caplog):
    class _CE:
        def __init__(self, *args, **kwargs):
            raise OSError("model not found / no network")

    monkeypatch.setattr("sentence_transformers.CrossEncoder", _CE)
    monkeypatch.setattr("drbrain.rag.rerank._hf_reachable", lambda: False)  # offline
    r = CrossEncoderReranker("bad/model")
    with caplog.at_level(logging.WARNING):
        assert r.available is False
    assert r._load_error is not None
    assert any("load failed" in rec.message for rec in caplog.records)
    with pytest.raises(RuntimeError):
        r.rerank("q", ["p"])


def test_cross_encoder_loads_from_cache(monkeypatch):
    calls = {}

    class _CE:
        def __init__(self, model_name, device=None, max_length=None, local_files_only=False):
            calls["init"] = (model_name, device, max_length, local_files_only)

        def predict(self, pairs, batch_size=None, show_progress_bar=None):
            calls["predict"] = pairs
            return [0.9, 0.1]

    monkeypatch.setattr("sentence_transformers.CrossEncoder", _CE)
    r = CrossEncoderReranker("m", device="auto", max_length=256)
    assert r.available is True
    # cache-first attempt, "auto" mapped to None (ST auto-selects CUDA)
    assert calls["init"] == ("m", None, 256, True)
    assert r.rerank("q", ["p1", "p2"]) == pytest.approx([0.9, 0.1])
    assert calls["predict"] == [["q", "p1"], ["q", "p2"]]


def test_cross_encoder_downloads_when_hf_reachable(monkeypatch):
    calls = []

    class _CE:
        def __init__(self, model_name, device=None, max_length=None, local_files_only=False):
            calls.append(local_files_only)
            if local_files_only:
                raise OSError("not in cache")

        def predict(self, pairs, batch_size=None, show_progress_bar=None):
            return [0.5]

    monkeypatch.setattr("sentence_transformers.CrossEncoder", _CE)
    monkeypatch.setattr("drbrain.rag.rerank._hf_reachable", lambda: True)  # online
    r = CrossEncoderReranker("m", device="cpu")
    assert r.available is True
    assert calls == [True, False]  # cache miss → download attempt


def test_cross_encoder_skips_download_when_offline(monkeypatch, caplog):
    calls = []

    class _CE:
        def __init__(self, model_name, device=None, max_length=None, local_files_only=False):
            calls.append(local_files_only)
            raise OSError("not in cache")

    monkeypatch.setattr("sentence_transformers.CrossEncoder", _CE)
    monkeypatch.setattr("drbrain.rag.rerank._hf_reachable", lambda: False)  # offline
    r = CrossEncoderReranker("m", device="cpu")
    with caplog.at_level(logging.WARNING):
        assert r.available is False
    # only the cache attempt ran — the download branch was gated off
    assert calls == [True]
    assert any("unreachable" in rec.message for rec in caplog.records)


def test_cross_encoder_device_normalization():
    assert CrossEncoderReranker("m", device="auto").device is None
    assert CrossEncoderReranker("m", device=None).device is None
    assert CrossEncoderReranker("m", device="").device is None
    assert CrossEncoderReranker("m", device="cpu").device == "cpu"
    assert CrossEncoderReranker("m", device="cuda:0").device == "cuda:0"


# ── build_reranker factory ───────────────────────────────────────────────────


def test_build_reranker_reads_config():
    r = build_reranker(_cfg())
    assert r is not None
    # T9: the configured id is resolved to a locally-loadable path when the
    # model is cached (modelscope/HF); on hosts without a cache it stays the
    # raw id — accept either.
    assert "Qwen3-Reranker-0.6B" in r.model_name
    assert r.device is None  # embed.device default "auto" → None


def test_build_reranker_none_when_model_unset():
    cfg = Config(llamaindex=LlamaIndexConfig(rerank=True, rerank_model=""))
    assert build_reranker(cfg) is None


# ── RerankPostprocessor ──────────────────────────────────────────────────────


def test_rerank_postprocessor_rescores_and_reorders():
    nodes = [_nws("n1", score=0.01), _nws("n2", score=0.02), _nws("n3", score=0.03)]
    pp = RerankPostprocessor(top_k=10, reranker=_FakeReranker(scores=[0.2, 0.9, 0.5]))
    out = pp.postprocess_nodes(nodes, query_bundle=_qb())
    assert [n.node.node_id for n in out] == ["p1:n2", "p1:n3", "p1:n1"]
    assert [n.score for n in out] == pytest.approx([0.9, 0.5, 0.2])
    # same node objects (metadata/source annotations survive)
    assert {id(n.node) for n in out} == {id(n.node) for n in nodes}
    # the reranker saw (query, passage) pairs in coarse order
    assert pp.reranker.calls[0][0] == "test query"
    assert pp.reranker.calls[0][1] == [n.node.text for n in nodes]


def test_rerank_postprocessor_truncates_to_top_k():
    nodes = [_nws(f"n{i}", score=float(i)) for i in range(25)]
    pp = RerankPostprocessor(top_k=20, reranker=_FakeReranker(scores=[0.0] * 25))
    out = pp.postprocess_nodes(nodes, query_bundle=_qb())
    assert len(out) == 20
    assert [n.node.node_id for n in out] == [f"p1:n{i}" for i in range(20)]


def test_rerank_postprocessor_empty_input():
    pp = RerankPostprocessor(top_k=10, reranker=_FakeReranker())
    assert pp.postprocess_nodes([], query_bundle=_qb()) == []


def test_rerank_postprocessor_noop_without_reranker():
    nodes = [_nws("n1")]
    pp = RerankPostprocessor(top_k=10, reranker=None)
    assert pp.postprocess_nodes(nodes, query_bundle=_qb()) == nodes


def test_rerank_postprocessor_noop_when_unavailable():
    nodes = [_nws("n1")]
    pp = RerankPostprocessor(top_k=10, reranker=_FakeReranker(available=False))
    assert pp.postprocess_nodes(nodes, query_bundle=_qb()) == nodes


def test_rerank_postprocessor_noop_without_query():
    nodes = [_nws("n1")]
    pp = RerankPostprocessor(top_k=10, reranker=_FakeReranker(scores=[1.0]))
    assert pp.postprocess_nodes(nodes, query_bundle=None) == nodes


def test_rerank_postprocessor_noop_on_rerank_error(caplog):
    nodes = [_nws("n1"), _nws("n2")]
    pp = RerankPostprocessor(top_k=10, reranker=_FakeReranker(error=RuntimeError("boom")))
    with caplog.at_level(logging.WARNING):
        out = pp.postprocess_nodes(nodes, query_bundle=_qb())
    assert out == nodes
    assert any("rerank failed" in rec.message for rec in caplog.records)


def test_rerank_postprocessor_noop_on_score_mismatch(caplog):
    nodes = [_nws("n1"), _nws("n2")]
    pp = RerankPostprocessor(top_k=10, reranker=_FakeReranker(scores=[1.0]))  # 2 passages, 1 score
    with caplog.at_level(logging.WARNING):
        out = pp.postprocess_nodes(nodes, query_bundle=_qb())
    assert out == nodes
    assert any("scores for" in rec.message for rec in caplog.records)


# ── DeduplicatePostprocessor ─────────────────────────────────────────────────


def test_dedup_keeps_first_by_node_id():
    nodes = [
        NodeWithScore(node=_node("n1"), score=0.9),
        NodeWithScore(node=_node("n2"), score=0.8),
        NodeWithScore(node=_node("n1"), score=0.7),  # duplicate id, lower score
    ]
    out = DeduplicatePostprocessor().postprocess_nodes(nodes)
    assert [n.node.node_id for n in out] == ["p1:n1", "p1:n2"]
    assert out[0].score == pytest.approx(0.9)  # first occurrence wins


def test_dedup_distinct_nodes_untouched():
    nodes = [_nws("n1"), _nws("n2"), _nws("n3")]
    out = DeduplicatePostprocessor().postprocess_nodes(nodes)
    assert [n.node.node_id for n in out] == ["p1:n1", "p1:n2", "p1:n3"]


def test_dedup_content_hash_fallback():
    # Same content → same content hash; node_id emptied so the hash is the key.
    a = TextNode(text="same content", id_="x1")
    b = TextNode(text="same content", id_="x2")
    c = TextNode(text="different content", id_="x3")
    a.node_id = ""
    b.node_id = ""
    c.node_id = ""
    assert a.hash == b.hash and a.hash != c.hash
    nodes = [NodeWithScore(node=n, score=1.0) for n in (a, b, c)]
    out = DeduplicatePostprocessor().postprocess_nodes(nodes)
    assert len(out) == 2
    assert out[0].node is a  # first occurrence kept


# ── build_query_engine assembly (T8 chain) ──────────────────────────────────


def test_rerank_true_mounts_three_postprocessors_in_order(monkeypatch):
    monkeypatch.setattr(
        "drbrain.rag.fusion.get_retrievers", lambda cfg, db: {"vector": _FakeRetriever()}
    )
    monkeypatch.setattr("drbrain.rag.rerank.build_reranker", lambda cfg: _FakeReranker())
    engine = build_query_engine(_cfg(rerank=True, cutoff=0.7, rerank_top_k=20), db=None)
    pps = engine._node_postprocessors
    assert [type(p).__name__ for p in pps] == [
        "RerankPostprocessor",
        "SimilarityCutoffPostprocessor",
        "DeduplicatePostprocessor",
    ]
    assert pps[0].top_k == 20
    assert isinstance(pps[0].reranker, _FakeReranker)
    assert pps[1].similarity_cutoff == 0.7


def test_rerank_false_mounts_cutoff_only(monkeypatch):
    monkeypatch.setattr(
        "drbrain.rag.fusion.get_retrievers", lambda cfg, db: {"vector": _FakeRetriever()}
    )
    engine = build_query_engine(_cfg(rerank=False, cutoff=0.7), db=None)
    pps = engine._node_postprocessors
    assert len(pps) == 1
    assert isinstance(pps[0], SimilarityCutoffPostprocessor)


def test_rerank_false_no_cutoff_mounts_nothing(monkeypatch):
    monkeypatch.setattr(
        "drbrain.rag.fusion.get_retrievers", lambda cfg, db: {"vector": _FakeRetriever()}
    )
    engine = build_query_engine(_cfg(rerank=False, cutoff=None), db=None)
    assert engine._node_postprocessors == []


def test_rerank_true_bumps_fusion_top_k(monkeypatch):
    captured = {}

    def _recorder(
        cfg,
        vector_index=None,
        bm25_retriever=None,
        custom_retrievers=None,
        top_k=None,
        weights=None,
    ):
        captured["top_k"] = top_k
        return _FakeRetriever()

    monkeypatch.setattr(
        "drbrain.rag.fusion.get_retrievers", lambda cfg, db: {"vector": _FakeRetriever()}
    )
    monkeypatch.setattr("drbrain.rag.fusion.build_fusion_retriever", _recorder)
    build_query_engine(_cfg(rerank=True, rerank_top_k=25), db=None)
    assert captured["top_k"] == 25


def test_rerank_false_keeps_caller_top_k(monkeypatch):
    captured = {}

    def _recorder(
        cfg,
        vector_index=None,
        bm25_retriever=None,
        custom_retrievers=None,
        top_k=None,
        weights=None,
    ):
        captured["top_k"] = top_k
        return _FakeRetriever()

    monkeypatch.setattr(
        "drbrain.rag.fusion.get_retrievers", lambda cfg, db: {"vector": _FakeRetriever()}
    )
    monkeypatch.setattr("drbrain.rag.fusion.build_fusion_retriever", _recorder)
    build_query_engine(_cfg(rerank=False), db=None, top_k=7)
    assert captured["top_k"] == 7


def test_rerank_chain_executes_rerank_cutoff_dedup(monkeypatch):
    """Offline end-to-end chain: rerank reorders → cutoff drops low-leg node → dedup collapses dup."""
    nodes = [
        _fused_node("n1", vector_score=0.9),
        _fused_node("n2", vector_score=0.8),
        _fused_node("n3", vector_score=0.85),
        _fused_node("n1", vector_score=0.9),  # duplicate of n1 (same node_id)
        _fused_node("n4", vector_score=0.1),  # below cutoff on the original leg score
    ]
    # reranker reverses the coarse order
    monkeypatch.setattr(
        "drbrain.rag.fusion.get_retrievers",
        lambda cfg, db: {"vector": _FakeRetriever(nodes)},
    )
    monkeypatch.setattr(
        "drbrain.rag.rerank.build_reranker",
        lambda cfg: _FakeReranker(scores=[0.1, 0.9, 0.5, 0.2, 0.7]),
    )
    engine = build_query_engine(_cfg(rerank=True, cutoff=0.7), db=None)
    out = engine._apply_node_postprocessors(nodes, _qb())
    # n2 (0.9) → n3 (0.5) → n1dup (0.2); n4 dropped by cutoff, second n1 dropped by dedup
    assert [n.node.node_id for n in out] == ["p1:n2", "p1:n3", "p1:n1"]
    assert [n.score for n in out] == pytest.approx([0.9, 0.5, 0.2])


# ── rank-comparison statistics (A/B tool helpers) ───────────────────────────


def test_top_k_overlap():
    assert top_k_overlap(["a", "b", "c"], ["a", "b", "d"], 3) == pytest.approx(2 / 4)  # Jaccard
    assert top_k_overlap(["a", "b", "c"], ["a", "b", "c"], 2) == 1.0
    assert top_k_overlap(["a"], ["x"], 5) == 0.0
    assert top_k_overlap([], [], 5) == 0.0


def test_mean_rank_displacement():
    # a: rank1→3, b: 2→1, c: 3→2 → displacements 2,1,1 → mean 4/3
    assert mean_rank_displacement(["a", "b", "c"], ["b", "c", "a"]) == pytest.approx(4 / 3)
    assert mean_rank_displacement(["a", "b"], ["a", "b"]) == 0.0
    assert mean_rank_displacement(["a"], ["x"]) == 0.0


def test_kendall_tau():
    assert kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == pytest.approx(-1.0)
    assert kendall_tau(["a"], ["x"]) == 0.0  # no common ids → 0


# ── integration: real tiny cross-encoder (skip unless cached/downloadable) ───


@pytest.mark.integration
@pytest.mark.timeout(180)
def test_integration_real_cross_encoder_reranks():
    """Load a real tiny cross-encoder and verify ordering follows relevance.

    Candidate order: env override (``DRBRAIN_RERANK_TEST_MODEL``) → tiny
    ms-marco MiniLM. Loading is cache-first and the download branch is gated
    on a quick huggingface.co reachability probe (offline CI fails fast
    instead of hanging on HF retries). Skipped when nothing loads.
    """
    reranker = None
    candidates = [
        os.environ.get("DRBRAIN_RERANK_TEST_MODEL"),
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ]
    for name in candidates:
        if not name:
            continue
        try:
            candidate = CrossEncoderReranker(name, device="cpu", batch_size=8, max_length=128)
            if candidate.available:
                reranker = candidate
                break
        except Exception:  # noqa: BLE001 - try the next candidate
            continue
    if reranker is None:
        pytest.skip("no cross-encoder model cached and no network to download one")

    query = "perovskite solar cell power conversion efficiency"
    passages = [
        "we study the synthesis of perovskite solar cells and report a power conversion "
        "efficiency of 25% under standard illumination.",
        "the film thickness was measured with a scanning electron microscope in cross section.",
    ]
    scores = reranker.rerank(query, passages)
    assert len(scores) == 2
    # the lexically relevant passage must score above the unrelated one
    assert scores[0] > scores[1]
