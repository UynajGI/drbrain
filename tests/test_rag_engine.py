"""T5 tests: query engine assembly + ask compatibility layer.

Covers :mod:`drbrain.rag.engine`:

* :func:`resolve_engine` — CLI fallback rule (disabled → legacy, explicit
  legacy → legacy, enabled + importable → llamaindex)
* :func:`build_query_engine` — assembly (fusion retriever + refine synthesizer
  + similarity-cutoff postprocessor), None on disabled/no-legs
* :func:`ask_llamaindex` — compat dict shape, source back-links, streaming
  chunk protocol (str and object chunks), RuntimeError on unbuildable engine
* :func:`SimilarityCutoffPostprocessor` — fused-score-aware cutoff semantics
* :func:`extract_sources` / :func:`nodes_to_paper_results` — source annotation
  and paper-level aggregation for the hybrid/query branches

All unit tests run offline (mocked retriever/synthesizer/response, no network,
no GPU). One live end-to-end test (test-run papers + real LLM) is marked
``integration`` and skipped by default.
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from drbrain.config import Config, LlamaIndexConfig
from drbrain.rag.engine import (
    SimilarityCutoffPostprocessor,
    ask_llamaindex,
    build_hybrid_retriever,
    build_query_engine,
    extract_sources,
    nodes_to_paper_results,
    resolve_engine,
)

_HAS_LLAMA_INDEX = importlib.util.find_spec("llama_index") is not None

if _HAS_LLAMA_INDEX:
    from llama_index.core.base.response.schema import Response, StreamingResponse
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, TextNode

pytestmark = pytest.mark.skipif(not _HAS_LLAMA_INDEX, reason="llama_index not installed")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _cfg(
    enabled: bool = True,
    streaming: bool = True,
    cutoff: float | None = 0.7,
    rerank: bool = False,  # T8: default False keeps the pre-T8 assembly assertions valid
    rerank_top_k: int = 20,
) -> Config:
    return Config(
        llamaindex=LlamaIndexConfig(
            enabled=enabled,
            streaming=streaming,
            similarity_cutoff=cutoff,
            rerank=rerank,
            rerank_top_k=rerank_top_k,
        )
    )


def _node(
    node_id: str = "n1",
    paper_id: str = "p1",
    title: str = "Title 1",
    text: str = "some section body",
    sources: list | None = None,
) -> TextNode:
    meta = {"paper_id": paper_id, "node_id": node_id, "title": title}
    if sources is not None:
        meta["sources"] = sources
    return TextNode(text=text, id_=f"{paper_id}:{node_id}", metadata=meta)


def _fused_node(
    node_id: str = "n1",
    paper_id: str = "p1",
    title: str = "Title 1",
    rrf: float = 0.03,
    vector_score: float = 0.9,
    bm25_score: float = 3.5,
) -> NodeWithScore:
    """A node as the T4 FusionRetriever would annotate it (RRF score + contributions)."""
    node = _node(node_id, paper_id, title)
    node.metadata["source"] = "vector"
    node.metadata["contributions"] = {
        "vector": {"rank": 1, "score": vector_score, "weight": 1.0},
        "bm25": {"rank": 2, "score": bm25_score, "weight": 1.0},
    }
    return NodeWithScore(node=node, score=rrf)


class _FakeRetriever(BaseRetriever):
    """Minimal BaseRetriever for assembly tests."""

    def __init__(self) -> None:
        super().__init__()

    def _retrieve(self, query_bundle):
        return [NodeWithScore(node=_node(), score=0.9)]


def _fake_response(answer: str = "a synthesized answer") -> Response:
    return Response(response=answer, source_nodes=[NodeWithScore(node=_node(), score=0.9)])


def _fake_stream_response(chunks: list, answer: str) -> StreamingResponse:
    return StreamingResponse(
        response_gen=iter(chunks),
        source_nodes=[NodeWithScore(node=_node(), score=0.9)],
        metadata={},
        response_txt=answer,
    )


class _FakeEngine:
    """Stand-in for RetrieverQueryEngine returning a canned response."""

    def __init__(self, response) -> None:
        self._response = response

    def query(self, question: str):
        return self._response


# ── resolve_engine (CLI fallback rule) ───────────────────────────────────────


def test_resolve_engine_disabled_falls_back_to_legacy():
    assert resolve_engine(_cfg(enabled=False), "llamaindex") == "legacy"


def test_resolve_engine_explicit_legacy_wins():
    assert resolve_engine(_cfg(enabled=True), "legacy") == "legacy"
    assert resolve_engine(_cfg(enabled=True), " LEGACY ") == "legacy"


def test_resolve_engine_enabled_returns_llamaindex():
    assert resolve_engine(_cfg(enabled=True), "llamaindex") == "llamaindex"


def test_resolve_engine_dict_config_without_section_falls_back():
    # CLI contexts in existing tests pass plain dicts without `llamaindex:`.
    assert resolve_engine({}, "llamaindex") == "legacy"


# ── build_query_engine assembly ──────────────────────────────────────────────


def test_build_query_engine_none_when_disabled(monkeypatch):
    monkeypatch.setattr("drbrain.rag.engine._engine_ready", lambda cfg: False)
    assert build_query_engine(_cfg(enabled=False), db=None) is None


def test_build_query_engine_none_without_legs(monkeypatch):
    # get_retrievers returns nothing → fusion has no legs → None.
    monkeypatch.setattr("drbrain.rag.fusion.get_retrievers", lambda cfg, db: {})
    assert build_query_engine(_cfg(enabled=True), db=None) is None


def test_build_query_engine_assembles_engine(monkeypatch):
    from llama_index.core.query_engine import RetrieverQueryEngine

    monkeypatch.setattr(
        "drbrain.rag.fusion.get_retrievers", lambda cfg, db: {"vector": _FakeRetriever()}
    )
    engine = build_query_engine(_cfg(enabled=True, streaming=False, cutoff=0.7), db=None)
    assert isinstance(engine, RetrieverQueryEngine)
    # refine synthesizer with the requested streaming flag
    assert engine._response_synthesizer._streaming is False
    # similarity cutoff postprocessor attached
    assert len(engine._node_postprocessors) == 1
    pp = engine._node_postprocessors[0]
    assert isinstance(pp, SimilarityCutoffPostprocessor)
    assert pp.similarity_cutoff == 0.7


def test_build_query_engine_streaming_default_from_config(monkeypatch):
    monkeypatch.setattr(
        "drbrain.rag.fusion.get_retrievers", lambda cfg, db: {"vector": _FakeRetriever()}
    )
    engine = build_query_engine(_cfg(enabled=True, streaming=True), db=None)
    assert engine._response_synthesizer._streaming is True


def test_build_query_engine_no_postprocessor_when_cutoff_none(monkeypatch):
    monkeypatch.setattr(
        "drbrain.rag.fusion.get_retrievers", lambda cfg, db: {"vector": _FakeRetriever()}
    )
    engine = build_query_engine(_cfg(enabled=True, cutoff=None), db=None)
    # RetrieverQueryEngine normalizes None → [], so no postprocessors attached.
    assert engine._node_postprocessors == []


def test_build_hybrid_retriever_returns_fusion(monkeypatch):
    monkeypatch.setattr(
        "drbrain.rag.fusion.get_retrievers",
        lambda cfg, db: {"vector": _FakeRetriever(), "bm25": _FakeRetriever()},
    )
    retriever = build_hybrid_retriever(_cfg(enabled=True), db=None)
    assert retriever is not None
    assert retriever._sources == ["bm25", "vector"]  # leg labels preserved


# ── ask_llamaindex compat layer ──────────────────────────────────────────────


def test_ask_llamaindex_returns_compat_dict(monkeypatch):
    monkeypatch.setattr(
        "drbrain.rag.engine.build_query_engine",
        lambda cfg, db, streaming=False, top_k=None: _FakeEngine(_fake_response()),
    )
    result = ask_llamaindex(_cfg(enabled=True), db=None, question="q?", top_k=3, streaming=False)
    assert isinstance(result, dict)
    assert result["question"] == "q?"
    assert result["answer"] == "a synthesized answer"
    assert result["engine"] == "llamaindex"
    src = result["sources"][0]
    assert src["paper_id"] == "p1"
    assert src["node_id"] == "n1"
    assert src["title"] == "Title 1"
    assert src["score"] == pytest.approx(0.9)
    assert src["sources"] == []  # no fusion source annotation on a plain node


def test_ask_llamaindex_result_json_serializable(monkeypatch):
    monkeypatch.setattr(
        "drbrain.rag.engine.build_query_engine",
        lambda cfg, db, streaming=False, top_k=None: _FakeEngine(_fake_response()),
    )
    result = ask_llamaindex(_cfg(enabled=True), db=None, question="q?", streaming=False)
    json.dumps(result)  # must not raise


def test_ask_llamaindex_streaming_yields_chunks_then_final(monkeypatch):
    monkeypatch.setattr(
        "drbrain.rag.engine.build_query_engine",
        lambda cfg, db, streaming=True, top_k=None: _FakeEngine(
            _fake_stream_response(["chunk one ", "chunk two"], "chunk one chunk two")
        ),
    )
    items = list(ask_llamaindex(_cfg(enabled=True), db=None, question="q?", streaming=True))
    assert items[:-1] == [{"chunk": "chunk one "}, {"chunk": "chunk two"}]
    final = items[-1]
    assert final["answer"] == "chunk one chunk two"
    assert final["engine"] == "llamaindex"
    assert final["sources"][0]["paper_id"] == "p1"
    assert final["sources"][0]["score"] == pytest.approx(0.9)


def test_ask_llamaindex_streaming_object_chunks(monkeypatch):
    # DrbrainLLM's single-chunk streaming yields ChatResponse-like objects.
    chunk = SimpleNamespace(delta="object chunk", text="object chunk")
    monkeypatch.setattr(
        "drbrain.rag.engine.build_query_engine",
        lambda cfg, db, streaming=True, top_k=None: _FakeEngine(
            _fake_stream_response([chunk], "object chunk")
        ),
    )
    items = list(ask_llamaindex(_cfg(enabled=True), db=None, question="q?", streaming=True))
    assert items[0] == {"chunk": "object chunk"}
    assert items[-1]["answer"] == "object chunk"


def test_ask_llamaindex_streaming_empty_stream_falls_back_to_response_txt(monkeypatch):
    monkeypatch.setattr(
        "drbrain.rag.engine.build_query_engine",
        lambda cfg, db, streaming=True, top_k=None: _FakeEngine(
            _fake_stream_response([], "full answer from response_txt")
        ),
    )
    items = list(ask_llamaindex(_cfg(enabled=True), db=None, question="q?", streaming=True))
    assert items[-1]["answer"] == "full answer from response_txt"


def test_ask_llamaindex_raises_when_engine_unavailable(monkeypatch):
    monkeypatch.setattr(
        "drbrain.rag.engine.build_query_engine", lambda cfg, db, streaming=False, top_k=None: None
    )
    with pytest.raises(RuntimeError):
        ask_llamaindex(_cfg(enabled=True), db=None, question="q?", streaming=False)


# ── SimilarityCutoffPostprocessor ────────────────────────────────────────────


def test_cutoff_keeps_fused_node_with_high_leg_score():
    pp = SimilarityCutoffPostprocessor(similarity_cutoff=0.7)
    kept = pp.postprocess_nodes([_fused_node(rrf=0.03, vector_score=0.9)])
    assert len(kept) == 1


def test_cutoff_drops_fused_node_with_low_leg_score():
    pp = SimilarityCutoffPostprocessor(similarity_cutoff=0.7)
    # both legs scored below the cutoff → dropped
    kept = pp.postprocess_nodes([_fused_node(rrf=0.03, vector_score=0.3, bm25_score=0.4)])
    assert kept == []


def test_cutoff_keeps_fused_node_when_any_leg_high():
    # RRF score is tiny, but the vector leg similarity is above cutoff → kept.
    pp = SimilarityCutoffPostprocessor(similarity_cutoff=0.7)
    kept = pp.postprocess_nodes([_fused_node(rrf=0.03, vector_score=0.9, bm25_score=0.4)])
    assert len(kept) == 1


def test_cutoff_uses_own_score_when_not_fused():
    pp = SimilarityCutoffPostprocessor(similarity_cutoff=0.7)
    good = NodeWithScore(node=_node(), score=0.8)
    bad = NodeWithScore(node=_node(node_id="n2"), score=0.2)
    kept = pp.postprocess_nodes([good, bad])
    assert [n.node.node_id for n in kept] == ["p1:n1"]


def test_cutoff_keeps_score_none_nodes():
    pp = SimilarityCutoffPostprocessor(similarity_cutoff=0.7)
    kept = pp.postprocess_nodes([NodeWithScore(node=_node(node_id="n2"), score=None)])
    assert len(kept) == 1


def test_cutoff_none_is_passthrough():
    pp = SimilarityCutoffPostprocessor(similarity_cutoff=None)
    nodes = [_fused_node(rrf=0.03, vector_score=0.1)]
    assert pp.postprocess_nodes(nodes) == nodes


# ── extract_sources / nodes_to_paper_results ─────────────────────────────────


def test_extract_sources_falls_back_to_node_id():
    node = TextNode(text="x", id_="p9:n9")
    rows = extract_sources([NodeWithScore(node=node, score=None)])
    assert rows[0]["paper_id"] == ""
    assert rows[0]["node_id"] == "p9:n9"
    assert rows[0]["title"] == ""
    assert rows[0]["score"] is None


def test_extract_sources_uses_metadata_sources():
    rows = extract_sources([NodeWithScore(node=_node(sources=["bm25", "vector"]), score=0.5)])
    assert rows[0]["sources"] == ["bm25", "vector"]


def test_nodes_to_paper_results_aggregates_and_ranks():
    nodes = [
        NodeWithScore(node=_node(node_id="n1", title="T1"), score=0.02),
        NodeWithScore(node=_node(node_id="n2", title="T1"), score=0.01),
        NodeWithScore(node=_node(node_id="n3", paper_id="p2", title="T2"), score=0.04),
    ]
    results = nodes_to_paper_results(nodes)
    assert [r["paper_id"] for r in results] == ["p2", "p1"]
    assert results[0]["score"] == pytest.approx(0.04)
    assert results[1]["score"] == pytest.approx(0.02)  # best score per paper
    assert results[1]["sections"] == 2
    assert results[1]["rank"] == 2
    assert results[0]["rank"] == 1


def test_nodes_to_paper_results_top_k():
    nodes = [
        NodeWithScore(node=_node(node_id="n1", paper_id="p1"), score=0.1),
        NodeWithScore(node=_node(node_id="n2", paper_id="p2"), score=0.2),
        NodeWithScore(node=_node(node_id="n3", paper_id="p3"), score=0.3),
    ]
    assert [r["paper_id"] for r in nodes_to_paper_results(nodes, top_k=2)] == ["p3", "p2"]


def test_nodes_to_paper_results_paper_id_from_node_id():
    node = TextNode(text="x", id_="p9:n9", metadata={"title": "T"})
    results = nodes_to_paper_results([NodeWithScore(node=node, score=0.5)])
    assert results[0]["paper_id"] == "p9"


# ── integration: real test-run corpus + real LLM ─────────────────────────────


@pytest.mark.integration
def test_integration_ask_llamaindex_real(tmp_path, monkeypatch):
    """End-to-end: build a small index from real test-run papers, then ask.

    Uses the opencode.ai ``deepseek-v4-flash`` test key from ``test-run/``
    (never hardcoded) and the real Qwen embedding model. Skipped unless the
    test-run config exists.
    """
    test_cfg_path = Path(__file__).resolve().parents[1] / "test-run" / "config.yaml"
    if not test_cfg_path.exists():
        pytest.skip("test-run/config.yaml (opencode test key) not present")
    # Explicit nonexistent local_path: skip the repo-root config.local.yaml
    # overlay so the opencode test key stays models[0].
    cfg = Config.from_yaml(
        str(test_cfg_path), local_path=test_cfg_path.parent / "config.local.yaml"
    )
    assert cfg.llm.models, "test-run config must define llm.models"
    assert "opencode" in (cfg.llm.models[0].get("base_url") or ""), (
        "expected opencode.ai test key as models[0]"
    )
    base = test_cfg_path.parent
    cfg.dirs.papers = str(base / cfg.dirs.papers)
    cfg.dirs.cache = str(tmp_path)
    # test-run/config.yaml predates the llamaindex section → enable explicitly.
    # rerank=False (T8): the rerank chain would otherwise try to load the
    # real reranker model (not cached) during this live query.
    cfg.llamaindex = LlamaIndexConfig(
        enabled=True,
        streaming=False,
        storage_dir=str(tmp_path / "llamaindex"),
        similarity_cutoff=None,  # cutoff semantics are covered by unit tests
        rerank=False,
    )

    # Build the index from the smallest test-run paper (3 tree nodes).
    paper_id = "10.1002_adma.202308655"
    from drbrain.rag.indexer import build_index

    class _PaperDB:
        def get_all_papers(self):
            return [{"local_id": paper_id}]

    stats = build_index(cfg, _PaperDB(), paper_ids=[paper_id])
    assert stats["nodes"] >= 1, f"index build produced no nodes: {stats}"

    # Live ask through the real engine.
    result = ask_llamaindex(
        cfg, _PaperDB(), "What are the key contributions of this paper?", top_k=3, streaming=False
    )
    assert isinstance(result, dict)
    assert result["answer"], "llamaindex ask returned an empty answer"
    assert result["sources"], "llamaindex ask returned no sources"
    assert result["engine"] == "llamaindex"
    assert all(s["paper_id"] for s in result["sources"])
