"""Additive retrieval and rerank telemetry contracts."""

from __future__ import annotations

import importlib.util

import pytest

from drbrain.rag.fusion import FusionRetriever

_HAS_LLAMA_INDEX = importlib.util.find_spec("llama_index") is not None

if _HAS_LLAMA_INDEX:
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

    from drbrain.rag.rerank import RerankPostprocessor

pytestmark = pytest.mark.skipif(not _HAS_LLAMA_INDEX, reason="llama_index not installed")


class _Leg(BaseRetriever):
    def __init__(self, nodes=None, error: Exception | None = None) -> None:
        super().__init__()
        self._nodes = list(nodes or [])
        self._error = error

    def _retrieve(self, query_bundle):
        if self._error is not None:
            raise self._error
        return list(self._nodes)


class _Reranker:
    available = True

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        return [float(i) for i in range(len(passages))]


def _node(node_id: str, score: float) -> NodeWithScore:
    return NodeWithScore(node=TextNode(text=node_id, id_=node_id), score=score)


def test_fusion_trace_reports_each_leg_and_acl_drop():
    denied = _node("denied", 1.0)
    denied.node.metadata["tenant"] = "other"
    allowed = _node("allowed", 0.5)
    allowed.node.metadata["tenant"] = "acme"
    retriever = FusionRetriever(
        [_Leg([denied]), _Leg([allowed]), _Leg(error=TimeoutError("slow"))],
        sources=["vector", "bm25", "graph"],
        top_k=1,
        acl_filter={"tenant": "acme"},
    )

    result = retriever.retrieve(QueryBundle("query"))
    trace = retriever.get_last_trace()

    assert [node.node.node_id for node in result] == ["allowed"]
    assert [leg["source"] for leg in trace["legs"]] == ["vector", "bm25", "graph"]
    assert [leg["status"] for leg in trace["legs"]] == ["ok", "ok", "timeout"]
    assert trace["fusion"]["acl_filtered"] == 1
    assert trace["fusion"]["returned"] == 1
    assert all(isinstance(leg["duration_ms"], float) for leg in trace["legs"])


def test_rerank_trace_includes_counts_duration_and_status():
    processor = RerankPostprocessor(top_k=2, reranker=_Reranker())
    result = processor.postprocess_nodes([_node("one", 0.5), _node("two", 0.4)], QueryBundle("q"))

    trace = processor.get_last_trace()
    assert [item.node.node_id for item in result] == ["two", "one"]
    assert trace["status"] == "ok"
    assert trace["input_nodes"] == 2
    assert trace["candidates"] == 2
    assert trace["output_nodes"] == 2
    assert trace["duration_ms"] >= 0.0
