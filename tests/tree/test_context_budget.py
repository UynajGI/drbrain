"""T44: rank-only fusion, bounded rerank head, and a real context budget."""

from __future__ import annotations

import pytest

from drbrain.rag.context import (
    DEFAULT_MAX_DOCS,
    ContextBudgetPostprocessor,
    estimate_tokens,
    select_context,
)
from drbrain.rag.fusion import FusionRetriever
from drbrain.rag.rerank import RERANK_HEAD_MAX, RERANK_HEAD_MIN, clamp_rerank_head


def _doc(node_id: str, paper_id: str = "p1", chars: int = 8) -> dict:
    text = (f"{node_id} " + "x" * chars)[:chars]  # unique per node id
    return {"node_id": node_id, "paper_id": paper_id, "text": text, "score": 0.5}


class TestRankOnlyFusion:
    def test_fusion_uses_ranks_not_score_magnitudes(self):
        pytest.importorskip("llama_index.core.schema")
        from llama_index.core.schema import NodeWithScore, TextNode

        def leg(nodes):
            class _Static:
                def retrieve(self, query_bundle):
                    return [
                        NodeWithScore(
                            node=TextNode(
                                text=f"text {node_id}",
                                id_=f"p1:{node_id}",
                                metadata={"paper_id": "p1", "node_id": node_id},
                            ),
                            score=score,
                        )
                        for node_id, score in nodes
                    ]

            return _Static()

        small = FusionRetriever(
            [leg([("a", 0.9), ("b", 0.1)]), leg([("b", 0.0002), ("a", 0.0001)])],
            sources=["bm25", "vector"],
            top_k=5,
        ).retrieve("q")
        huge = FusionRetriever(
            [leg([("a", 1000.0), ("b", 1.0)]), leg([("b", 5e6), ("a", 1.0)])],
            sources=["bm25", "vector"],
            top_k=5,
        ).retrieve("q")
        assert [n.node.node_id for n in small] == [n.node.node_id for n in huge]
        assert [n.score for n in small] == [n.score for n in huge]

    def test_rrf_mode_ignores_retriever_weights(self):
        pytest.importorskip("llama_index.core.schema")
        from llama_index.core.schema import NodeWithScore, TextNode

        def leg(nodes):
            class _Static:
                def retrieve(self, query_bundle):
                    return [
                        NodeWithScore(
                            node=TextNode(
                                text=f"text {node_id}",
                                id_=f"p1:{node_id}",
                                metadata={"paper_id": "p1", "node_id": node_id},
                            ),
                            score=score,
                        )
                        for node_id, score in nodes
                    ]

            return _Static()

        plain = FusionRetriever(
            [leg([("a", 0.9), ("b", 0.8)])], sources=["tree"], top_k=5
        ).retrieve("q")
        weighted = FusionRetriever(
            [leg([("a", 0.9), ("b", 0.8)])], sources=["tree"], top_k=5, weights={"tree": 99.0}
        ).retrieve("q")
        assert [n.node.node_id for n in plain] == [n.node.node_id for n in weighted]
        assert [n.score for n in plain] == [n.score for n in weighted]


class TestRerankHead:
    def test_head_is_bounded_to_20_50(self):
        assert clamp_rerank_head(5) == RERANK_HEAD_MIN
        assert clamp_rerank_head(80) == RERANK_HEAD_MAX
        assert clamp_rerank_head(25) == 25
        assert clamp_rerank_head(None) == RERANK_HEAD_MIN
        assert clamp_rerank_head("not-a-number") == RERANK_HEAD_MIN

    def test_negative_logit_is_not_a_cosine_below_cutoff(self):
        pytest.importorskip("llama_index.core.schema")
        from llama_index.core.schema import NodeWithScore, TextNode

        from drbrain.rag.engine import SimilarityCutoffPostprocessor

        def node(node_id: str, logit: float, contribution: float | None):
            metadata = {"paper_id": "p1", "node_id": node_id, "score_kind": "rerank"}
            if contribution is not None:
                metadata["contributions"] = {"vector": {"rank": 1, "score": contribution}}
            return NodeWithScore(
                node=TextNode(text=f"body {node_id}", metadata=metadata), score=logit
            )

        processor = SimilarityCutoffPostprocessor(similarity_cutoff=0.7)
        # A good hit reranked to a negative logit must survive: the cutoff
        # evaluates the original coarse similarity, not the logit.
        strong = node("strong", -4.0, contribution=0.9)
        weak = node("weak", 9.0, contribution=0.3)  # high logit, weak coarse match
        untagged = node("untagged", -1.0, contribution=None)  # logit is the only score
        kept = processor.postprocess_nodes([strong, weak, untagged])
        assert [n.node.metadata["node_id"] for n in kept] == ["strong", "untagged"]


class TestSelectContext:
    def test_estimate_tokens(self):
        assert estimate_tokens("") == 1
        assert estimate_tokens("abcd") == 1
        assert estimate_tokens("a" * 8) == 2

    def test_returned_documents_match_the_token_budget(self):
        docs = [_doc(f"n{i}") for i in range(5)]  # 2 tokens each
        selection = select_context(docs, token_budget=5, max_docs=10)
        assert len(selection.items) == 2
        assert selection.tokens_used == 4 <= selection.token_budget
        assert selection.tokens_used == sum(item.tokens for item in selection.items)
        assert selection.dropped == 3

    def test_max_docs_caps_the_returned_count(self):
        docs = [_doc(f"n{i}", chars=8) for i in range(12)]
        assert len(select_context(docs, token_budget=8000, max_docs=8).items) == 8
        assert len(select_context(docs, token_budget=8000, max_docs=10).items) == 10

    def test_duplicate_provenance_is_deduplicated(self):
        docs = [
            _doc("n1"),
            _doc("n1"),  # same node id
            {"node_id": "n2", "paper_id": "p1", "text": "same body text", "score": 0.4},
            {"node_id": "n3", "paper_id": "p1", "text": "same body text", "score": 0.3},
        ]
        selection = select_context(docs, token_budget=8000)
        assert [item.node_id for item in selection.items] == ["n1", "n2"]
        assert selection.deduplicated == 2

    def test_selection_spreads_across_papers(self):
        docs = [_doc("p1a", "p1"), _doc("p1b", "p1"), _doc("p2a", "p2"), _doc("p3a", "p3")]
        selection = select_context(docs, token_budget=6)  # exactly three 2-token docs
        assert [item.paper_id for item in selection.items] == ["p1", "p2", "p3"]

    def test_oversized_head_document_is_truncated_to_budget(self):
        docs = [_doc("big", chars=400)]  # 100 tokens
        selection = select_context(docs, token_budget=10)
        assert len(selection.items) == 1
        assert selection.items[0].truncated and selection.truncated
        assert selection.tokens_used == 10 <= selection.token_budget
        assert len(selection.items[0].text) == 40

    def test_oversized_non_head_documents_are_dropped(self):
        docs = [_doc("big", chars=400), _doc("small", chars=8)]
        selection = select_context(docs, token_budget=10)
        assert [item.node_id for item in selection.items] == ["small"]
        assert not selection.truncated

    def test_to_json_reports_budget_and_counts(self):
        selection = select_context([_doc("n1")], token_budget=100)
        payload = selection.to_json()
        assert payload["count"] == 1
        assert payload["tokens_used"] == 2 <= payload["token_budget"] == 100
        assert payload["items"][0]["node_id"] == "n1"


class TestContextBudgetPostprocessor:
    def _nodes(self, count: int, chars: int = 8):
        pytest.importorskip("llama_index.core.schema")
        from llama_index.core.schema import NodeWithScore, TextNode

        return [
            NodeWithScore(
                node=TextNode(
                    text=(f"n{i} " + "x" * chars)[:chars],
                    id_=f"p1:n{i}",
                    metadata={"paper_id": "p1", "node_id": f"n{i}"},
                ),
                score=1.0 - i * 0.01,
            )
            for i in range(count)
        ]

    def test_postprocessor_applies_the_budget(self):
        nodes = self._nodes(12)
        processor = ContextBudgetPostprocessor(max_docs=10, token_budget=8000)
        out = processor.postprocess_nodes(nodes)
        assert len(out) == min(10, DEFAULT_MAX_DOCS)
        assert all("context_tokens" in n.node.metadata for n in out)
        trace = processor.get_last_trace()
        assert trace["selected"] == len(out) and trace["input_nodes"] == 12

    def test_postprocessor_flags_truncation(self):
        nodes = self._nodes(1, chars=400)
        processor = ContextBudgetPostprocessor(max_docs=10, token_budget=10)
        out = processor.postprocess_nodes(nodes)
        assert len(out) == 1
        assert out[0].node.metadata["context_truncated"] is True
        assert len(out[0].node.text) == 40
