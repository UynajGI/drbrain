"""Offline regression cases for the RAG layer's backend-independent promises."""

from __future__ import annotations

import sqlite3

import pytest

from drbrain.config import Config, DBConfig, EmbedConfig, LlamaIndexConfig
from drbrain.rag import sql_retrie


@pytest.fixture
def sql_corpus(tmp_path, monkeypatch):
    path = tmp_path / "drbrain_rag.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE node_texts (
                node_key TEXT PRIMARY KEY, paper_id TEXT, node_id TEXT,
                text TEXT, content_hash TEXT);
            CREATE VIRTUAL TABLE node_texts_fts USING fts5(
                text, content='node_texts', content_rowid='rowid');
            CREATE TABLE tree_vectors (
                node_id TEXT PRIMARY KEY, paper_id TEXT, embedding BLOB,
                content_hash TEXT, tree_layer TEXT);
            CREATE TABLE paper_categories (paper_id TEXT PRIMARY KEY, categories TEXT);
            INSERT INTO node_texts VALUES ('a:1','a','1','first reference','hash-a');
            INSERT INTO node_texts VALUES ('b:1','b','1','second reference','hash-b');
            INSERT INTO node_texts_fts(node_texts_fts) VALUES ('rebuild');
            INSERT INTO paper_categories VALUES ('a','physics');
            INSERT INTO paper_categories VALUES ('b','biology');
            """
        )
    cfg = Config(
        db=DBConfig(path=str(tmp_path / "drbrain.db")),
        embed=EmbedConfig(provider="none"),
        llamaindex=LlamaIndexConfig(
            rag_engine="sql",
            retrievers=["bm25"],
            rerank=False,
            storage_dir=str(tmp_path / "indexes"),
        ),
    )
    monkeypatch.setattr(sql_retrie, "_default_rag_db", lambda _: path)
    return cfg, path


@pytest.mark.asyncio
async def test_sql_agent_tool_keeps_valid_json_and_pinned_generation(sql_corpus, monkeypatch):
    import json

    from drbrain.rag import agent_tools
    from drbrain.rag.indexer import build_index

    cfg, _ = sql_corpus
    generation = build_index(cfg, None)["generation"]
    rows = [{"text": "reference " * 1500, "generation": generation}]
    calls = []

    def retrieve(*args, **kwargs):
        calls.append(kwargs)
        return rows

    monkeypatch.setattr(agent_tools, "retrieve_documents", retrieve)
    tool = agent_tools._build_retrieval_tool(cfg, None, None, rag_generation=generation)
    assert tool is not None
    output = await tool.acall(query="reference")
    # Oversized observations are dropped at the row boundary so the tool
    # always returns a bounded, valid JSON payload.
    assert json.loads(output.content) == []
    assert len(output.content) <= 12000
    assert calls[0]["generation"] == generation


def test_sql_fingerprint_covers_nonfirst_content_and_vectors(sql_corpus):
    _, path = sql_corpus
    with sqlite3.connect(path) as conn:
        first = sql_retrie._generation_id(conn)
        conn.execute("UPDATE node_texts SET text='changed' WHERE paper_id='b'")
        second = sql_retrie._generation_id(conn)
        assert second != first
        conn.execute("INSERT INTO tree_vectors VALUES ('a:1','a',X'0000','v1','pageindex')")
        assert sql_retrie._generation_id(conn) != second


def test_sql_publish_keeps_previous_content_and_pins_configuration(sql_corpus):
    from drbrain.rag.agent import retrieve_documents
    from drbrain.rag.indexer import build_index, capture_index_generation, retain_index_generation

    cfg, path = sql_corpus
    first = build_index(cfg, None)["generation"]
    assert capture_index_generation(cfg) == first
    assert retain_index_generation(cfg, first, "run-1")
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE node_texts SET text='replacement' WHERE paper_id='b'")
        conn.execute("INSERT INTO node_texts_fts(node_texts_fts) VALUES ('rebuild')")
    cfg.embed.model = "different-model"
    second = build_index(cfg, None)["generation"]
    assert second != first
    # Restoring the original embedding configuration makes the old snapshot usable.
    cfg.embed.model = EmbedConfig().model
    old = retrieve_documents(cfg, None, None, "second", generation=first)
    assert old and old[0]["text"] == "second reference"
    assert old[0]["generation"] == first
    with pytest.raises((ValueError, RuntimeError)):
        retrieve_documents(cfg, None, None, "reference", generation="missing-generation")


def test_sql_pinned_requests_reject_live_sources(sql_corpus):
    from drbrain.rag.agent import retrieve_documents
    from drbrain.rag.indexer import build_index

    cfg, _ = sql_corpus
    generation = build_index(cfg, None)["generation"]
    cfg.llamaindex.retrievers = ["bm25", "claims"]
    with pytest.raises(ValueError, match="claims|mutable|live"):
        retrieve_documents(cfg, None, None, "reference", generation=generation)


def test_sql_filters_are_enforced_and_unknown_filters_rejected(sql_corpus):
    cfg, _ = sql_corpus
    rows = sql_retrie.retrieve_documents_sql(
        cfg, None, "reference", filters={"paper_ids": ["b"]}, top_k=1
    )
    assert [row["paper_id"] for row in rows] == ["b"]
    with pytest.raises(ValueError, match="filter"):
        sql_retrie.retrieve_documents_sql(cfg, None, "reference", filters={"imaginary": "x"})


def test_sql_reports_failure_instead_of_empty(sql_corpus, monkeypatch):
    from drbrain.rag.status import RetrievalError

    cfg, _ = sql_corpus
    cfg.llamaindex.retrievers = ["vector"]

    def fail(*args, **kwargs):
        raise TimeoutError("offline fixture")

    monkeypatch.setattr(sql_retrie, "_rerank_with_vectors", fail)
    with pytest.raises(RetrievalError):
        sql_retrie.retrieve_documents_sql(cfg, None, "reference")


def test_partial_sql_outage_is_visible_even_when_other_leg_is_empty(sql_corpus, monkeypatch):
    cfg, _ = sql_corpus
    cfg.llamaindex.retrievers = ["bm25", "vector"]
    monkeypatch.setattr(
        sql_retrie,
        "_rerank_with_vectors",
        lambda *args: (_ for _ in ()).throw(TimeoutError("offline fixture")),
    )
    rows = sql_retrie.retrieve_documents_sql(cfg, None, "absent")
    assert rows == []
    assert rows.result.status == "degraded"
    assert rows.result.legs[-1].reason == "timeout"


def test_failed_sql_publication_keeps_previous_generation(sql_corpus, monkeypatch):
    from drbrain.rag import sql_snapshot
    from drbrain.rag.indexer import build_index, capture_index_generation, get_index_health

    cfg, _ = sql_corpus
    cfg.llamaindex.enabled = True
    assert get_index_health(cfg)["ready"] is False
    generation = build_index(cfg, None)["generation"]
    assert get_index_health(cfg)["ready"] is True
    monkeypatch.setattr(
        sql_snapshot, "copy_snapshot", lambda *args: (_ for _ in ()).throw(OSError("copy failure"))
    )
    with pytest.raises(OSError):
        build_index(cfg, None)
    assert capture_index_generation(cfg) == generation


@pytest.mark.timeout(3)
def test_long_plain_text_redaction_preserves_text_and_redacts_secrets():
    from drbrain.security import redact_sensitive_text

    text = "filler " * 20000
    assert redact_sensitive_text(text) == text
    assert "topsecret" not in redact_sensitive_text(text + " api_key=topsecret")


def test_fragment_offsets_recover_exact_text():
    from llama_index.core.schema import Document

    from drbrain.rag.indexer import _chunk_document

    original = "Heading\n" + "paragraph words " * 50 + "\n\n" + "second part " * 30
    doc = Document(
        text=original,
        id_="paper:node",
        metadata={
            "title": "Heading",
            "paper_id": "paper",
            "node_id": "node",
            "line_start": 10,
            "line_end": 80,
        },
    )
    fragments = _chunk_document(doc, max_node_tokens=25)
    assert len(fragments) > 1
    for fragment in fragments:
        md = fragment.metadata
        assert md["parent_node_id"] == "node"
        assert fragment.text == original[md["char_start"] : md["char_end"]]
        assert md["node_id"] != "node"
        assert "line_start" not in md and "line_end" not in md
        assert md["parent_line_start"] == 10


@pytest.mark.parametrize("scores", [[float("nan"), 1.0], [float("inf"), 1.0], [None, 1.0]])
def test_invalid_rerank_scores_preserve_coarse_order(scores):
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

    from drbrain.rag.rerank import RerankPostprocessor

    class Reranker:
        available = True

        def rerank(self, query, passages):
            return scores

    nodes = [
        NodeWithScore(node=TextNode(text="first"), score=0.5),
        NodeWithScore(node=TextNode(text="second"), score=0.4),
    ]
    processor = RerankPostprocessor(top_k=2, reranker=Reranker())
    output = processor.postprocess_nodes(nodes, query_bundle=QueryBundle(query_str="question"))
    assert [(row.node.node_id, row.score) for row in output] == [
        (row.node.node_id, row.score) for row in nodes
    ]


@pytest.mark.parametrize("backend", ["sql", "llamaindex"])
def test_backends_share_scope_count_and_empty_contract(sql_corpus, monkeypatch, backend):
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, TextNode

    from drbrain.rag import fusion
    from drbrain.rag.contracts import RetrievalRequest
    from drbrain.rag.retrieval import retrieve

    cfg, _ = sql_corpus
    cfg.llamaindex.rag_engine = backend

    class LocalRetriever(BaseRetriever):
        def _retrieve(self, bundle):
            return [
                NodeWithScore(
                    node=TextNode(
                        text=f"{pid} reference",
                        id_=pid,
                        metadata={"paper_id": pid, "node_id": "1", "categories": category},
                    ),
                    score=score,
                )
                for pid, category, score in [("a", "physics", 1.0), ("b", "biology", 0.5)]
            ]

    monkeypatch.setattr(fusion, "get_retrievers", lambda *a, **kw: {"bm25": LocalRetriever()})
    generation = "g-fixture" if backend == "llamaindex" else None
    request = RetrievalRequest("reference", 1, generation, {"paper_ids": ["b"]})
    result = retrieve(cfg, None, None, request)
    assert result.status == "ok"
    assert [row["paper_id"] for row in result.records] == ["b"]
    empty = retrieve(
        cfg, None, None, RetrievalRequest("reference", 5, generation, {"paper_ids": []})
    )
    assert empty.records == [] and empty.status == "empty"


def test_core_retrieval_modules_do_not_import_agent_or_loop():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "drbrain" / "rag"
    for name in ("contracts.py", "retrieval.py", "index_generations.py", "eval_metrics.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        modules = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert not any(
            module.startswith(("drbrain.rag.agent", "drbrain.loop")) for module in modules
        )


def test_query_engine_adapter_uses_the_sql_snapshot(sql_corpus):
    from drbrain.rag.engine import build_hybrid_retriever
    from drbrain.rag.indexer import build_index

    cfg, _ = sql_corpus
    cfg.llamaindex.enabled = True
    generation = build_index(cfg, None)["generation"]
    retriever = build_hybrid_retriever(cfg, None, top_k=1)
    nodes = retriever.retrieve("reference")
    assert len(nodes) == 1
    assert nodes[0].node.metadata["generation"] == generation


def test_failed_index_build_restores_process_gc_state(tmp_path):
    import gc

    from drbrain.rag.indexer import build_index

    cfg = Config(
        llamaindex=LlamaIndexConfig(rag_engine="llamaindex", storage_dir=str(tmp_path / "index"))
    )
    before = gc.isenabled()
    gc.enable()
    try:
        with pytest.raises(AttributeError):
            build_index(cfg, None)
        assert gc.isenabled()
    finally:
        if before:
            gc.enable()
        else:
            gc.disable()


def test_similarity_cutoff_does_not_misinterpret_rrf_rank_scores():
    from llama_index.core.schema import NodeWithScore, TextNode

    from drbrain.rag.engine import SimilarityCutoffPostprocessor

    coarse = NodeWithScore(node=TextNode(text="coarse", metadata={"score_kind": "rrf"}), score=0.02)
    ranked = NodeWithScore(
        node=TextNode(text="ranked", metadata={"score_kind": "rerank"}), score=0.02
    )
    processor = SimilarityCutoffPostprocessor(similarity_cutoff=0.7)
    assert processor.postprocess_nodes([coarse, ranked]) == [coarse]
