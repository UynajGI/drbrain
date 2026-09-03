"""SQL-native retrieval leg tests (hermetic: temp DB + monkeypatched embedding)."""

from __future__ import annotations

import sqlite3
import struct

import pytest

from drbrain.config import DBConfig, EmbedConfig, LlamaIndexConfig
from drbrain.rag import sql_retrie

DIM = 1024  # must match the production `length(embedding) = 4096` filter


def _vec(seed: float) -> bytes:
    return struct.pack(f"<{DIM}f", *([seed] * DIM))


@pytest.fixture()
def rag_db(tmp_path, monkeypatch):
    db_path = tmp_path / "drbrain_rag.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE node_texts (
            node_key TEXT PRIMARY KEY, paper_id TEXT NOT NULL, node_id TEXT NOT NULL,
            text TEXT NOT NULL, content_hash TEXT NOT NULL);
        CREATE VIRTUAL TABLE node_texts_fts USING fts5(
            text, content='node_texts', content_rowid='rowid');
        CREATE TABLE tree_vectors (
            node_id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, embedding BLOB NOT NULL,
            content_hash TEXT NOT NULL DEFAULT '', tree_layer TEXT NOT NULL DEFAULT '');
        CREATE TABLE tree_summaries (
            node_id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, summary_text TEXT NOT NULL DEFAULT '',
            source_node_ids TEXT NOT NULL DEFAULT '', tree_layer INTEGER NOT NULL DEFAULT 0);
        """
    )
    rows = [
        ("pA:0000", "pA", "0000", "kagome metal flat band ARPES evidence\nabstract body", "hA0"),
        ("pA:0001", "pA", "0001", "section on topological bands\nbody", "hA1"),
        ("pB:0000", "pB", "0000", "salt corrosion of steel pipelines\nbody", "hB0"),
    ]
    conn.executemany("INSERT INTO node_texts VALUES (?,?,?,?,?)", rows)
    conn.execute("INSERT INTO node_texts_fts(node_texts_fts) VALUES ('rebuild')")
    conn.executemany(
        "INSERT INTO tree_vectors VALUES (?,?,?,?,?)",
        [
            ("pA:0000", "pA", _vec(1.0), "hA0", "pageindex"),
            ("pA:0001", "pA", _vec(0.5), "hA1", "pageindex"),
            ("raptor_pA_L1_x", "pA", _vec(0.9), "hR", "raptor_L1"),
        ],
    )
    conn.executemany(
        "INSERT INTO tree_summaries VALUES (?,?,?,?,?)",
        [("raptor_pA_L1_x", "pA", "kagome flat band summary text", "", 1)],
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(sql_retrie, "_default_rag_db", lambda cfg: db_path)
    return db_path


@pytest.fixture()
def cfg():
    return _simple_namespace_config()


def _simple_namespace_config():
    from types import SimpleNamespace

    return SimpleNamespace(
        db=DBConfig(path="dummy.sqlite"),
        embed=EmbedConfig(device="cpu"),
        llamaindex=LlamaIndexConfig(rag_engine="sql", retrievers=["bm25", "vector"], rerank=False),
    )


def _patch_embed(monkeypatch, qvec):
    import drbrain.services.embedding as emb

    monkeypatch.setattr(emb, "_embed_batch", lambda texts, embed_cfg: [list(qvec)] * len(texts))


def test_fts_query_quoting():
    assert sql_retrie._fts_query("flat band, MoS2!") == '"flat" OR "band" OR "MoS2"'
    assert sql_retrie._fts_query("？？？") is None


def test_fuse_prefers_double_hits():
    a = [("x", 1.0), ("y", 0.9)]
    b = [("y", 1.0), ("z", 0.8)]
    fused = sql_retrie._fuse([a, b])
    assert fused[0][0] == "y"  # double hit outranks single hits


def test_retrieve_sql_two_legs(rag_db, cfg, monkeypatch):
    _patch_embed(monkeypatch, [1.0] * DIM)
    rows = sql_retrie.retrieve_documents_sql(cfg, None, "kagome flat band", top_k=2)
    assert rows, "expected results"
    assert rows[0]["paper_id"] == "pA"
    assert {row["paper_id"] for row in rows} <= {"pA", "pB"}
    vector_seen = any("vector" in row["legs"] for row in rows)
    assert vector_seen, "vector leg should contribute with patched embeddings"
    for row in rows:
        assert row["source"] == "sql-fusion"
        assert "evidence_id" in row  # evidence chain compatibility
        assert row["text"]


def test_diversity_guarantee_appends_leg_best(rag_db, cfg, monkeypatch):
    _patch_embed(monkeypatch, [1.0] * DIM)
    cfg.llamaindex.retrievers = ["bm25", "vector", "raptor"]
    rows = sql_retrie.retrieve_documents_sql(cfg, None, "kagome flat band", top_k=1)
    assert len(rows) == 2  # head(1) + raptor guarantee
    raptor_rows = [r for r in rows if "raptor" in r["legs"]]
    assert raptor_rows, "raptor leg must surface via guarantee"
    assert raptor_rows[0]["node_id"].startswith("raptor_")
    assert "summary" in raptor_rows[0]["text"]  # text resolved from tree_summaries


def test_graph_leg_rows(rag_db, cfg, monkeypatch):
    import drbrain.extractor.agent_tools as tools

    monkeypatch.setattr(
        tools,
        "search_concepts",
        lambda db, q, limit=5: [{"label": "flat band", "type": "Property", "score": 0.9}],
    )
    monkeypatch.setattr(tools, "get_neighbors", lambda graph, label, hops=1, direction="both": [])
    cfg.llamaindex.retrievers = ["bm25", "vector", "graph"]
    _patch_embed(monkeypatch, [1.0] * DIM)

    class FakeDB:  # concepts lookup row
        class conn:  # noqa: N801
            @staticmethod
            def execute(sql, params=()):
                if "FROM concepts" in sql:
                    return [("pA", "Property", "flat band", 0.9, "sec-1")]
                raise AssertionError(f"unexpected query: {sql}")

    rows = sql_retrie.retrieve_documents_sql(cfg, FakeDB, "kagome flat band", top_k=3)
    graph_rows = [r for r in rows if "graph" in r["legs"]]
    assert graph_rows, "graph leg must surface"
    assert graph_rows[0]["node_id"] == "concept:flat band"
    assert "Property" in graph_rows[0]["text"]


def test_reranker_cache(monkeypatch, cfg):
    cfg.llamaindex.rerank = True
    cfg.llamaindex.rerank_model = "Qwen/Qwen3-Reranker-0.6B"
    calls = []
    import drbrain.rag.rerank as rr

    monkeypatch.setattr(rr, "build_reranker", lambda c: calls.append(1) or object())
    sql_retrie._RERANKER_CACHE.clear()
    assert sql_retrie._get_reranker(cfg) is not None
    assert sql_retrie._get_reranker(cfg) is not None
    assert len(calls) == 1  # cached, built once
    sql_retrie._RERANKER_CACHE.clear()


def test_generation_id_stable(rag_db):
    conn = sqlite3.connect(rag_db)
    g1 = sql_retrie._generation_id(conn)
    g2 = sql_retrie._generation_id(conn)
    conn.close()
    assert g1 == g2 and g1.startswith("sql-")
