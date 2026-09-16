"""T46: ask is a read-only, route-selectable query over prepared indexes."""

from __future__ import annotations

import json
import sqlite3
import struct
from types import SimpleNamespace
from unittest import mock

import pytest
import typer

from drbrain.config import Config, EmbedConfig, LlamaIndexConfig
from drbrain.rag.engine import (
    AskIndexNotPreparedError,
    _route_info,
    _route_with_status,
    ask_llamaindex,
)

DIM = 8  # matches the fixture vectors below


def _cfg(*, legs=("bm25", "vector"), engine="sql", storage_dir="", tree_candidates=100):
    return Config(
        llamaindex=LlamaIndexConfig(
            rag_engine=engine,
            retrievers=list(legs),
            enabled=True,
            storage_dir=storage_dir or "data/llamaindex",
            tree_candidates=tree_candidates,
        ),
        embed=EmbedConfig(dim=DIM, model="fake"),
    )


def _vec(seed: float) -> bytes:
    return struct.pack(f"<{DIM}f", *([seed] * DIM))


def _rag_fixture_db(tmp_path):
    path = tmp_path / "drbrain_rag.db"
    conn = sqlite3.connect(path)
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
        CREATE TABLE paper_categories (paper_id TEXT PRIMARY KEY, categories TEXT NOT NULL DEFAULT '');
        """
    )
    conn.executemany(
        "INSERT INTO node_texts VALUES (?,?,?,?,?)",
        [
            ("pA:0000", "pA", "0000", "kagome metal flat band ARPES evidence", "hA0"),
            ("pA:0001", "pA", "0001", "section on topological bands", "hA1"),
        ],
    )
    conn.execute("INSERT INTO node_texts_fts(node_texts_fts) VALUES ('rebuild')")
    conn.execute("INSERT INTO paper_categories VALUES ('pA', 'cond-mat.str-el')")
    conn.commit()
    conn.close()
    return path


class TestRouteSelection:
    def test_legacy_names_fold_and_the_override_takes_effect(self):
        cfg = _cfg(legs=["bm25", "vector"])
        routed, route = _route_info(cfg, ["pageindex", "raptor"])
        assert route["requested"] == ["pageindex", "raptor"]
        assert route["legs"] == ["tree"]
        assert route["notes"]
        assert routed.llamaindex.retrievers == ["tree"]
        # The original cfg is untouched.
        assert cfg.llamaindex.retrievers == ["bm25", "vector"]

    def test_without_override_the_configured_route_is_reported(self):
        cfg = _cfg(legs=["tree"])
        routed, route = _route_info(cfg, None)
        assert route["requested"] == ["tree"] and route["legs"] == ["tree"]
        assert routed is cfg

    def test_route_status_merges_the_leg_trace(self):
        route = _route_with_status(
            {"legs": ["tree"]},
            {"retrieval": {"legs": [{"source": "tree", "status": "ok"}]}},
        )
        assert route["leg_status"] == {"tree": "ok"}

    def test_conflicting_legacy_names_are_refused(self):
        from drbrain.rag.legs import LegConfigError

        cfg = _cfg()
        with pytest.raises(LegConfigError):
            _route_info(cfg, ["tree", "pageindex"])


class TestNotPrepared:
    def test_ask_raises_the_typed_not_prepared_error(self, tmp_path):
        cfg = _cfg(engine="sql", storage_dir=str(tmp_path / "llamaindex"))
        with pytest.raises(AskIndexNotPreparedError) as excinfo:
            ask_llamaindex(cfg, db=None, question="q", streaming=False)
        assert excinfo.value.hint == "drbrain index build"
        assert excinfo.value.engine == "sql"

    def test_ask_cli_reports_source_unavailable_without_building(self, tmp_path):
        import drbrain.rag.pageindex_native as native
        import drbrain.services.index_model as index_model
        from drbrain.cli.analysis_commands import ask_cmd

        cfg = _cfg(engine="sql", storage_dir=str(tmp_path / "llamaindex"))
        cfg.db.path = str(tmp_path / "db.sqlite")
        ctx = mock.MagicMock(spec=typer.Context)
        ctx.obj = {"config": cfg}
        captured: list[str] = []
        with (
            mock.patch.object(
                native, "ensure_document", side_effect=AssertionError("pageindex build called")
            ),
            mock.patch.object(
                index_model.IndexModel, "call_text", side_effect=AssertionError("IndexModel called")
            ),
            mock.patch("typer.echo", side_effect=lambda m="", *a, **k: captured.append(str(m))),
            pytest.raises(typer.Exit) as excinfo,
        ):
            ask_cmd(
                ctx,
                ["what", "about", "flat", "bands"],
                top_k=5,
                legs="",
                json_output=True,
                pageindex_native=False,
                pageindex_paper=None,
            )
        assert excinfo.value.exit_code == 1
        payload = json.loads(captured[0])
        assert payload["status"] == "source_unavailable"
        assert payload["hint"] == "drbrain index build"
        assert payload["sources"] == [] and payload["evidence_ids"] == []


class TestTreeOnly:
    def _run(self, tmp_path, monkeypatch, *, legs, tree_candidates=7):
        from drbrain.rag import sql_retrie
        from drbrain.tree.leg import TreeLegHit, TreeLegOutcome

        db_path = _rag_fixture_db(tmp_path)
        calls: dict = {}

        def fake_run_tree_leg(cfg_arg, *, query, top_k, verify=None, **kwargs):
            # The unified leg is exercised end to end in tests/tree/test_tree_leg.py;
            # here only the route/cap wiring is under test.
            calls["top_k"] = top_k
            hits = [
                TreeLegHit(
                    node_id="0000",
                    local_id="pA",
                    node_revision=1,
                    content_hash="hA0",
                    block_id="b0",
                    char_start=0,
                    char_end=5,
                    tokens=3,
                    score=0.9,
                    text="kagome leaf text",
                )
            ]
            if verify is not None:
                hits = [hit for hit in hits if verify(hit)]
            return TreeLegOutcome(
                status="ok" if hits else "unavailable",
                generation="gen-test",
                hits=hits,
            )

        monkeypatch.setattr(
            "drbrain.rag.sql_snapshot.resolve_sql_snapshot", lambda cfg, generation: db_path
        )
        monkeypatch.setattr("drbrain.tree.leg.run_tree_leg", fake_run_tree_leg)
        cfg = _cfg(legs=legs, tree_candidates=tree_candidates)
        rows = sql_retrie.retrieve_documents_sql(
            cfg, None, "kagome flat band", generation="g-test", top_k=2
        )
        return rows, calls

    def test_tree_only_route_queries_the_prepared_index(self, tmp_path, monkeypatch):
        rows, calls = self._run(tmp_path, monkeypatch, legs=["tree"])
        assert rows, "tree-only retrieval must return rows"
        assert all(row["legs"] == ["tree"] for row in rows)
        assert all(row["evidence_id"] for row in rows)
        assert rows[0]["node_id"] == "0000" and rows[0]["paper_id"] == "pA"

    def test_tree_expansion_respects_the_configured_cap(self, tmp_path, monkeypatch):
        _, calls = self._run(tmp_path, monkeypatch, legs=["pageindex"], tree_candidates=5)
        assert calls["top_k"] == 5

    def test_legacy_pageindex_name_uses_the_tree_leg(self, tmp_path, monkeypatch):
        rows, _ = self._run(tmp_path, monkeypatch, legs=["pageindex"])
        assert rows and all("tree" in row["legs"] for row in rows)


class TestEmptyAnswer:
    def test_empty_synthesis_is_reported_not_successful(self, monkeypatch):
        from llama_index.core.schema import NodeWithScore, TextNode

        node = NodeWithScore(
            node=TextNode(
                text="evidence text",
                id_="pA:ev1",
                metadata={"paper_id": "pA", "node_id": "0000", "evidence_id": "pA:ev1"},
            ),
            score=1.0,
        )
        response = mock.MagicMock()
        response.source_nodes = [node]
        engine = mock.MagicMock()
        engine.query.return_value = response
        engine._drbrain_observability = {
            "fusion": mock.MagicMock(
                get_last_trace=lambda: {"legs": [{"source": "tree", "status": "ok"}]}
            )
        }
        response.response = ""
        monkeypatch.setattr("drbrain.rag.engine.build_query_engine", lambda *a, **k: engine)
        monkeypatch.setattr("drbrain.rag.engine._response_text", lambda r: "")
        monkeypatch.setattr("drbrain.rag.engine._last_finish_reason", lambda: "length")
        result = ask_llamaindex(_cfg(legs=["tree"]), db=None, question="q", streaming=False)
        assert result["status"] == "empty_answer"
        assert result["answer_truncated"] is True
        assert result["route"]["leg_status"] == {"tree": "ok"}

    def test_answer_reports_route_and_truncation_flag(self, monkeypatch):
        from llama_index.core.schema import NodeWithScore, TextNode

        node = NodeWithScore(
            node=TextNode(
                text="evidence text",
                id_="pA:ev1",
                metadata={"paper_id": "pA", "node_id": "0000", "evidence_id": "pA:ev1"},
            ),
            score=1.0,
        )
        response = mock.MagicMock()
        response.source_nodes = [node]
        response.response = "the answer"
        engine = mock.MagicMock()
        engine.query.return_value = response
        engine._drbrain_observability = {
            "fusion": mock.MagicMock(
                get_last_trace=lambda: {"legs": [{"source": "tree", "status": "ok"}]}
            )
        }
        monkeypatch.setattr("drbrain.rag.engine.build_query_engine", lambda *a, **k: engine)
        monkeypatch.setattr("drbrain.rag.engine._response_text", lambda r: "the answer")
        monkeypatch.setattr("drbrain.rag.engine._last_finish_reason", lambda: "")
        result = ask_llamaindex(_cfg(legs=["tree"]), db=None, question="q", streaming=False)
        assert result["answer"] == "the answer"
        assert result["answer_truncated"] is False
        assert result["route"]["legs"] == ["tree"]
        assert result["engine"] == "llamaindex"


class TestChatTruncation:
    def test_drbrain_llm_records_finish_reason(self, monkeypatch):
        pytest.importorskip("llama_index.core")
        from drbrain.rag.llm import DrbrainLLM

        cfg = Config()
        cfg.llm.models = [{"provider": "openai", "model": "m", "api_key": "k"}]
        llm = DrbrainLLM(cfg)
        assert llm.last_finish_reason == ""
        monkeypatch.setattr(
            "drbrain.extractor.llm_client.call_text_with_meta",
            lambda prompt, models, max_tokens=4096: {"text": "ok", "finish_reason": "length"},
        )
        with mock.patch.object(llm, "_completion", lambda text: SimpleNamespace(text=text)):
            llm.complete("prompt")
        assert llm.last_finish_reason == "length"
