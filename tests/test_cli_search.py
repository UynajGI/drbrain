"""CLI contract tests for ``drbrain search`` and the compatibility aliases.

The redesign makes ``search`` the evidence-retrieval entry point over the same
chain as ``ask`` (no answer synthesis), moves the historical BM25 command to
``library search``, and keeps ``query``/``hybrid``/``fsearch`` callable as
hidden aliases with one migration line on stderr.

These tests pin: the JSON contract (sources, text locators, route, index
version), ``--paper`` scoping down to the tree leg, ``--source`` external rows,
the missing-index remedy, and the untouched legacy JSON shapes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest import mock

import yaml
from typer.testing import CliRunner

from drbrain.cli.main import app
from drbrain.storage.database import Database

runner = CliRunner()

# Usage errors may carry ANSI styling (some CI environments force colours), so
# text assertions must run on the plain rendering.
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)


def _write_config(tmp_path: Path, *, retrievers=None) -> None:
    config = {
        "db": {"path": "data/drbrain.db"},
        "dirs": {
            "inbox": "data/spool/inbox",
            "papers": "data/papers",
            "logs": "data/logs",
        },
        "llm": {"models": []},
        "llamaindex": {
            "enabled": True,
            "rag_engine": "sql",
            "tree_storage": "data/tree",
            "retrievers": retrievers or ["bm25", "vector", "pageindex", "raptor"],
        },
        "embed": {"provider": "local", "model": "fake-embed", "dim": 3},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _invoke(tmp_path: Path, *args: str):
    return runner.invoke(app, ["--root", str(tmp_path), *args])


def _evidence_rows(generation: str = "gen-test") -> object:
    """A real ``RetrievalRows`` built through the evidence contract."""
    from drbrain.rag.contracts import LegResult, finish_retrieval
    from drbrain.rag.evidence import build_evidence_record

    row = {
        "paper_id": "p1",
        "node_id": "0000",
        "title": "Flat bands",
        "text": "kagome flat band evidence",
        "source": "unified-tree",
        "score": 0.9,
        "block_id": "cb-1",
        "char_start": 0,
        "char_end": 25,
    }
    row.update(
        build_evidence_record(
            generation=generation,
            query="kagome",
            retriever="unified-tree",
            rank=1,
            score=0.9,
            source={**row, "text": row["text"]},
            excerpt=row["text"],
        )
    )
    return finish_retrieval(
        [row],
        generation=generation,
        legs=[LegResult("tree", "ok", 1, 1.5, "")],
        capabilities={"backend": "unified"},
    )


class TestSearchContract:
    def test_json_carries_route_generation_and_evidence(self, tmp_path):
        _write_config(tmp_path)
        captured: dict = {}

        def fake_retrieve(
            cfg, db, graph, query, *, generation=None, filters=None, top_k=5, acl_filter=None
        ):
            captured.update(query=query, filters=filters, top_k=top_k)
            return _evidence_rows()

        with mock.patch("drbrain.rag.retrieval.retrieve_documents", side_effect=fake_retrieve):
            result = _invoke(tmp_path, "search", "kagome flat band", "--json")

        assert result.exit_code == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["query"] == "kagome flat band"
        assert payload["engine"] == "sql"
        assert payload["route"]["legs"] == ["bm25", "vector", "tree"]
        assert payload["route"]["extras"] == []
        assert payload["generations"]["result"] == "gen-test"
        assert "tree" in payload["generations"] and "sql" in payload["generations"]
        assert payload["legs"] == [
            {"source": "tree", "status": "ok", "count": 1, "duration_ms": 1.5, "reason": ""}
        ]
        row = payload["evidence"][0]
        for key in (
            "evidence_id",
            "paper_id",
            "node_id",
            "title",
            "text",
            "score",
            "source",
            "content_checksum",
            "block_id",
            "char_start",
            "char_end",
        ):
            assert key in row, key
        assert "answer" not in payload
        assert captured["query"] == "kagome flat band"
        assert captured["filters"] is None

    def test_paper_scope_is_passed_to_retrieval(self, tmp_path):
        _write_config(tmp_path)
        captured: dict = {}

        def fake_retrieve(cfg, db, graph, query, *, filters=None, **kwargs):
            captured["filters"] = filters
            return _evidence_rows()

        with mock.patch("drbrain.rag.retrieval.retrieve_documents", side_effect=fake_retrieve):
            result = _invoke(tmp_path, "search", "q", "--paper", "pA", "--paper", "pB", "--json")

        assert result.exit_code == 0, result.stderr
        assert captured["filters"] == {"paper_ids": ["pA", "pB"]}

    def test_plain_output_names_sources_and_locators(self, tmp_path):
        _write_config(tmp_path)
        with mock.patch(
            "drbrain.rag.retrieval.retrieve_documents",
            side_effect=lambda *a, **k: _evidence_rows(),
        ):
            result = _invoke(tmp_path, "search", "kagome")

        assert result.exit_code == 0, result.stderr
        assert 'Search: "kagome"' in result.stdout
        assert "route: bm25,vector,tree" in result.stdout
        assert "generation: gen-test" in result.stdout
        assert "block_id=cb-1" in result.stdout

    def test_missing_index_reports_the_index_build_remedy(self, tmp_path):
        _write_config(tmp_path)
        from drbrain.rag.status import RetrievalUnavailableError

        with mock.patch(
            "drbrain.rag.retrieval.retrieve_documents",
            side_effect=RetrievalUnavailableError("no active unified tree generation"),
        ):
            result = _invoke(tmp_path, "search", "kagome", "--json")

        assert result.exit_code == 1
        assert "drbrain index build" in result.stderr
        payload = json.loads(result.stdout)
        assert payload["evidence"] == []
        assert payload["legs"][0]["status"] == "unavailable"
        assert "no active unified tree generation" in payload["legs"][0]["reason"]

    def test_search_never_calls_the_answer_synthesizer(self, tmp_path):
        _write_config(tmp_path)
        with (
            mock.patch(
                "drbrain.rag.retrieval.retrieve_documents",
                side_effect=lambda *a, **k: _evidence_rows(),
            ),
            mock.patch(
                "drbrain.rag.engine.build_query_engine",
                side_effect=AssertionError("search must not build a query engine"),
            ),
            mock.patch(
                "drbrain.cli.analysis_commands.ask_cmd",
                side_effect=AssertionError("search must not invoke ask"),
            ),
        ):
            result = _invoke(tmp_path, "search", "kagome", "--json")

        assert result.exit_code == 0, result.stderr
        assert json.loads(result.stdout)["evidence"]

    def test_unknown_source_is_rejected(self, tmp_path):
        _write_config(tmp_path)
        result = _invoke(tmp_path, "search", "q", "--source", "web")
        assert result.exit_code == 2
        assert "--source" in _plain(result.output)


class TestSearchExternalSource:
    def test_arxiv_rows_carry_no_local_locator(self, tmp_path):
        _write_config(tmp_path)
        arxiv = [
            {
                "title": "Kagome paper",
                "authors": ["A. Author"],
                "year": 2024,
                "doi": "10.1/x",
                "arxiv_id": "2401.00001",
                "summary": "abstract text",
                "published": "2024-01-01",
            }
        ]
        with mock.patch("drbrain.services.fsearch.search_arxiv", return_value=arxiv):
            result = _invoke(tmp_path, "search", "kagome", "--source", "arxiv", "--json")

        assert result.exit_code == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["evidence"][0]["source"] == "arxiv"
        assert payload["evidence"][0]["url"] == "https://arxiv.org/abs/2401.00001"
        assert payload["evidence"][0]["doi"] == "10.1/x"
        assert payload["evidence"][0]["arxiv_id"] == "2401.00001"
        assert "paper_id" not in payload["evidence"][0]
        assert "node_id" not in payload["evidence"][0]
        assert payload["legs"][0]["source"] == "arxiv"

    def test_all_merges_local_and_external_rows(self, tmp_path):
        _write_config(tmp_path)
        arxiv = [
            {
                "title": "External",
                "authors": [],
                "year": 2024,
                "doi": "",
                "arxiv_id": "2401.1",
                "summary": "s",
                "published": "2024-01-01",
            }
        ]
        with (
            mock.patch(
                "drbrain.rag.retrieval.retrieve_documents",
                side_effect=lambda *a, **k: _evidence_rows(),
            ),
            mock.patch("drbrain.services.fsearch.search_arxiv", return_value=arxiv),
        ):
            result = _invoke(tmp_path, "search", "kagome", "--source", "all", "--json")

        assert result.exit_code == 0, result.stderr
        payload = json.loads(result.stdout)
        assert {row["source"] for row in payload["evidence"]} == {"unified-tree", "arxiv"}
        assert {leg["source"] for leg in payload["legs"]} == {"tree", "arxiv"}


class TestCompatibilityAliases:
    def test_library_search_keeps_the_historical_json_shape(self, tmp_path):
        _write_config(tmp_path)
        db = Database(tmp_path / "data" / "drbrain.db")
        try:
            db.insert_paper("p1", "Paper", 2024, "uploaded")
            db.insert_concept("p1", "Method", "transformer", 0.9, year=2024)
            db.commit()
        finally:
            db.close()

        result = _invoke(tmp_path, "library", "search", "transformer", "--json")

        assert result.exit_code == 0, result.stderr
        rows = json.loads(result.stdout)
        assert isinstance(rows, list)
        assert "transformer" in [row["label"] for row in rows]

    def test_legacy_search_symbol_stays_importable(self):
        from drbrain.cli.query_commands import search_cmd  # noqa: F401

    def test_hidden_aliases_print_one_migration_line(self, tmp_path):
        _write_config(tmp_path)
        for alias, hint in (
            ("query", "drbrain search"),
            ("hybrid", "drbrain search"),
        ):
            result = _invoke(tmp_path, alias, "nonexistent")
            assert result.exit_code == 1, alias
            assert f"[drbrain] '{alias}' has moved: use '{hint}'" in result.stderr
            # The legacy failure report still reaches the caller unchanged.
            assert f"[{alias}]" in result.output

    def test_new_search_prints_no_migration_line(self, tmp_path):
        _write_config(tmp_path)
        with mock.patch(
            "drbrain.rag.retrieval.retrieve_documents",
            side_effect=lambda *a, **k: _evidence_rows(),
        ):
            result = _invoke(tmp_path, "search", "kagome", "--json")

        assert "has moved" not in result.stderr

    def test_alias_registration_keeps_the_legacy_flags(self):
        """``functools.wraps`` must keep Typer's inspect.signature view intact."""
        from drbrain.cli import main as cli_main

        commands = {command.name: command for command in cli_main.app.registered_commands}
        query = commands["query"]
        assert query.hidden is True
        assert "limit" in query.callback.__wrapped__.__code__.co_varnames
        assert commands["search"].hidden is not True
        assert "library" in {group.name for group in cli_main.app.registered_groups}
