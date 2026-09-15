"""CLI contract tests for ``drbrain index`` (build / status / verify, bare index).

Phase 1 of the CLI/RAG usage refactor: the ``index`` sub-app owns the main-line
index lifecycle.  These tests pin the contracts the design asks for — the
three states a corpus can be in are reported separately (ingested is not
indexed, indexed is not retrievable), ``index build`` reports every stage and
exits 1 on a failed stage, ``index verify`` reports what ``search``/``ask``
read, and bare ``drbrain index`` keeps its historical JSON/exit contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import yaml
from typer.testing import CliRunner

from drbrain.cli.main import app
from drbrain.services.canonical_content import write_canonical_content
from drbrain.storage.database import Database

runner = CliRunner()

DIMENSION = 3


class _FakeEmbedder:
    def __call__(self, texts):
        return [[float(len(text) % 7), float(len(text) % 5), 0.5] for text in list(texts)]


class _FakeSummaryModel:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str, *, max_tokens: int):
        self.calls += 1
        return SimpleNamespace(text="summary", finish_reason="stop")


def _write_config(tmp_path: Path, *, retrievers=None, rag_engine: str = "sql") -> None:
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
            "rag_engine": rag_engine,
            "tree_storage": "data/tree",
            "retrievers": retrievers or ["bm25", "vector", "tree"],
        },
        "embed": {"provider": "local", "model": "fake-embed", "dim": DIMENSION},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _write_corpus(tmp_path: Path, *local_ids: str) -> None:
    """Register canonical content so the corpus counts as *ingested*."""
    db_path = tmp_path / "data" / "drbrain.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = Database(db_path)
    try:
        for local_id in local_ids or ("p1", "p2"):
            text = "".join(
                f"# Section {index}\n\n" + " ".join([f"{local_id}body{index}"] * 30) + "\n\n"
                for index in range(4)
            )
            if db.get_paper(local_id) is None:
                db.insert_paper(local_id, "T", 2024, "uploaded")
                db.commit()
            result = write_canonical_content(db, local_id, text, media_type="md", parser="test")
            assert result["ok"]
    finally:
        db.close()


def _invoke(tmp_path: Path, *args: str):
    return runner.invoke(app, ["--root", str(tmp_path), *args])


def _patch_build(monkeypatch, *, summary=None, raise_model: bool = False):
    """Keep the build offline: fake embedder + fake/absent index model."""
    embedder = _FakeEmbedder()
    monkeypatch.setattr(
        "drbrain.services.embedding._embed_batch",
        lambda texts, _cfg: embedder(texts),
    )
    if raise_model:

        def _unavailable(_config):
            raise RuntimeError("index model endpoint unreachable")

        monkeypatch.setattr("drbrain.tree.prepare._resolve_index_summary_model", _unavailable)
    else:
        model = summary or _FakeSummaryModel()
        monkeypatch.setattr(
            "drbrain.tree.prepare._resolve_index_summary_model", lambda _config: model
        )
    return embedder


class TestIndexStatus:
    def test_ingested_is_not_indexed_and_not_retrievable(self, tmp_path):
        _write_config(tmp_path)
        _write_corpus(tmp_path)

        result = _invoke(tmp_path, "index", "status", "--json")

        assert result.exit_code == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["states"]["ingested"]["ready"] is True
        assert report["states"]["indexed"]["ready"] is False
        assert report["states"]["retrievable"]["ready"] is False
        assert report["status"] in {"partial", "not_ready"}
        for leg in ("lexical", "fts", "vector", "tree"):
            assert "ready" in report["legs"][leg]
            assert "reasons" in report["legs"][leg]
        assert "no_published_generation" in report["legs"]["tree"]["reasons"]
        # Unpromoted leaves are pending assignment work, not a readiness gate.
        assert report["pending"]["leaves_missing_parent"] >= 1

    def test_status_is_read_only(self, tmp_path):
        _write_config(tmp_path)
        _write_corpus(tmp_path)

        db_path = tmp_path / "data" / "drbrain.db"
        result = _invoke(tmp_path, "index", "status", "--json")

        assert result.exit_code == 0
        assert not (tmp_path / "data" / "tree").exists()
        db = Database(db_path)
        try:
            assert db.get_last_run("index") is None
        finally:
            db.close()

    def test_human_status_names_the_three_states(self, tmp_path):
        _write_config(tmp_path)
        _write_corpus(tmp_path)

        result = _invoke(tmp_path, "index", "status")

        assert result.exit_code == 0
        assert "Ingested:" in result.stdout
        assert "Indexed:" in result.stdout
        assert "Retrievable:" in result.stdout

    def test_llamaindex_engine_requires_its_own_generation(self, tmp_path, monkeypatch):
        _write_config(tmp_path, rag_engine="llamaindex")
        _write_corpus(tmp_path)
        backend = {
            "ready": False,
            "status": "not_ready",
            "reasons": ["index_missing"],
            "generation": None,
            "storage_dir": str(tmp_path / "data" / "llamaindex"),
        }
        with mock.patch("drbrain.rag.indexer.get_index_health", return_value=backend):
            report = json.loads(_invoke(tmp_path, "index", "status", "--json").stdout)

        assert report["states"]["retrievable"]["ready"] is False
        assert "llamaindex_generation_not_ready" in report["states"]["retrievable"]["reasons"]

    def test_llamaindex_engine_ready_when_generation_and_tree_are_healthy(
        self, tmp_path, monkeypatch
    ):
        _write_config(tmp_path, rag_engine="llamaindex")
        _write_corpus(tmp_path)
        _patch_build(monkeypatch)
        backend = {
            "ready": True,
            "status": "ready",
            "reasons": [],
            "generation": "gen-li-1",
            "storage_dir": str(tmp_path / "data" / "llamaindex"),
        }
        with (
            mock.patch("drbrain.rag.indexer.get_index_health", return_value=backend),
        ):
            build = _invoke(tmp_path, "index", "build", "--json")
            report = json.loads(_invoke(tmp_path, "index", "status", "--json").stdout)

        assert build.exit_code == 0, build.stderr
        assert report["states"]["retrievable"]["ready"] is True
        assert report["status"] == "ready"

    def test_failed_build_is_not_ready_and_a_recovered_build_clears_it(self, tmp_path, monkeypatch):
        """A failed stage is never reported as ready, even with a fallback tree.

        The recorded last-build outcome gates the tree leg; the next successful
        run overwrites it and readiness returns without any extra state.
        """
        _write_config(tmp_path)
        _write_corpus(tmp_path)
        _patch_build(monkeypatch, raise_model=True)
        assert _invoke(tmp_path, "index", "build", "--json").exit_code == 1

        report = json.loads(_invoke(tmp_path, "index", "status", "--json").stdout)
        tree = report["legs"]["tree"]
        assert tree["ready"] is False
        assert tree["last_build"]["ok"] is False
        assert "hierarchy" in tree["last_build"]["failed_stages"]
        assert any(reason.startswith("last_build_failed") for reason in tree["reasons"])

        _patch_build(monkeypatch)
        assert _invoke(tmp_path, "index", "build", "--json").exit_code == 0

        recovered = json.loads(_invoke(tmp_path, "index", "status", "--json").stdout)
        assert recovered["legs"]["tree"]["ready"] is True
        assert recovered["legs"]["tree"]["last_build"]["ok"] is True


class TestIndexBuild:
    def test_build_reports_stages_and_publishes_then_status_is_ready(self, tmp_path, monkeypatch):
        _write_config(tmp_path)
        _write_corpus(tmp_path)
        _patch_build(monkeypatch)

        result = _invoke(tmp_path, "index", "build", "--json")

        assert result.exit_code == 0, result.stderr
        payload = json.loads(result.stdout)
        assert set(
            [
                "ok",
                "changed",
                "published",
                "failed_stages",
                "fts",
                "vectors",
                "hierarchy",
                "publication",
                "duration_ms",
                "lexical",
            ]
        ) <= set(payload)
        assert payload["ok"] is True
        assert payload["failed_stages"] == []
        assert payload["published"]
        assert set(payload["lexical"]) <= {"documents", "indexed", "up_to_date"}
        assert (tmp_path / "data" / "tree" / "active.json").is_file()

        status = _invoke(tmp_path, "index", "status", "--json")
        report = json.loads(status.stdout)
        assert status.exit_code == 0
        assert report["states"]["ingested"]["ready"] is True
        assert report["states"]["indexed"]["ready"] is True
        assert report["states"]["retrievable"]["ready"] is True
        assert report["status"] == "ready"
        assert report["generation"] == payload["published"]

    def test_failed_stage_exits_one_and_is_never_ready(self, tmp_path, monkeypatch):
        _write_config(tmp_path)
        _write_corpus(tmp_path)
        _patch_build(monkeypatch, raise_model=True)

        result = _invoke(tmp_path, "index", "build", "--json")

        assert result.exit_code == 1, result.stderr
        payload = json.loads(result.stdout)
        assert payload["ok"] is False
        assert "hierarchy" in payload["failed_stages"]
        assert payload["published"] is None

        status = json.loads(_invoke(tmp_path, "index", "status", "--json").stdout)
        assert status["states"]["indexed"]["ready"] is False
        assert status["states"]["retrievable"]["ready"] is False

    def test_build_plain_output_lists_stage_statuses(self, tmp_path, monkeypatch):
        _write_config(tmp_path)
        _write_corpus(tmp_path)
        _patch_build(monkeypatch)

        result = _invoke(tmp_path, "index", "build")

        assert result.exit_code == 0, result.stderr
        assert "Index build (incremental): ok" in result.stdout
        for stage in ("lexical", "fts", "vectors", "hierarchy", "publication"):
            assert stage in result.stdout

    def test_build_has_no_shard_db_override(self, tmp_path, monkeypatch):
        """``index build`` serves the main corpus only — no shard override.

        The shard pipelines keep their legacy ``embed --tree --db`` stage until
        the merge path understands the unified tables, so no ``--db`` flag is
        offered here.
        """
        _write_config(tmp_path)
        _write_corpus(tmp_path)
        _patch_build(monkeypatch)

        result = _invoke(tmp_path, "index", "build", "--db", str(tmp_path / "shard.db"), "--json")

        assert result.exit_code == 2
        assert "--db" in result.output

    def test_build_does_not_touch_the_llamaindex_engine_path(self, tmp_path, monkeypatch):
        """`index build` serves the main corpus; `rag index` owns that engine.

        The LlamaIndex generation is prepared by the compatibility command the
        missing-index hint names for that engine, so a build must not
        half-publish it (or, worse, republish the deprecated SQL snapshot).
        """
        _write_config(tmp_path, rag_engine="llamaindex")
        _write_corpus(tmp_path)
        _patch_build(monkeypatch)
        with mock.patch(
            "drbrain.rag.indexer.build_index",
            side_effect=AssertionError("index build must not prepare the llamaindex generation"),
        ):
            result = _invoke(tmp_path, "index", "build", "--json")

        assert result.exit_code == 0, result.stderr
        payload = json.loads(result.stdout)
        assert "llamaindex" not in payload
        assert payload["failed_stages"] == []

    def test_build_without_an_embedding_profile_fails_closed(self, tmp_path):
        _write_config(tmp_path)
        cfg_path = tmp_path / "config.yaml"
        payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        payload["embed"] = {"provider": "none", "model": ""}
        cfg_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

        result = _invoke(tmp_path, "index", "build", "--json")

        assert result.exit_code == 1
        assert "embedding profile unavailable" in result.stderr


class TestIndexVerify:
    def test_verify_ok_for_a_published_generation(self, tmp_path, monkeypatch):
        _write_config(tmp_path)
        _write_corpus(tmp_path)
        _patch_build(monkeypatch)
        assert _invoke(tmp_path, "index", "build", "--json").exit_code == 0

        result = _invoke(tmp_path, "index", "verify", "--json")

        assert result.exit_code == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["ok"] is True
        assert report["generation"]
        assert report["errors"] == []
        names = {check["name"] for check in report["checks"]}
        assert {"tree_generation", "content_fts", "node_vectors", "leaf_reachability"} <= names
        by_name = {check["name"]: check for check in report["checks"]}
        assert by_name["node_vectors"]["detail"]["pending"] == 0
        assert by_name["generation_freshness"]["ok"] is True
        assert by_name["last_build"]["ok"] is True

    def test_verify_reports_the_failed_last_build_and_recovers(self, tmp_path, monkeypatch):
        """A recorded failed build is an error even while an older generation reads."""
        _write_config(tmp_path)
        _write_corpus(tmp_path)
        _patch_build(monkeypatch)
        assert _invoke(tmp_path, "index", "build", "--json").exit_code == 0
        assert _invoke(tmp_path, "index", "verify", "--json").exit_code == 0

        db = Database(tmp_path / "data" / "drbrain.db")
        try:
            db.set_vector_metadata(
                "tree.prepare.last",
                json.dumps({"ok": False, "failed_stages": ["hierarchy"], "published": None}),
            )
            db.commit()
        finally:
            db.close()

        result = _invoke(tmp_path, "index", "verify", "--json")

        assert result.exit_code == 1
        report = json.loads(result.stdout)
        assert any(error.startswith("last_build") for error in report["errors"])

        # The next successful run overwrites the record and clears the error.
        assert _invoke(tmp_path, "index", "build", "--json").exit_code == 0
        recovered = _invoke(tmp_path, "index", "verify", "--json")
        assert recovered.exit_code == 0, recovered.stderr
        assert json.loads(recovered.stdout)["ok"] is True

    def test_verify_fails_closed_without_a_generation(self, tmp_path):
        _write_config(tmp_path)
        _write_corpus(tmp_path)

        result = _invoke(tmp_path, "index", "verify")

        assert result.exit_code == 1
        assert "tree_generation" in result.stdout

        report = json.loads(_invoke(tmp_path, "index", "verify", "--json").stdout)
        assert report["ok"] is False
        assert any("tree_generation" in error for error in report["errors"])

    def test_verify_checks_the_engine_generation(self, tmp_path, monkeypatch):
        _write_config(tmp_path, rag_engine="llamaindex")
        _write_corpus(tmp_path)
        backend = {
            "ready": False,
            "status": "not_ready",
            "reasons": ["index_missing"],
            "generation": None,
            "storage_dir": str(tmp_path / "data" / "llamaindex"),
        }
        with mock.patch("drbrain.rag.indexer.get_index_health", return_value=backend):
            report = json.loads(_invoke(tmp_path, "index", "verify", "--json").stdout)

        by_name = {check["name"]: check for check in report["checks"]}
        assert by_name["llamaindex_generation"]["ok"] is False
        assert report["ok"] is False
        assert any("llamaindex_generation" in error for error in report["errors"])

    def test_verify_notes_that_no_engine_generation_is_pinned(self, tmp_path, monkeypatch):
        _write_config(tmp_path)
        _write_corpus(tmp_path)
        _patch_build(monkeypatch)
        assert _invoke(tmp_path, "index", "build", "--json").exit_code == 0

        report = json.loads(_invoke(tmp_path, "index", "verify", "--json").stdout)

        by_name = {check["name"]: check for check in report["checks"]}
        assert by_name["engine_generation"]["ok"] is True
        assert by_name["engine_generation"]["detail"]["status"] == "unavailable"
        assert report["ok"] is True

    def test_verify_detects_vectors_missing_for_ready_nodes(self, tmp_path, monkeypatch):
        _write_config(tmp_path)
        _write_corpus(tmp_path)
        _patch_build(monkeypatch)
        assert _invoke(tmp_path, "index", "build", "--json").exit_code == 0

        # A new leaf with no vector at all: index build would have to embed it.
        db = Database(tmp_path / "data" / "drbrain.db")
        try:
            db.insert_paper("p3", "T3", 2025, "uploaded")
            db.commit()
            text = "# New\n\n" + " ".join(["freshleaf"] * 20)
            write_canonical_content(db, "p3", text, media_type="md", parser="test")
        finally:
            db.close()

        report = json.loads(_invoke(tmp_path, "index", "verify", "--json").stdout)

        by_name = {check["name"]: check for check in report["checks"]}
        assert by_name["node_vectors"]["ok"] is False
        assert by_name["node_vectors"]["detail"]["pending"] >= 1
        assert report["ok"] is False


class TestBareIndexContract:
    def test_first_run_builds_and_second_run_reports_up_to_date(self, tmp_path):
        _write_config(tmp_path)
        _write_corpus(tmp_path)

        first = _invoke(tmp_path, "index", "--json")
        assert first.exit_code == 0, first.stderr
        built = json.loads(first.stdout)
        assert built["indexed"] is True
        assert built["documents"] == 2
        assert "up_to_date" not in built

        second = _invoke(tmp_path, "index", "--json")
        assert second.exit_code == 0
        skipped = json.loads(second.stdout)
        assert skipped == {"documents": 2, "indexed": False, "up_to_date": True}

    def test_plain_output_contract(self, tmp_path):
        _write_config(tmp_path)
        _write_corpus(tmp_path)

        assert "Indexed 2 documents" in _invoke(tmp_path, "index").stdout
        assert "Index up to date (2 documents" in _invoke(tmp_path, "index").stdout

    def test_rebuild_flag_forces_a_rebuild(self, tmp_path):
        _write_config(tmp_path)
        _write_corpus(tmp_path)
        assert _invoke(tmp_path, "index", "--json").exit_code == 0

        result = _invoke(tmp_path, "index", "--rebuild", "--json")

        assert result.exit_code == 0
        assert json.loads(result.stdout)["indexed"] is True

    def test_bare_index_still_importable(self):
        """``index_cmd`` stays importable for the historical CLI surface."""
        from drbrain.cli.query_commands import index_cmd  # noqa: F401


class TestAskRemedyHint:
    @pytest.mark.parametrize(
        ("engine", "expected"),
        [
            ("sql", "drbrain index build"),
            # The persisted LlamaIndex generation is still prepared by the
            # compatibility command, so its hint must name that one.
            ("llamaindex", "drbrain rag index"),
        ],
    )
    def test_ask_prepare_hint_names_the_command_that_prepares_the_index(self, engine, expected):
        from drbrain.config import Config
        from drbrain.rag.engine import ask_prepare_hint

        config = Config()
        config.llamaindex.enabled = True
        config.llamaindex.rag_engine = engine

        assert ask_prepare_hint(config) == expected

    def test_llamaindex_hint_is_a_registered_command(self):
        """The hint must name a command that exists (hidden aliases count)."""
        import typer.main

        from drbrain.cli.main import app

        group = typer.main.get_command(app)
        assert "rag" in group.commands
        sub = group.commands["rag"]
        assert "index" in sub.commands
        assert sub.commands["index"].hidden is True
