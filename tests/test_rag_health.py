"""Read-only production readiness checks for persisted RAG indexes."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from drbrain.config import Config, LlamaIndexConfig


def _cfg(tmp_path, *, enabled: bool = True) -> Config:
    return Config(
        llamaindex=LlamaIndexConfig(enabled=enabled, storage_dir=str(tmp_path / "llamaindex"))
    )


def _write_ready_artifacts(cfg: Config) -> None:
    root = cfg.llamaindex.storage_dir
    from pathlib import Path

    path = Path(root)
    (path / "vector").mkdir(parents=True)
    (path / "bm25").mkdir()
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "embed_model": cfg.embed.model,
                "vector_store": cfg.llamaindex.vector_store,
                "papers": {
                    "paper-a": {"paper-a:abstract": "abc"},
                    "paper-b": {"paper-b:abstract": "def", "paper-b:results": "ghi"},
                },
            }
        ),
        encoding="utf-8",
    )
    (path / "vector" / "docstore.json").write_text("{}", encoding="utf-8")
    (path / "vector" / "default__vector_store.json").write_text("{}", encoding="utf-8")
    (path / "bm25" / "corpus.jsonl").write_text("{}\n", encoding="utf-8")


def test_get_index_health_reports_ready_storage(monkeypatch, tmp_path):
    from drbrain.rag.indexer import get_index_health

    cfg = _cfg(tmp_path)
    _write_ready_artifacts(cfg)
    monkeypatch.setattr("drbrain.rag.indexer.load_index", lambda _cfg: (object(), object()))

    result = get_index_health(cfg)

    assert result["ready"] is True
    assert result["status"] == "ready"
    assert result["reasons"] == []
    assert result["checks"]["manifest"]["paper_count"] == 2
    assert result["checks"]["manifest"]["parent_node_count"] == 3
    assert result["checks"]["vector"]["loadable"] is True
    assert result["checks"]["bm25"]["loadable"] is True


def test_get_index_health_is_read_only_when_disabled(monkeypatch, tmp_path):
    from drbrain.rag.indexer import get_index_health

    monkeypatch.setattr(
        "drbrain.rag.indexer.load_index",
        lambda _cfg: (_ for _ in ()).throw(AssertionError("must not load when disabled")),
    )

    result = get_index_health(_cfg(tmp_path, enabled=False))

    assert result["ready"] is False
    assert result["status"] == "not_ready"
    assert result["reasons"] == ["config_disabled"]


def test_get_index_health_rejects_corrupt_or_mismatched_manifest(tmp_path):
    from drbrain.rag.indexer import get_index_health

    cfg = _cfg(tmp_path)
    root = tmp_path / "llamaindex"
    root.mkdir()
    (root / "manifest.json").write_text("{not valid json", encoding="utf-8")

    corrupt = get_index_health(cfg)
    assert corrupt["ready"] is False
    assert "manifest_invalid" in corrupt["reasons"]

    _write_ready_artifacts(cfg)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["embed_model"] = "other-model"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    mismatch = get_index_health(cfg)
    assert mismatch["ready"] is False
    assert "embed_model_mismatch" in mismatch["reasons"]


def test_rag_health_cli_emits_json_with_automation_exit_status(monkeypatch, tmp_path):
    from drbrain.cli.rag_commands import rag_app

    healthy = {"ready": True, "status": "ready", "storage_dir": "x", "checks": {}, "reasons": []}
    monkeypatch.setattr("drbrain.rag.indexer.get_index_health", lambda _cfg: healthy)
    runner = CliRunner()

    result = runner.invoke(rag_app, ["health", "--json"], obj={"config": _cfg(tmp_path)})

    assert result.exit_code == 0
    assert json.loads(result.output) == healthy

    unhealthy = {**healthy, "ready": False, "status": "not_ready", "reasons": ["index_missing"]}
    monkeypatch.setattr("drbrain.rag.indexer.get_index_health", lambda _cfg: unhealthy)
    result = runner.invoke(rag_app, ["health", "--json"], obj={"config": _cfg(tmp_path)})

    assert result.exit_code == 1
    assert json.loads(result.output) == unhealthy
