"""Runtime and secret boundaries for CLI-owned outputs."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import typer

from drbrain.cli.analysis_commands import survey_cmd
from drbrain.cli.check_commands import check_cmd
from drbrain.cli.concept_graph_commands import cg_map_cmd, cg_predict_cmd, cg_recommend_cmd
from drbrain.cli.export_commands import export_cmd, export_okf_cmd, restore_cmd
from drbrain.cli.graph_commands import export_cmd as graph_export_cmd
from drbrain.cli.ingest_commands import ingest_cmd, report_cmd
from drbrain.cli.query_commands import query_cmd
from drbrain.cli.rag_commands import rag_eval_cmd
from drbrain.runtime import RuntimeContext


def _ctx(config: dict, runtime: RuntimeContext | None = None) -> SimpleNamespace:
    obj: dict = {"config": config}
    if runtime is not None:
        obj["runtime"] = runtime
    return SimpleNamespace(obj=obj)


def _config(root: Path) -> dict:
    return {
        "db": {"path": str(root / "data" / "drbrain.db")},
        "dirs": {
            "inbox": str(root / "data" / "spool" / "inbox"),
            "papers": str(root / "data" / "papers"),
            "reports": str(root / "data" / "reports"),
        },
        "api": {"deepxiv_token": "configured-secret"},
        "llm": {"models": []},
    }


def test_ingest_scopes_deepxiv_token_and_restores_absent(monkeypatch, tmp_path, capsys):
    """A configured token is available to providers only during one ingest."""
    monkeypatch.delenv("DEEPXIV_TOKEN", raising=False)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")
    seen: list[str | None] = []

    def fake_ingest(*args, **kwargs):
        seen.append(os.environ.get("DEEPXIV_TOKEN"))
        return {"ok": True, "report": {}}

    with patch("drbrain.cli.ingest_commands._ingest_single_paper", side_effect=fake_ingest):
        ingest_cmd(_ctx(_config(tmp_path)), [str(pdf)], json_output=False)

    assert seen == ["configured-secret"]
    assert "DEEPXIV_TOKEN" not in os.environ
    assert "configured-secret" not in capsys.readouterr().out


def test_ingest_preserves_caller_deepxiv_token(monkeypatch, tmp_path):
    """An existing process token is neither replaced nor removed."""
    monkeypatch.setenv("DEEPXIV_TOKEN", "caller-secret")
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")
    seen: list[str | None] = []

    def fake_ingest(*args, **kwargs):
        seen.append(os.environ.get("DEEPXIV_TOKEN"))
        return {"ok": True, "report": {}}

    with patch("drbrain.cli.ingest_commands._ingest_single_paper", side_effect=fake_ingest):
        ingest_cmd(_ctx(_config(tmp_path)), [str(pdf)], json_output=False)

    assert seen == ["caller-secret"]
    assert os.environ["DEEPXIV_TOKEN"] == "caller-secret"


def test_query_exception_does_not_echo_configured_key(tmp_path, capsys):
    """LlamaIndex retrieval failures are bounded and exact-key scrubbed."""
    secret = "sk-query-provider-secret"
    config = _config(tmp_path)
    config["llm"]["models"] = [{"provider": "openai", "model": "test", "api_key": secret}]

    with (
        patch("drbrain.rag.engine.resolve_engine", return_value="llamaindex"),
        patch(
            "drbrain.cli.query_commands._query_llamaindex_cli",
            side_effect=RuntimeError(f"provider rejected {secret}"),
        ),
    ):
        with pytest.raises(typer.Exit):
            query_cmd(_ctx(config), "test", limit=1, json_output=False, jsonl=False)

    captured = capsys.readouterr()
    assert secret not in captured.err
    assert "[REDACTED]" in captured.err


def test_check_hides_embedding_key_embedded_in_api_base(tmp_path, capsys):
    """The check report must not print a credential-bearing embedding URL."""
    secret = "sk-embedding-provider-secret"
    config = _config(tmp_path)
    config["embed"] = {
        "provider": "openai-compat",
        "api_base": f"https://embed.example/v1/{secret}",
        "api_key": secret,
    }
    config["api"]["deepxiv_token"] = ""

    with (
        patch("drbrain.cli.check_commands.shutil.which", return_value=None),
        patch(
            "drbrain.cli.check_commands.importlib.import_module",
            side_effect=ImportError,
        ),
    ):
        with pytest.raises(typer.Exit):
            check_cmd(_ctx(config))

    output = capsys.readouterr().out
    assert secret not in output
    assert "API base" in output


def test_report_root_is_runtime_bound_before_read(runtime_context):
    root, runtime = runtime_context
    outside = root.parent / "reports-outside"
    outside.mkdir()
    config = _config(root)
    config["dirs"]["reports"] = str(outside)

    with pytest.raises(ValueError, match="escapes runtime root"):
        report_cmd(_ctx(config, runtime), "p1")


def test_restore_target_is_runtime_bound_before_extract(runtime_context):
    root, runtime = runtime_context
    outside = root.parent / "restore-outside"
    outside.mkdir()
    config = _config(root)

    with patch("drbrain.storage.backup.restore_backup") as restore:
        with pytest.raises(ValueError, match="escapes runtime root"):
            restore_cmd(
                _ctx(config, runtime),
                str(root / "backup.tar.gz"),
                target=str(outside),
                force=False,
                json_output=False,
            )
    restore.assert_not_called()


@pytest.fixture
def runtime_context(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    return root, RuntimeContext.create(root, run_id="boundary-test")


@pytest.mark.parametrize(
    ("command", "kwargs", "patch_target"),
    [
        (
            export_cmd,
            {"local_id": "p1", "format": "bib", "output": "x.bib"},
            "drbrain.cli.export_commands.open_db",
        ),
        (export_okf_cmd, {"output": "bundle"}, "drbrain.cli.export_commands.open_db"),
        (
            graph_export_cmd,
            {"format": "graphml", "output": "x.graphml"},
            "drbrain.cli.graph_commands.open_db",
        ),
        (survey_cmd, {"output": "survey.md"}, "drbrain.cli.analysis_commands.Database"),
        (
            rag_eval_cmd,
            {"split": "dev", "metrics": "retriever", "k": "5", "out": "baseline.md"},
            "drbrain.cli.rag_commands.open_db",
        ),
        (cg_map_cmd, {"output": "map.html"}, "drbrain.cli.concept_graph_commands.open_db"),
        (
            cg_predict_cmd,
            {
                "feat_cutoff": 2016,
                "train_end": 2019,
                "test_end": 2022,
                "model": "baseline",
                "output_pairs": "pairs.json",
            },
            "drbrain.cli.concept_graph_commands.open_db",
        ),
        (
            cg_recommend_cmd,
            {"author": "A", "output": "recommend.md"},
            "drbrain.cli.concept_graph_commands.open_db",
        ),
    ],
)
def test_cli_outputs_are_runtime_relative_and_fail_before_io(
    runtime_context, command, kwargs, patch_target
):
    """Each user-controlled output is checked before its command opens data."""
    root, runtime = runtime_context
    config = _config(root)
    outside = root.parent / "outside"
    outside.mkdir()
    output_key = next(key for key in ("output_pairs", "output", "out") if key in kwargs)
    kwargs = {**kwargs, output_key: str(outside / kwargs[output_key])}
    ctx = _ctx(config, runtime)

    with patch(patch_target) as touched:
        with pytest.raises(ValueError, match="escapes runtime root"):
            command(ctx, **kwargs)
    touched.assert_not_called()
