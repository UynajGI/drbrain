"""CLI compatibility contracts for the main-line convergence (Phase 3).

The design moves the knowledge-graph pipeline into the ``graph`` namespace and
hides the historical index entries.  These tests pin both sides:

* ``graph build`` / ``graph embed`` / ``graph closure`` are real, visible
  commands registering the *same function objects* as the top-level names;
* the top-level aliases stay callable, hidden from help, with a stderr notice;
* the pipeline steps invoke the new main-line child commands while keeping
  their step labels and exit contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml
from typer.testing import CliRunner

from drbrain.cli.main import app

runner = CliRunner()


def _write_config(tmp_path: Path) -> None:
    config = {
        "db": {"path": "data/drbrain.db"},
        "dirs": {"inbox": "data/spool/inbox", "papers": "data/papers", "logs": "data/logs"},
        "llm": {"models": []},
        "llamaindex": {"enabled": True, "rag_engine": "sql", "tree_storage": "data/tree"},
        "embed": {"provider": "none"},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _invoke(tmp_path: Path, *args: str):
    return runner.invoke(app, ["--root", str(tmp_path), *args])


class TestGraphNamespace:
    def test_graph_namespace_registers_the_pipeline_commands(self):
        from drbrain.cli.build_commands import build_cmd, embed_cmd
        from drbrain.cli.graph_commands import graph_app
        from drbrain.cli.ingest_commands import closure_cmd

        callbacks = {command.name: command.callback for command in graph_app.registered_commands}
        assert callbacks["build"] is build_cmd
        assert callbacks["embed"] is embed_cmd
        assert callbacks["closure"] is closure_cmd

    def test_graph_help_lists_the_pipeline_commands(self):
        result = runner.invoke(app, ["graph", "--help"])
        assert result.exit_code == 0
        for name in ("build", "embed", "closure", "neighbors", "query"):
            assert name in result.stdout

    def test_graph_closure_runs_with_the_top_level_flags(self, tmp_path):
        _write_config(tmp_path)
        result = _invoke(tmp_path, "graph", "closure", "--json")
        assert result.exit_code == 0, result.stderr
        payload = json.loads(result.stdout)
        assert isinstance(payload, (dict, list))

    def test_top_level_pipeline_aliases_are_hidden_and_notice_migration(self, tmp_path):
        _write_config(tmp_path)
        for alias, hint in (
            ("build", "drbrain graph build"),
            ("closure", "drbrain graph closure"),
        ):
            result = _invoke(tmp_path, alias)
            assert result.exit_code == 0, (alias, result.stderr)
            assert f"[drbrain] '{alias}' has moved: use '{hint}'" in result.stderr

        embed = _invoke(tmp_path, "embed")
        assert "[drbrain] 'embed' has moved: use 'drbrain graph embed'" in embed.stderr

    def test_main_help_hides_the_top_level_pipeline_aliases(self):
        import typer.main

        group = typer.main.get_command(app)
        commands = group.commands
        for alias in ("build", "embed", "closure", "query", "hybrid", "fsearch"):
            assert alias in commands, alias
            assert commands[alias].hidden is True, alias
        for visible in ("search", "ask", "library", "index", "graph", "rag"):
            assert visible in commands, visible
            assert getattr(commands[visible], "hidden", False) is not True, visible


class TestRagNamespace:
    def test_rag_help_hides_the_index_aliases(self):
        result = runner.invoke(app, ["rag", "--help"])
        assert result.exit_code == 0
        lines = [line.strip("│ ").strip() for line in result.stdout.splitlines()]
        for hidden in ("prepare", "index", "health"):
            assert not any(line.startswith(hidden + " ") for line in lines), hidden
        for visible in ("eval", "baselines", "pageindex-index", "pageindex-chat"):
            assert any(line.startswith(visible + " ") for line in lines), visible

    def test_rag_index_keeps_its_argv_and_exit_contract(self, tmp_path):
        _write_config(tmp_path)
        stats = {"papers": 0, "nodes": 0, "embedded": 0}
        with mock.patch("drbrain.rag.indexer.build_index", return_value=stats) as build:
            result = _invoke(tmp_path, "rag", "index", "--json")

        assert result.exit_code == 0, result.stderr
        assert build.called
        assert json.loads(result.stdout) == stats
        assert "[drbrain] 'rag index' has moved: use 'drbrain index build'" in result.stderr


class TestPipelineChildCommands:
    def _ctx(self, tmp_path: Path):
        runtime = SimpleNamespace(config_path=None, root=str(tmp_path))
        return SimpleNamespace(obj={"runtime": runtime, "config": {}})

    def _run(self, tmp_path: Path, steps: str, *, full: bool = False):
        from drbrain.cli.ingest_commands import pipeline_cmd

        calls: list[list[str]] = []

        def fake_run(args, **kwargs):
            calls.append(list(args))
            import subprocess

            return subprocess.CompletedProcess(args, 0)

        with mock.patch("subprocess.run", side_effect=fake_run):
            pipeline_cmd(
                self._ctx(tmp_path),
                preset=None,
                steps=steps,
                list_steps_flag=False,
                dry_run=False,
                full=full,
            )
        # Drop ``python -m drbrain.cli.main`` and the ``--root <root>`` pair.
        return [self._child_argv(call) for call in calls]

    @staticmethod
    def _child_argv(call: list[str]) -> list[str]:
        args = call[3:]
        if args[:1] == ["--root"]:
            args = args[2:]
        return args

    def test_steps_map_to_the_new_main_line(self, tmp_path):
        subcommands = self._run(tmp_path, "ingest,build,embed,closure,rag")

        assert subcommands == [
            ["ingest"],
            ["graph", "build"],
            ["graph", "embed"],
            ["graph", "closure"],
            ["index", "build"],
        ]

    def test_full_mode_adds_the_force_flags(self, tmp_path):
        subcommands = self._run(tmp_path, "build,closure,rag", full=True)
        assert subcommands[0] == ["graph", "build", "--all"]
        assert subcommands[1] == ["graph", "closure", "--full"]
        assert subcommands[2] == ["index", "build", "--force"]

    def test_step_labels_and_exit_contract_are_unchanged(self, tmp_path, capsys):
        import subprocess

        import pytest
        import typer

        from drbrain.cli.ingest_commands import pipeline_cmd

        def fake_run(args, **kwargs):
            return subprocess.CompletedProcess(args, 7)

        with mock.patch("subprocess.run", side_effect=fake_run):
            with pytest.raises(typer.Exit) as excinfo:
                pipeline_cmd(
                    self._ctx(tmp_path),
                    preset=None,
                    steps="build,embed",
                    list_steps_flag=False,
                    dry_run=False,
                    full=False,
                )

        captured = capsys.readouterr()
        assert excinfo.value.exit_code == 7
        assert "Pipeline failed at step 'build'" in captured.err
