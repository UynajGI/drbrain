"""CLI tests for analysis commands (reason/evolve/landscape/transfer/paradigm)."""

from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

runner = CliRunner()


def _cfg(**ov):
    c = {
        "db": {"path": ":memory:"},
        "dirs": {"papers": "/tmp/p", "cache": "/tmp/c"},
        "llm": {"models": [{"provider": "test", "model": "m"}]},
    }
    c.update(ov)
    return c


def _make_app(cmd_fn, config):
    app = typer.Typer()

    @app.callback()
    def cb(ctx: typer.Context):
        ctx.obj = {"config": config}

    app.command("test")(cmd_fn)
    return app


class TestReasonListWorkflows:
    @pytest.mark.skip(reason="typer flag naming issue with CliRunner")
    def test_lists_all_workflows(self):
        from drbrain.cli.analysis_commands import reason_cmd

        app = _make_app(reason_cmd, _cfg())
        r = runner.invoke(app, ["test", "--list-workflows"])
        assert r.exit_code == 0
        assert "causal" in r.output
        assert "temporal" in r.output
        assert "hypothesis" in r.output
        assert "review" in r.output


class TestReasonWorkflow:
    def test_unknown_workflow(self):
        from drbrain.cli.analysis_commands import reason_cmd

        app = _make_app(reason_cmd, _cfg())
        r = runner.invoke(app, ["test", "q?", "--workflow", "bogus"])
        assert r.exit_code == 1
        assert "Unknown workflow" in r.output

    def test_no_llm_configured(self):
        from drbrain.cli.analysis_commands import reason_cmd

        app = _make_app(reason_cmd, _cfg(llm={"models": []}))
        r = runner.invoke(app, ["test", "q?"])
        assert r.exit_code == 1
        assert "No LLM models" in r.output


class TestEvolveCmd:
    def test_invalid_direction(self):
        from drbrain.cli.analysis_commands import evolve_cmd

        app = _make_app(evolve_cmd, _cfg())
        r = runner.invoke(app, ["test", "concept", "--direction", "bogus"])
        assert r.exit_code == 1

    def test_not_found(self):
        from drbrain.cli.analysis_commands import evolve_cmd

        app = _make_app(evolve_cmd, _cfg())
        with patch("drbrain.graph.genealogy.evolve_concept", return_value=[]):
            r = runner.invoke(app, ["test", "missing"])
            assert r.exit_code == 0
            assert "No concept found" in r.output

    def test_json_output(self):
        from drbrain.cli.analysis_commands import evolve_cmd

        app = _make_app(evolve_cmd, _cfg())
        with patch("drbrain.graph.genealogy.evolve_concept", return_value=[]):
            r = runner.invoke(app, ["test", "x", "--json"])
            assert r.exit_code == 0


class TestParadigmCmd:
    def test_no_shifts(self):
        from drbrain.cli.analysis_commands import paradigm_cmd

        app = _make_app(paradigm_cmd, _cfg())
        with patch("drbrain.graph.genealogy.detect_paradigm_shifts", return_value=[]):
            r = runner.invoke(app, ["test", "concept"])
            assert r.exit_code == 0
            assert "No paradigm shifts" in r.output

    def test_json_empty(self):
        from drbrain.cli.analysis_commands import paradigm_cmd

        app = _make_app(paradigm_cmd, _cfg())
        with patch("drbrain.graph.genealogy.detect_paradigm_shifts", return_value=[]):
            r = runner.invoke(app, ["test", "concept", "--json"])
            assert r.exit_code == 0


class TestTransfersCmd:
    def test_no_options(self):
        from drbrain.cli.analysis_commands import transfers_cmd

        app = _make_app(transfers_cmd, _cfg())
        r = runner.invoke(app, ["test"])
        assert r.exit_code == 1


class TestLandscapeCmd:
    def test_empty_json(self):
        from drbrain.cli.analysis_commands import landscape_cmd

        app = _make_app(landscape_cmd, _cfg())
        with patch(
            "drbrain.graph.genealogy.landscape_workspace",
            return_value={"timeline": [], "gaps": [], "debates": []},
        ):
            r = runner.invoke(app, ["test", "--json"])
            assert r.exit_code == 0
