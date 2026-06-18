"""CLI tests for graph commands (describe/query/export/traverse)."""

import tempfile
from unittest.mock import MagicMock, patch

import typer
from typer.testing import CliRunner

runner = CliRunner()


def _cfg(**ov):
    c = {"db": {"path": ":memory:"}, "dirs": {"papers": "/tmp/p"}, "llm": {"models": []}}
    c.update(ov)
    return c


def _make_graph_app(config):
    """Build a parent app with graph sub-app injected with config."""
    from drbrain.cli.graph_commands import graph_app

    parent = typer.Typer()

    @parent.callback()
    def cb(ctx: typer.Context):
        ctx.obj = {"config": config}

    parent.add_typer(graph_app, name="graph")
    return parent


class TestGraphDescribe:
    def test_node_not_found(self):
        app = _make_graph_app(_cfg())
        mock_g = MagicMock()
        mock_g.graph = MagicMock()
        mock_g.graph.__contains__ = lambda self, x: False
        with (
            patch("drbrain.cli.graph_commands.GraphEngine", return_value=mock_g),
            patch("drbrain.cli.graph_commands.open_db"),
        ):
            r = runner.invoke(app, ["graph", "describe", "nonexistent"])
            assert r.exit_code == 1


class TestGraphQuery:
    def test_invalid_json(self):
        app = _make_graph_app(_cfg())
        r = runner.invoke(app, ["graph", "query", "{bad json"])
        assert r.exit_code == 1
        assert "Invalid JSON" in r.output

    def test_no_results(self):
        app = _make_graph_app(_cfg())
        with patch("drbrain.cli.graph_commands.query_embed", return_value=[]):
            r = runner.invoke(
                app, ["graph", "query", '{"type":"project","entity":"X","relation":"Y"}']
            )
            assert r.exit_code == 0
            assert "No results" in r.output

    def test_json_output(self):
        app = _make_graph_app(_cfg())
        mock_results = [{"label": "ResultA", "score": 0.9}, {"label": "ResultB", "score": 0.7}]
        with patch("drbrain.cli.graph_commands.query_embed", return_value=mock_results):
            r = runner.invoke(
                app, ["graph", "query", '{"type":"project","entity":"X","relation":"Y"}', "--json"]
            )
            assert r.exit_code == 0


class TestGraphExport:
    def test_unknown_format(self):
        app = _make_graph_app(_cfg())
        r = runner.invoke(app, ["graph", "export", "--format", "bogus", "--output", "/tmp/x"])
        assert r.exit_code == 1

    def test_empty_graph(self):
        app = _make_graph_app(_cfg())
        mock_g = MagicMock()
        mock_g.graph.number_of_nodes.return_value = 0
        with (
            patch("drbrain.cli.graph_commands.GraphEngine", return_value=mock_g),
            patch("drbrain.cli.graph_commands.open_db"),
        ):
            r = runner.invoke(
                app, ["graph", "export", "--format", "graphml", "--output", "/tmp/test.graphml"]
            )
            assert r.exit_code == 0
            assert "empty" in r.output.lower()

    def test_export_graphml(self):
        app = _make_graph_app(_cfg())
        mock_g = MagicMock()
        mock_g.graph.number_of_nodes.return_value = 3
        with (
            patch("drbrain.cli.graph_commands.GraphEngine", return_value=mock_g),
            patch("drbrain.cli.graph_commands.open_db"),
            patch("drbrain.storage.graph_export.export_graphml"),
        ):
            with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as f:
                r = runner.invoke(
                    app, ["graph", "export", "--format", "graphml", "--output", f.name]
                )
            assert r.exit_code == 0
            assert "Exported" in r.output

    def test_export_cypher(self):
        app = _make_graph_app(_cfg())
        mock_g = MagicMock()
        mock_g.graph.number_of_nodes.return_value = 3
        with (
            patch("drbrain.cli.graph_commands.GraphEngine", return_value=mock_g),
            patch("drbrain.cli.graph_commands.open_db"),
            patch("drbrain.storage.graph_export.export_cypher"),
        ):
            with tempfile.NamedTemporaryFile(suffix=".cypher", delete=False) as f:
                r = runner.invoke(
                    app, ["graph", "export", "--format", "cypher", "--output", f.name]
                )
            assert r.exit_code == 0
            assert "Exported" in r.output

    def test_export_jsonld(self):
        app = _make_graph_app(_cfg())
        mock_g = MagicMock()
        mock_g.graph.number_of_nodes.return_value = 3
        with (
            patch("drbrain.cli.graph_commands.GraphEngine", return_value=mock_g),
            patch("drbrain.cli.graph_commands.open_db"),
            patch("drbrain.storage.graph_export.export_jsonld"),
        ):
            with tempfile.NamedTemporaryFile(suffix=".jsonld", delete=False) as f:
                r = runner.invoke(
                    app, ["graph", "export", "--format", "jsonld", "--output", f.name]
                )
            assert r.exit_code == 0
            assert "Exported" in r.output
