"""Tests for the standalone 'drbrain search' BM25 command."""

import json
import tempfile
from pathlib import Path
from unittest import mock

import typer

from drbrain.storage.database import Database


def _make_config(db_path: str) -> dict:
    return {
        "db": {"path": db_path},
        "llm": {"models": []},
        "dirs": {
            "inbox": "data/spool/inbox",
            "papers": "data/papers",
            "reports": "data/reports",
            "cache": "data/cache",
            "logs": "data/logs",
        },
        "api": {},
        "mineru": {},
        "extract": {"max_concurrent": 1},
        "bm25": {"k1": 1.5, "b": 0.75},
        "queue": {"weak_threshold": 0.5, "auto_accept": False},
    }


def _make_ctx(cfg: dict):
    ctx = mock.MagicMock(spec=typer.Context)
    ctx.obj = {"config": cfg}
    return ctx


class TestSearchCmdDirect:
    """Tests calling search_cmd directly (not via typer CLI runner)."""

    def test_search_returns_matching_concepts(self):
        """search_cmd finds concepts matching the query string."""
        from drbrain.cli.query_commands import search_cmd

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            db = Database(db_path)
            db.insert_paper("p1", "Neural Network Paper", 2024, "uploaded")
            db.insert_concept("p1", "Method", "transformer architecture", 0.9, year=2024)
            db.insert_concept("p1", "Problem", "scalability issue", 0.7, year=2024)
            db.insert_concept("p1", "Method", "attention mechanism", 0.85, year=2024)
            db.commit()
            db.close()

            cfg = _make_config(str(db_path))
            ctx = _make_ctx(cfg)
            # Capture output
            import io

            buf = io.StringIO()
            with mock.patch("typer.echo", side_effect=lambda *a, **kw: buf.write(a[0] + "\n")):
                search_cmd(ctx, "transformer", limit=5, type=None, json_output=False)

            output = buf.getvalue()
            assert "transformer" in output

    def test_search_with_type_filter(self):
        """search_cmd --type filters by document type."""
        from drbrain.cli.query_commands import search_cmd

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            db = Database(db_path)
            db.insert_paper("p1", "Test Paper", 2024, "uploaded")
            db.insert_concept("p1", "Method", "gradient descent", 0.9, year=2024)
            db.insert_concept("p1", "Problem", "gradient explosion", 0.7, year=2024)
            db.commit()
            db.close()

            cfg = _make_config(str(db_path))
            ctx = _make_ctx(cfg)

            import io

            buf = io.StringIO()
            with mock.patch("typer.echo", side_effect=lambda *a, **kw: buf.write(a[0] + "\n")):
                search_cmd(ctx, "gradient", limit=10, type="Problem", json_output=False)

            output = buf.getvalue()
            assert "Problem" in output
            assert "explosion" in output

    def test_search_json_output(self):
        """search_cmd --json returns valid JSON."""
        from drbrain.cli.query_commands import search_cmd

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            db = Database(db_path)
            db.insert_paper("p1", "Paper", 2024, "uploaded")
            db.insert_concept("p1", "Method", "transformer", 0.9, year=2024)
            db.commit()
            db.close()

            cfg = _make_config(str(db_path))
            ctx = _make_ctx(cfg)

            import io

            buf = io.StringIO()
            with mock.patch("typer.echo", side_effect=lambda *a, **kw: buf.write(a[0] + "\n")):
                search_cmd(ctx, "transformer", limit=5, type=None, json_output=True)

            output = buf.getvalue()
            results = json.loads(output)
            assert isinstance(results, list)
            assert len(results) >= 1
            labels = [r["label"] for r in results]
            assert "transformer" in labels

    def test_search_no_results(self):
        """search_cmd returns results with score 0 for non-matching terms."""
        from drbrain.cli.query_commands import search_cmd

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            db = Database(db_path)
            db.insert_paper("p1", "Paper", 2024, "uploaded")
            db.insert_concept("p1", "Method", "transformer", 0.9, year=2024)
            db.commit()
            db.close()

            cfg = _make_config(str(db_path))
            ctx = _make_ctx(cfg)

            import io

            buf = io.StringIO()
            with mock.patch("typer.echo", side_effect=lambda *a, **kw: buf.write(a[0] + "\n")):
                search_cmd(ctx, "zzz_nonexistent_xyz", limit=5, type=None, json_output=True)

            output = buf.getvalue()
            results = json.loads(output)
            # BM25 returns all docs with score 0 when no terms match
            assert len(results) >= 1
            assert all(r["score"] == 0.0 for r in results)

    def test_search_respects_limit(self):
        """search_cmd --limit caps the number of results."""
        from drbrain.cli.query_commands import search_cmd

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            db = Database(db_path)
            db.insert_paper("p1", "Paper", 2024, "uploaded")
            for i in range(5):
                db.insert_concept("p1", "Method", f"method {i} neural network", 0.9, year=2024)
            db.commit()
            db.close()

            cfg = _make_config(str(db_path))
            ctx = _make_ctx(cfg)

            import io

            buf = io.StringIO()
            with mock.patch("typer.echo", side_effect=lambda *a, **kw: buf.write(a[0] + "\n")):
                search_cmd(ctx, "neural network", limit=2, type=None, json_output=True)

            output = buf.getvalue()
            results = json.loads(output)
            assert len(results) <= 2
