"""Tests for drbrain graph subcommands: neighbors, path."""

import io
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

from drbrain.cli.graph_commands import neighbors_cmd


def _make_minimal_config(db_path: str, papers_dir: str) -> dict:
    return {
        "db": {"path": db_path},
        "dirs": {
            "inbox": "data/inbox",
            "papers": papers_dir,
            "reports": "/tmp/reports",
            "cache": "data/cache",
            "logs": "data/logs",
        },
        "bm25": {"k1": 1.5, "b": 0.75},
        "queue": {"weak_threshold": 0.7, "auto_accept": 0.9},
        "api": {"s2_rate_limit": 100, "cache_ttl": 86400},
    }


def test_graph_neighbors_concept():
    """graph neighbors shows concept neighbors with path info."""
    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td) / "papers"
        papers_dir.mkdir()
        db_path = f"{td}/test.db"

        from drbrain.storage.database import Database

        db = Database(db_path)
        db.insert_paper("paper_a", "Paper A", 2023, "uploaded")
        db.insert_concept("paper_a", "Method", "method_x", 0.9, year=2023)
        db.insert_concept("paper_a", "Gap", "gap_y", 0.8, year=2023)
        db.insert_edge("method_x", "gap_y", "addresses", "paper_a", 1.0)
        db.commit()
        db.close()

        cfg = _make_minimal_config(db_path, str(papers_dir))

        old_stdout = sys.stdout
        capture = io.StringIO()
        sys.stdout = capture
        try:
            with mock.patch("drbrain.cli.graph_commands.load_config", return_value=cfg):
                neighbors_cmd(
                    node_label="method_x",
                    hops=1,
                    relation=None,
                    direction="both",
                    json_output=True,
                    workspace=None,
                )
        finally:
            sys.stdout = old_stdout

        results = json.loads(capture.getvalue())
        graph_results = [r for r in results if r.get("_via_graph")]
        assert any(r["local_id"] == "gap_y" and r["type"] == "Gap" for r in graph_results)


def test_graph_neighbors_node_not_found():
    """graph neighbors with nonexistent node shows error."""
    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td) / "papers"
        papers_dir.mkdir()
        db_path = f"{td}/test.db"

        from drbrain.storage.database import Database

        db = Database(db_path)
        db.close()

        cfg = _make_minimal_config(db_path, str(papers_dir))

        old_stdout = sys.stdout
        capture = io.StringIO()
        sys.stdout = capture
        try:
            with mock.patch("drbrain.cli.graph_commands.load_config", return_value=cfg):
                neighbors_cmd(
                    node_label="nonexistent",
                    hops=1,
                    relation=None,
                    direction="both",
                    json_output=False,
                    workspace=None,
                )
        finally:
            sys.stdout = old_stdout

        output = capture.getvalue()
        assert "not found" in output.lower()
