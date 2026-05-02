"""Tests for drbrain graph subcommands: neighbors, path."""

import io
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import typer

from drbrain.cli.graph_commands import neighbors_cmd, path_cmd


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
                try:
                    neighbors_cmd(
                        node_label="nonexistent",
                        hops=1,
                        relation=None,
                        direction="both",
                        json_output=False,
                        workspace=None,
                    )
                except typer.Exit:
                    pass
        finally:
            sys.stdout = old_stdout

        output = capture.getvalue()
        assert "not found" in output.lower()


def test_graph_path_direct_connection():
    """path between directly connected nodes."""
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
                path_cmd(
                    src_label="method_x",
                    dst_label="gap_y",
                    max_length=6,
                    json_output=True,
                    workspace=None,
                )
        finally:
            sys.stdout = old_stdout

        result = json.loads(capture.getvalue())
        assert result["length"] == 1
        assert len(result["path"]) == 1
        assert result["path"][0]["src"] == "method_x"
        assert result["path"][0]["relation"] == "addresses"
        assert result["path"][0]["dst"] == "gap_y"


def test_graph_path_two_hop():
    """path between nodes two hops apart."""
    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td) / "papers"
        papers_dir.mkdir()
        db_path = f"{td}/test.db"

        from drbrain.storage.database import Database

        db = Database(db_path)
        db.insert_paper("paper_a", "Paper A", 2023, "uploaded")
        db.insert_concept("paper_a", "Method", "m1", 0.9, year=2023)
        db.insert_concept("paper_a", "Method", "m2", 0.85, year=2023)
        db.insert_concept("paper_a", "Method", "m3", 0.8, year=2023)
        db.insert_edge("m1", "m2", "extends", "paper_a", 1.0)
        db.insert_edge("m2", "m3", "replaces", "paper_a", 1.0)
        db.commit()
        db.close()

        cfg = _make_minimal_config(db_path, str(papers_dir))

        old_stdout = sys.stdout
        capture = io.StringIO()
        sys.stdout = capture
        try:
            with mock.patch("drbrain.cli.graph_commands.load_config", return_value=cfg):
                path_cmd(
                    src_label="m1",
                    dst_label="m3",
                    max_length=6,
                    json_output=True,
                    workspace=None,
                )
        finally:
            sys.stdout = old_stdout

        result = json.loads(capture.getvalue())
        assert result["length"] == 2
        assert len(result["path"]) == 2


def test_graph_path_src_not_found():
    """path with nonexistent source shows error."""
    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td) / "papers"
        papers_dir.mkdir()
        db_path = f"{td}/test.db"

        from drbrain.storage.database import Database

        db = Database(db_path)
        db.close()

        cfg = _make_minimal_config(db_path, str(papers_dir))

        try:
            with mock.patch("drbrain.cli.graph_commands.load_config", return_value=cfg):
                path_cmd(
                    src_label="nonexistent",
                    dst_label="anything",
                    max_length=6,
                    json_output=False,
                    workspace=None,
                )
            assert False, "Should have raised Exit"
        except Exception as e:
            assert hasattr(e, "exit_code") and e.exit_code == 1


def test_graph_path_no_path():
    """path between disconnected nodes shows 'no path' message."""
    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td) / "papers"
        papers_dir.mkdir()
        db_path = f"{td}/test.db"

        from drbrain.storage.database import Database

        db = Database(db_path)
        db.insert_paper("paper_a", "Paper A", 2023, "uploaded")
        db.insert_concept("paper_a", "Method", "m1", 0.9, year=2023)
        db.insert_concept("paper_a", "Gap", "g1", 0.8, year=2023)
        db.insert_concept("paper_a", "Method", "dummy_m1", 0.5, year=2023)
        db.insert_concept("paper_a", "Method", "dummy_g1", 0.5, year=2023)
        # Connect each node to its own dummy so both are in the graph
        # but are in disconnected subgraphs (no path between m1 and g1)
        db.insert_edge("m1", "dummy_m1", "extends", "paper_a", 1.0)
        db.insert_edge("g1", "dummy_g1", "addresses", "paper_a", 1.0)
        db.commit()
        db.close()

        cfg = _make_minimal_config(db_path, str(papers_dir))

        old_stdout = sys.stdout
        capture = io.StringIO()
        sys.stdout = capture
        try:
            with mock.patch("drbrain.cli.graph_commands.load_config", return_value=cfg):
                path_cmd(
                    src_label="m1",
                    dst_label="g1",
                    max_length=6,
                    json_output=False,
                    workspace=None,
                )
        finally:
            sys.stdout = old_stdout

        output = capture.getvalue()
        assert "No path found" in output
