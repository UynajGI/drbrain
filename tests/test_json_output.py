"""Tests for --json flag on all CLI commands."""
import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from brbrain.cli.main import app
from brbrain.storage.database import Database

runner = CliRunner()


def _make_db(td: str) -> Database:
    db = Database(Path(td) / "test.db")
    db.insert_paper("p1", "Test Paper 1", 2024, "uploaded")
    db.insert_paper("p2", "Test Paper 2", 2023, "uploaded")
    db.insert_concept("p1", "Method", "Transformer", 0.95, year=2024)
    db.insert_concept("p2", "Method", "GNN", 0.9, year=2023)
    db.insert_concept("p1", "Problem", "Long-range dependency", 0.8, year=2024)
    db.insert_edge("p1", "p2", "extends", "p1", 1.0)
    db.insert_argument(
        "p1", "Transformer outperforms RNN on sequence tasks",
        "proposes", "Transformer", "Method", "empirical", "WMT14", 0.95,
    )
    db.insert_seed("stale_problem", "Test seed", 0.5)
    db.insert_queue_item("p1", "concept", json.dumps({"label": "weak_concept", "type": "Method"}), 0.4)
    db.commit()
    return db


def _write_config(td: str, db_path: str) -> None:
    cfg = {
        "llm": {"models": [{"provider": "ollama", "model": "qwen2.5:7b", "api_key": None, "base_url": "http://localhost:11434"}]},
        "mineru": {"token": "", "model": "vlm", "is_ocr": False, "enable_formula": True, "enable_table": True},
        "db": {"path": db_path},
        "dirs": {"pdfs": "data/pdfs", "reports": "data/reports", "cache": "data/cache", "logs": "data/logs"},
        "api": {"s2_rate_limit": 100, "cache_ttl": 86400, "crossref_email": "", "openalex_token": ""},
        "bm25": {"k1": 1.5, "b": 0.75},
        "queue": {"weak_threshold": 0.7, "auto_accept": 0.9},
    }
    with open(Path(td) / "config.yaml", "w") as f:
        yaml.dump(cfg, f)


def _run(td: str, args: list[str]):
    """Run CLI command in temp directory with config."""
    return runner.invoke(app, args, catch_exceptions=False)


class TestListJson:
    def test_list_json_outputs_valid_json(self):
        with tempfile.TemporaryDirectory() as td:
            db = _make_db(td)
            db.close()
            _write_config(td, str(Path(td) / "test.db"))
            old = os.getcwd()
            os.chdir(td)
            try:
                result = runner.invoke(app, ["list", "--json"])
                assert result.exit_code == 0
                data = json.loads(result.output)
                assert isinstance(data, list)
                assert len(data) == 2
                assert "local_id" in data[0]
                assert "title" in data[0]
            finally:
                os.chdir(old)

    def test_list_without_json_still_works(self):
        with tempfile.TemporaryDirectory() as td:
            db = _make_db(td)
            db.close()
            _write_config(td, str(Path(td) / "test.db"))
            old = os.getcwd()
            os.chdir(td)
            try:
                result = runner.invoke(app, ["list"])
                assert result.exit_code == 0
                assert "Papers" in result.output
            finally:
                os.chdir(old)


class TestStatsJson:
    def test_stats_json_outputs_valid_json(self):
        with tempfile.TemporaryDirectory() as td:
            db = _make_db(td)
            db.close()
            _write_config(td, str(Path(td) / "test.db"))
            old = os.getcwd()
            os.chdir(td)
            try:
                result = runner.invoke(app, ["stats", "--json"])
                assert result.exit_code == 0
                data = json.loads(result.output)
                assert "papers" in data
                assert "concepts" in data
                assert "edges" in data
                assert isinstance(data["papers"], int)
            finally:
                os.chdir(old)


class TestQueryJson:
    def test_query_json_outputs_valid_json(self):
        with tempfile.TemporaryDirectory() as td:
            db = _make_db(td)
            db.close()
            _write_config(td, str(Path(td) / "test.db"))
            old = os.getcwd()
            os.chdir(td)
            try:
                result = runner.invoke(app, ["query", "Transformer", "--json"])
                assert result.exit_code == 0
                data = json.loads(result.output)
                assert isinstance(data, list)
                assert any("Transformer" in str(r.get("label", "")) for r in data)
            finally:
                os.chdir(old)

    def test_query_json_with_type_filter(self):
        with tempfile.TemporaryDirectory() as td:
            db = _make_db(td)
            db.close()
            _write_config(td, str(Path(td) / "test.db"))
            old = os.getcwd()
            os.chdir(td)
            try:
                result = runner.invoke(app, ["query", "Transformer", "--json", "--type-filter=Method"])
                assert result.exit_code == 0
                data = json.loads(result.output)
                for item in data:
                    assert item["type"] == "Method"
            finally:
                os.chdir(old)

    def test_query_jsonl_stream(self):
        """Query --jsonl outputs newline-delimited JSON."""
        with tempfile.TemporaryDirectory() as td:
            db = _make_db(td)
            db.close()
            _write_config(td, str(Path(td) / "test.db"))
            old = os.getcwd()
            os.chdir(td)
            try:
                result = runner.invoke(app, ["query", "Transformer", "--jsonl"])
                assert result.exit_code == 0
                lines = result.output.strip().split("\n")
                assert len(lines) >= 1
                for line in lines:
                    data = json.loads(line)
                    assert "label" in data
            finally:
                os.chdir(old)


class TestSeedJson:
    def test_seed_json_outputs_valid_json(self):
        with tempfile.TemporaryDirectory() as td:
            db = _make_db(td)
            db.close()
            _write_config(td, str(Path(td) / "test.db"))
            old = os.getcwd()
            os.chdir(td)
            try:
                result = runner.invoke(app, ["seed", "--json"])
                assert result.exit_code == 0
                data = json.loads(result.output)
                assert isinstance(data, list)
                # Can be empty with no graph data, but should be valid JSON list
            finally:
                os.chdir(old)


class TestQueueJson:
    def test_queue_json_outputs_valid_json(self):
        with tempfile.TemporaryDirectory() as td:
            db = _make_db(td)
            db.close()
            _write_config(td, str(Path(td) / "test.db"))
            old = os.getcwd()
            os.chdir(td)
            try:
                result = runner.invoke(app, ["queue", "--json"])
                assert result.exit_code == 0
                data = json.loads(result.output)
                assert isinstance(data, list)
            finally:
                os.chdir(old)


class TestTimelineJson:
    def test_timeline_json_outputs_valid_json(self):
        with tempfile.TemporaryDirectory() as td:
            db = _make_db(td)
            db.close()
            _write_config(td, str(Path(td) / "test.db"))
            old = os.getcwd()
            os.chdir(td)
            try:
                result = runner.invoke(app, ["timeline", "Transformer", "--json"])
                assert result.exit_code == 0
                data = json.loads(result.output)
                assert "label" in data or "concept" in data
            finally:
                os.chdir(old)

    def test_timeline_json_no_data(self):
        """Timeline --json for nonexistent concept returns empty result."""
        with tempfile.TemporaryDirectory() as td:
            db = _make_db(td)
            db.close()
            _write_config(td, str(Path(td) / "test.db"))
            old = os.getcwd()
            os.chdir(td)
            try:
                result = runner.invoke(app, ["timeline", "nonexistent", "--json"])
                assert result.exit_code == 0
                data = json.loads(result.output)
                assert "evolution" in data
            finally:
                os.chdir(old)


class TestClosureJson:
    def test_closure_json_outputs_valid_json(self):
        with tempfile.TemporaryDirectory() as td:
            db = _make_db(td)
            db.close()
            _write_config(td, str(Path(td) / "test.db"))
            old = os.getcwd()
            os.chdir(td)
            try:
                result = runner.invoke(app, ["closure", "--json"])
                assert result.exit_code == 0
                data = json.loads(result.output)
                assert "inferred" in data or isinstance(data, list)
            finally:
                os.chdir(old)
