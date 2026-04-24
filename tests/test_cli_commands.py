"""Tests for CLI commands: expand, report, closure, seed, list, stats, query, export, queue, timeline."""
import json
import tempfile
from pathlib import Path
from unittest import mock

import typer
from brbrain.storage.database import Database
from brbrain.graph.engine import GraphEngine


def _make_minimal_config(db_path: str, reports_dir: str) -> dict:
    return {
        "db": {"path": db_path},
        "llm": {"models": [{"provider": "openai", "model": "gpt-4", "api_key": "x"}]},
        "mineru": {"token": "", "model": "vlm", "is_ocr": False, "enable_formula": True, "enable_table": True},
        "dirs": {"reports": reports_dir, "pdfs": "data/pdfs", "cache": "data/cache", "logs": "data/logs"},
        "api": {"s2_rate_limit": 100, "cache_ttl": 86400},
        "queue": {"weak_threshold": 0.7, "auto_accept": 0.9},
        "bm25": {"k1": 1.5, "b": 0.75},
    }


def _mock_load_config(cfg: dict):
    return mock.patch("brbrain.cli.commands.load_config", return_value=cfg)


# -- expand_cmd --

def test_expand_cmd_not_found():
    """expand_cmd raises Exit when paper not found."""
    from brbrain.cli.commands import expand_cmd
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        cfg = _make_minimal_config(str(db_path), str(reports_dir))

        with _mock_load_config(cfg):
            try:
                expand_cmd("nonexistent")
                assert False, "Should have raised Exit"
            except typer.Exit as e:
                assert e.exit_code == 1


def test_expand_cmd_success():
    """expand_cmd expands citation neighborhood."""
    from brbrain.cli.commands import expand_cmd
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        reports_dir.mkdir()
        cfg = _make_minimal_config(str(db_path), str(reports_dir))

        db = Database(str(db_path))
        db.insert_paper("p1", "Test Paper", 2024, "uploaded")
        db.insert_paper_ids("p1", s2_id="s2_123")
        db.commit()
        db.close()

        with _mock_load_config(cfg), \
             mock.patch("brbrain.extractor.citation.expand_citations", return_value=([], [])):
            expand_cmd("p1")


# -- report_cmd --

def test_report_cmd_not_found():
    """report_cmd raises Exit when no report file."""
    from brbrain.cli.commands import report_cmd
    with tempfile.TemporaryDirectory() as td:
        cfg = _make_minimal_config("/tmp/x.db", str(Path(td) / "reports"))
        with _mock_load_config(cfg):
            try:
                report_cmd("nonexistent")
                assert False, "Should have raised Exit"
            except typer.Exit as e:
                assert e.exit_code == 1


def test_report_cmd_displays_report():
    """report_cmd reads and displays existing report."""
    from brbrain.cli.commands import report_cmd
    with tempfile.TemporaryDirectory() as td:
        reports_dir = Path(td) / "reports"
        reports_dir.mkdir()

        report_data = {
            "paper": {"local_id": "p1", "title": "Test Paper", "year": 2024, "status": "uploaded",
                      "ids": {"doi": None, "arxiv": None}},
            "concepts": {"problems": [{"label": "X", "confidence": 0.9}], "methods": [],
                         "conclusions": [], "debates": [], "gaps": [], "actors": []},
            "arguments": [],
            "references": [], "citations": [],
            "summary": {"refs_in_graph": 0, "cits_in_graph": 0, "total_refs": 0, "total_cits": 0,
                        "graph_coverage": 1.0},
            "boundary_alert": {"low_coverage": False},
            "validation": {"items_rejected": 0, "items_queued": 0, "tbox_violations": [], "rbox_violations": []},
        }
        (reports_dir / "p1.json").write_text(json.dumps(report_data))

        cfg = _make_minimal_config("/tmp/x.db", str(reports_dir))
        with _mock_load_config(cfg):
            report_cmd("p1")  # Should not raise


# -- closure_cmd --

def test_closure_cmd_empty_graph():
    """closure_cmd runs on empty graph."""
    from brbrain.cli.commands import closure_cmd
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        with _mock_load_config(cfg):
            closure_cmd()  # Should not raise


def test_closure_cmd_with_edges():
    """closure_cmd infers edges from existing data."""
    from brbrain.cli.commands import closure_cmd
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))

        db = Database(str(db_path))
        db.insert_paper("p1", "A", 2024, "uploaded")
        db.insert_paper("p2", "B", 2024, "uploaded")
        db.insert_edge("p1", "Conclusion_X", "challenges", "p1")
        db.insert_edge("p2", "Conclusion_X", "supports", "p2")
        db.commit()
        db.close()

        with _mock_load_config(cfg):
            closure_cmd()


# -- seed_cmd --

def test_seed_cmd_empty_graph():
    """seed_cmd runs on empty graph, finds no seeds."""
    from brbrain.cli.commands import seed_cmd
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        with _mock_load_config(cfg):
            seed_cmd()


# -- list_cmd --

def test_list_cmd_no_papers():
    """list_cmd handles empty database."""
    from brbrain.cli.commands import list_cmd
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        with _mock_load_config(cfg):
            list_cmd()  # Should not raise


def test_list_cmd_with_papers():
    """list_cmd displays papers in table."""
    from brbrain.cli.commands import list_cmd
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))

        db = Database(str(db_path))
        db.insert_paper("p1", "Paper A", 2024, "uploaded")
        db.commit()
        db.close()

        with _mock_load_config(cfg):
            list_cmd()


# -- stats_cmd --

def test_stats_cmd_empty_db():
    """stats_cmd shows zeros for empty database."""
    from brbrain.cli.commands import stats_cmd
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        with _mock_load_config(cfg):
            stats_cmd()


def test_stats_cmd_with_data():
    """stats_cmd shows correct counts."""
    from brbrain.cli.commands import stats_cmd
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))

        db = Database(str(db_path))
        db.insert_paper("p1", "A", 2024, "uploaded")
        db.insert_concept("p1", "Problem", "X", 0.9, year=2024)
        db.insert_edge("p1", "p2", "cites", "p1")
        db.commit()
        db.close()

        with _mock_load_config(cfg):
            stats_cmd()


# -- query_cmd --

def test_query_cmd_no_results():
    """query_cmd handles no results."""
    from brbrain.cli.commands import query_cmd
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        with _mock_load_config(cfg):
            query_cmd("nonexistent concept")


def test_query_cmd_with_results():
    """query_cmd finds concepts via BM25."""
    from brbrain.cli.commands import query_cmd
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))

        db = Database(str(db_path))
        db.insert_paper("p1", "A", 2024, "uploaded")
        db.insert_concept("p1", "Problem", "Transformer attention", 0.9, year=2024)
        db.commit()
        db.close()

        with _mock_load_config(cfg):
            query_cmd("transformer", type_filter=None, arg_type=None,
                      year_start=None, year_end=None, limit=20,
                      json_output=False, jsonl=False)


# -- export_cmd --

def test_export_cmd_json():
    """export_cmd outputs JSON format."""
    from brbrain.cli.commands import export_cmd
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))

        db = Database(str(db_path))
        db.insert_paper("p1", "A", 2024, "uploaded")
        db.commit()
        db.close()

        with _mock_load_config(cfg):
            export_cmd("json")  # Should not raise


def test_export_cmd_unsupported_format():
    """export_cmd raises Exit for unsupported format."""
    from brbrain.cli.commands import export_cmd
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        with _mock_load_config(cfg):
            try:
                export_cmd("csv")
                assert False, "Should have raised Exit"
            except typer.Exit as e:
                assert e.exit_code == 1


# -- queue_cmd --

def test_queue_cmd_empty():
    """queue_cmd shows empty queue message."""
    from brbrain.cli.commands import queue_cmd
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        with _mock_load_config(cfg):
            queue_cmd()


def test_queue_cmd_with_items():
    """queue_cmd displays pending items."""
    from brbrain.cli.commands import queue_cmd
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))

        db = Database(str(db_path))
        db.insert_paper("p1", "A", 2024, "uploaded")
        db.insert_queue_item("p1", "concept", '{"label": "X", "type": "Problem"}', 0.6)
        db.commit()
        db.close()

        with _mock_load_config(cfg):
            queue_cmd()


# -- queue_resolve_cmd --

def test_queue_resolve_accept():
    """queue_resolve_cmd accepts a queue item."""
    from brbrain.cli.commands import queue_resolve_cmd
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))

        db = Database(str(db_path))
        db.insert_paper("p1", "A", 2024, "uploaded")
        qid = db.insert_queue_item("p1", "concept", '{"label": "X"}', 0.6)
        db.commit()
        db.close()

        with _mock_load_config(cfg):
            queue_resolve_cmd(qid, accept=True, reject=False)


def test_queue_resolve_reject():
    """queue_resolve_cmd rejects a queue item."""
    from brbrain.cli.commands import queue_resolve_cmd
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))

        db = Database(str(db_path))
        db.insert_paper("p1", "A", 2024, "uploaded")
        qid = db.insert_queue_item("p1", "concept", '{"label": "X"}', 0.6)
        db.commit()
        db.close()

        with _mock_load_config(cfg):
            queue_resolve_cmd(qid, accept=False, reject=True)


def test_queue_resolve_both_flags():
    """queue_resolve_cmd raises Exit when both accept and reject."""
    from brbrain.cli.commands import queue_resolve_cmd
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        with _mock_load_config(cfg):
            try:
                queue_resolve_cmd(1, accept=True, reject=True)
                assert False, "Should have raised Exit"
            except typer.Exit as e:
                assert e.exit_code == 1


def test_queue_resolve_neither_flag():
    """queue_resolve_cmd raises Exit when neither accept nor reject."""
    from brbrain.cli.commands import queue_resolve_cmd
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        with _mock_load_config(cfg):
            try:
                queue_resolve_cmd(1, accept=False, reject=False)
                assert False, "Should have raised Exit"
            except typer.Exit as e:
                assert e.exit_code == 1


# -- timeline_cmd --

def test_timeline_cmd_no_data():
    """timeline_cmd handles concept with no data."""
    from brbrain.cli.commands import timeline_cmd
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        with _mock_load_config(cfg):
            timeline_cmd("NonexistentConcept")


def test_timeline_cmd_with_data():
    """timeline_cmd shows concept evolution."""
    from brbrain.cli.commands import timeline_cmd
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))

        current_year = 2026
        db = Database(str(db_path))
        db.insert_paper("p1", "A", current_year, "uploaded")
        db.insert_concept("p1", "Method", "Transformer", 0.9, year=current_year)
        db.commit()
        db.close()

        with _mock_load_config(cfg):
            timeline_cmd("Transformer")


# -- JSON output --

def test_ingest_json_on_empty_dir():
    """ingest_cmd with json_output=True outputs error JSON when no PDFs found."""
    from brbrain.cli.commands import ingest_cmd
    with tempfile.TemporaryDirectory() as td:
        empty_dir = Path(td) / "empty"
        empty_dir.mkdir()
        try:
            ingest_cmd([str(empty_dir)], json_output=True)
            assert False, "Should have raised Exit"
        except typer.Exit as e:
            assert e.exit_code == 1


# -- merged state --

def test_check_and_merge_duplicates():
    """_check_and_merge_duplicates finds existing placeholder with same DOI."""
    from brbrain.cli.commands import _check_and_merge_duplicates
    from brbrain.dedup.resolver import PaperIDs
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        db = Database(str(db_path))
        # Existing placeholder with same DOI
        db.insert_paper("p_placeholder", "Old Title", 2020, "placeholder")
        db.insert_paper_ids("p_placeholder", doi="10.1234/test")
        db.commit()

        ids = PaperIDs(doi="10.1234/test", arxiv=None)
        result = _check_and_merge_duplicates(db, ids, "New Title", 2024)
        assert result == "p_placeholder"
        db.close()


def test_merge_papers():
    """_merge_papers merges concepts, arguments, and edges from one paper to another."""
    from brbrain.cli.commands import _merge_papers
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        db = Database(str(db_path))
        db.insert_paper("p_keep", "Keep", 2024, "uploaded")
        db.insert_paper("p_merge", "Merge Me", 2023, "placeholder")
        db.insert_concept("p_merge", "Problem", "X", 0.9, year=2023)
        db.insert_argument("p_merge", "claim", "supports", "Y", "Method")
        db.insert_edge("p_merge", "p_keep", "extends", "p_merge")
        db.commit()

        _merge_papers(db, "p_keep", "p_merge")

        # Concept moved
        concepts = db.get_concepts_by_paper("p_keep")
        assert len(concepts) == 1
        assert concepts[0]["label"] == "X"

        # Edge redirected
        edges = db.conn.execute("SELECT src_id, dst_id FROM edges WHERE src_id='p_keep'").fetchall()
        assert any(e[0] == "p_keep" for e in edges)

        # Merged paper deleted
        paper = db.get_paper("p_merge")
        assert paper is None
        db.close()



# -- _log_error --

def test_log_error_writes_to_file():
    """_log_error creates validation.log in configured logs directory."""
    from brbrain.cli.commands import _log_error
    with tempfile.TemporaryDirectory() as td:
        cfg = {"dirs": {"logs": str(Path(td) / "logs")}}
        _log_error(cfg, "Test error message")
        log_path = Path(td) / "logs" / "validation.log"
        assert log_path.exists()
        content = log_path.read_text()
        assert "Test error message" in content


def test_merge_papers_redirects_edge_source_paper():
    """_merge_papers updates source_paper field in edges."""
    from brbrain.cli.commands import _merge_papers
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        db = Database(str(db_path))
        db.insert_paper("p_keep", "Keep", 2024, "uploaded")
        db.insert_paper("p_merge", "Merge Me", 2023, "placeholder")
        db.insert_edge("p_keep", "p_keep", "cites", "p_merge")
        db.commit()

        _merge_papers(db, "p_keep", "p_merge")

        source = db.conn.execute(
            "SELECT source_paper FROM edges WHERE src_id='p_keep' AND dst_id='p_keep'"
        ).fetchone()[0]
        assert source == "p_keep"
        db.close()

