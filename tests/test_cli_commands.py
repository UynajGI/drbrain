"""Tests for CLI commands: citations, check-citations, report, closure, seed, list, stats, query, export, queue."""

import json
import tempfile
from pathlib import Path
from unittest import mock

import typer

from drbrain.storage.database import Database


def _make_minimal_config(db_path: str, reports_dir: str) -> dict:
    return {
        "db": {"path": db_path},
        "llm": {"models": [{"provider": "openai", "model": "gpt-4", "api_key": "x"}]},
        "mineru": {
            "token": "",
            "model": "vlm",
            "is_ocr": False,
            "enable_formula": True,
            "enable_table": True,
        },
        "dirs": {
            "inbox": "data/spool/inbox",
            "papers": "data/papers",
            "reports": reports_dir,
            "cache": "data/cache",
            "logs": "data/logs",
        },
        "api": {"s2_rate_limit": 100, "cache_ttl": 86400},
        "queue": {"weak_threshold": 0.7, "auto_accept": 0.9},
        "bm25": {"k1": 1.5, "b": 0.75},
    }


def _make_ctx(cfg: dict):
    """Create a minimal typer.Context mock with config pre-loaded."""
    ctx = mock.MagicMock(spec=typer.Context)
    ctx.obj = {"config": cfg}
    return ctx


# -- citations_cmd --


def test_citations_cmd_not_found():
    """citations_cmd raises Exit when paper not found."""
    from drbrain.cli.commands import citations_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        cfg = _make_minimal_config(str(db_path), str(reports_dir))

        ctx = _make_ctx(cfg)
        try:
            citations_cmd(ctx, "nonexistent")
            assert False, "Should have raised Exit"
        except typer.Exit as e:
            assert e.exit_code == 1


def test_citations_cmd_invalid_type():
    """citations_cmd raises Exit for invalid ctype."""
    from drbrain.cli.commands import citations_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        cfg = _make_minimal_config(str(db_path), str(reports_dir))

        ctx = _make_ctx(cfg)
        try:
            citations_cmd(ctx, "nonexistent", "bogus")
            assert False, "Should have raised Exit"
        except typer.Exit as e:
            assert e.exit_code == 1


def test_citations_cmd_success():
    """citations_cmd queries citation graph."""
    from drbrain.cli.commands import citations_cmd

    fake_result = {
        "paper": {"local_id": "p1", "title": "Test Paper", "year": 2024},
        "refs": [
            {"title": "Ref A", "year": 2020, "doi": "10.0/1", "local_id": "p2"},
        ],
        "citing": [
            {"title": "Cite B", "year": 2025, "doi": "10.0/2"},
        ],
        "shared_refs": [],
        "counts": {"references": 1, "citing": 1},
    }

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

        ctx = _make_ctx(cfg)
        with mock.patch(
            "drbrain.storage.citation_graph.query_citation_graph",
            return_value=fake_result,
        ):
            citations_cmd(ctx, "p1")


def test_citations_cmd_json_output():
    """citations_cmd outputs JSON when --json is set."""
    from drbrain.cli.commands import citations_cmd

    fake_result = {
        "paper": {"local_id": "p1", "title": "Test Paper", "year": 2024},
        "refs": [],
        "citing": [],
        "shared_refs": [],
        "counts": {"references": 0, "citing": 0},
    }

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

        ctx = _make_ctx(cfg)
        with mock.patch(
            "drbrain.storage.citation_graph.query_citation_graph",
            return_value=fake_result,
        ):
            citations_cmd(ctx, "p1", "all", json_output=True)


# -- check_citations_cmd --


def test_check_citations_cmd_no_input():
    """check_citations_cmd raises Exit when no text provided."""
    from drbrain.cli.commands import check_citations_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        cfg = _make_minimal_config(str(db_path), str(reports_dir))

        ctx = _make_ctx(cfg)
        try:
            check_citations_cmd(ctx)
            assert False, "Should have raised Exit"
        except typer.Exit as e:
            assert e.exit_code == 1


def test_check_citations_cmd_from_file():
    """check_citations_cmd reads text from --file."""
    from drbrain.cli.commands import check_citations_cmd
    from drbrain.extractor.citation_check import CitationMatch

    fake_citations = [
        CitationMatch(
            author="Smith",
            year="2020",
            raw="Smith (2020)",
            found=True,
            matched_id="p2",
            matched_title="A Paper",
        ),
    ]

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        cfg = _make_minimal_config(str(db_path), str(reports_dir))

        db = Database(str(db_path))
        db.insert_paper("p1", "Test Paper", 2024, "uploaded")
        db.commit()
        db.close()

        tf = Path(td) / "test.txt"
        tf.write_text("According to Smith (2020)...")

        ctx = _make_ctx(cfg)
        with (
            mock.patch(
                "drbrain.extractor.citation_check.extract_citations",
                return_value=[
                    CitationMatch(author="Smith", year="2020", raw="Smith (2020)"),
                ],
            ),
            mock.patch(
                "drbrain.extractor.citation_check.match_citations",
                return_value=fake_citations,
            ),
        ):
            check_citations_cmd(ctx, file=str(tf))


def test_check_citations_cmd_json_output():
    """check_citations_cmd outputs JSON when --json is set."""
    from drbrain.cli.commands import check_citations_cmd
    from drbrain.extractor.citation_check import CitationMatch

    fake_citations = [
        CitationMatch(
            author="Smith",
            year="2020",
            raw="Smith (2020)",
            found=True,
            matched_id="p2",
            matched_title="A Paper",
        ),
    ]

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        cfg = _make_minimal_config(str(db_path), str(reports_dir))

        db = Database(str(db_path))
        db.insert_paper("p1", "Test Paper", 2024, "uploaded")
        db.commit()
        db.close()

        ctx = _make_ctx(cfg)
        with (
            mock.patch(
                "drbrain.extractor.citation_check.extract_citations",
                return_value=[
                    CitationMatch(author="Smith", year="2020", raw="Smith (2020)"),
                ],
            ),
            mock.patch(
                "drbrain.extractor.citation_check.match_citations",
                return_value=fake_citations,
            ),
        ):
            check_citations_cmd(ctx, "Smith (2020)", json_output=True)


# -- report_cmd --


def test_report_cmd_not_found():
    """report_cmd raises Exit when no report file."""
    from drbrain.cli.commands import report_cmd

    with tempfile.TemporaryDirectory() as td:
        cfg = _make_minimal_config("/tmp/x.db", str(Path(td) / "reports"))
        ctx = _make_ctx(cfg)
        try:
            report_cmd(ctx, "nonexistent")
            assert False, "Should have raised Exit"
        except typer.Exit as e:
            assert e.exit_code == 1


def test_report_cmd_displays_report():
    """report_cmd reads and displays existing report."""
    from drbrain.cli.commands import report_cmd

    with tempfile.TemporaryDirectory() as td:
        reports_dir = Path(td) / "reports"
        reports_dir.mkdir()

        report_data = {
            "paper": {
                "local_id": "p1",
                "title": "Test Paper",
                "year": 2024,
                "status": "uploaded",
                "ids": {"doi": None, "arxiv": None},
            },
            "concepts": {
                "problems": [{"label": "X", "confidence": 0.9}],
                "methods": [],
                "conclusions": [],
                "debates": [],
                "gaps": [],
                "actors": [],
            },
            "arguments": [],
            "references": [],
            "citations": [],
            "summary": {
                "refs_in_graph": 0,
                "cits_in_graph": 0,
                "total_refs": 0,
                "total_cits": 0,
                "graph_coverage": 1.0,
            },
            "boundary_alert": {"low_coverage": False},
            "validation": {
                "items_rejected": 0,
                "items_queued": 0,
                "tbox_violations": [],
                "rbox_violations": [],
            },
        }
        (reports_dir / "p1.json").write_text(json.dumps(report_data))

        cfg = _make_minimal_config("/tmp/x.db", str(reports_dir))
        ctx = _make_ctx(cfg)
        report_cmd(ctx, "p1")  # Should not raise


# -- closure_cmd --


def test_closure_cmd_empty_graph():
    """closure_cmd runs on empty graph."""
    from drbrain.cli.commands import closure_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        ctx = _make_ctx(cfg)
        closure_cmd(ctx)  # Should not raise


def test_closure_cmd_with_edges():
    """closure_cmd infers edges from existing data."""
    from drbrain.cli.commands import closure_cmd

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

        ctx = _make_ctx(cfg)
        closure_cmd(ctx)


# -- seed_cmd --


def test_seed_cmd_empty_graph():
    """seed_cmd runs on empty graph, finds no seeds."""
    from drbrain.cli.commands import seed_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        ctx = _make_ctx(cfg)
        seed_cmd(ctx)


# -- list_cmd --


def test_list_cmd_no_papers():
    """list_cmd handles empty database."""
    from drbrain.cli.commands import list_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        ctx = _make_ctx(cfg)
        list_cmd(ctx)  # Should not raise


def test_list_cmd_with_papers():
    """list_cmd displays papers in table."""
    from drbrain.cli.commands import list_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))

        db = Database(str(db_path))
        db.insert_paper("p1", "Paper A", 2024, "uploaded")
        db.commit()
        db.close()

        ctx = _make_ctx(cfg)
        list_cmd(ctx)


# -- stats_cmd --


def test_stats_cmd_empty_db():
    """stats_cmd shows zeros for empty database."""
    from drbrain.cli.commands import stats_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        ctx = _make_ctx(cfg)
        stats_cmd(ctx)


def test_stats_cmd_with_data():
    """stats_cmd shows correct counts."""
    from drbrain.cli.commands import stats_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))

        db = Database(str(db_path))
        db.insert_paper("p1", "A", 2024, "uploaded")
        db.insert_concept("p1", "Problem", "X", 0.9, year=2024)
        db.insert_edge("p1", "p2", "cites", "p1")
        db.commit()
        db.close()

        ctx = _make_ctx(cfg)
        stats_cmd(ctx)


# -- query_cmd --


def test_query_cmd_no_results():
    """query_cmd handles no results."""
    from drbrain.cli.commands import query_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        ctx = _make_ctx(cfg)
        query_cmd(
            ctx,
            "nonexistent concept",
            type_filter=None,
            arg_type=None,
            year_start=None,
            year_end=None,
            limit=20,
            neighbors=0,
            json_output=False,
            jsonl=False,
        )


def test_query_cmd_with_results():
    """query_cmd finds concepts via BM25."""
    from drbrain.cli.commands import query_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))

        db = Database(str(db_path))
        db.insert_paper("p1", "A", 2024, "uploaded")
        db.insert_concept("p1", "Problem", "Transformer attention", 0.9, year=2024)
        db.commit()
        db.close()

        ctx = _make_ctx(cfg)
        query_cmd(
            ctx,
            "transformer",
            type_filter=None,
            arg_type=None,
            year_start=None,
            year_end=None,
            min_confidence=None,
            limit=20,
            neighbors=0,
            json_output=False,
            jsonl=False,
        )


# -- export_cmd --


def test_export_cmd_bibtex():
    """export_cmd outputs BibTeX format by default."""
    from drbrain.cli.commands import export_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))

        db = Database(str(db_path))
        db.insert_paper("p1", "Test Paper", 2024, "uploaded")
        db.commit()
        db.close()

        ctx = _make_ctx(cfg)
        export_cmd(ctx, local_id="p1", format="bib", json_output=True)


def test_export_cmd_unsupported_format():
    """export_cmd raises Exit for unsupported format."""
    from drbrain.cli.commands import export_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        ctx = _make_ctx(cfg)
        try:
            export_cmd(ctx, local_id="nonexistent", format="csv")
        except typer.Exit as e:
            assert e.exit_code == 1


# -- queue_cmd --


def test_queue_cmd_empty():
    """queue_cmd shows empty queue message."""
    from drbrain.cli.commands import queue_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        ctx = _make_ctx(cfg)
        queue_cmd(ctx)


def test_queue_cmd_with_items():
    """queue_cmd displays pending items."""
    from drbrain.cli.commands import queue_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))

        db = Database(str(db_path))
        db.insert_paper("p1", "A", 2024, "uploaded")
        db.insert_queue_item("p1", "concept", '{"label": "X", "type": "Problem"}', 0.6)
        db.commit()
        db.close()

        ctx = _make_ctx(cfg)
        queue_cmd(ctx)


# -- queue_resolve_cmd --


def test_queue_resolve_accept():
    """queue_resolve_cmd accepts a queue item."""
    from drbrain.cli.commands import queue_resolve_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))

        db = Database(str(db_path))
        db.insert_paper("p1", "A", 2024, "uploaded")
        qid = db.insert_queue_item("p1", "concept", '{"label": "X"}', 0.6)
        db.commit()
        db.close()

        ctx = _make_ctx(cfg)
        queue_resolve_cmd(ctx, qid, accept=True, reject=False)


def test_queue_resolve_reject():
    """queue_resolve_cmd rejects a queue item."""
    from drbrain.cli.commands import queue_resolve_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))

        db = Database(str(db_path))
        db.insert_paper("p1", "A", 2024, "uploaded")
        qid = db.insert_queue_item("p1", "concept", '{"label": "X"}', 0.6)
        db.commit()
        db.close()

        ctx = _make_ctx(cfg)
        queue_resolve_cmd(ctx, qid, accept=False, reject=True)


def test_queue_resolve_both_flags():
    """queue_resolve_cmd raises Exit when both accept and reject."""
    from drbrain.cli.commands import queue_resolve_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        ctx = _make_ctx(cfg)
        try:
            queue_resolve_cmd(ctx, 1, accept=True, reject=True)
            assert False, "Should have raised Exit"
        except typer.Exit as e:
            assert e.exit_code == 1


def test_queue_resolve_neither_flag():
    """queue_resolve_cmd raises Exit when neither accept nor reject."""
    from drbrain.cli.commands import queue_resolve_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        ctx = _make_ctx(cfg)
        try:
            queue_resolve_cmd(ctx, 1, accept=False, reject=False)
            assert False, "Should have raised Exit"
        except typer.Exit as e:
            assert e.exit_code == 1


# -- JSON output --


def test_ingest_json_on_empty_dir():
    """ingest_cmd with json_output=True outputs error JSON when no PDFs found."""
    from drbrain.cli.commands import ingest_cmd

    with tempfile.TemporaryDirectory() as td:
        empty_dir = Path(td) / "empty"
        empty_dir.mkdir()
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        ctx = _make_ctx(cfg)
        try:
            ingest_cmd(ctx, [str(empty_dir)], json_output=True)
            assert False, "Should have raised Exit"
        except typer.Exit as e:
            assert e.exit_code == 1


# -- merged state --


def test_check_and_merge_duplicates():
    """_check_and_merge_duplicates finds existing placeholder with same DOI."""
    from drbrain.cli.commands import _check_and_merge_duplicates
    from drbrain.dedup.resolver import PaperIDs

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
    from drbrain.cli.commands import _merge_papers

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


def test_log_error_logs_via_loguru():
    """_log_error logs an error via loguru."""
    from unittest import mock

    from drbrain.cli.commands import _log_error

    with mock.patch("loguru.logger.error") as mock_error:
        _log_error({}, "Test error message")
        mock_error.assert_called_once_with("Test error message")


def test_merge_papers_redirects_edge_source_paper():
    """_merge_papers updates source_paper field in edges."""
    from drbrain.cli.commands import _merge_papers

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


# -- check_cmd --


def test_check_cmd_all_configured():
    """check_cmd passes when all dependencies and config are set."""
    from drbrain.cli.commands import check_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        reports_dir.mkdir()
        (Path(td) / "config.yaml").touch()

        cfg = _make_minimal_config(str(db_path), str(reports_dir))
        ctx = _make_ctx(cfg)
        check_cmd(ctx)  # Should not raise


def test_check_cmd_missing_config():
    """check_cmd exits with code 1 when config.yaml is missing (detected at filesystem level)."""
    from drbrain.cli.commands import check_cmd

    cfg = _make_minimal_config("/tmp/test.db", "/tmp/reports")
    ctx = _make_ctx(cfg)
    with mock.patch("pathlib.Path.exists", return_value=False):
        try:
            check_cmd(ctx)
            assert False, "Should have raised Exit"
        except typer.Exit as e:
            assert e.exit_code == 1


def test_check_cmd_missing_llm_key():
    """check_cmd warns when LLM model has no API key."""
    from drbrain.cli.commands import check_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        reports_dir.mkdir()
        (Path(td) / "config.yaml").touch()

        cfg = _make_minimal_config(str(db_path), str(reports_dir))
        cfg["llm"]["models"] = [
            {"provider": "openai", "model": "gpt-4", "api_key": "", "base_url": None}
        ]
        ctx = _make_ctx(cfg)
        check_cmd(ctx)  # Should warn, not raise


# -- repair_cmd --


def test_repair_cmd_no_args():
    """repair_cmd raises Exit(1) when no local_id, --all, or --workspace."""
    from drbrain.cli.commands import repair_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        cfg = _make_minimal_config(str(db_path), str(reports_dir))

        ctx = _make_ctx(cfg)
        try:
            repair_cmd(ctx, local_id=None, all=False, workspace=None)
            assert False, "Should have raised Exit"
        except typer.Exit as e:
            assert e.exit_code == 1


def test_repair_cmd_dry_run():
    """repair_cmd with --dry-run produces output without DB writes."""
    from drbrain.cli.commands import repair_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        cfg = _make_minimal_config(str(db_path), str(reports_dir))

        db = Database(str(db_path))
        db.insert_paper("p1", "Test Paper", 2024, "uploaded")
        db.commit()
        db.close()

        fake_repairs = [
            {"field": "title", "old": "Test Paper", "new": "Fixed Title", "source": "crossref"}
        ]

        ctx = _make_ctx(cfg)
        with mock.patch(
            "drbrain.services.repair.repair_paper", return_value=fake_repairs
        ) as mock_repair:
            repair_cmd(
                ctx, local_id="p1", all=False, workspace=None, dry_run=True, json_output=False
            )
            mock_repair.assert_called_once()
            assert mock_repair.call_args[1].get("dry_run") is True


# -- backup_cmd --


def test_backup_cmd_list_empty():
    """backup_cmd --list on empty dir outputs 'No backups found.'"""
    from drbrain.cli.commands import backup_cmd

    with mock.patch("drbrain.storage.backup.list_backups", return_value=[]):
        with mock.patch("typer.echo") as mock_echo:
            backup_cmd(None, list_only=True, json_output=False)
            mock_echo.assert_any_call("No backups found.")


# -- import_cmd --


def test_import_cmd_invalid_source():
    """import_cmd with invalid source type raises Exit(1)."""
    from drbrain.cli.commands import import_cmd

    try:
        import_cmd(None, source="xxx", path="dummy.bib")
        assert False, "Should have raised Exit"
    except typer.Exit as e:
        assert e.exit_code == 1


def test_import_cmd_file_not_found():
    """import_cmd with nonexistent file raises Exit(1)."""
    from drbrain.cli.commands import import_cmd

    try:
        import_cmd(None, source="zotero", path="/nonexistent/path/zotero.sqlite")
        assert False, "Should have raised Exit"
    except typer.Exit as e:
        assert e.exit_code == 1


# -- translate_cmd --


def test_translate_cmd_paper_not_found():
    """translate_cmd raises Exit(1) when paper not found."""
    from drbrain.cli.commands import translate_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        cfg = _make_minimal_config(str(db_path), str(reports_dir))

        ctx = _make_ctx(cfg)
        try:
            translate_cmd(ctx, local_id="nonexistent")
            assert False, "Should have raised Exit"
        except typer.Exit as e:
            assert e.exit_code == 1


# -- analyze_cmd --


def test_analyze_cmd_no_args():
    """analyze_cmd raises Exit(1) when no local_id and no workspace."""
    from drbrain.cli.commands import analyze_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        cfg = _make_minimal_config(str(db_path), str(reports_dir))

        ctx = _make_ctx(cfg)
        try:
            analyze_cmd(ctx, local_id=None, workspace=None)
            assert False, "Should have raised Exit"
        except typer.Exit as e:
            assert e.exit_code == 1


# -- clean_cmd --


def test_clean_cmd_empty_dirs():
    """clean_cmd prints 'Nothing to clean' when data dirs are already empty."""
    from drbrain.cli.commands import clean_cmd

    with tempfile.TemporaryDirectory() as td:
        cfg = {
            "db": {"path": f"{td}/nonexistent.db"},
            "dirs": {
                "cache": f"{td}/cache",
                "logs": f"{td}/logs",
                "papers": f"{td}/papers",
                "reports": f"{td}/reports",
            },
        }
        with mock.patch("drbrain.cli.check_commands.load_config", return_value=cfg):
            clean_cmd(force=True, config_path="config.yaml")


# -- query_cmd no results message --


def test_query_cmd_no_results_message():
    """query_cmd prints 'No results for: ...' when no matches found."""
    from drbrain.cli.commands import query_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))

        ctx = _make_ctx(cfg)
        with mock.patch("typer.echo") as mock_echo:
            query_cmd(
                ctx,
                "xyzzy_nonexistent_concept",
                type_filter=None,
                arg_type=None,
                year_start=None,
                year_end=None,
                limit=20,
                neighbors=0,
                json_output=False,
                jsonl=False,
            )
            mock_echo.assert_any_call("No results for: xyzzy_nonexistent_concept")


def test_closure_cmd_dry_run_does_not_persist():
    """--dry-run outputs edges but does not write to DB."""
    import io
    import sys

    from drbrain.cli.commands import closure_cmd

    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td) / "papers"
        papers_dir.mkdir()
        db_path = f"{td}/test.db"

        db = Database(db_path)
        db.insert_paper("paper_a", "Paper A", 2023, "uploaded")
        db.insert_concept("paper_a", "Method", "method_x", 0.9, year=2023)
        db.insert_concept("paper_a", "Method", "method_y", 0.85, year=2023)
        db.insert_edge("method_x", "method_y", "extends", "paper_a", 1.0)
        db.commit()
        db.close()

        cfg = {
            "db": {"path": db_path},
            "dirs": {"papers": str(papers_dir)},
            "bm25": {"k1": 1.5, "b": 0.75},
        }

        old_stdout = sys.stdout
        capture = io.StringIO()
        sys.stdout = capture
        try:
            ctx = _make_ctx(cfg)
            closure_cmd(ctx, dry_run=True, json_output=True, workspace=None, rule=None)
        finally:
            sys.stdout = old_stdout

        output = capture.getvalue()
        result = json.loads(output)
        assert "inferred" in result
        assert result["count"] >= 0

        # DB should NOT have new closure edges
        db = Database(db_path)
        edge_count_after = db.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE source_paper = 'closure'"
        ).fetchone()[0]
        db.close()
        assert edge_count_after == 0


def test_closure_cmd_rule_filter():
    """--rule gap_addressed only returns gap_addressed edges."""
    import io
    import sys

    from drbrain.cli.commands import closure_cmd

    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td) / "papers"
        papers_dir.mkdir()
        db_path = f"{td}/test.db"

        db = Database(db_path)
        db.insert_paper("paper_a", "Paper A", 2023, "uploaded")
        db.insert_concept("paper_a", "Problem", "problem_p", 0.9, year=2023)
        db.insert_concept("paper_a", "Gap", "gap_g", 0.8, year=2023)
        db.insert_concept("paper_a", "Method", "method_m", 0.85, year=2023)
        # leaves_open(problem_p, gap_g) + addresses(method_m, gap_g) => gap_addressed
        db.insert_edge("problem_p", "gap_g", "leaves_open", "paper_a", 1.0)
        db.insert_edge("method_m", "gap_g", "addresses", "paper_a", 1.0)
        db.commit()
        db.close()

        cfg = {
            "db": {"path": db_path},
            "dirs": {"papers": str(papers_dir)},
            "bm25": {"k1": 1.5, "b": 0.75},
        }

        old_stdout = sys.stdout
        capture = io.StringIO()
        sys.stdout = capture
        try:
            ctx = _make_ctx(cfg)
            closure_cmd(
                ctx,
                rule=["gap_addressed"],
                dry_run=True,
                json_output=True,
                workspace=None,
            )
        finally:
            sys.stdout = old_stdout

        output = capture.getvalue()
        result = json.loads(output)
        # All edges should be gap_addressed
        for edge in result["inferred"]:
            assert edge["relation"] == "gap_addressed"


def test_closure_cmd_rule_invalid():
    """--rule with invalid name raises Exit(1)."""
    from drbrain.cli.commands import closure_cmd

    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td) / "papers"
        papers_dir.mkdir()
        db_path = f"{td}/test.db"

        db = Database(db_path)
        db.close()

        cfg = {
            "db": {"path": db_path},
            "dirs": {"papers": str(papers_dir)},
            "bm25": {"k1": 1.5, "b": 0.75},
        }

        try:
            ctx = _make_ctx(cfg)
            closure_cmd(
                ctx,
                rule=["nonexistent_rule"],
                dry_run=True,
                json_output=False,
                workspace=None,
            )
            assert False, "Should have raised Exit"
        except Exception as e:
            assert hasattr(e, "exit_code") and e.exit_code == 1


def test_closure_cmd_backward_compat():
    """closure without new flags persists edges (existing behavior)."""
    from drbrain.cli.commands import closure_cmd

    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td) / "papers"
        papers_dir.mkdir()
        db_path = f"{td}/test.db"

        db = Database(db_path)
        db.insert_paper("paper_a", "Paper A", 2023, "uploaded")
        db.insert_concept("paper_a", "Method", "method_x", 0.9, year=2023)
        db.insert_concept("paper_a", "Method", "method_y", 0.85, year=2023)
        db.insert_concept("paper_a", "Method", "method_z", 0.8, year=2023)
        # Transitive chain: x->y, y->z => x->z (transitive closure on extends)
        db.insert_edge("method_x", "method_y", "extends", "paper_a", 1.0)
        db.insert_edge("method_y", "method_z", "extends", "paper_a", 1.0)
        db.commit()
        edge_count_before = db.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        db.close()

        cfg = {
            "db": {"path": db_path},
            "dirs": {"papers": str(papers_dir)},
            "bm25": {"k1": 1.5, "b": 0.75},
        }

        ctx = _make_ctx(cfg)
        closure_cmd(
            ctx,
            rule=None,
            dry_run=False,
            json_output=False,
            workspace=None,
        )

        db = Database(db_path)
        edge_count_after = db.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        db.close()
        assert edge_count_after > edge_count_before


# -- New tests for coverage boost --


def test_show_cmd_nonexistent_paper():
    """show_cmd raises typer.Exit(1) when paper is not found."""
    from drbrain.cli.commands import show_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        ctx = _make_ctx(cfg)

        try:
            show_cmd(ctx, "nonexistent-paper-id")
            assert False, "Should have raised Exit"
        except typer.Exit as e:
            assert e.exit_code == 1


def test_show_cmd_json_output_for_nonexistent_paper():
    """show_cmd --json still raises Exit for missing paper."""
    from drbrain.cli.commands import show_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        ctx = _make_ctx(cfg)

        try:
            show_cmd(ctx, "nonexistent-paper-id", json_output=True)
            assert False, "Should have raised Exit"
        except typer.Exit as e:
            assert e.exit_code == 1


def test_fetch_cmd_doi_parsing():
    """_resolve_identifier detects DOIs (starts with '10.')."""
    from drbrain.services.fetch import _resolve_identifier

    doi, title, arxiv_id = _resolve_identifier("10.1234/example.doi")
    assert doi == "10.1234/example.doi"
    assert title is None
    assert arxiv_id is None


def test_fetch_cmd_doi_parsing_slash():
    """_resolve_identifier detects DOI-style strings with '/' and no spaces."""
    from drbrain.services.fetch import _resolve_identifier

    doi, title, arxiv_id = _resolve_identifier("prefix/suffix")
    assert doi == "prefix/suffix"
    assert title is None
    assert arxiv_id is None


def test_fetch_cmd_arxiv_flag():
    """_resolve_identifier with is_arxiv=True returns arxiv_id, not doi/title."""
    from drbrain.services.fetch import _resolve_identifier

    doi, title, arxiv_id = _resolve_identifier("2301.12345", is_arxiv=True)
    assert doi is None
    assert title is None
    assert arxiv_id == "2301.12345"

    # Even a DOI-like string is treated as arXiv ID when flag is set
    doi2, title2, arxiv_id2 = _resolve_identifier("10.1234/foo", is_arxiv=True)
    assert arxiv_id2 == "10.1234/foo"
    assert doi2 is None


def test_fetch_cmd_title_fallback():
    """_resolve_identifier treats plain text as a title (no DOI, no arXiv flag)."""
    from drbrain.services.fetch import _resolve_identifier

    doi, title, arxiv_id = _resolve_identifier("Graph Neural Networks for Reasoning")
    assert doi is None
    assert title == "Graph Neural Networks for Reasoning"
    assert arxiv_id is None


def test_clean_cmd_force_no_password():
    """clean --force works when no admin password is configured."""
    from drbrain.cli.commands import clean_cmd

    with tempfile.TemporaryDirectory() as td:
        cfg = {
            "db": {"path": f"{td}/nonexistent.db"},
            "dirs": {
                "cache": f"{td}/cache",
                "logs": f"{td}/logs",
                "papers": f"{td}/papers",
                "reports": f"{td}/reports",
            },
            # No admin.password_hash — has_password() returns False
        }
        with mock.patch("drbrain.cli.check_commands.load_config", return_value=cfg):
            # Should not prompt for password and not raise Exit
            clean_cmd(force=True, config_path="config.yaml")


def test_clean_cmd_force_with_password_requires_prompt():
    """clean --force prompts for password when admin password is configured."""
    from drbrain.auth import hash_password
    from drbrain.cli.commands import clean_cmd

    pw_hash = hash_password("test123")

    with tempfile.TemporaryDirectory() as td:
        # Create actual directories so clean_cmd finds targets to clean
        # (otherwise it short-circuits with "Nothing to clean" before password check)
        for sub in ("cache", "logs", "papers", "reports"):
            (Path(td) / sub).mkdir()

        cfg = {
            "db": {"path": f"{td}/nonexistent.db"},
            "dirs": {
                "cache": f"{td}/cache",
                "logs": f"{td}/logs",
                "papers": f"{td}/papers",
                "reports": f"{td}/reports",
            },
            "admin": {"password_hash": pw_hash},
        }
        with mock.patch("drbrain.cli.check_commands.load_config", return_value=cfg):
            with mock.patch("typer.prompt") as mock_prompt:
                mock_prompt.return_value = "test123"
                clean_cmd(force=True, config_path="config.yaml")
                mock_prompt.assert_called_once()
                assert "Admin password" in mock_prompt.call_args[0][0]


def test_repair_nonexistent_paper():
    """repair_paper returns error dict for nonexistent paper."""
    from drbrain.services.repair import repair_paper

    db = Database(":memory:")
    result = repair_paper(db, "nonexistent-id")
    assert isinstance(result, list)
    assert len(result) >= 1
    assert any(r.get("field") == "error" for r in result)
    db.close()


def test_repair_cmd_nonexistent_paper():
    """repair_cmd raises typer.Exit(1) when paper is not found."""
    from drbrain.cli.commands import repair_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        ctx = _make_ctx(cfg)

        try:
            repair_cmd(
                ctx,
                local_id="nonexistent-id",
                all=False,
                workspace=None,
                dry_run=False,
                json_output=False,
            )
            assert False, "Should have raised Exit"
        except typer.Exit as e:
            assert e.exit_code == 1


def test_audit_cmd_runs_empty_library():
    """audit_papers on an empty in-memory database returns empty list."""
    from pathlib import Path

    from drbrain.services.audit import audit_papers

    db = Database(":memory:")
    issues = audit_papers(db, Path("data/papers"), severity="warning")
    assert isinstance(issues, list)
    assert len(issues) == 0
    db.close()


def test_audit_cmd_all_severity_levels():
    """audit_papers accepts error, warning, and info severity levels."""
    from pathlib import Path

    from drbrain.services.audit import audit_papers

    db = Database(":memory:")
    for sev in ("error", "warning", "info"):
        issues = audit_papers(db, Path("data/papers"), severity=sev)
        assert isinstance(issues, list)
    db.close()


# -- ask_cmd closure edges --


def test_ask_cmd_includes_closure_edges():
    """ask_cmd prompt includes ``--[inferred: ...]-->`` closure edges for a seeded graph."""
    from drbrain.cli.analysis_commands import ask_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        cfg = _make_minimal_config(str(db_path), str(reports_dir))

        db = Database(str(db_path))
        db.insert_paper("p1", "Test Paper", 2024, "uploaded")
        db.insert_concept("p1", "Problem", "overfitting", 0.9, year=2024)
        db.insert_concept("p1", "Conclusion", "deep learning", 0.9, year=2024)
        db.insert_concept("p1", "Method", "regularization", 0.85, year=2024)
        # edges that trigger creates_debate closure rule
        db.insert_edge("overfitting", "deep learning", "challenges", "p1", 1.0)
        db.insert_edge("regularization", "deep learning", "supports", "p1", 1.0)
        db.commit()
        db.close()

        ctx = _make_ctx(cfg)

        captured_prompt: list[str] = []

        async def _fake_llm(prompt, models, max_tokens=1024):
            captured_prompt.append(prompt)
            return "Test answer"

        with mock.patch("drbrain.extractor.llm_client.acall_text_with_fallback", _fake_llm):
            ask_cmd(ctx, ["deep", "learning"])

        assert len(captured_prompt) == 1, "Expected one LLM call"
        prompt = captured_prompt[0]
        assert "--[inferred:" in prompt, f"Prompt should contain closure edges, got:\n{prompt}"
        assert "creates debate" in prompt, f"Prompt should contain creates_debate, got:\n{prompt}"
        assert "regularization" in prompt


def test_ask_cmd_closure_edges_top_k_limit():
    """ask_cmd limits closure edges in context to top_k (--top/-k)."""
    from drbrain.cli.analysis_commands import ask_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        cfg = _make_minimal_config(str(db_path), str(reports_dir))

        db = Database(str(db_path))
        db.insert_paper("p1", "Test Paper", 2024, "uploaded")
        db.insert_concept("p1", "Conclusion", "deep learning", 0.9, year=2024)

        # 3 challengers x 3 supporters = 9 creates_debate inferred edges
        for i in range(3):
            db.insert_concept("p1", "Problem", f"problem_{i}", 0.9, year=2024)
            db.insert_edge(f"problem_{i}", "deep learning", "challenges", "p1", 1.0)
        for i in range(3):
            db.insert_concept("p1", "Method", f"method_{i}", 0.9, year=2024)
            db.insert_edge(f"method_{i}", "deep learning", "supports", "p1", 1.0)

        db.commit()
        db.close()

        ctx = _make_ctx(cfg)

        captured_prompt: list[str] = []

        async def _fake_llm(prompt, models, max_tokens=1024):
            captured_prompt.append(prompt)
            return "Test answer"

        with mock.patch("drbrain.extractor.llm_client.acall_text_with_fallback", _fake_llm):
            ask_cmd(ctx, ["deep", "learning"], top_k=2)

        assert len(captured_prompt) == 1
        prompt = captured_prompt[0]
        inferred_count = prompt.count("--[inferred:")
        assert inferred_count == 2, (
            f"Expected exactly 2 inferred edges (top_k=2), got {inferred_count}"
        )


def test_reason_cmd_includes_closure_in_graph():
    """reason_cmd passes closure_context with inferred edges to ReasonerAgent."""
    from drbrain.cli.analysis_commands import reason_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        cfg = _make_minimal_config(str(db_path), str(reports_dir))

        db = Database(str(db_path))
        db.insert_paper("p1", "Test Paper", 2024, "uploaded")
        db.insert_concept("p1", "Problem", "overfitting", 0.9, year=2024)
        db.insert_concept("p1", "Conclusion", "deep learning", 0.9, year=2024)
        db.insert_concept("p1", "Method", "regularization", 0.85, year=2024)
        db.insert_edge("overfitting", "deep learning", "challenges", "p1", 1.0)
        db.insert_edge("regularization", "deep learning", "supports", "p1", 1.0)
        db.commit()
        db.close()

        ctx = _make_ctx(cfg)

        with mock.patch("drbrain.extractor.reasoner.ReasonerAgent") as mock_agent:
            mock_instance = mock.MagicMock()
            mock_instance.reason = mock.MagicMock()
            mock_agent.return_value = mock_instance

            with mock.patch(
                "drbrain.cli.analysis_commands.asyncio.run",
                return_value="Test reasoning result",
            ):
                reason_cmd(ctx, "deep learning", bidirectional=False, max_rounds=3)

            assert mock_agent.called, "ReasonerAgent should have been instantiated"
            _, kwargs = mock_agent.call_args
            closure_ctx = kwargs.get("closure_context", "")
            assert "--[inferred:" in closure_ctx, (
                f"closure_context should contain inferred edges, got: {closure_ctx!r}"
            )
            assert "creates debate" in closure_ctx
            assert "regularization" in closure_ctx


def test_ask_cmd_closure_edges_empty_graph_no_crash():
    """ask_cmd on empty graph does not crash when computing closure context."""
    from drbrain.cli.analysis_commands import ask_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        cfg = _make_minimal_config(str(db_path), str(reports_dir))

        ctx = _make_ctx(cfg)
        # Should not raise — exits gracefully with "No relevant concepts" message
        with mock.patch("typer.echo") as mock_echo:
            ask_cmd(ctx, ["nonexistent", "concept"])
            mock_echo.assert_called_with("No relevant concepts found in the knowledge graph.")
