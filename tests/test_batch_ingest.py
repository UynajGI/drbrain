"""Tests for batch ingest orchestration (file enumeration, counting, skip-on-fail).

These tests verify how ingest_cmd dispatches PDFs to _ingest_single_paper and
aggregates results. The per-paper pipeline (PDF parse, identify, tree build,
LLM extraction) is mocked at the _ingest_single_paper boundary so the tests
exercise only batch logic — not the 6+ external calls the real pipeline makes
(MinerU, OpenAlex, CrossRef, async LLM calls, etc.).

Each mock inserts a real paper row into the DB so the tests can assert on the
post-ingest DB state that ingest_cmd's callers rely on.
"""

import tempfile
from pathlib import Path
from unittest import mock

import typer

from drbrain.cli.commands import ingest_cmd


def _make_minimal_config(db_path: str, reports_dir: str) -> dict:
    """Return config dict for testing."""
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
        "openalex_token": None,
    }


def _make_ctx(cfg: dict):
    """Create a minimal typer.Context mock with config pre-loaded."""
    ctx = mock.MagicMock(spec=typer.Context)
    ctx.obj = {"config": cfg}
    return ctx


def _make_success_factory(start_idx: int = 0):
    """Build a side_effect that inserts a real paper row for each PDF.

    Returns a (side_effect_fn, call_counter) pair. The side_effect mimics the
    DB-writing portion of _ingest_single_paper (insert_paper + insert_paper_ids)
    so callers can assert on post-ingest DB state. ``call_counter`` is a list
    whose [0] tracks how many times the mock was invoked.
    """

    import uuid

    counter = [start_idx]
    calls = [0]

    def side_effect(pdf_path, cfg, db, dedup, **kwargs):
        calls[0] += 1
        idx = counter[0]
        counter[0] += 1
        local_id = f"p{uuid.uuid4().hex[:6]}"
        db.insert_paper(
            local_id,
            f"Test Paper {idx}",
            2024 + idx,
            "uploaded",
        )
        db.insert_paper_ids(local_id, arxiv=f"2401.{10000 + idx}")
        db.commit()
        return {
            "ok": True,
            "local_id": local_id,
            "report": {"local_id": local_id, "title": f"Test Paper {idx}"},
        }

    return side_effect, calls


def _failing_on(filename_substr: str, start_idx: int = 0):
    """Build a side_effect that reports failure for files matching ``filename_substr``.

    Mirrors the real _ingest_single_paper contract: failures are returned as
    ``{"ok": False, ...}`` rather than raised, so ingest_cmd's batch loop can
    continue. Successful invocations insert a real paper row.
    """

    import uuid

    counter = [start_idx]
    calls = [0]

    def side_effect(pdf_path, cfg, db, dedup, **kwargs):
        calls[0] += 1
        if filename_substr in str(pdf_path):
            return {
                "ok": False,
                "local_id": None,
                "error": "simulated parse failure",
            }
        idx = counter[0]
        counter[0] += 1
        local_id = f"p{uuid.uuid4().hex[:6]}"
        db.insert_paper(
            local_id,
            f"Test Paper {idx}",
            2024 + idx,
            "uploaded",
        )
        db.insert_paper_ids(local_id, arxiv=f"2401.{10000 + idx}")
        db.commit()
        return {
            "ok": True,
            "local_id": local_id,
            "report": {"local_id": local_id, "title": f"Test Paper {idx}"},
        }

    return side_effect, calls


# ── Tests ─────────────────────────────────────────────────────────────────


def test_ingest_single_file():
    """ingest_cmd processes a single PDF file successfully."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        reports_dir.mkdir()

        pdf_path = Path(td) / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 dummy")

        ctx = _make_ctx(_make_minimal_config(str(db_path), str(reports_dir)))
        side_effect, _ = _make_success_factory()
        with mock.patch(
            "drbrain.cli.ingest_commands._ingest_single_paper", side_effect=side_effect
        ):
            ingest_cmd(ctx, [str(pdf_path)])

        from drbrain.storage.database import Database

        db = Database(str(db_path))
        papers = [p for p in db.get_all_papers() if p["status"] == "uploaded"]
        assert len(papers) == 1
        assert papers[0]["title"] == "Test Paper 0"
        db.close()


def test_ingest_directory():
    """ingest_cmd processes all PDFs in a directory."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        reports_dir.mkdir()
        pdfs_dir = Path(td) / "pdfs"
        pdfs_dir.mkdir()

        (pdfs_dir / "paper1.pdf").write_bytes(b"%PDF-1.4 dummy1")
        (pdfs_dir / "paper2.pdf").write_bytes(b"%PDF-1.4 dummy2")

        ctx = _make_ctx(_make_minimal_config(str(db_path), str(reports_dir)))
        side_effect, _ = _make_success_factory()
        with mock.patch(
            "drbrain.cli.ingest_commands._ingest_single_paper", side_effect=side_effect
        ):
            ingest_cmd(ctx, [str(pdfs_dir)])

        from drbrain.storage.database import Database

        db = Database(str(db_path))
        papers = [p for p in db.get_all_papers() if p["status"] == "uploaded"]
        assert len(papers) == 2
        titles = {p["title"] for p in papers}
        assert titles == {"Test Paper 0", "Test Paper 1"}
        db.close()


def test_ingest_skips_failed_papers():
    """ingest_cmd continues processing when one paper fails."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        reports_dir.mkdir()
        pdfs_dir = Path(td) / "pdfs"
        pdfs_dir.mkdir()

        (pdfs_dir / "ok1.pdf").write_bytes(b"%PDF-1.4 dummy")
        (pdfs_dir / "fail.pdf").write_bytes(b"%PDF-1.4 fail")
        (pdfs_dir / "ok2.pdf").write_bytes(b"%PDF-1.4 dummy")

        ctx = _make_ctx(_make_minimal_config(str(db_path), str(reports_dir)))
        side_effect, calls = _failing_on("fail")
        with mock.patch(
            "drbrain.cli.ingest_commands._ingest_single_paper", side_effect=side_effect
        ):
            ingest_cmd(ctx, [str(pdfs_dir)])

        # Should have attempted all 3 files.
        assert calls[0] == 3

        from drbrain.storage.database import Database

        db = Database(str(db_path))
        papers = [p for p in db.get_all_papers() if p["status"] == "uploaded"]
        # Only the two non-failing papers should land in the DB.
        assert len(papers) == 2
        db.close()


def test_ingest_multiple_files():
    """ingest_cmd handles multiple individual file paths."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        reports_dir.mkdir()

        pdf1 = Path(td) / "a.pdf"
        pdf2 = Path(td) / "b.pdf"
        pdf1.write_bytes(b"%PDF-1.4 dummy")
        pdf2.write_bytes(b"%PDF-1.4 dummy")

        ctx = _make_ctx(_make_minimal_config(str(db_path), str(reports_dir)))
        side_effect, _ = _make_success_factory()
        with mock.patch(
            "drbrain.cli.ingest_commands._ingest_single_paper", side_effect=side_effect
        ):
            ingest_cmd(ctx, [str(pdf1), str(pdf2)])

        from drbrain.storage.database import Database

        db = Database(str(db_path))
        papers = [p for p in db.get_all_papers() if p["status"] == "uploaded"]
        assert len(papers) == 2
        db.close()


def test_ingest_exits_when_no_pdfs():
    """ingest_cmd raises Exit when no PDF files found."""
    import click

    with tempfile.TemporaryDirectory() as td:
        empty_dir = Path(td) / "empty"
        empty_dir.mkdir()
        db_path = Path(td) / "test.db"

        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        ctx = _make_ctx(cfg)
        try:
            ingest_cmd(ctx, [str(empty_dir)])
            assert False, "Should have raised Exit"
        except click.exceptions.Exit as e:
            assert e.exit_code == 1
