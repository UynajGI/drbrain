"""Tests for cli/repair_commands.py covering repair_cmd, import_cmd, enrich_cmd.

Note: command functions use typer.OptionInfo / typer.ArgumentInfo as parameter
defaults, so callers MUST pass every argument explicitly (False/None) rather
than relying on defaults, otherwise truthy OptionInfo sends control down the
wrong branch.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest
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
    ctx = mock.MagicMock(spec=typer.Context)
    ctx.obj = {"config": cfg}
    return ctx


def _setup_db(db_path: str, papers=None):
    db = Database(str(db_path))
    for p in papers or []:
        db.insert_paper(p["local_id"], p["title"], p.get("year"), "uploaded")
    db.commit()
    db.close()


def _echo_text() -> str:
    """Helper unavailable here; tests build printed text inline."""


# -- repair_cmd --


def test_repair_no_args_exits():
    """repair_cmd without args exits with code 1."""
    from drbrain.cli.repair_commands import repair_cmd

    with tempfile.TemporaryDirectory() as td:
        cfg = _make_minimal_config(str(Path(td) / "t.db"), str(Path(td) / "reports"))
        ctx = _make_ctx(cfg)
        with pytest.raises(typer.Exit) as exc:
            repair_cmd(
                ctx, local_id=None, all=False, workspace=None, dry_run=False, json_output=False
            )
        assert exc.value.exit_code == 1


def test_repair_dry_run_outputs_preview(capsys):
    """repair_cmd --dry-run emits '[DRY RUN]' and calls repair_paper with dry_run=True."""
    from drbrain.cli.repair_commands import repair_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "t.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        _setup_db(str(db_path), [{"local_id": "p1", "title": "Test", "year": 2024}])

        fake = [{"field": "title", "old": "Test", "new": "Fixed", "source": "crossref"}]
        ctx = _make_ctx(cfg)
        with mock.patch("drbrain.services.repair.repair_paper", return_value=fake) as m_repair:
            repair_cmd(
                ctx, local_id="p1", all=False, workspace=None, dry_run=True, json_output=False
            )

        m_repair.assert_called_once()
        assert m_repair.call_args.kwargs.get("dry_run") is True
        assert "[DRY RUN]" in capsys.readouterr().out


def test_repair_all_empty_db(capsys):
    """repair_cmd --all with empty DB reports 0 fields across 0 papers."""
    from drbrain.cli.repair_commands import repair_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "t.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        _setup_db(str(db_path), papers=[])

        ctx = _make_ctx(cfg)
        with mock.patch("drbrain.services.repair.repair_paper", return_value=[]):
            repair_cmd(
                ctx, local_id=None, all=True, workspace=None, dry_run=False, json_output=False
            )

        assert "0 fields across 0 papers" in capsys.readouterr().out


def test_repair_nonexistent_paper_exits():
    """repair_cmd with unknown local_id exits with code 1."""
    from drbrain.cli.repair_commands import repair_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "t.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        _setup_db(str(db_path), papers=[])

        ctx = _make_ctx(cfg)
        with pytest.raises(typer.Exit) as exc:
            repair_cmd(
                ctx, local_id="ghost", all=False, workspace=None, dry_run=False, json_output=False
            )
        assert exc.value.exit_code == 1


def test_repair_json_output(capsys):
    """repair_cmd --json emits JSON list with repairs."""
    from drbrain.cli.repair_commands import repair_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "t.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        _setup_db(str(db_path), [{"local_id": "p1", "title": "T", "year": 2024}])

        fake = [{"field": "year", "old": 2024, "new": 2023, "source": "arxiv"}]
        ctx = _make_ctx(cfg)
        with mock.patch("drbrain.services.repair.repair_paper", return_value=fake):
            repair_cmd(
                ctx, local_id="p1", all=False, workspace=None, dry_run=False, json_output=True
            )

        parsed = json.loads(capsys.readouterr().out)
        assert parsed[0]["paper"] == "p1"
        assert parsed[0]["repairs"][0]["field"] == "year"


def test_repair_applies_with_repairs(capsys):
    """repair_cmd without dry_run emits 'Repaired' and lists fixes."""
    from drbrain.cli.repair_commands import repair_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "t.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        _setup_db(str(db_path), [{"local_id": "p1", "title": "T", "year": 2024}])

        fake = [{"field": "title", "old": "T", "new": "Better", "source": "openalex"}]
        ctx = _make_ctx(cfg)
        with mock.patch("drbrain.services.repair.repair_paper", return_value=fake):
            repair_cmd(
                ctx, local_id="p1", all=False, workspace=None, dry_run=False, json_output=False
            )

        out = capsys.readouterr().out
        assert "Repaired 1 fields" in out
        assert "Better" in out


# -- enrich_cmd --


def test_enrich_requires_arg():
    """enrich_cmd with no arg exits."""
    from drbrain.cli.repair_commands import enrich_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "t.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        ctx = _make_ctx(cfg)
        with pytest.raises(typer.Exit):
            enrich_cmd(ctx, local_id=None, all=False, dry_run=False, json_output=False)


def test_enrich_all_empty_db(capsys):
    """enrich_cmd --all on empty DB reports 0 papers checked."""
    from drbrain.cli.repair_commands import enrich_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "t.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        _setup_db(str(db_path), papers=[])

        ctx = _make_ctx(cfg)
        enrich_cmd(ctx, local_id=None, all=True, dry_run=True, json_output=False)

        assert "Checked 0 paper(s)" in capsys.readouterr().out


# -- import_cmd --


def test_import_invalid_source_exits():
    """import_cmd with bad source exits with code 1."""
    from drbrain.cli.repair_commands import import_cmd

    with pytest.raises(typer.Exit) as exc:
        import_cmd(
            None,
            source="xxx",
            path="dummy.bib",
            dry_run=False,
            json_output=False,
            list_collections=False,
            collection=None,
            api_key=None,
            library_id=None,
            library_type="user",
            no_pdf=False,
            import_collections=False,
        )
    assert exc.value.exit_code == 1


def test_import_bibtex_file_not_found_exits():
    """import_cmd bibtex with missing file exits with code 1."""
    from drbrain.cli.repair_commands import import_cmd

    with pytest.raises(typer.Exit) as exc:
        import_cmd(
            None,
            source="bibtex",
            path="/nonexistent/x.bib",
            dry_run=False,
            json_output=False,
            list_collections=False,
            collection=None,
            api_key=None,
            library_id=None,
            library_type="user",
            no_pdf=False,
            import_collections=False,
        )
    assert exc.value.exit_code == 1
