"""T14: display and export body reads come from the canonical store first."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import typer

from drbrain.app import service
from drbrain.services.canonical_content import write_canonical_content
from drbrain.storage.database import Database
from drbrain.storage.paper_view import body_outline, read_body

TEXT = "# Title\n\nIntro.\n\n## Methods\n\nStep one.\n\n### Detail\n\nDeep body.\n"


def _db(tmp_path) -> Database:
    return Database(tmp_path / "db.sqlite")


def _paper(db: Database, local_id: str = "p1", title: str = "Display Paper") -> None:
    if db.get_paper(local_id) is None:
        db.insert_paper(local_id, title, 2024, "uploaded")
        db.commit()


def _canonical(db: Database, local_id: str = "p1", text: str = TEXT) -> None:
    _paper(db, local_id)
    result = write_canonical_content(db, local_id, text, media_type="md", parser="test")
    assert result["ok"]


def _legacy_dir(root: Path, local_id: str = "p1") -> Path:
    directory = root / local_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "raw.md").write_text("# Old Title\n\nlegacy body text\n", encoding="utf-8")
    tree = {
        "structure": [
            {
                "title": "Root",
                "node_id": "0000",
                "text": "legacy body text",
                "nodes": [{"title": "Child", "node_id": "0001", "nodes": []}],
            }
        ]
    }
    (directory / "tree.json").write_text(json.dumps(tree), encoding="utf-8")
    return directory


def _snapshot(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


def _cfg() -> dict:
    return {
        "db": {"path": "data/test.db"},
        "llm": {"models": []},
        "bm25": {"k1": 1.5, "b": 0.75},
        "dirs": {"papers": "data/papers"},
        "autoresearch": {"enabled": False, "run_dir": "workspace/runs", "plugins_dir": ""},
    }


def _runtime_root(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "root"
    (root / "data").mkdir(parents=True)
    monkeypatch.setenv("DRBRAIN_ROOT", str(root))
    return root


class TestCanonicalFirst:
    def test_db_only_paper_reads_from_canonical(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        papers_root = tmp_path / "papers"
        papers_root.mkdir()
        view = read_body(db, "p1", papers_root=papers_root)
        assert view.source == "canonical" and view.available
        assert view.text == TEXT
        assert view.revision == 1
        assert view.sections and view.sections[0]["anchor"] == "Title"

    def test_db_only_reads_create_no_files(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        papers_root = tmp_path / "papers"
        papers_root.mkdir()
        before = _snapshot(tmp_path)
        for _ in range(2):
            assert read_body(db, "p1", papers_root=papers_root).text == TEXT
            assert body_outline(db, "p1", papers_root=papers_root)
        assert _snapshot(tmp_path) == before
        # Display must not materialize persistent MD/JSON for a DB-only paper.
        assert not (papers_root / "p1").exists()

    def test_canonical_outline_uses_sections(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        outline = body_outline(db, "p1")
        assert [(entry["title"], entry["depth"]) for entry in outline] == [
            ("Title", 0),
            ("Methods", 1),
            ("Detail", 2),
        ]
        assert all(set(entry) == {"node_id", "title", "depth", "children"} for entry in outline)
        methods = outline[1]
        assert methods["node_id"] == "Methods" and methods["children"] == 1
        assert outline[0]["children"] == 1 and outline[2]["children"] == 0

    def test_webui_paper_detail_uses_canonical_outline(self, tmp_path, monkeypatch):
        root = _runtime_root(tmp_path, monkeypatch)
        db = Database(root / "data" / "test.db")
        _canonical(db)
        db.close()
        detail = service.paper_detail(_cfg(), "p1")
        assert detail["paper"]["title"] == "Display Paper"
        assert [entry["title"] for entry in detail["outline"]] == ["Title", "Methods", "Detail"]
        # No tree.json was materialized for the display to work.
        assert not (root / "data" / "papers" / "p1").exists()


class TestLegacyFallback:
    def test_legacy_paper_falls_back_read_only(self, tmp_path):
        db = _db(tmp_path)
        _paper(db)
        root = tmp_path / "papers"
        directory = _legacy_dir(root)
        view = read_body(db, "p1", papers_root=root)
        assert view.source == "raw.md" and "legacy body text" in view.text
        assert view.sections and view.sections[0]["title"] == "Root"
        # Reads leave the legacy material untouched.
        assert (directory / "raw.md").is_file() and (directory / "tree.json").is_file()

    def test_legacy_tree_json_outline_still_displays(self, tmp_path):
        # Real tree.json is {"structure": [...]}; the legacy walk must read it.
        db = _db(tmp_path)
        _paper(db)
        root = tmp_path / "papers"
        _legacy_dir(root)
        assert body_outline(db, "p1", papers_root=root) == [
            {"node_id": "0000", "title": "Root", "depth": 0, "children": 1},
            {"node_id": "0001", "title": "Child", "depth": 1, "children": 0},
        ]

    def test_legacy_outline_depth_is_bounded_like_before(self, tmp_path):
        db = _db(tmp_path)
        _paper(db)
        root = tmp_path / "papers"
        directory = root / "p1"
        directory.mkdir(parents=True)
        node: dict = {"title": "n5", "node_id": "0005", "nodes": []}
        for level in range(4, -1, -1):
            node = {"title": f"n{level}", "node_id": f"000{level}", "nodes": [node]}
        (directory / "tree.json").write_text(json.dumps({"structure": [node]}), encoding="utf-8")
        outline = body_outline(db, "p1", papers_root=root)
        assert [entry["depth"] for entry in outline] == [0, 1, 2, 3, 4]

    def test_missing_material_is_an_empty_view(self, tmp_path):
        db = _db(tmp_path)
        _paper(db)
        root = tmp_path / "papers"
        root.mkdir()
        view = read_body(db, "p1", papers_root=root)
        assert not view.available and view.source == ""
        assert view.warnings
        assert body_outline(db, "p1", papers_root=root) == []


class TestExplicitExport:
    def _ctx(self, db_path: Path):
        cfg = {
            "db": {"path": str(db_path)},
            "llm": {"models": []},
            "dirs": {"papers": "data/papers", "reports": "reports"},
        }
        ctx = mock.MagicMock(spec=typer.Context)
        ctx.obj = {"config": cfg}
        return ctx

    def test_export_still_works_for_db_only_paper(self, tmp_path):
        from drbrain.cli.export_commands import export_cmd

        db_path = tmp_path / "test.db"
        db = Database(db_path)
        _canonical(db)
        db.close()
        papers_root = tmp_path / "data" / "papers"
        papers_root.mkdir(parents=True)
        before = _snapshot(tmp_path)
        captured: list[str] = []

        def _capture(message="", *args, **kwargs):
            captured.append(str(message))

        with mock.patch("typer.echo", side_effect=_capture):
            export_cmd(
                self._ctx(db_path),
                local_id="p1",
                format="md",
                all=False,
                output=None,
                style="apa",
                json_output=False,
            )
        assert "Display Paper" in "\n".join(captured)
        assert _snapshot(tmp_path) == before

    def test_show_works_for_db_only_paper(self, tmp_path):
        from drbrain.cli.query_commands import show_cmd

        db_path = tmp_path / "test.db"
        db = Database(db_path)
        _canonical(db)
        db.close()
        captured: list[str] = []
        with mock.patch(
            "typer.echo", side_effect=lambda message="", *a, **k: captured.append(str(message))
        ):
            show_cmd(self._ctx(db_path), local_id="p1", json_output=True)
        payload = json.loads("\n".join(captured))
        assert payload["paper"]["title"] == "Display Paper"
