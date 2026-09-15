"""T53: pre-migration evidence stays readable; export is explicit and local."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
import typer

from drbrain.services.canonical_content import write_canonical_content
from drbrain.services.legacy_evidence import resolve_evidence
from drbrain.storage.database import Database
from drbrain.storage.paper_view import export_paper_view, export_structure

TEXT = "# Title\n\nIntro paragraph text about kagome metals.\n\n## Methods\n\nStep one of the method.\n"
SNIPPET = "Intro paragraph text about kagome metals."


def _db(tmp_path) -> Database:
    return Database(tmp_path / "db.sqlite")


def _canonical(db: Database, local_id: str = "p1", text: str = TEXT) -> None:
    if db.get_paper(local_id) is None:
        db.insert_paper(local_id, "T", 2024, "uploaded")
        db.commit()
    result = write_canonical_content(db, local_id, text, media_type="md", parser="test")
    assert result["ok"]


def _leaf(db: Database, local_id: str = "p1") -> dict:
    leaves = db.list_tree_nodes(kind="leaf", state="ready", local_id=local_id, limit=100)
    assert leaves
    return leaves[0]


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


class TestEvidenceCompatibility:
    def test_canonical_evidence_reads_the_current_revision(self, tmp_path):
        from drbrain.storage.node_projection import read_node_text

        db = _db(tmp_path)
        _canonical(db)
        leaf = _leaf(db)
        expected = read_node_text(db.conn, leaf["node_id"])
        assert expected
        db.record_evidence("p1", leaf["node_id"], snippet="whatever")
        db.commit()
        view = resolve_evidence(db, f"p1:{leaf['node_id']}")
        assert view is not None and view.source == "canonical" and view.available
        assert view.text == expected and view.revision == 1

    def test_pre_migration_evidence_reads_the_original_revision(self, tmp_path):
        db = _db(tmp_path)
        db.insert_paper("p1", "T", 2024, "uploaded")
        db.commit()
        root = tmp_path / "papers"
        _legacy_dir(root)
        # Recorded before migration: the old PageIndex node id + snippet.
        db.record_evidence("p1", "0000", snippet="legacy body text", page="1")
        db.commit()
        view = resolve_evidence(db, "p1:0000", papers_root=root)
        assert view is not None
        assert view.source == "legacy"
        assert view.text == "legacy body text"  # the original revision's text

    def test_alias_maps_an_old_id_onto_todays_leaf(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        # No files: the stored snippet still locates the canonical leaf.
        db.record_evidence("p1", "0000", snippet=SNIPPET)
        db.commit()
        view = resolve_evidence(db, "p1:0000")
        assert view is not None and view.source == "alias"
        assert view.alias_node_id.startswith("nl-")
        assert SNIPPET in view.text

    def test_unresolvable_evidence_falls_back_to_the_snippet(self, tmp_path):
        db = _db(tmp_path)
        db.insert_paper("p1", "T", 2024, "uploaded")
        db.commit()
        db.record_evidence("p1", "0000", snippet="a snippet that matches nothing at all")
        db.commit()
        view = resolve_evidence(db, "p1:0000")
        assert view is not None and view.source == "snippet"
        assert view.text == "a snippet that matches nothing at all"

    def test_reads_leave_no_files_behind(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        leaf = _leaf(db)
        db.record_evidence("p1", leaf["node_id"], snippet=SNIPPET)
        db.record_evidence("p1", "0000", snippet=SNIPPET)
        db.commit()
        before = _snapshot(tmp_path)
        resolve_evidence(db, f"p1:{leaf['node_id']}")
        resolve_evidence(db, "p1:0000")
        assert _snapshot(tmp_path) == before
        assert not (tmp_path / "papers").exists()


class TestExplicitExport:
    def test_export_bundle_is_self_contained(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        bundle = export_paper_view(db, "p1")
        assert bundle.source == "canonical" and bundle.text == TEXT
        payload = json.loads(json.dumps(bundle.to_json()))  # JSON-safe round trip
        titles = [node["title"] for node in payload["structure"]]
        assert "Title" in titles and "Methods" in titles
        leaves = [node for node in payload["structure"] if "text" in node]
        assert leaves and leaves[0]["text"]
        assert not (tmp_path / "papers").exists()

    def test_export_legacy_paper_reemits_the_tree_json(self, tmp_path):
        db = _db(tmp_path)
        db.insert_paper("p1", "T", 2024, "uploaded")
        db.commit()
        root = tmp_path / "papers"
        directory = _legacy_dir(root)
        original = json.loads((directory / "tree.json").read_text(encoding="utf-8"))
        assert export_structure(db, "p1", papers_root=root) == original["structure"]
        bundle = export_paper_view(db, "p1", papers_root=root)
        assert bundle.source == "raw.md" and "legacy body text" in bundle.text

    def test_missing_material_exports_empty(self, tmp_path):
        db = _db(tmp_path)
        db.insert_paper("p1", "T", 2024, "uploaded")
        db.commit()
        bundle = export_paper_view(db, "p1", papers_root=None)
        assert not bundle.text and not bundle.structure


class TestExportCli:
    def _ctx(self, tmp_path):
        ctx = mock.MagicMock(spec=typer.Context)
        ctx.obj = {"config": {"db": {"path": str(tmp_path / "db.sqlite")}}}
        return ctx

    def test_export_cli_writes_only_when_asked(self, tmp_path):
        from drbrain.cli.storage_commands import storage_export_cmd

        db = _db(tmp_path)
        _canonical(db)

        captured: list[str] = []
        with mock.patch("typer.echo", side_effect=lambda m="", *a, **k: captured.append(str(m))):
            storage_export_cmd(
                self._ctx(tmp_path),
                paper="p1",
                format="md",
                output="",
                papers_root="",
            )
        assert captured and captured[0] == TEXT
        assert list(tmp_path.glob("*.md")) == []  # stdout mode writes nothing

        target = tmp_path / "out" / "p1.md"
        with mock.patch("typer.echo"):
            storage_export_cmd(
                self._ctx(tmp_path),
                paper="p1",
                format="md",
                output=str(target),
                papers_root="",
            )
        assert target.read_text(encoding="utf-8") == TEXT

    def test_export_cli_rejects_empty_and_bad_format(self, tmp_path):
        from drbrain.cli.storage_commands import storage_export_cmd

        db = _db(tmp_path)
        db.insert_paper("p1", "T", 2024, "uploaded")
        db.commit()
        with mock.patch("typer.echo"), pytest.raises(typer.Exit) as excinfo:
            storage_export_cmd(
                self._ctx(tmp_path), paper="p1", format="md", output="", papers_root=""
            )
        assert excinfo.value.exit_code == 1
        with pytest.raises(typer.BadParameter):
            storage_export_cmd(
                self._ctx(tmp_path), paper="p1", format="xml", output="", papers_root=""
            )
