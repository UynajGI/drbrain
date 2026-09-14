"""T13: read-only legacy material adapter (no writes, no silent conflict picks)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from drbrain.storage.legacy_content import (
    LegacyContentError,
    discover,
    load,
    section_text,
)

TREE = {
    "doc_name": "raw",
    "line_count": 12,
    "structure": [
        {
            "title": "Root",
            "node_id": "0000",
            "text": "# Root\n\nintro",
            "line_num": 1,
            "nodes": [
                {"title": "Methods", "node_id": "0001", "text": "## Methods\n\nstep", "line_num": 4}
            ],
        }
    ],
}


def _write_legacy(root: Path, local_id: str, *, raw: str | None = "# Doc\n\nbody\n", tree=None):
    directory = root / local_id
    directory.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        (directory / "raw.md").write_text(raw, encoding="utf-8")
    if tree is not None:
        (directory / "tree.json").write_text(json.dumps(tree), encoding="utf-8")
    return directory


class TestDiscovery:
    def test_find_canonical_and_reconstruct_from_tree(self, tmp_path):
        _write_legacy(tmp_path, "p1", raw=None, tree=TREE)
        refs = discover(tmp_path, "p1")
        assert len(refs) == 1
        document = load(refs[0])
        assert document.text_source == "tree.json"
        assert "step" in document.text
        assert document.warnings and "raw.md missing" in document.warnings[0]
        assert [section["node_id"] for section in document.sections] == ["0000", "0001"]

    def test_headings_are_flattened_with_paths(self, tmp_path):
        _write_legacy(tmp_path, "p1", tree=TREE)
        document = load(discover(tmp_path, "p1")[0])
        methods = next(section for section in document.sections if section["node_id"] == "0001")
        assert methods["heading_path"] == ["Root", "Methods"]
        assert section_text(document, "0001").startswith("## Methods")

    def test_scan_is_limited_and_sorted(self, tmp_path):
        for name in ("p1", "p2", "p3"):
            _write_legacy(tmp_path, name, tree=TREE)
        refs = discover(tmp_path, limit=2)
        assert len(refs) == 2
        assert [ref.local_id for ref in refs] == ["p1", "p2"]

    def test_missing_document_returns_empty(self, tmp_path):
        assert discover(tmp_path, "nope") == []


class TestAmbiguityAndSafety:
    def test_ambiguous_legacy_layouts_raise(self, tmp_path):
        doi = "10.1/foo"
        nested = tmp_path / "10.1" / "foo"
        nested.mkdir(parents=True)
        (nested / "tree.json").write_text(json.dumps(TREE), encoding="utf-8")
        (nested / "raw.md").write_text("# nested\n", encoding="utf-8")
        flat = tmp_path / "10.1_foo"
        flat.mkdir()
        (flat / "tree.json").write_text(json.dumps(TREE), encoding="utf-8")
        (flat / "raw.md").write_text("# flat\n", encoding="utf-8")
        with pytest.raises(ValueError, match="ambiguous"):
            discover(tmp_path, doi)

    def test_symlinked_directory_is_skipped(self, tmp_path):
        real = _write_legacy(tmp_path, "p1", tree=TREE)
        outside = tmp_path.parent / f"{tmp_path.name}-outside"
        outside.mkdir()
        (outside / "tree.json").write_text(json.dumps(TREE), encoding="utf-8")
        link = tmp_path / "link"
        try:
            os.symlink(outside, link)
        except OSError:  # pragma: no cover - platform without symlink support
            pytest.skip("symlinks unavailable")
        refs = discover(tmp_path)
        assert {ref.local_id for ref in refs} == {"p1"}
        assert real.exists()
        assert not any(ref.paper_dir.is_symlink() for ref in refs)

    def test_reads_never_write(self, tmp_path, monkeypatch):
        directory = _write_legacy(tmp_path, "p1", tree=TREE)
        before = {
            path: (
                path.stat().st_mtime_ns,
                path.stat().st_size,
            )
            for path in directory.rglob("*")
            if path.is_file()
        }
        files_before = {path.name for path in tmp_path.rglob("*")}
        document = load(discover(tmp_path, "p1")[0])
        assert document.text
        after = {
            path: (path.stat().st_mtime_ns, path.stat().st_size)
            for path in directory.rglob("*")
            if path.is_file()
        }
        assert before == after
        assert files_before == {path.name for path in tmp_path.rglob("*")}


class TestMalformed:
    def test_bad_json_is_reported_not_raised(self, tmp_path):
        directory = _write_legacy(tmp_path, "p1", raw="# still readable\n")
        (directory / "tree.json").write_text("{not json", encoding="utf-8")
        document = load(discover(tmp_path, "p1")[0])
        assert document.text == "# still readable\n"
        assert any("unreadable tree.json" in warning for warning in document.warnings)
        assert document.tree is None

    def test_missing_node_lookup_raises(self, tmp_path):
        _write_legacy(tmp_path, "p1", tree=TREE)
        document = load(discover(tmp_path, "p1")[0])
        with pytest.raises(LegacyContentError, match="no legacy node"):
            section_text(document, "9999")

    def test_empty_directory_yields_warnings(self, tmp_path):
        directory = tmp_path / "p1"
        directory.mkdir()
        refs = discover(tmp_path, "p1")
        document = load(refs[0])
        assert document.text == ""
        assert any("no raw.md and no tree.json" in warning for warning in document.warnings)
