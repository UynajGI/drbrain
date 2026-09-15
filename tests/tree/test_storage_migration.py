"""T50: deterministic migration dry-run planning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from drbrain.services.storage_migration import (
    ACTIONS,
    build_migration_plan,
    plan_rejects_concurrent_change,
)
from drbrain.storage.database import Database
from drbrain.tree.blocks import build_content_blocks


def _canonical_paper(db: Database, local_id: str, text: str = "# Sec\n\nbody text\n") -> None:
    db.insert_paper(local_id, "T", 2024, "uploaded")
    blocks = build_content_blocks(text, local_id=local_id, revision=1, media_type="md")
    db.upsert_document_revision(
        local_id,
        1,
        source_hash="s",
        canonical_hash=hashlib.sha256(text.encode()).hexdigest(),
        media_type="md",
    )
    db.insert_content_blocks(blocks)
    db.commit()


def _legacy_dir(
    root: Path, local_id: str, *, raw: str | None = "# Legacy\n\nbody\n", tree=True
) -> Path:
    directory = root / local_id
    directory.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        (directory / "raw.md").write_text(raw, encoding="utf-8")
    if tree:
        (directory / "tree.json").write_text(
            json.dumps({"structure": [{"title": "Root", "node_id": "0000"}]}), encoding="utf-8"
        )
    return directory


def _items(plan):
    return {item.local_id: item for item in plan.items}


class TestPlanDeterminism:
    def test_two_scans_produce_identical_plans(self, tmp_path):
        db_path = tmp_path / "db.sqlite"
        db = Database(db_path)
        _canonical_paper(db, "p1")
        db.close()
        papers = tmp_path / "papers"
        _legacy_dir(papers, "p2")
        first = build_migration_plan(db_path=db_path, papers_root=papers)
        second = build_migration_plan(db_path=db_path, papers_root=papers)
        assert first.to_json() == second.to_json()
        assert first.plan_id() == second.plan_id()

    def test_scan_leaves_inputs_untouched(self, tmp_path):
        db_path = tmp_path / "db.sqlite"
        db = Database(db_path)
        _canonical_paper(db, "p1")
        db.close()
        papers = tmp_path / "papers"
        directory = _legacy_dir(papers, "p2")
        before_db = db_path.read_bytes()
        before = {
            path: (path.stat().st_mtime_ns, path.stat().st_size)
            for path in directory.rglob("*")
            if path.is_file()
        }
        build_migration_plan(db_path=db_path, papers_root=papers)
        assert db_path.read_bytes() == before_db
        assert before == {
            path: (path.stat().st_mtime_ns, path.stat().st_size)
            for path in directory.rglob("*")
            if path.is_file()
        }

    def test_plan_json_shape_is_stable(self, tmp_path):
        db_path = tmp_path / "db.sqlite"
        db = Database(db_path)
        db.close()
        papers = tmp_path / "papers"
        _legacy_dir(papers, "p2")
        payload = build_migration_plan(db_path=db_path, papers_root=papers).to_json()
        assert set(payload) == {"schema_version", "summary", "items", "inputs", "plan_id"}
        assert set(payload["summary"]) == set(ACTIONS)
        assert payload["plan_id"].startswith("mig-")
        item = payload["items"][0]
        assert set(item) == {
            "local_id",
            "action",
            "reason",
            "source",
            "expected_revision",
            "expected_hash",
            "detail",
        }


class TestActions:
    def test_canonical_ready_is_reused(self, tmp_path):
        db_path = tmp_path / "db.sqlite"
        db = Database(db_path)
        _canonical_paper(db, "p1")
        db.close()
        plan = build_migration_plan(db_path=db_path)
        item = _items(plan)["p1"]
        assert item.action == "reuse" and item.reason == "canonical-ready"
        assert item.expected_revision == 1 and item.expected_hash

    def test_legacy_only_is_import_with_expected_source(self, tmp_path):
        db_path = tmp_path / "db.sqlite"
        db = Database(db_path)
        db.close()
        papers = tmp_path / "papers"
        directory = _legacy_dir(papers, "p2")
        plan = build_migration_plan(db_path=db_path, papers_root=papers)
        item = _items(plan)["p2"]
        assert item.action == "import" and item.reason == "legacy-raw+tree"
        assert item.expected_revision == 1
        expected = hashlib.sha256((directory / "raw.md").read_bytes()).hexdigest()
        assert item.expected_hash == expected

    def test_canonical_plus_legacy_stays_reuse_with_detail(self, tmp_path):
        db_path = tmp_path / "db.sqlite"
        db = Database(db_path)
        _canonical_paper(db, "p1")
        db.close()
        papers = tmp_path / "papers"
        _legacy_dir(papers, "p1")
        item = _items(build_migration_plan(db_path=db_path, papers_root=papers))["p1"]
        assert item.action == "reuse" and item.reason == "canonical-ready+legacy-copy"
        assert item.detail["legacy_tree_ok"] is True

    def test_broken_canonical_is_recompute(self, tmp_path):
        db_path = tmp_path / "db.sqlite"
        db = Database(db_path)
        _canonical_paper(db, "p1")
        db.conn.execute("DELETE FROM content_blocks")
        db.commit()
        db.close()
        item = _items(build_migration_plan(db_path=db_path))["p1"]
        assert item.action == "recompute" and item.reason == "canonical-inconsistent"

    def test_stale_canonical_is_recompute(self, tmp_path):
        db_path = tmp_path / "db.sqlite"
        db = Database(db_path)
        _canonical_paper(db, "p1")
        db.set_document_revision_state("p1", 1, "stale")
        db.commit()
        db.close()
        item = _items(build_migration_plan(db_path=db_path))["p1"]
        assert item.action == "recompute" and item.reason == "canonical-stale"

    def test_ambiguous_layout_is_conflict(self, tmp_path):
        db_path = tmp_path / "db.sqlite"
        db = Database(db_path)
        db.close()
        papers = tmp_path / "papers"
        # Two directory layouts that both claim the same DOI-ish id: the plan
        # must flag the conflict instead of choosing one.
        nested = papers / "10.1" / "foo"
        nested.mkdir(parents=True)
        (nested / "raw.md").write_text("# nested\n", encoding="utf-8")
        flat = papers / "10.1_foo"
        flat.mkdir(parents=True)
        (flat / "raw.md").write_text("# flat\n", encoding="utf-8")
        item = _items(build_migration_plan(db_path=db_path, papers_root=papers))["10.1/foo"]
        assert item.action == "conflict" and item.reason == "ambiguous-layout"
        assert len(item.detail["directories"]) == 2

    def test_tree_only_and_empty_dirs_are_conflicts(self, tmp_path):
        db_path = tmp_path / "db.sqlite"
        db = Database(db_path)
        db.close()
        papers = tmp_path / "papers"
        _legacy_dir(papers, "p2", raw=None, tree=True)
        (papers / "p3").mkdir(parents=True)
        (papers / "p3" / "tree.json").write_text("{not json", encoding="utf-8")
        items = _items(build_migration_plan(db_path=db_path, papers_root=papers))
        assert items["p2"].action == "conflict" and items["p2"].reason == "tree-only"
        assert items["p3"].action == "conflict"
        assert items["p3"].reason == "unusable-legacy-material"

    def test_missing_artifacts_are_skipped(self, tmp_path):
        db_path = tmp_path / "db.sqlite"
        db = Database(db_path)
        db.insert_paper("p9", "T", 2024, "uploaded")
        db.commit()
        db.close()
        item = _items(build_migration_plan(db_path=db_path))["p9"]
        assert item.action == "skip" and item.reason == "no-artifacts"


class TestConcurrencyGuard:
    def test_hash_change_is_detected(self, tmp_path):
        db_path = tmp_path / "db.sqlite"
        db = Database(db_path)
        db.close()
        papers = tmp_path / "papers"
        directory = _legacy_dir(papers, "p2")
        item = _items(build_migration_plan(db_path=db_path, papers_root=papers))["p2"]
        current = hashlib.sha256((directory / "raw.md").read_bytes()).hexdigest()
        assert not plan_rejects_concurrent_change(item, current_hash=current)
        (directory / "raw.md").write_text("# changed after planning\n", encoding="utf-8")
        changed = hashlib.sha256((directory / "raw.md").read_bytes()).hexdigest()
        assert plan_rejects_concurrent_change(item, current_hash=changed)
