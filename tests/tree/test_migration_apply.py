"""T51: resumable, idempotent per-paper migration apply."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from drbrain.services.storage_migration import (
    apply_migration_plan,
    build_migration_plan,
)
from drbrain.storage.database import Database
from drbrain.storage.inbox import file_sha256


def _legacy(root: Path, local_id: str, body: str) -> Path:
    directory = root / local_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "raw.md").write_text(body, encoding="utf-8")
    (directory / "tree.json").write_text(
        json.dumps({"structure": [{"title": "Root", "node_id": "0000"}]}), encoding="utf-8"
    )
    return directory


def _db(tmp_path: Path, *papers: str) -> Database:
    db = Database(tmp_path / "db.sqlite")
    for local_id in papers:
        db.insert_paper(local_id, "T", 2024, "uploaded")
    db.commit()
    return db


class TestImport:
    def test_legacy_raw_is_imported_and_source_untouched(self, tmp_path):
        papers = tmp_path / "papers"
        body = "# Legacy\n\noriginal body text\n"
        directory = _legacy(papers, "p1", body)
        before_hash = file_sha256(directory / "raw.md")
        before_mtime = (directory / "raw.md").stat().st_mtime_ns
        db = _db(tmp_path, "p1")
        plan = build_migration_plan(
            db_path=db.db_path if False else tmp_path / "db.sqlite", papers_root=papers
        )
        outcome = apply_migration_plan(db, plan, papers_root=papers)
        assert outcome.applied == ["p1"] and not outcome.failed
        revision = db.get_document_revision("p1")
        assert revision["revision"] == 1 and revision["media_type"] == "md"
        blocks = db.get_content_blocks("p1")
        assert "".join(block["text"] for block in blocks) == body
        assert len(db.list_tree_nodes(kind="leaf", state="ready")) == len(blocks)
        # The legacy source is untouched (hash, mtime, presence).
        assert file_sha256(directory / "raw.md") == before_hash
        assert (directory / "raw.md").stat().st_mtime_ns == before_mtime

    def test_second_apply_is_a_noop(self, tmp_path):
        papers = tmp_path / "papers"
        _legacy(papers, "p1", "# One\n\nbody\n")
        db_path = tmp_path / "db.sqlite"
        db = _db(tmp_path, "p1")
        plan = build_migration_plan(db_path=db_path, papers_root=papers)
        first = apply_migration_plan(db, plan, papers_root=papers)
        assert first.applied == ["p1"]
        revision_before = db.get_document_revision("p1")["revision"]
        blocks_before = db.count_content_blocks("p1")
        plan_again = build_migration_plan(db_path=db_path, papers_root=papers)
        second = apply_migration_plan(db, plan_again, papers_root=papers)
        assert second.applied == [] and second.reused == ["p1"]
        assert db.get_document_revision("p1")["revision"] == revision_before
        assert db.count_content_blocks("p1") == blocks_before

    def test_directory_only_paper_gets_a_placeholder_record(self, tmp_path):
        papers = tmp_path / "papers"
        _legacy(papers, "p-dir-only", "# Stray Heading\n\nbody text\n")
        db_path = tmp_path / "db.sqlite"
        # The database has no record for this directory yet.
        db = _db(tmp_path)
        plan = build_migration_plan(db_path=db_path, papers_root=papers)
        outcome = apply_migration_plan(db, plan, papers_root=papers)
        assert outcome.applied == ["p-dir-only"]
        assert outcome.created_records == ["p-dir-only"]
        record = db.get_paper("p-dir-only")
        assert record is not None and record["status"] == "placeholder"
        assert record["title"] == "Stray Heading"
        assert db.get_document_revision("p-dir-only")["revision"] == 1

    def test_source_change_is_refused_without_writing(self, tmp_path):
        papers = tmp_path / "papers"
        directory = _legacy(papers, "p1", "# One\n\nplanned body\n")
        db_path = tmp_path / "db.sqlite"
        db = _db(tmp_path, "p1")
        plan = build_migration_plan(db_path=db_path, papers_root=papers)
        # The file changes after planning: apply must refuse this item.
        (directory / "raw.md").write_text("# One\n\nunplanned body\n", encoding="utf-8")
        outcome = apply_migration_plan(db, plan, papers_root=papers)
        assert outcome.applied == []
        assert outcome.failed == [{"local_id": "p1", "error": "source-changed"}]
        assert db.get_document_revision("p1") is None

    def test_failure_is_isolated_per_paper(self, tmp_path):
        papers = tmp_path / "papers"
        _legacy(papers, "p1", "# One\n\nfirst body\n")
        directory = _legacy(papers, "p2", "# Two\n\nsecond body\n")
        db_path = tmp_path / "db.sqlite"
        db = _db(tmp_path, "p1", "p2")
        plan = build_migration_plan(db_path=db_path, papers_root=papers)
        (directory / "raw.md").write_text("# changed\n", encoding="utf-8")
        outcome = apply_migration_plan(db, plan, papers_root=papers)
        assert outcome.applied == ["p1"]
        assert outcome.failed == [{"local_id": "p2", "error": "source-changed"}]
        assert db.get_document_revision("p1") is not None


class TestResume:
    def test_interrupted_run_resumes_from_checkpoint(self, tmp_path):
        papers = tmp_path / "papers"
        for index in range(3):
            _legacy(papers, f"p{index}", f"# Doc {index}\n\nbody {index}\n")
        db_path = tmp_path / "db.sqlite"
        db = _db(tmp_path, "p0", "p1", "p2")
        plan = build_migration_plan(db_path=db_path, papers_root=papers)
        first = apply_migration_plan(db, plan, papers_root=papers, max_items=1)
        assert first.paused and len(first.applied) == 1 and len(first.remaining) == 2
        job = db.get_tree_job(first.job_id)
        assert job["state"] == "paused"
        second = apply_migration_plan(db, plan, papers_root=papers)
        assert second.job_id == first.job_id
        assert not second.paused and second.remaining == []
        assert set(second.applied) | set(first.applied) == {"p0", "p1", "p2"}
        for local_id in ("p0", "p1", "p2"):
            assert db.get_document_revision(local_id)["revision"] == 1
        assert db.count_content_blocks() == sum(
            db.count_content_blocks(local_id) for local_id in ("p0", "p1", "p2")
        )
        assert db.get_tree_job(first.job_id)["state"] == "done"

    def test_resume_does_not_rewrite_finished_items(self, tmp_path):
        papers = tmp_path / "papers"
        _legacy(papers, "p0", "# A\n\nbody a\n")
        _legacy(papers, "p1", "# B\n\nbody b\n")
        db_path = tmp_path / "db.sqlite"
        db = _db(tmp_path, "p0", "p1")
        plan = build_migration_plan(db_path=db_path, papers_root=papers)
        apply_migration_plan(db, plan, papers_root=papers, max_items=1)
        applied_before = db.get_document_revision("p0")["canonical_hash"]
        apply_migration_plan(db, plan, papers_root=papers)
        assert db.get_document_revision("p0")["canonical_hash"] == applied_before
        assert db.get_document_revision("p0")["revision"] == 1

    def test_competing_worker_is_rejected(self, tmp_path):
        papers = tmp_path / "papers"
        _legacy(papers, "p0", "# A\n\nbody a\n")
        _legacy(papers, "p1", "# B\n\nbody b\n")
        db_path = tmp_path / "db.sqlite"
        db = _db(tmp_path, "p0", "p1")
        plan = build_migration_plan(db_path=db_path, papers_root=papers)

        def crash_after_first(item, status):
            if status == "applied":
                raise RuntimeError("simulated crash")

        # A crash mid-run leaves the job claimed (running, lease unexpired).
        try:
            apply_migration_plan(db, plan, papers_root=papers, on_progress=crash_after_first)
        except RuntimeError as exc:
            assert "simulated crash" in str(exc)
        jobs = [job for job in db.list_tree_jobs() if job["kind"] == "migrate"]
        assert jobs and jobs[0]["state"] == "running"
        try:
            apply_migration_plan(db, plan, papers_root=papers, owner="other-worker")
        except RuntimeError as exc:
            assert "already claimed" in str(exc)
        else:  # pragma: no cover - only when the lease expired between calls
            raise AssertionError("a second worker was allowed to claim a running job")


class TestRecompute:
    def test_stale_canonical_is_rederived_from_legacy(self, tmp_path):
        papers = tmp_path / "papers"
        _legacy(papers, "p1", "# Fresh\n\nnew body from legacy\n")
        db_path = tmp_path / "db.sqlite"
        db = Database(db_path)
        db.insert_paper("p1", "T", 2024, "uploaded")
        # A stale revision exists: recompute must add a new one, never delete.
        from drbrain.services.canonical_content import write_canonical_content

        write_canonical_content(db, "p1", "# Old\n\nold body\n", media_type="md", parser="test")
        db.set_document_revision_state("p1", 1, "stale")
        db.commit()
        plan = build_migration_plan(db_path=db_path, papers_root=papers)
        item = {entry.local_id: entry for entry in plan.items}["p1"]
        assert item.action == "recompute" and item.reason == "canonical-stale"
        outcome = apply_migration_plan(db, plan, papers_root=papers)
        assert outcome.applied == ["p1"]
        assert db.get_document_revision("p1", 1)["state"] == "stale"
        latest = db.get_document_revision("p1")
        assert latest["revision"] == 2
        from drbrain.storage.content import read_text

        assert read_text(db, "p1") == "# Fresh\n\nnew body from legacy\n"
        assert read_text(db, "p1", 1) == "# Old\n\nold body\n"


class TestDispositions:
    def test_conflicts_and_skips_are_reported_untouched(self, tmp_path):
        papers = tmp_path / "papers"
        # Nested + flattened layouts for one logical id: an explicit conflict.
        nested = papers / "10.1" / "foo"
        nested.mkdir(parents=True)
        (nested / "raw.md").write_text("# nested\n", encoding="utf-8")
        flat = papers / "10.1_foo"
        flat.mkdir(parents=True)
        (flat / "raw.md").write_text("# flat\n", encoding="utf-8")
        db_path = tmp_path / "db.sqlite"
        db = _db(tmp_path, "p9")
        plan = build_migration_plan(db_path=db_path, papers_root=papers)
        outcome = apply_migration_plan(db, plan, papers_root=papers)
        reasons = {entry["local_id"]: entry["reason"] for entry in outcome.skipped}
        assert reasons["10.1/foo"] == "ambiguous-layout"
        assert reasons["p9"] == "no-artifacts"
        assert (nested / "raw.md").exists() and (flat / "raw.md").exists()
        assert db.get_document_revision("10.1/foo") is None

    def test_reuse_items_are_counted_without_writes(self, tmp_path):
        papers = tmp_path / "papers"
        _legacy(papers, "p1", "# One\n\nbody\n")
        db_path = tmp_path / "db.sqlite"
        db = _db(tmp_path, "p1")
        from drbrain.services.canonical_content import write_canonical_content

        write_canonical_content(db, "p1", "# Already\n\ncanonical body\n", media_type="md")
        plan = build_migration_plan(db_path=db_path, papers_root=papers)
        outcome = apply_migration_plan(db, plan, papers_root=papers)
        assert outcome.reused == ["p1"] and outcome.applied == []
        assert db.get_document_revision("p1")["revision"] == 1
        assert (
            db.get_document_revision("p1")["canonical_hash"]
            == hashlib.sha256(b"# Already\n\ncanonical body\n").hexdigest()
        )
