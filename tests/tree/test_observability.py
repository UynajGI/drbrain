"""T42: stage snapshots, aggregate state and recorded stage reports."""

from __future__ import annotations

import hashlib

from drbrain.storage.database import Database
from drbrain.tree.blocks import BlockPolicy, build_content_blocks
from drbrain.tree.contracts import (
    ChildRef,
    LeafRef,
    NodeRecord,
    StageReport,
    leaf_node_id,
    region_node_id,
)
from drbrain.tree.observability import (
    read_stage,
    record_stage,
    stage_reports,
    tree_snapshot,
)


def _seed_document(db: Database, local_id: str = "p1", *, state: str = "ready") -> str:
    text = "# Sec\n\nbody text of the document\n"
    db.insert_paper(local_id, "T", 2024, "uploaded")
    blocks = build_content_blocks(
        text,
        local_id=local_id,
        revision=1,
        media_type="md",
        policy=BlockPolicy(min_chars=0),
    )
    db.upsert_document_revision(
        local_id,
        1,
        source_hash="s",
        canonical_hash=hashlib.sha256(text.encode()).hexdigest(),
        media_type="md",
        state=state,
    )
    db.insert_content_blocks(blocks)
    ref = LeafRef(
        local_id=local_id,
        revision=1,
        block_id=blocks[1].block_id,
        char_start=0,
        char_end=len(blocks[1].text),
    )
    leaf = NodeRecord(
        node_id=leaf_node_id(ref),
        revision=1,
        kind="leaf",
        state="ready",
        layer=0,
        content_hash=blocks[1].text_hash,
        leaf=ref,
    )
    db.insert_tree_node(leaf, publish=True)
    return leaf.node_id


class TestSnapshot:
    def test_empty_store_is_absent(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        snapshot = tree_snapshot(db)
        assert snapshot.stage_state().value == "absent"
        assert snapshot.to_json()["blocks"] == 0

    def test_ready_store_reports_ready(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _seed_document(db)
        payload = tree_snapshot(db, storage_dir=tmp_path / "storage").to_json()
        assert payload["state"] == "ready"
        assert payload["documents"]["ready"] == 1
        assert payload["nodes"]["ready"] == 1
        assert payload["node_kinds"]["leaf"] == 1

    def test_staging_node_reports_running(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        leaf_id = _seed_document(db)
        refs = (ChildRef(child_id=leaf_id, child_revision=1, ordinal=0),)
        contract = {"p": "q"}
        staging = NodeRecord(
            node_id=region_node_id(refs, contract),
            revision=1,
            kind="region",
            state="staging",
            layer=1,
            content_hash=hashlib.sha256(b"s").hexdigest(),
            summary="s",
            children=refs,
            contract=contract,
        )
        db.insert_tree_node(staging)
        assert tree_snapshot(db).stage_state().value == "running"

    def test_failed_summary_is_degraded_and_visible(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _seed_document(db)
        db.put_summary_cache("sum-x", state="failed", reason="empty_summary")
        payload = tree_snapshot(db).to_json()
        assert payload["state"] == "degraded"
        assert payload["errors"]["summaries_failed"] == 1
        assert payload["summaries"]["failed"] == 1

    def test_stale_document_outranks_ready(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _seed_document(db, "p1")
        _seed_document(db, "p2", state="stale")
        assert tree_snapshot(db).stage_state().value == "stale"

    def test_failed_document_is_failed(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _seed_document(db, "p1")
        db.set_document_revision_state("p1", 1, "failed")
        assert tree_snapshot(db).stage_state().value == "failed"

    def test_snapshot_reflects_generation_pointer(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _seed_document(db)
        from drbrain.tree.publish import publish_tree_generation

        storage = tmp_path / "storage"
        result = publish_tree_generation(db, storage, profile_id="emb-test")
        payload = tree_snapshot(db, storage_dir=storage).to_json()
        assert payload["generation"] == result["generation"]


class TestStageReports:
    def test_record_and_read_roundtrip(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        report = StageReport(
            stage="build",
            state="partial",
            counts={"nodes": 3},
            created=2,
            failed=1,
        )
        payload = record_stage(db, report)
        assert payload["state"] == "partial"
        assert read_stage(db, "build")["counts"]["nodes"] == 3
        assert "build" in stage_reports(db)

    def test_missing_stage_is_none(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        assert read_stage(db, "prepare") is None
        assert stage_reports(db) == {}

    def test_unknown_key_is_ignored(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        db.conn.execute(
            "INSERT OR REPLACE INTO vector_metadata (key, value) VALUES (?, ?)",
            ("tree_stage:broken", "{not json"),
        )
        db.conn.commit()
        assert read_stage(db, "broken") is None
        assert "broken" not in stage_reports(db)
