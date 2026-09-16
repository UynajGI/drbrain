"""T52: migrating confirmable legacy tree_vectors into the shared store."""

from __future__ import annotations

import dataclasses
import struct

import pytest

from drbrain.services.canonical_content import write_canonical_content
from drbrain.storage.database import Database
from drbrain.tree.embedding_identity import EmbeddingProfile
from drbrain.tree.vector_migration import (
    LegacyVectorIdentity,
    LegacyVectorRow,
    canonical_target_resolver,
    classify_legacy_vector,
    decode_legacy_vector,
    iter_legacy_vector_rows,
    migrate_legacy_vectors,
)

PROFILE = EmbeddingProfile(
    provider="local",
    model="qwen3-embedding-0.6b",
    dimension=3,
    max_seq_length=512,
    normalize=True,
)


def _db(tmp_path) -> Database:
    return Database(tmp_path / "db.sqlite")


def _canonical(db: Database, local_id: str = "p1", text: str = "# Heading\n\nbody one\n") -> None:
    if db.get_paper(local_id) is None:
        db.insert_paper(local_id, "T", 2024, "uploaded")
        db.commit()
    result = write_canonical_content(db, local_id, text, media_type="md", parser="test")
    assert result["ok"]


def _leaf(db: Database, local_id: str = "p1") -> dict:
    leaves = db.list_tree_nodes(kind="leaf", state="ready", local_id=local_id, limit=100)
    assert leaves
    return leaves[0]


def _legacy_row(
    db: Database,
    node_id: str,
    paper_id: str,
    content_hash: str,
    vector=(1.0, 0.0, 0.0),
) -> LegacyVectorRow:
    blob = struct.pack(f"<{len(vector)}f", *vector)
    db.conn.execute(
        "INSERT OR REPLACE INTO tree_vectors "
        "(node_id, paper_id, embedding, content_hash, tree_layer) VALUES (?, ?, ?, ?, ?)",
        (node_id, paper_id, blob, content_hash, "pageindex"),
    )
    db.commit()
    return LegacyVectorRow(
        node_id=node_id,
        paper_id=paper_id,
        content_hash=content_hash,
        tree_layer="pageindex",
        blob=blob,
    )


def _digest(db: Database, local_id: str = "p1") -> tuple[dict, str]:
    leaf = _leaf(db, local_id)
    return leaf, leaf["content_hash"][:16]


class RecordingStore:
    def __init__(self) -> None:
        self.entries: list = []

    def upsert(self, entries):
        entries = list(entries)
        self.entries.extend(entries)
        return len(entries)


class TestClassification:
    def test_text_match_with_unknown_model_is_pending(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        leaf, digest = _digest(db)
        row = _legacy_row(db, "p1:0000", "p1", digest)
        resolve = canonical_target_resolver(db)
        # The text identity itself is verifiable: the legacy hash resolves.
        assert [target.node_id for target in resolve("p1", digest)] == [leaf["node_id"]]
        for declared in (None, LegacyVectorIdentity()):
            verdict = classify_legacy_vector(
                row, profile=PROFILE, declared=declared, resolve_target=resolve
            )
            assert verdict.decision == "pending", declared
            assert verdict.reason == "identity-unknown"

    def test_id_mismatch_is_pending(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        _, digest = _digest(db)
        resolve = canonical_target_resolver(db)
        declared = LegacyVectorIdentity.from_profile(PROFILE)
        for node_id in ("p2:0000", "0000"):
            row = _legacy_row(db, node_id, "p1", digest)
            verdict = classify_legacy_vector(
                row, profile=PROFILE, declared=declared, resolve_target=resolve
            )
            assert verdict.decision == "pending", node_id
            assert verdict.reason == "id-mismatch"

    def test_dimension_mismatch_is_pending(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        _, digest = _digest(db)
        row = _legacy_row(db, "p1:0000", "p1", digest, vector=(1.0, 0.0))
        verdict = classify_legacy_vector(
            row,
            profile=PROFILE,
            declared=LegacyVectorIdentity.from_profile(PROFILE),
            resolve_target=canonical_target_resolver(db),
        )
        assert verdict.decision == "pending" and verdict.reason == "dimension-mismatch"

    def test_normalization_mismatch_is_pending(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        _, digest = _digest(db)
        row = _legacy_row(db, "p1:0000", "p1", digest)
        declared = dataclasses.replace(LegacyVectorIdentity.from_profile(PROFILE), normalize=False)
        verdict = classify_legacy_vector(
            row,
            profile=PROFILE,
            declared=declared,
            resolve_target=canonical_target_resolver(db),
        )
        assert verdict.decision == "pending" and verdict.reason == "normalization-mismatch"

    def test_undeclared_identity_field_is_pending(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        _, digest = _digest(db)
        row = _legacy_row(db, "p1:0000", "p1", digest)
        declared = LegacyVectorIdentity(provider="local", model=PROFILE.model, normalize=True)
        verdict = classify_legacy_vector(
            row,
            profile=PROFILE,
            declared=declared,
            resolve_target=canonical_target_resolver(db),
        )
        assert verdict.decision == "pending" and verdict.reason == "undeclared-max_seq_length"

    def test_unusable_blob_is_pending(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        _, digest = _digest(db)
        db.conn.execute(
            "INSERT INTO tree_vectors (node_id, paper_id, embedding, content_hash, tree_layer) "
            "VALUES (?, ?, ?, ?, ?)",
            ("p1:0000", "p1", b"\x01\x02\x03", digest, "pageindex"),
        )
        db.commit()
        row = LegacyVectorRow("p1:0000", "p1", digest, "pageindex", b"\x01\x02\x03")
        verdict = classify_legacy_vector(
            row,
            profile=PROFILE,
            declared=LegacyVectorIdentity.from_profile(PROFILE),
            resolve_target=canonical_target_resolver(db),
        )
        assert verdict.decision == "pending" and verdict.reason == "unusable-blob"

    def test_unresolvable_text_is_pending(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        row = _legacy_row(db, "p1:0000", "p1", "0" * 16)
        verdict = classify_legacy_vector(
            row,
            profile=PROFILE,
            declared=LegacyVectorIdentity.from_profile(PROFILE),
            resolve_target=canonical_target_resolver(db),
        )
        assert verdict.decision == "pending" and verdict.reason == "no-canonical-match"

    def test_confirmed_row_carries_targets_and_vector(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        leaf, digest = _digest(db)
        row = _legacy_row(db, "p1:0000", "p1", digest, vector=(0.25, 0.5, 1.0))
        verdict = classify_legacy_vector(
            row,
            profile=PROFILE,
            declared=LegacyVectorIdentity.from_profile(PROFILE),
            resolve_target=canonical_target_resolver(db),
        )
        assert verdict.decision == "reuse"
        assert [target.node_id for target in verdict.targets] == [leaf["node_id"]]
        assert verdict.vector == (0.25, 0.5, 1.0)


class TestResolver:
    def test_resolution_is_paper_scoped_and_truncated_hash_matched(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db, "p1")
        leaf, digest = _digest(db, "p1")
        resolve = canonical_target_resolver(db)
        assert [target.node_id for target in resolve("p1", digest)] == [leaf["node_id"]]
        assert resolve("p2", digest) == ()
        assert resolve("p1", "f" * 16) == ()
        assert leaf["content_hash"][:16] == digest  # 16-hex legacy hash basis

    def test_stale_nodes_are_not_targets(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        leaf, digest = _digest(db)
        assert canonical_target_resolver(db)("p1", digest)
        db.update_tree_node_state(leaf["node_id"], "stale")
        # The resolver reflects the published state at construction time.
        assert canonical_target_resolver(db)("p1", digest) == ()


class TestRowReader:
    def test_rows_are_read_in_order(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        _, digest = _digest(db)
        _legacy_row(db, "p1:0002", "p1", digest, vector=(1.0, 0.0, 0.0))
        _legacy_row(db, "p1:0001", "p1", digest, vector=(0.5, 0.5, 0.0))
        rows = list(iter_legacy_vector_rows(db))
        assert [row.node_id for row in rows] == ["p1:0001", "p1:0002"]
        assert rows[0].blob == struct.pack("<3f", 0.5, 0.5, 0.0)
        assert rows[0].tree_layer == "pageindex"

    def test_missing_table_reads_empty(self, tmp_path):
        db = _db(tmp_path)
        db.conn.execute("DROP TABLE tree_vectors")
        db.commit()
        assert list(iter_legacy_vector_rows(db)) == []


class TestMigration:
    def test_confirmed_vector_is_reused_without_embedding(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        leaf, digest = _digest(db)
        _legacy_row(db, "p1:0000", "p1", digest, vector=(0.25, 0.5, 1.0))
        store = RecordingStore()
        report = migrate_legacy_vectors(
            db,
            store,
            profile=PROFILE,
            declared_identity=LegacyVectorIdentity.from_profile(PROFILE),
        )
        assert report.rows == 1
        assert report.reused == [{"legacy": "p1:0000", "target": leaf["node_id"], "local_id": "p1"}]
        assert not report.pending and not report.failed
        assert len(store.entries) == 1
        entry = store.entries[0]
        assert entry.node_id == leaf["node_id"]
        assert entry.node_revision == leaf["revision"]
        assert entry.kind == "leaf"
        assert entry.content_hash == leaf["content_hash"]
        assert entry.profile_id == PROFILE.profile_id()
        assert entry.vector == (0.25, 0.5, 1.0)
        meta = db.get_node_vector(leaf["node_id"])
        assert meta["state"] == "ready" and meta["dimension"] == 3

    def test_second_run_reports_already_and_writes_nothing(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        leaf, digest = _digest(db)
        _legacy_row(db, "p1:0000", "p1", digest)
        declared = LegacyVectorIdentity.from_profile(PROFILE)
        migrate_legacy_vectors(db, RecordingStore(), profile=PROFILE, declared_identity=declared)
        store = RecordingStore()
        again = migrate_legacy_vectors(db, store, profile=PROFILE, declared_identity=declared)
        assert again.reused == []
        assert again.already == [{"legacy": "p1:0000", "target": leaf["node_id"]}]
        assert store.entries == []

    def test_pending_rows_are_listed_not_written(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        _, digest = _digest(db)
        _legacy_row(db, "p1:0000", "p1", digest)
        store = RecordingStore()
        report = migrate_legacy_vectors(db, store, profile=PROFILE, declared_identity=None)
        assert report.pending == [{"legacy": "p1:0000", "reason": "identity-unknown"}]
        assert store.entries == []
        assert db.count_node_vectors() == 0

    def test_sqlite_float_copies_are_not_recreated(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        leaf, digest = _digest(db)
        _legacy_row(db, "p1:0000", "p1", digest, vector=(1.0, 0.0, 0.0))
        query = "SELECT node_id, embedding, content_hash FROM tree_vectors ORDER BY node_id"
        before = db.conn.execute(query).fetchall()
        tables_before = _table_names(db)
        report = migrate_legacy_vectors(
            db,
            RecordingStore(),
            profile=PROFILE,
            declared_identity=LegacyVectorIdentity.from_profile(PROFILE),
        )
        assert report.reused
        assert db.conn.execute(query).fetchall() == before
        tables_after = _table_names(db)
        assert "tree_vectors_vec" not in tables_after
        assert not {"tree_vectors_vec", "tree_vectors_vec_f32_bak"} & (tables_after - tables_before)
        columns = {row[1] for row in db.conn.execute("PRAGMA table_info(node_vectors)").fetchall()}
        assert not {"embedding", "vector", "blob", "vec"} & columns
        assert db.get_node_vector(leaf["node_id"])["state"] == "ready"

    def test_store_failure_leaves_staging_and_resumes(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        leaf, digest = _digest(db)
        _legacy_row(db, "p1:0000", "p1", digest)

        class FailingStore:
            def upsert(self, entries):
                raise RuntimeError("index down")

        declared = LegacyVectorIdentity.from_profile(PROFILE)
        failed = migrate_legacy_vectors(
            db, FailingStore(), profile=PROFILE, declared_identity=declared
        )
        assert failed.failed == [
            {"legacy": "p1:0000", "target": leaf["node_id"], "error": "index down"}
        ]
        assert db.get_node_vector(leaf["node_id"])["state"] == "staging"
        # A later run picks the unfinished entry back up and finishes it.
        store = RecordingStore()
        resumed = migrate_legacy_vectors(db, store, profile=PROFILE, declared_identity=declared)
        assert resumed.reused == [
            {"legacy": "p1:0000", "target": leaf["node_id"], "local_id": "p1"}
        ]
        assert db.get_node_vector(leaf["node_id"])["state"] == "ready"
        assert len(store.entries) == 1


class TestDecode:
    def test_roundtrip_and_rejects_odd_lengths(self):
        assert decode_legacy_vector(struct.pack("<3f", 1.0, -2.5, 0.0)) == (1.0, -2.5, 0.0)
        assert decode_legacy_vector(b"") is None
        assert decode_legacy_vector(None) is None
        assert decode_legacy_vector(b"\x00\x00\x00") is None


class TestEndToEnd:
    def test_confirmed_vector_lands_in_shared_store(self, tmp_path):
        pytest.importorskip("zvec", reason="zvec package required for the ANN tests")
        from drbrain.tree.vector_store import UnifiedVectorStore

        db = _db(tmp_path)
        _canonical(db)
        leaf, digest = _digest(db)
        vector = (0.25, 0.5, 1.0)
        _legacy_row(db, "p1:0000", "p1", digest, vector=vector)
        declared = LegacyVectorIdentity.from_profile(PROFILE)
        with UnifiedVectorStore(tmp_path / "zvec", create=True, dimension=3) as store:
            report = migrate_legacy_vectors(db, store, profile=PROFILE, declared_identity=declared)
            assert [entry["target"] for entry in report.reused] == [leaf["node_id"]]
            hits = store.query(list(vector), top_k=5)
            assert hits and hits[0].node_id == leaf["node_id"]
            assert hits[0].score > 0.999
            assert hits[0].profile_id == PROFILE.profile_id()
        with UnifiedVectorStore(tmp_path / "zvec", dimension=3) as store:
            again = migrate_legacy_vectors(db, store, profile=PROFILE, declared_identity=declared)
            assert again.already and not again.reused


def _table_names(db: Database) -> set[str]:
    return {
        str(row[0])
        for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }
