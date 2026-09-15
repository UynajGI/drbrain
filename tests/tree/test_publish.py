"""T37: immutable generation publication for the unified store."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from drbrain.storage.database import Database
from drbrain.tree.blocks import build_content_blocks
from drbrain.tree.contracts import LeafRef, NodeRecord, leaf_node_id
from drbrain.tree.publish import (
    ACTIVE_POINTER_NAME,
    GENERATIONS_DIR_NAME,
    MANIFEST_NAME,
    SNAPSHOT_NAME,
    PublicationError,
    get_active_tree_generation,
    publish_tree_generation,
    resolve_tree_generation,
    verify_tree_generation,
)


def _seed(db: Database, local_id: str = "p1", revision: int = 1) -> str:
    text = f"# Sec\n\nbody of {local_id} revision {revision}\n"
    db.insert_paper(local_id, "T", 2024, "uploaded")
    blocks = build_content_blocks(
        text, local_id=local_id, revision=revision, media_type="md", parser="test"
    )
    db.upsert_document_revision(
        local_id,
        revision,
        source_hash=f"src-{local_id}-{revision}",
        canonical_hash=hashlib.sha256(text.encode()).hexdigest(),
        media_type="md",
    )
    db.insert_content_blocks(blocks)
    ref = LeafRef(
        local_id=local_id,
        revision=revision,
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
    db.upsert_node_vector(
        leaf.node_id,
        node_revision=1,
        kind="leaf",
        profile_id="emb-test",
        content_hash=blocks[1].text_hash,
        dimension=3,
        local_id=local_id,
        state="ready",
    )
    return leaf.node_id


class TestPublication:
    def test_publish_and_resolve(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _seed(db)
        storage = tmp_path / "storage"
        result = publish_tree_generation(db, storage, profile_id="emb-test")
        assert result["published"]
        generation = result["generation"]
        assert get_active_tree_generation(storage) == generation
        resolved = resolve_tree_generation(storage, generation)
        assert Path(resolved["snapshot"]).is_file()
        manifest = resolved["manifest"]
        assert manifest["watermarks"]["content"]["documents"] == 1
        assert manifest["watermarks"]["nodes"]["nodes"] == 1
        assert manifest["watermarks"]["vectors"]["vectors"] == 1
        assert verify_tree_generation(storage, generation)["ok"]

    def test_staging_nodes_are_not_in_the_manifest(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        leaf_id = _seed(db)
        # A second, unpublished node must not leak into the manifest.
        from drbrain.tree.contracts import ChildRef, region_node_id

        refs = (ChildRef(child_id=leaf_id, child_revision=1, ordinal=0),)
        contract = {"prompt": "p"}
        region = NodeRecord(
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
        db.insert_tree_node(region)  # staging, not published
        result = publish_tree_generation(db, tmp_path / "storage", profile_id="emb-test")
        assert result["watermarks"]["nodes"]["nodes"] == 1

    def test_pointer_moves_and_old_generation_stays_readable(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _seed(db)
        storage = tmp_path / "storage"
        first = publish_tree_generation(db, storage, profile_id="emb-test")
        assert verify_tree_generation(storage, first["generation"])["ok"]
        _seed(db, "p2")
        second = publish_tree_generation(db, storage, profile_id="emb-test")
        assert second["generation"] != first["generation"]
        assert get_active_tree_generation(storage) == second["generation"]
        # The old generation is still resolvable and consistent.
        assert resolve_tree_generation(storage, first["generation"])["manifest"]["generation"]
        assert verify_tree_generation(storage, first["generation"])["ok"]

    def test_publish_requires_matching_vector_state(self, tmp_path):
        """Metadata/ANN mismatch must block publication, not warn."""
        db = Database(tmp_path / "db.sqlite")
        _seed(db)
        # Metadata says two ready vectors but the ANN dir holds one.
        db.upsert_node_vector(
            "nl-phantom",
            node_revision=1,
            kind="leaf",
            profile_id="emb-test",
            content_hash="h",
            dimension=3,
            state="ready",
        )
        vector_dir = tmp_path / "zvec"
        _build_ann(vector_dir, entries=1)
        with pytest.raises(PublicationError, match="refusing to publish"):
            publish_tree_generation(
                db, tmp_path / "storage", profile_id="emb-test", vector_dir=vector_dir
            )
        # Nothing was published and no staging directory remains.
        assert get_active_tree_generation(tmp_path / "storage") is None
        generations = tmp_path / "storage" / GENERATIONS_DIR_NAME
        assert not any(path.name.startswith(".staging-") for path in generations.iterdir())


class TestReaderGuards:
    def test_staging_directory_is_not_resolvable(self, tmp_path):
        storage = tmp_path / "storage"
        (storage / GENERATIONS_DIR_NAME / ".staging-gen-1").mkdir(parents=True)
        with pytest.raises(PublicationError, match="staging"):
            resolve_tree_generation(storage, ".staging-gen-1")

    def test_missing_manifest_is_refused(self, tmp_path):
        storage = tmp_path / "storage"
        (storage / GENERATIONS_DIR_NAME / "gen-1").mkdir(parents=True)
        with pytest.raises(PublicationError, match="no manifest"):
            resolve_tree_generation(storage, "gen-1")

    def test_unknown_generation_is_refused(self, tmp_path):
        with pytest.raises(PublicationError, match="unknown generation"):
            resolve_tree_generation(tmp_path / "storage", "gen-404")

    def test_verify_detects_tampering(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        leaf_id = _seed(db)
        storage = tmp_path / "storage"
        result = publish_tree_generation(db, storage, profile_id="emb-test")
        snapshot = Path(resolve_tree_generation(storage, result["generation"])["snapshot"])
        conn = sqlite3.connect(snapshot)
        conn.execute("UPDATE tree_nodes SET state = 'stale' WHERE node_id = ?", (leaf_id,))
        conn.commit()
        conn.close()
        report = verify_tree_generation(storage, result["generation"])
        assert not report["ok"] and "nodes" in report["mismatches"]

    def test_manifest_is_json_and_pointer_is_relative(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _seed(db)
        storage = tmp_path / "storage"
        result = publish_tree_generation(db, storage, profile_id="emb-test")
        manifest_file = storage / GENERATIONS_DIR_NAME / result["generation"] / MANIFEST_NAME
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert payload["generation"] == result["generation"]
        pointer = json.loads((storage / ACTIVE_POINTER_NAME).read_text(encoding="utf-8"))
        assert pointer["generation"] == result["generation"]
        assert (storage / GENERATIONS_DIR_NAME / result["generation"] / SNAPSHOT_NAME).is_file()


def _build_ann(path: Path, *, entries: int) -> None:
    from drbrain.tree.vector_store import UnifiedVectorStore, VectorEntry

    with UnifiedVectorStore(path, create=True, dimension=3) as store:
        store.upsert(
            [
                VectorEntry(
                    node_id=f"nl-{index}",
                    node_revision=1,
                    kind="leaf",
                    local_id="p1",
                    layer=0,
                    content_hash=f"h{index}",
                    profile_id="emb-test",
                    vector=(1.0, 0.0, 0.0),
                )
                for index in range(entries)
            ]
        )
