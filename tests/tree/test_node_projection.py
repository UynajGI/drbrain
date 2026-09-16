"""T15: one canonical node projection shared by embedding and RAG consumers."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from drbrain.services.canonical_content import write_canonical_content
from drbrain.storage.database import Database
from drbrain.storage.node_projection import (
    collect_canonical_node_records,
    collect_node_records,
    collect_tree_node_records,
)

TEXT = "# Title\n\nIntro paragraph.\n\n## Methods\n\nStep one.\n\n## Results\n\nOutcome text.\n"


def _db(tmp_path) -> Database:
    return Database(tmp_path / "db.sqlite")


def _canonical(db: Database, local_id: str = "p1", text: str = TEXT) -> None:
    if db.get_paper(local_id) is None:
        db.insert_paper(local_id, "T", 2024, "uploaded")
        db.commit()
    result = write_canonical_content(db, local_id, text, media_type="md", parser="test")
    assert result["ok"]


def _legacy_paper(root: Path, local_id: str = "p1") -> Path:
    directory = root / local_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "raw.md").write_text("# Old\n\nlegacy body\n", encoding="utf-8")
    tree = {"structure": [{"title": "Root", "node_id": "0000", "text": "legacy body", "nodes": []}]}
    (directory / "tree.json").write_text(json.dumps(tree), encoding="utf-8")
    return directory


def _add_region(db: Database, local_id: str = "p1", summary: str = "A short summary.") -> str:
    from drbrain.tree.contracts import ChildRef, NodeRecord, region_node_id

    leaves = db.list_tree_nodes(kind="leaf", state="ready", local_id=local_id, limit=100)
    refs = tuple(
        ChildRef(child_id=leaf["node_id"], child_revision=1, ordinal=index)
        for index, leaf in enumerate(leaves[:2])
    )
    contract = {"purpose": "test"}
    region = NodeRecord(
        node_id=region_node_id(refs, contract),
        revision=1,
        kind="region",
        state="ready",
        layer=1,
        content_hash=hashlib.sha256(summary.encode("utf-8")).hexdigest(),
        summary=summary,
        children=refs,
        contract=contract,
    )
    db.insert_tree_node(region, publish=True)
    return region.node_id


class TestCanonicalProjection:
    def test_leaves_are_exact_blocks_with_canonical_identity(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        records = sorted(collect_canonical_node_records(db.conn, "p1"), key=lambda r: r["ordinal"])
        blocks = sorted(db.get_content_blocks("p1"), key=lambda row: row["ordinal"])
        assert [record["text"] for record in records] == [row["text"] for row in blocks]
        assert [record["text_hash"] for record in records] == [row["text_hash"] for row in blocks]
        for record in records:
            assert record["origin"] == "canonical" and record["kind"] == "leaf"
            assert record["node_id"].startswith("nl-")
            assert record["text_hash"] == hashlib.sha256(record["text"].encode("utf-8")).hexdigest()

    def test_parent_and_children_are_not_the_same_leaf_text(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        summary = "A short summary."
        _add_region(db, summary=summary)
        leaves_only = collect_canonical_node_records(db.conn, "p1")
        assert leaves_only and all(record["kind"] == "leaf" for record in leaves_only)
        with_regions = collect_canonical_node_records(db.conn, "p1", include_regions=True)
        regions = [record for record in with_regions if record["kind"] == "region"]
        assert len(regions) == 1 and regions[0]["text"] == summary
        # The region summary replaces the parent body: every block-backed leaf
        # appears exactly once, and no record duplicates another's text.
        leaf_texts = [record["text"] for record in with_regions if record["kind"] == "leaf"]
        assert leaf_texts == [record["text"] for record in leaves_only]
        assert len(set(leaf_texts)) == len(leaf_texts)
        assert summary not in leaf_texts

    def test_only_the_latest_ready_revision_is_projected(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db, "p1", "# Old\n\nold body\n")
        old_hashes = {row["text_hash"] for row in db.get_content_blocks("p1")}
        db.set_document_revision_state("p1", 1, "stale")
        result = write_canonical_content(db, "p1", "# New\n\nnew body\n", media_type="md")
        assert result["ok"] and result["revision"] == 2
        records = collect_canonical_node_records(db.conn, "p1")
        assert records
        current = {row["text_hash"] for row in db.get_content_blocks("p1", 2)}
        assert {record["text_hash"] for record in records} == current
        assert not ({record["text_hash"] for record in records} & old_hashes)

    def test_uninitialized_connection_has_no_canonical_view(self, tmp_path):
        conn = sqlite3.connect(tmp_path / "raw.db")
        try:
            assert collect_canonical_node_records(conn, "p1") == []
        finally:
            conn.close()


class TestSharedIdentityAcrossConsumers:
    def test_embedding_and_rag_share_node_ids_and_hashes(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        paper_dir = tmp_path / "papers" / "p1"  # no files: canonical serves it

        from drbrain.services.embedding import _collect_tree_nodes

        embed_nodes = _collect_tree_nodes(paper_dir, conn=db.conn, paper_id="p1")
        assert embed_nodes and all(node["node_id"].startswith("nl-") for node in embed_nodes)

        pytest.importorskip("llama_index.core.schema")
        from drbrain.rag.index_nodes import collect_tree_nodes

        docs = collect_tree_nodes(paper_dir, paper_id="p1", conn=db.conn)
        by_id = {doc.metadata["node_id"]: doc for doc in docs}
        assert set(by_id) == {node["node_id"] for node in embed_nodes}
        for node in embed_nodes:
            doc = by_id[node["node_id"]]
            assert doc.text == node["text"]
            assert (
                doc.metadata["text_hash"]
                == hashlib.sha256(node["text"].encode("utf-8")).hexdigest()
            )
            assert doc.metadata["origin"] == "canonical"

    def test_rag_documents_include_region_summaries(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        region_id = _add_region(db, summary="Summary only.")
        pytest.importorskip("llama_index.core.schema")
        from drbrain.rag.index_nodes import collect_tree_nodes

        docs = collect_tree_nodes(tmp_path / "papers" / "p1", paper_id="p1", conn=db.conn)
        by_id = {doc.metadata["node_id"]: doc for doc in docs}
        assert by_id[region_id].text == "Summary only."
        assert by_id[region_id].metadata["origin"] == "canonical"


class TestLegacyCompatibility:
    def test_legacy_records_keep_the_compatibility_shape(self, tmp_path):
        db = _db(tmp_path)
        db.insert_paper("p1", "T", 2024, "uploaded")
        db.commit()
        root = tmp_path / "papers"
        directory = _legacy_paper(root)
        records = collect_node_records(db.conn, "p1", paper_dir=directory)
        assert records == collect_tree_node_records(directory, paper_id="p1")
        assert records[0]["origin"] == "legacy"
        assert (
            records[0]["text_hash"]
            == hashlib.sha256(records[0]["text"].encode("utf-8")).hexdigest()
        )

    def test_no_canonical_and_no_files_is_empty(self, tmp_path):
        db = _db(tmp_path)
        assert collect_node_records(db.conn, "p9") == []

    def test_embedding_without_a_connection_stays_on_the_legacy_entry(self, tmp_path):
        db = _db(tmp_path)
        db.insert_paper("p1", "T", 2024, "uploaded")
        db.commit()
        directory = _legacy_paper(tmp_path / "papers")
        from drbrain.services.embedding import _collect_tree_nodes

        nodes = _collect_tree_nodes(directory)
        assert [node["node_id"] for node in nodes] == ["0000"]
        assert "legacy body" in nodes[0]["text"]

    def test_embedding_reads_canonical_leaves_for_db_only_papers(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        from drbrain.services.embedding import _collect_tree_nodes

        nodes = _collect_tree_nodes(tmp_path / "papers" / "p1", conn=db.conn, paper_id="p1")
        assert nodes and all(node["node_id"].startswith("nl-") for node in nodes)
