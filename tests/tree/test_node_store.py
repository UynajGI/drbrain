"""T11: unified node registry and membership edges."""

from __future__ import annotations

import hashlib
import sqlite3

import pytest

from drbrain.storage.database import Database
from drbrain.tree.blocks import BlockPolicy, build_content_blocks
from drbrain.tree.contracts import (
    ChildRef,
    LeafRef,
    NodeRecord,
    leaf_node_id,
    region_node_id,
)


def _doc(db: Database, local_id: str, text: str) -> list:
    db.insert_paper(local_id, "T", 2024, "uploaded")
    blocks = build_content_blocks(
        text,
        local_id=local_id,
        revision=1,
        media_type="md",
        parser="test",
        policy=BlockPolicy(min_chars=0),
    )
    db.upsert_document_revision(
        local_id,
        1,
        source_hash=f"src-{local_id}",
        canonical_hash=hashlib.sha256(text.encode()).hexdigest(),
        backend="test",
        media_type="md",
    )
    db.insert_content_blocks(blocks)
    return blocks


def _leaf(blocks, index: int = 0, local_id: str = "p1") -> NodeRecord:
    block = blocks[index]
    ref = LeafRef(
        local_id=local_id,
        revision=1,
        block_id=block.block_id,
        char_start=0,
        char_end=len(block.text),
    )
    return NodeRecord(
        node_id=leaf_node_id(ref),
        revision=1,
        kind="leaf",
        state="staging",
        layer=0,
        content_hash=block.text_hash,
        leaf=ref,
        heading_path=block.heading_path,
    )


def _region(children: list[NodeRecord], *, layer: int = 1, summary: str = "summary", contract=None):
    contract = contract or {"prompt": "summarize-v1", "model": "m", "budget": 128}
    refs = tuple(
        ChildRef(child_id=child.node_id, child_revision=1, ordinal=i)
        for i, child in enumerate(children)
    )
    return NodeRecord(
        node_id=region_node_id(refs, contract),
        revision=1,
        kind="region",
        state="staging",
        layer=layer,
        content_hash=hashlib.sha256(summary.encode()).hexdigest(),
        summary=summary,
        children=refs,
        contract=contract,
        origin="semantic",
    )


class TestLeaves:
    def test_leaf_roundtrip_and_publish(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        blocks = _doc(db, "p1", "# Title\n\nBody text.\n")
        leaf = _leaf(blocks)
        db.insert_tree_node(leaf)
        row = db.get_tree_node(leaf.node_id)
        assert row["state"] == "staging" and row["kind"] == "leaf"
        assert row["block_id"] == blocks[0].block_id and row["doc_revision"] == 1
        db.publish_tree_node(leaf.node_id)
        assert db.get_tree_node(leaf.node_id)["state"] == "ready"

    def test_leaf_references_must_be_resolved(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        blocks = _doc(db, "p1", "# Title\n\nBody.\n")
        ref = LeafRef(local_id="p1", revision=1, block_id=blocks[0].block_id)  # char_end None
        node = NodeRecord(
            node_id=leaf_node_id(ref),
            revision=1,
            kind="leaf",
            state="staging",
            layer=0,
            content_hash=blocks[0].text_hash,
            leaf=ref,
        )
        with pytest.raises(ValueError, match="resolved"):
            db.insert_tree_node(node)

    def test_unknown_block_is_rejected(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _doc(db, "p1", "# Title\n\nBody.\n")
        ref = LeafRef(local_id="p1", revision=1, block_id="cb-missing", char_start=0, char_end=3)
        node = NodeRecord(
            node_id=leaf_node_id(ref),
            revision=1,
            kind="leaf",
            state="staging",
            layer=0,
            content_hash="x" * 64,
            leaf=ref,
        )
        with pytest.raises(ValueError, match="unknown block"):
            db.insert_tree_node(node)


class TestRegions:
    def test_publish_requires_ready_children(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        blocks = _doc(db, "p1", "# A\n\none\n\n# B\n\ntwo\n")
        leaves = [_leaf(blocks, 0), _leaf(blocks, 2)]
        for leaf in leaves:
            db.insert_tree_node(leaf)
        region = _region(leaves)
        db.insert_tree_node(region)
        with pytest.raises(ValueError, match="publish it first"):
            db.publish_tree_node(region.node_id)
        for leaf in leaves:
            db.publish_tree_node(leaf.node_id)
        db.publish_tree_node(region.node_id)
        assert db.get_tree_node(region.node_id)["state"] == "ready"
        children = db.get_tree_children(region.node_id)
        assert [child["child_id"] for child in children] == [leaf.node_id for leaf in leaves]
        assert all(child["child_revision"] == 1 for child in children)

    def test_layer_must_exceed_children(self, tmp_path):
        """A region may not sit at the same layer as one of its members."""
        db = Database(tmp_path / "db.sqlite")
        blocks = _doc(db, "p1", "# A\n\none\n\n# B\n\ntwo\n")
        leaves = [_leaf(blocks, 0), _leaf(blocks, 2)]
        for leaf in leaves:
            db.insert_tree_node(leaf, publish=True)
        first = _region(leaves, layer=1, summary="first")
        db.insert_tree_node(first, publish=True)
        flat = _region([leaves[0], first], layer=1, summary="flat", contract={"p": "x"})
        db.insert_tree_node(flat)
        with pytest.raises(ValueError, match="must be below"):
            db.publish_tree_node(flat.node_id)

    def test_monotonic_layers_prevent_cycles(self, tmp_path):
        """A region cannot absorb another region of the same or higher layer."""
        db = Database(tmp_path / "db.sqlite")
        blocks = _doc(db, "p1", "# A\n\none\n\n# B\n\ntwo\n")
        leaves = [_leaf(blocks, 0), _leaf(blocks, 2)]
        for leaf in leaves:
            db.insert_tree_node(leaf, publish=True)
        first = _region(leaves, layer=1, summary="first")
        db.insert_tree_node(first, publish=True)
        same_layer = _region([leaves[0], first], layer=1, summary="clash", contract={"p": "2"})
        db.insert_tree_node(same_layer)
        with pytest.raises(ValueError, match="must be below"):
            db.publish_tree_node(same_layer.node_id)
        upper = _region([leaves[0], first], layer=2, summary="upper", contract={"p": "2"})
        db.insert_tree_node(upper, publish=True)
        assert db.tree_node_ancestors(leaves[0].node_id) == sorted(
            [first.node_id, upper.node_id], key=lambda item: (item != upper.node_id, item)
        ) or set(db.tree_node_ancestors(leaves[0].node_id)) == {first.node_id, upper.node_id}

    def test_multi_parent_and_reverse_lookup(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        blocks = _doc(db, "p1", "# A\n\none\n\n# B\n\ntwo\n")
        leaves = [_leaf(blocks, 0), _leaf(blocks, 2)]
        for leaf in leaves:
            db.insert_tree_node(leaf, publish=True)
        left = _region(leaves, layer=1, summary="left", contract={"p": "l"})
        right = _region(leaves, layer=1, summary="right", contract={"p": "r"})
        db.insert_tree_node(left, publish=True)
        db.insert_tree_node(right, publish=True)
        parents = db.get_tree_parents(leaves[0].node_id)
        assert {parent["parent_id"] for parent in parents} == {left.node_id, right.node_id}
        assert db.leaves_missing_parent() == []

    def test_cross_paper_region_has_no_fabricated_pages(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        blocks_a = _doc(db, "p1", "# A\n\none\n")
        blocks_b = _doc(db, "p2", "# B\n\ntwo\n")
        leaf_a = _leaf(blocks_a, 0, "p1")
        leaf_b = _leaf(blocks_b, 0, "p2")
        for leaf in (leaf_a, leaf_b):
            db.insert_tree_node(leaf, publish=True)
        region = _region([leaf_a, leaf_b], layer=1, summary="joint")
        db.insert_tree_node(region, publish=True)
        row = db.get_tree_node(region.node_id)
        assert row["local_id"] == "" and row["block_id"] is None
        assert row["char_start"] is None and row["char_end"] is None

    def test_duplicate_members_rejected(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        blocks = _doc(db, "p1", "# A\n\none\n")
        leaf = _leaf(blocks)
        db.insert_tree_node(leaf, publish=True)
        with pytest.raises(ValueError, match="duplicate member"):
            _region([leaf, leaf])

    def test_dangling_child_rolls_back(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        blocks = _doc(db, "p1", "# A\n\none\n")
        leaf = _leaf(blocks)
        db.insert_tree_node(leaf, publish=True)
        ghost = ChildRef(child_id="nl-ghost", child_revision=1, ordinal=1)
        refs = (
            ChildRef(child_id=leaf.node_id, child_revision=1, ordinal=0),
            ghost,
        )
        contract = {"prompt": "p"}
        node = NodeRecord(
            node_id=region_node_id(refs, contract),
            revision=1,
            kind="region",
            state="staging",
            layer=1,
            content_hash="c" * 64,
            summary="s",
            children=refs,
            contract=contract,
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.insert_tree_node(node)
        assert db.get_tree_node(node.node_id) is None

    def test_same_identity_new_summary_bumps_revision(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        blocks = _doc(db, "p1", "# A\n\none\n\n# B\n\ntwo\n")
        leaves = [_leaf(blocks, 0), _leaf(blocks, 2)]
        for leaf in leaves:
            db.insert_tree_node(leaf, publish=True)
        contract = {"prompt": "p", "model": "m"}
        first = _region(leaves, layer=1, summary="v1", contract=contract)
        db.insert_tree_node(first, publish=True)
        second = _region(leaves, layer=1, summary="v2", contract=contract)
        assert second.node_id == first.node_id
        revision = db.insert_tree_node(second)
        assert revision == 2
        row = db.get_tree_node(first.node_id)
        assert row["summary"] == "v2" and row["state"] == "staging"
        db.publish_tree_node(first.node_id)
        assert db.get_tree_node(first.node_id)["state"] == "ready"

    def test_idempotent_insert_is_noop(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        blocks = _doc(db, "p1", "# A\n\none\n\n# B\n\ntwo\n")
        leaves = [_leaf(blocks, 0), _leaf(blocks, 2)]
        for leaf in leaves:
            db.insert_tree_node(leaf, publish=True)
        region = _region(leaves)
        db.insert_tree_node(region, publish=True)
        again = db.insert_tree_node(region, publish=True)
        assert again == 1
        assert db.get_tree_node(region.node_id)["state"] == "ready"
        assert db.count_tree_nodes(kind="region") == 1


class TestListing:
    def test_state_filtering(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        blocks = _doc(db, "p1", "# A\n\none\n")
        leaf = _leaf(blocks)
        db.insert_tree_node(leaf)
        assert db.count_tree_nodes(state="ready") == 0
        assert [row["node_id"] for row in db.list_tree_nodes(state="staging")] == [leaf.node_id]
        db.update_tree_node_state(leaf.node_id, "failed")
        assert db.get_tree_node(leaf.node_id)["state"] == "failed"
        with pytest.raises(ValueError, match="unsupported node state"):
            db.update_tree_node_state(leaf.node_id, "almost")

    def test_unknown_node_operations(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        assert db.get_tree_node("nl-none") is None
        assert db.get_tree_children("nl-none") == []
        assert db.get_tree_parents("nl-none") == []
        with pytest.raises(ValueError, match="unknown tree node"):
            db.publish_tree_node("nl-none")
