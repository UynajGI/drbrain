"""Block repack: re-segmentation keeps the revision, its text, and its leaves.

The 10k flow test found leaf granularity at one structural fragment per
leaf (118-character median).  The repack re-segments a revision in place
under the current ``BlockPolicy``; the canonical text and its hash must not
move, every leaf is rebuilt, and stale child references are cleared.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from drbrain.services.canonical_content import write_canonical_content
from drbrain.services.storage_repack import apply_repack, plan_repack
from drbrain.storage.database import Database
from drbrain.tree.blocks import BlockPolicy, build_content_blocks
from drbrain.tree.contracts import ChildRef, LeafRef, NodeRecord, leaf_node_id, region_node_id

FRAGMENTED = "# Title\n\n" + "".join(f"Fragment {index}.\n\n" for index in range(24))


def _db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "db.sqlite")
    db.insert_paper("p1", "Title", 2024, "uploaded")
    return db


def _legacy_revision(db: Database, text: str) -> list[str]:
    """Write one revision the way the pre-merge builder would have.

    ``write_canonical_content`` now merges to paragraph size, so a
    fine-grained revision can only come from before the repack existed —
    this reproduces exactly that state (same tables, same leaf contract).
    """
    revision = db.next_document_revision("p1")
    blocks = build_content_blocks(
        text,
        local_id="p1",
        revision=revision,
        media_type="md",
        parser="legacy",
        policy=BlockPolicy(min_chars=0),
    )
    leaves: list[str] = []
    with db.transaction():
        db.upsert_document_revision(
            "p1",
            revision,
            source_hash="",
            canonical_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            backend="legacy",
            media_type="md",
            parser_revision="blocks-v1",
        )
        db.insert_content_blocks(blocks)
        for block in blocks:
            ref = LeafRef(
                local_id="p1",
                revision=revision,
                block_id=block.block_id,
                char_start=0,
                char_end=len(block.text),
            )
            db.insert_tree_node(
                NodeRecord(
                    node_id=leaf_node_id(ref),
                    revision=1,
                    kind="leaf",
                    state="ready",
                    layer=0,
                    content_hash=block.text_hash,
                    leaf=ref,
                    heading_path=block.heading_path,
                ),
                publish=True,
            )
            leaves.append(leaf_node_id(ref))
    return leaves


def _leaf_id(row: dict) -> str:
    return leaf_node_id(
        LeafRef(
            local_id="p1",
            revision=1,
            block_id=str(row["block_id"]),
            char_start=0,
            char_end=len(str(row["text"])),
        )
    )


def test_repack_collapses_fragments_and_keeps_the_revision(tmp_path: Path) -> None:
    db = _db(tmp_path)
    legacy_leaves = _legacy_revision(db, FRAGMENTED)
    before_rows = db.get_content_blocks("p1", 1)
    before_revision = db.get_document_revision("p1", 1)
    assert len(before_rows) > 8
    assert len(legacy_leaves) == len(before_rows)

    plan = plan_repack(db)
    assert plan.failed == []
    assert len(plan.items) == 1
    item = plan.items[0]
    assert item.blocks_before == len(before_rows)
    assert item.blocks_after < item.blocks_before

    outcome = apply_repack(db, plan)
    assert outcome["applied"] == 1
    assert outcome["failed"] == []

    after_rows = db.get_content_blocks("p1", 1)
    assert len(after_rows) == item.blocks_after
    # the same revision, the same text, only the boundaries moved
    after_revision = db.get_document_revision("p1", 1)
    assert int(after_revision["revision"]) == int(before_revision["revision"])
    assert after_revision["canonical_hash"] == before_revision["canonical_hash"]
    assert "".join(str(row["text"]) for row in after_rows) == FRAGMENTED

    # every block has one published leaf; the old leaves are gone
    for row in after_rows:
        node = db.get_tree_node(_leaf_id(row))
        assert node is not None
        assert node["state"] == "ready"
    assert db.count_tree_nodes(kind="leaf", state="ready") == len(after_rows)
    surviving = {_leaf_id(row) for row in after_rows}
    for node_id in legacy_leaves:
        if node_id not in surviving:
            assert db.get_tree_node(node_id) is None

    # re-planning is a no-op: the segmentation already matches the policy
    again = plan_repack(db)
    assert again.items == []
    assert again.skipped == 1


def test_repack_clears_child_references_before_deleting_leaves(tmp_path: Path) -> None:
    db = _db(tmp_path)
    _legacy_revision(db, FRAGMENTED)
    rows = db.get_content_blocks("p1", 1)
    leaf_id = _leaf_id(rows[0])
    children = (ChildRef(child_id=leaf_id, child_revision=1, ordinal=0),)
    region = NodeRecord(
        node_id=region_node_id(children, {}),
        revision=1,
        kind="region",
        state="ready",
        layer=1,
        content_hash="region-hash",
        summary="summary",
        children=children,
        contract={},
        origin="manual",
    )
    db.insert_tree_node(region)
    db.publish_tree_node(region.node_id)

    assert db.delete_tree_nodes([leaf_id]) == 1
    assert db.get_tree_node(leaf_id) is None
    remaining = db.conn.execute(
        "SELECT COUNT(*) FROM tree_node_children WHERE child_id = ?", (leaf_id,)
    ).fetchone()[0]
    assert remaining == 0
    # the parent survives — only its dangling child reference was removed
    assert db.get_tree_node(region.node_id) is not None


def test_repack_skips_a_single_fragment_revision(tmp_path: Path) -> None:
    db = _db(tmp_path)
    write_canonical_content(db, "p1", "# T\n\n" + "Body. " * 120, media_type="md", parser="test")
    plan = plan_repack(db)
    assert plan.items == []
    assert plan.skipped == 1
    assert plan.failed == []
