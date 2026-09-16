"""Block repack: re-segmentation keeps the revision, its text, and its leaves.

The 10k flow test found leaf granularity at one structural fragment per
leaf (118-character median).  The repack re-segments a revision in place
under the current ``BlockPolicy``; the canonical text and its hash must not
move, every leaf is rebuilt, and stale child references are cleared.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from drbrain.services.canonical_content import write_canonical_content
from drbrain.services.storage_repack import RepackItem, apply_repack, plan_repack
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
    # the reported totals describe the revisions that actually applied
    assert outcome["blocks_before"] == item.blocks_before
    assert outcome["blocks_after"] == item.blocks_after

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


def test_repack_delete_cascades_to_orphaned_parent_regions(tmp_path: Path) -> None:
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

    # a region's identity derives from its full member list, so the parent
    # that lost a member is deleted together with the leaf (cascade)
    assert db.delete_tree_nodes([leaf_id]) == 2
    assert db.get_tree_node(leaf_id) is None
    assert db.get_tree_node(region.node_id) is None
    remaining = db.conn.execute(
        "SELECT COUNT(*) FROM tree_node_children WHERE child_id = ?", (leaf_id,)
    ).fetchone()[0]
    assert remaining == 0


def test_repack_skips_a_single_fragment_revision(tmp_path: Path) -> None:
    db = _db(tmp_path)
    write_canonical_content(db, "p1", "# T\n\n" + "Body. " * 120, media_type="md", parser="test")
    plan = plan_repack(db)
    assert plan.items == []
    assert plan.skipped == 1
    assert plan.failed == []


def test_apply_reports_only_applied_block_totals(tmp_path: Path) -> None:
    db = _db(tmp_path)
    _legacy_revision(db, FRAGMENTED)
    plan = plan_repack(db)
    applied_item = plan.items[0]
    # a vanished revision must not inflate the reported before/after totals
    plan.items.append(RepackItem(local_id="missing", revision=1, blocks_before=99, blocks_after=1))

    outcome = apply_repack(db, plan)

    assert outcome["applied"] == 1
    assert [entry["local_id"] for entry in outcome["failed"]] == ["missing"]
    assert outcome["blocks_before"] == applied_item.blocks_before
    assert outcome["blocks_after"] == applied_item.blocks_after


def test_cli_min_chars_zero_disables_merging() -> None:
    from drbrain.cli.storage_commands import _repack_policy

    assert _repack_policy(None).min_chars == BlockPolicy().min_chars
    assert _repack_policy(0).min_chars == 0
    assert _repack_policy(320).min_chars == 320


def test_cli_min_chars_rejects_negative_values() -> None:
    import typer

    from drbrain.cli.storage_commands import _repack_policy

    with pytest.raises(typer.BadParameter):
        _repack_policy(-1)


def test_page_marks_refuse_to_guess_after_cross_page_merges() -> None:
    from drbrain.services.storage_repack import _page_marks

    fine = [
        {"page_start": 1, "page_end": 1, "char_start": 0},
        {"page_start": 2, "page_end": 2, "char_start": 400},
    ]
    assert _page_marks(fine) == [(1, 0), (2, 400)]

    # page 2 has no row starting on it (a merged block covers it), so the
    # original boundary offsets cannot be recovered exactly
    merged = [
        {"page_start": 1, "page_end": 2, "char_start": 0},
        {"page_start": 3, "page_end": 3, "char_start": 400},
    ]
    assert _page_marks(merged) is None


def test_repack_retires_replaced_leaf_docs_from_the_shared_store(tmp_path: Path) -> None:
    pytest.importorskip("zvec", reason="zvec package required for the store test")
    from drbrain.tree.vector_store import UnifiedVectorStore, VectorEntry

    db = _db(tmp_path)
    legacy_leaves = _legacy_revision(db, FRAGMENTED)
    store = UnifiedVectorStore(tmp_path / "vectors", create=True, dimension=3)
    store.open()
    store.upsert(
        [
            VectorEntry(
                node_id=leaf_id,
                node_revision=1,
                kind="leaf",
                local_id="p1",
                layer=0,
                content_hash="legacy",
                profile_id="emb-test",
                vector=(1.0, 0.0, 0.0),
            )
            for leaf_id in legacy_leaves
        ]
    )

    plan = plan_repack(db)
    outcome = apply_repack(db, plan, store=store)

    assert outcome["applied"] == 1
    surviving = {_leaf_id(row) for row in db.get_content_blocks("p1", 1)}
    retired = [leaf_id for leaf_id in legacy_leaves if leaf_id not in surviving]
    assert retired  # the merge replaced something
    assert store.get(retired) == {}
    store.close()
