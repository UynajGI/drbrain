"""Batch coverage/profile lookups must equal the per-node path (cost-gate warm path)."""

from __future__ import annotations

import hashlib

from drbrain.storage.database import Database
from drbrain.tree.assign import (
    leaf_spans_of_node,
    leaf_spans_of_nodes,
    node_source_profile,
    source_profiles_for_nodes,
)
from drbrain.tree.blocks import build_content_blocks
from drbrain.tree.contracts import ChildRef, LeafRef, NodeRecord, leaf_node_id, region_node_id
from drbrain.tree.cost import coverage_for_members, unique_coverage


def _count(text: str) -> int:
    return max(1, len(text.split()))


def _doc(db: Database, local_id: str, sections: int = 3):
    text = "".join(
        f"# Section {i}\n\n" + " ".join([f"{local_id}s{i}"] * 20) + "\n\n" for i in range(sections)
    )
    db.insert_paper(local_id, "T", 2024, "uploaded")
    blocks = build_content_blocks(
        text, local_id=local_id, revision=1, media_type="md", parser="test"
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
    db.commit()
    return blocks


def _leaf(block, local_id: str) -> NodeRecord:
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
        state="ready",
        layer=0,
        content_hash=block.text_hash,
        leaf=ref,
        heading_path=block.heading_path,
    )


def _region(db: Database, leaves, layer: int = 1) -> NodeRecord:
    children = tuple(
        ChildRef(child_id=leaf.node_id, child_revision=1, ordinal=index)
        for index, leaf in enumerate(leaves)
    )
    record = NodeRecord(
        node_id=region_node_id(children, {}),
        revision=1,
        kind="region",
        state="staging",
        layer=layer,
        content_hash="region-hash",
        summary="summary",
        children=children,
        contract={},
        origin="manual",
    )
    db.insert_tree_node(record)
    db.publish_tree_node(record.node_id)
    return record


def _span_key(span) -> tuple:
    return (span.local_id, span.block_id, span.char_start, span.char_end)


def test_batch_lookups_equal_the_per_node_path(tmp_path) -> None:
    db = Database(tmp_path / "db.sqlite")
    doc1 = _doc(db, "p1")
    doc2 = _doc(db, "p2")
    leaf_a, leaf_b = _leaf(doc1[0], "p1"), _leaf(doc2[0], "p2")
    db.insert_tree_node(leaf_a, publish=True)
    db.insert_tree_node(leaf_b, publish=True)
    region = _region(db, [leaf_a, leaf_b])
    node_ids = [leaf_a.node_id, leaf_b.node_id, region.node_id]

    singles = {nid: leaf_spans_of_node(db, nid, count_tokens=_count) for nid in node_ids}
    batch = leaf_spans_of_nodes(db, node_ids, count_tokens=_count)
    for nid in node_ids:
        assert sorted(map(_span_key, batch[nid])) == sorted(map(_span_key, singles[nid]))
        assert sum(span.tokens for span in batch[nid]) == sum(span.tokens for span in singles[nid])

    single_profile = node_source_profile(db, region.node_id, count_tokens=_count)
    batch_profile = source_profiles_for_nodes(db, [region.node_id], count_tokens=_count)[
        region.node_id
    ]
    assert sorted(batch_profile.parts) == sorted(single_profile.parts)

    coverage = coverage_for_members(db, [leaf_a.node_id, leaf_b.node_id], count_tokens=_count)
    manual = unique_coverage(
        [span for nid in (leaf_a.node_id, leaf_b.node_id) for span in singles[nid]]
    )
    assert coverage.unique_tokens == manual.unique_tokens
    assert sorted(map(_span_key, coverage.spans)) == sorted(map(_span_key, manual.spans))
