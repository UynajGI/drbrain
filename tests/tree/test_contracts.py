"""T03/T06 contract tests: identity, coordinates, coverage, stages."""

from __future__ import annotations

import pytest

from drbrain.tree.contracts import (
    ChildRef,
    ContentBlock,
    DocumentRevision,
    LeafRef,
    NodeRecord,
    ReadReceipt,
    StageReport,
    StageState,
    combine_stage_states,
    content_block_id,
    contract_digest,
    is_queryable,
    leaf_node_id,
    region_node_id,
)


def _block(local_id: str, revision: int, ordinal: int, text: str, start: int, **kw):
    import hashlib

    return ContentBlock(
        block_id=content_block_id(local_id, revision, ordinal),
        local_id=local_id,
        revision=revision,
        ordinal=ordinal,
        text=text,
        text_hash=hashlib.sha256(text.encode()).hexdigest(),
        char_start=start,
        char_end=start + len(text),
        **kw,
    )


class TestIdentityStability:
    def test_block_and_leaf_ids_are_stable(self):
        ref = LeafRef(local_id="p1", revision=1, block_id="cb-x", char_start=3, char_end=9)
        assert leaf_node_id(ref) == leaf_node_id(
            LeafRef(local_id="p1", revision=1, block_id="cb-x", char_start=3, char_end=9)
        )

    def test_same_text_different_provenance_stays_distinct(self):
        a = leaf_node_id(LeafRef(local_id="p1", revision=1, block_id="cb-1"))
        b = leaf_node_id(LeafRef(local_id="p2", revision=1, block_id="cb-1"))
        assert a != b
        # A later revision of the same source is a different node identity.
        c = leaf_node_id(LeafRef(local_id="p1", revision=2, block_id="cb-1"))
        assert c != a

    def test_region_id_covers_members_and_contract(self):
        children = (
            ChildRef(child_id="nl-a", child_revision=1, ordinal=0),
            ChildRef(child_id="nl-b", child_revision=1, ordinal=1),
        )
        contract = {"prompt": "summarize-v1", "model": "spark-x25-4b", "budget": 512}
        base = region_node_id(children, contract)
        assert base == region_node_id(tuple(reversed(children)), contract)
        assert base != region_node_id(children, {**contract, "prompt": "summarize-v2"})
        other = (children[0], ChildRef(child_id="nl-c", child_revision=1, ordinal=1))
        assert base != region_node_id(other, contract)

    def test_contract_digest_is_order_insensitive(self):
        assert contract_digest({"a": 1, "b": 2}) == contract_digest({"b": 2, "a": 1})

    def test_region_id_rejects_duplicate_members(self):
        children = (
            ChildRef(child_id="nl-a", child_revision=1, ordinal=0),
            ChildRef(child_id="nl-a", child_revision=1, ordinal=1),
        )
        with pytest.raises(ValueError, match="duplicate member"):
            region_node_id(children, {"prompt": "p"})


class TestBlockCoverage:
    def test_blocks_reconstruct_canonical_text_verbatim(self):
        canonical = "# Title\n\nFirst para.\n\n## Methods\n\nTable 1: a | b\n\n residual "
        pieces = [
            "# Title\n\n",
            "First para.\n\n",
            "## Methods\n\n",
            "Table 1: a | b\n\n residual ",
        ]
        offset = 0
        blocks = []
        for ordinal, piece in enumerate(pieces):
            blocks.append(_block("p1", 1, ordinal, piece, offset))
            offset += len(piece)
        assert "".join(b.text for b in blocks) == canonical
        for prev, cur in zip(blocks, blocks[1:]):
            assert prev.char_end == cur.char_start

    def test_block_rejects_text_length_mismatch(self):
        import hashlib

        with pytest.raises(ValueError, match="char span"):
            ContentBlock(
                block_id="cb-1",
                local_id="p1",
                revision=1,
                ordinal=0,
                text="abc",
                text_hash=hashlib.sha256(b"abc").hexdigest(),
                char_start=0,
                char_end=4,
            )

    def test_block_rejects_wrong_hash_and_bad_pages(self):
        with pytest.raises(ValueError, match="text_hash"):
            ContentBlock(
                block_id="cb-1",
                local_id="p1",
                revision=1,
                ordinal=0,
                text="abc",
                text_hash="0" * 64,
                char_start=0,
                char_end=3,
            )
        with pytest.raises(ValueError, match="1-based"):
            _block("p1", 1, 0, "abc", 0, page_start=0, page_end=1)

    def test_overlapping_page_spans_do_not_duplicate_text(self):
        """Shared boundary pages are a locator overlap, never duplicated body."""
        a = _block("p1", 1, 0, "left", 0, page_start=1, page_end=2)
        b = _block("p1", 1, 1, "right", 4, page_start=2, page_end=3)
        assert a.text + b.text == "leftright"
        assert a.page_span() == (1, 2) and b.page_span() == (2, 3)


class TestCoordinates:
    def test_leaf_resolves_open_end_and_rejects_overflow(self):
        ref = LeafRef(local_id="p1", revision=1, block_id="cb-1", char_start=2)
        assert ref.resolved_char_end(10) == 10
        assert ref.resolve(10).char_end == 10
        with pytest.raises(ValueError, match="exceeds block"):
            LeafRef(local_id="p1", revision=1, block_id="cb-1", char_start=2, char_end=11).resolve(
                10
            )

    def test_document_revision_locator_families(self):
        rev = DocumentRevision(
            local_id="p1",
            revision=1,
            source_hash="a" * 64,
            canonical_hash="b" * 64,
            backend="latex-native",
            media_type="tex",
        )
        assert rev.state == "ready"
        with pytest.raises(ValueError, match="media_type"):
            DocumentRevision(
                local_id="p1",
                revision=1,
                source_hash="a",
                canonical_hash="b",
                backend="x",
                media_type="docx",
            )


class TestNodeRecords:
    def _leaf(self, text: str = "hello"):
        import hashlib

        block = _block("p1", 1, 0, text, 0)
        ref = LeafRef(local_id="p1", revision=1, block_id=block.block_id).resolve(len(text))
        return NodeRecord(
            node_id=leaf_node_id(ref),
            revision=1,
            kind="leaf",
            state="ready",
            layer=0,
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
            leaf=ref,
        )

    def test_leaf_record_roundtrip_and_fingerprint(self):
        node = self._leaf()
        assert node.fingerprint
        assert node.with_state("stale").fingerprint == node.fingerprint

    def test_leaf_rejects_children_and_summary(self):
        node = self._leaf()
        with pytest.raises(ValueError, match="cannot have children"):
            NodeRecord(
                node_id=node.node_id,
                revision=1,
                kind="leaf",
                state="ready",
                layer=0,
                content_hash=node.content_hash,
                leaf=node.leaf,
                children=(ChildRef(child_id="nl-x", child_revision=1),),
            )

    def test_region_requires_summary_and_members(self):
        leaf = self._leaf()
        contract = {"prompt": "summarize-v1"}
        children = (ChildRef(child_id=leaf.node_id, child_revision=1, ordinal=0),)
        rid = region_node_id(children, contract)
        with pytest.raises(ValueError, match="non-empty summary"):
            NodeRecord(
                node_id=rid,
                revision=1,
                kind="region",
                state="ready",
                layer=1,
                content_hash="c" * 64,
                children=children,
                contract=contract,
            )

    def test_cross_paper_region_has_no_page_masquerade(self):
        import hashlib

        leaf_a = self._leaf("alpha")
        ref_b = LeafRef(local_id="p9", revision=1, block_id="cb-z").resolve(4)
        leaf_b = NodeRecord(
            node_id=leaf_node_id(ref_b),
            revision=1,
            kind="leaf",
            state="ready",
            layer=0,
            content_hash=hashlib.sha256(b"beta").hexdigest(),
            leaf=ref_b,
        )
        contract = {"prompt": "summarize-v1", "model": "m"}
        children = (
            ChildRef(child_id=leaf_a.node_id, child_revision=1, ordinal=0),
            ChildRef(child_id=leaf_b.node_id, child_revision=1, ordinal=1),
        )
        node = NodeRecord(
            node_id=region_node_id(children, contract),
            revision=1,
            kind="region",
            state="ready",
            layer=1,
            content_hash=hashlib.sha256(b"summary").hexdigest(),
            summary="joint summary",
            children=children,
            contract=contract,
        )
        # The contract exposes no page/line fields on regions at all.
        assert not hasattr(node, "page_start")
        assert node.leaf is None


class TestStageProtocol:
    def test_worst_wins_and_queryability(self):
        assert combine_stage_states([]) == StageState.ABSENT
        assert combine_stage_states(["ready", "degraded"]) == StageState.DEGRADED
        assert combine_stage_states(["ready", "failed"]) == StageState.FAILED
        assert combine_stage_states(["partial", "stale"]) == StageState.STALE
        assert is_queryable("ready") and is_queryable("degraded")
        assert not is_queryable("partial") and not is_queryable("failed")
        assert not is_queryable("absent") and not is_queryable("stale")

    def test_stage_report_json_shape(self):
        report = StageReport(
            stage="prepare",
            state=StageState.PARTIAL,
            counts={"nodes": 3},
            created=2,
            failed=1,
            duration_ms=12.3456,
        )
        payload = report.to_json()
        assert payload["state"] == "partial"
        assert payload["created"] == 2 and payload["failed"] == 1
        assert payload["duration_ms"] == 12.346

    def test_read_receipt_binds_range(self):
        receipt = ReadReceipt(
            request_id="q1",
            tool="read",
            node_id="nl-x",
            node_revision=1,
            local_id="p1",
            block_id="cb-1",
            char_start=0,
            char_end=5,
            content_hash="d" * 64,
            tokens=3,
        )
        assert receipt.span_key == ("p1", "cb-1", 0, 5)
        with pytest.raises(ValueError, match="tool"):
            ReadReceipt(
                request_id="q1",
                tool="guess",
                node_id="nl-x",
                node_revision=1,
                local_id="p1",
                block_id="cb-1",
                char_start=0,
                char_end=5,
                content_hash="d" * 64,
                tokens=3,
            )
