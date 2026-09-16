"""T31-T33: one candidate pool, the cost gate, and staged acceptance."""

from __future__ import annotations

import hashlib

import pytest

from drbrain.storage.database import Database
from drbrain.tree.assign import AssignmentCandidate
from drbrain.tree.blocks import BlockPolicy, build_content_blocks
from drbrain.tree.contracts import ChildRef, LeafRef, NodeRecord, leaf_node_id, region_node_id
from drbrain.tree.cost import (
    CostParams,
    coverage_for_members,
    proposal_post_check,
    proposal_pre_screen,
)
from drbrain.tree.proposals import (
    CandidateProposal,
    merge_proposals,
    proposal_key,
    proposals_from_assignment,
    proposals_from_structure,
)
from drbrain.tree.summary import SummaryContract


def _count(text: str) -> int:
    return max(1, len(text.split()))


def _doc(db: Database, local_id: str, text: str):
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


def _leaf(block, local_id: str = "p1") -> NodeRecord:
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


DOC = (
    "# Methods\n\n"
    + " ".join(["sparse"] * 40)
    + " retrieval with bm25 over tokenized passages and a wide evaluation across corpora.\n\n"
    "# Results\n\n"
    + " ".join(["benchmark"] * 40)
    + " we improve recall by twelve points on the same benchmark suite.\n"
)


def _contract(**overrides) -> SummaryContract:
    base = {
        "model": "spark-x25-4b",
        "tokenizer": "o200k_base",
        "max_output_tokens": 128,
        "input_budget": 4000,
    }
    base.update(overrides)
    return SummaryContract(**base)


class TestProposalMerging:
    def test_same_members_and_contract_deduplicate_once(self):
        contract = _contract()
        structure = proposals_from_structure(
            [{"local_id": "p1", "heading_path": ["Methods", "Detail"]}],
            {("p1", "Methods > Detail"): ["nl-b", "nl-a"]},
            contract,
        )
        semantic = proposals_from_assignment(
            [AssignmentCandidate("g0", (("nl-a", 0.9), ("nl-b", 0.8)))], contract
        )
        merged = merge_proposals(structure, semantic)
        assert len(merged) == 1
        proposal = merged[0]
        assert proposal.origin == "mixed"
        assert proposal.member_ids == ("nl-a", "nl-b")
        assert proposal.structure == (("section", "Methods > Detail"), ("component", "g0"))

    def test_same_members_different_contract_stay_separate(self):
        members = ["nl-a", "nl-b"]
        first = CandidateProposal(tuple(members), _contract().canonical(), origin="semantic")
        second = CandidateProposal(
            tuple(members), _contract(prompt_id="tree-summarize-v2").canonical(), origin="semantic"
        )
        assert first.key != second.key
        assert len(merge_proposals([first], [second])) == 2

    def test_single_block_sections_are_skipped(self):
        proposals = proposals_from_structure(
            [{"local_id": "p1", "heading_path": ["Intro"]}],
            {("p1", "Intro"): ["nl-only"]},
            _contract(),
        )
        assert proposals == []

    def test_key_is_stable_under_member_order(self):
        contract = _contract().canonical()
        assert proposal_key(["b", "a"], contract) == proposal_key(["a", "b"], contract)

    def test_merge_rejects_different_keys(self):
        a = CandidateProposal(("nl-a", "nl-b"), _contract().canonical())
        b = CandidateProposal(("nl-a", "nl-b"), _contract(model="other").canonical())
        with pytest.raises(ValueError, match="different keys"):
            a.merged_with(b)


class TestCostGate:
    def _setup(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        blocks = _doc(db, "p1", DOC)
        # The substantive paragraphs, not the two heading-only blocks.
        leaves = [_leaf(blocks[1]), _leaf(blocks[3])]
        for leaf in leaves:
            db.insert_tree_node(leaf, publish=True)
        return db, leaves

    def test_coverage_counts_shared_origins_once(self, tmp_path):
        db, leaves = self._setup(tmp_path)
        region_refs = tuple(
            ChildRef(child_id=leaf.node_id, child_revision=1, ordinal=i)
            for i, leaf in enumerate(leaves)
        )
        region = NodeRecord(
            node_id=region_node_id(region_refs, {"p": "1"}),
            revision=1,
            kind="region",
            state="ready",
            layer=1,
            content_hash=hashlib.sha256(b"s").hexdigest(),
            summary="s",
            children=region_refs,
            contract={"p": "1"},
        )
        db.insert_tree_node(region, publish=True)
        direct = coverage_for_members(
            db, [leaves[0].node_id, leaves[1].node_id], count_tokens=_count
        )
        via_region = coverage_for_members(
            db, [leaves[0].node_id, region.node_id], count_tokens=_count
        )
        assert direct.unique_tokens == via_region.unique_tokens
        assert direct.unique_chars == via_region.unique_chars

    def test_pre_screen_rejects_single_member_and_accepts_pairs(self, tmp_path):
        db, leaves = self._setup(tmp_path)
        contract = _contract().canonical()
        single = CandidateProposal((leaves[0].node_id,), contract)
        assert proposal_pre_screen(db, single, count_tokens=_count).reason == "single_member"
        pair = CandidateProposal((leaves[0].node_id, leaves[1].node_id), contract)
        decision = proposal_pre_screen(
            db, pair, count_tokens=_count, params=CostParams(summary_output_budget=16)
        )
        assert decision.accepted, decision.reason

    def test_pre_screen_duplicate_group(self, tmp_path):
        db, leaves = self._setup(tmp_path)
        contract = _contract().canonical()
        pair = CandidateProposal((leaves[0].node_id, leaves[1].node_id), contract)
        decision = proposal_pre_screen(db, pair, count_tokens=_count, seen_member_keys={pair.key})
        assert decision.reason == "duplicate_group"

    def test_post_check_rejects_unread_members(self, tmp_path):
        db, leaves = self._setup(tmp_path)
        contract = _contract().canonical()
        pair = CandidateProposal((leaves[0].node_id, leaves[1].node_id), contract)
        one_sided = coverage_for_members(db, [leaves[0].node_id], count_tokens=_count)
        decision = proposal_post_check(
            db,
            pair,
            summary_text="a short summary",
            summary_tokens=3,
            finish_reason="stop",
            referenced_spans=one_sided.spans,
            count_tokens=_count,
        )
        assert decision.reason == "coverage_incomplete"
        full = coverage_for_members(db, list(pair.member_ids), count_tokens=_count)
        ok = proposal_post_check(
            db,
            pair,
            summary_text="a short summary",
            summary_tokens=3,
            finish_reason="stop",
            referenced_spans=full.spans,
            params=CostParams(summary_output_budget=16),
            count_tokens=_count,
        )
        assert ok.accepted, ok.reason

    def test_post_check_budget_and_gain(self, tmp_path):
        db, leaves = self._setup(tmp_path)
        contract = _contract().canonical()
        pair = CandidateProposal((leaves[0].node_id, leaves[1].node_id), contract)
        full = coverage_for_members(db, list(pair.member_ids), count_tokens=_count)
        over = proposal_post_check(
            db,
            pair,
            summary_text="x " * 500,
            summary_tokens=500,
            finish_reason="stop",
            referenced_spans=full.spans,
            params=CostParams(summary_output_budget=64),
            count_tokens=_count,
        )
        assert over.reason == "summary_over_budget"
        truncated = proposal_post_check(
            db,
            pair,
            summary_text="partial",
            summary_tokens=1,
            finish_reason="length",
            referenced_spans=full.spans,
            count_tokens=_count,
        )
        assert truncated.reason == "summary_truncated"
