"""T41: evidence ranking, duplicate collapse and unread-source refusal."""

from __future__ import annotations

import pytest

from drbrain.tree.contracts import ReadReceipt
from drbrain.tree.evidence import (
    EvidenceError,
    TreeEvidence,
    evidence_from_navigation,
    quoted_text,
    rank_evidence,
    validate_evidence,
)


class FakeResult:
    def __init__(self, evidence, receipts, query="q"):
        self.evidence = evidence
        self.receipts = receipts
        self.query = query


def _receipt(block_id="cb-1", start=0, end=10, node_id="nl-a") -> ReadReceipt:
    return ReadReceipt(
        request_id="q1",
        tool="read",
        node_id=node_id,
        node_revision=1,
        local_id="p1",
        block_id=block_id,
        char_start=start,
        char_end=end,
        content_hash="h" * 64,
        tokens=5,
    )


def _leaf_item(node_id="nl-a", score=1.0, block_id="cb-1", start=0, end=10):
    return {
        "node_id": node_id,
        "node_revision": 1,
        "kind": "leaf",
        "local_id": "p1",
        "score": score,
        "source": "leaf",
        "text": "alpha beta",
        "receipt": {
            "block_id": block_id,
            "char_start": start,
            "char_end": end,
            "content_hash": "h" * 64,
            "tokens": 5,
        },
    }


class TestAssembly:
    def test_leaf_evidence_requires_a_receipt(self):
        result = FakeResult([_leaf_item()], [], query="alpha")
        with pytest.raises(EvidenceError, match="no read receipt"):
            evidence_from_navigation(result)

    def test_leaf_evidence_carries_the_receipt_hash(self):
        result = FakeResult([_leaf_item()], [_receipt()], query="alpha")
        items = evidence_from_navigation(result)
        assert len(items) == 1
        assert items[0].content_hash == "h" * 64
        assert items[0].source == "leaf" and items[0].query == "alpha"

    def test_summary_evidence_is_marked_not_leaf(self):
        result = FakeResult(
            [
                {
                    "node_id": "nr-1",
                    "kind": "region",
                    "local_id": "",
                    "score": 0.9,
                    "source": "summary",
                    "text": "summary text",
                    "receipt": {"block_id": "summary:nr-1", "char_start": 0, "char_end": 12},
                }
            ],
            [],
        )
        items = evidence_from_navigation(result)
        assert items[0].source == "summary"
        assert quoted_text(items) == []  # summaries are not source quotes
        assert quoted_text(items, allow_summaries=True) == ["summary text"]


class TestRanking:
    def test_duplicate_spans_collapse_with_origins_recorded(self):
        first = TreeEvidence(
            node_id="nl-a",
            node_revision=1,
            kind="leaf",
            local_id="p1",
            block_id="cb-1",
            char_start=0,
            char_end=10,
            content_hash="h",
            tokens=5,
            source="leaf",
            score=0.5,
        )
        better = TreeEvidence(
            node_id="nl-b",
            node_revision=1,
            kind="leaf",
            local_id="p1",
            block_id="cb-1",
            char_start=0,
            char_end=10,
            content_hash="h",
            tokens=5,
            source="leaf",
            score=0.9,
        )
        ranked = rank_evidence([first, better])
        assert len(ranked) == 1
        assert ranked[0].score == 0.9
        assert "nl-a" in ranked[0].via and "nl-b" in ranked[0].via

    def test_distinct_spans_keep_score_order(self):
        low = TreeEvidence(
            node_id="nl-a",
            node_revision=1,
            kind="leaf",
            local_id="p1",
            block_id="cb-1",
            char_start=0,
            char_end=5,
            content_hash="h",
            tokens=2,
            source="leaf",
            score=0.2,
        )
        high = TreeEvidence(
            node_id="nl-b",
            node_revision=1,
            kind="leaf",
            local_id="p2",
            block_id="cb-9",
            char_start=3,
            char_end=9,
            content_hash="g",
            tokens=3,
            source="leaf",
            score=0.8,
        )
        ranked = rank_evidence([low, high])
        assert [item.node_id for item in ranked] == ["nl-b", "nl-a"]

    def test_validate_reports_coverage_and_duplicates(self):
        item = TreeEvidence(
            node_id="nl-a",
            node_revision=1,
            kind="leaf",
            local_id="p1",
            block_id="cb-1",
            char_start=0,
            char_end=5,
            content_hash="h",
            tokens=2,
            source="leaf",
        )
        report = validate_evidence([item, item])
        assert report["items"] == 2 and report["unique_spans"] == 1
        assert report["duplicate_spans"] == 1 and report["leaf_items"] == 2

    def test_summary_without_expansion_is_flagged(self):
        summary = TreeEvidence(
            node_id="nr-1",
            node_revision=1,
            kind="region",
            local_id="",
            block_id="summary:nr-1",
            char_start=0,
            char_end=5,
            content_hash="h",
            tokens=2,
            source="summary",
        )
        report = validate_evidence([summary])
        assert report["unbacked_summaries"] == ["nr-1"]

    def test_invalid_sources_are_rejected(self):
        with pytest.raises(ValueError, match="unsupported evidence source"):
            TreeEvidence(
                node_id="x",
                node_revision=1,
                kind="leaf",
                local_id="p",
                block_id="cb",
                char_start=0,
                char_end=1,
                content_hash="h",
                tokens=1,
                source="guess",
            )
        with pytest.raises(EvidenceError, match="concrete span"):
            TreeEvidence(
                node_id="x",
                node_revision=1,
                kind="leaf",
                local_id="p",
                block_id="",
                char_start=0,
                char_end=0,
                content_hash="h",
                tokens=1,
                source="leaf",
            )
