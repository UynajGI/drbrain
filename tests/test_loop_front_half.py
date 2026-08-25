"""Durability tests for the autoresearch proposal-to-queue front half."""

from __future__ import annotations

import pytest

from drbrain.loop.front_half import FRONT_HALF_NODE_SPECS, DurableFrontHalf
from drbrain.loop.store import RunLedger
from drbrain.loop.transitions import TransitionService


def _front_half(tmp_path) -> DurableFrontHalf:
    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    run = ledger.get_or_create_run("durable front half")
    front_half = DurableFrontHalf(TransitionService(ledger), run.run_id)
    front_half.ensure_node_contracts()
    return front_half


def _proposal() -> dict[str, object]:
    return {
        "claim_id": "cl-durable-proposal",
        "statement": "A recoverable proposal needs an independent review.",
        "prediction": "A persisted review unlocks one queue item.",
        "falsification": "A self-review unlocks the queue item.",
        "conditions": {"mode": "test"},
    }


def test_front_half_persists_contracts_for_every_migrated_node(tmp_path):
    front_half = _front_half(tmp_path)

    snapshot = front_half.snapshot()

    assert set(snapshot["node_specs"]) == {spec.name for spec in FRONT_HALF_NODE_SPECS}
    assert snapshot["node_specs"]["critique"]["max_attempts"] >= 1
    assert snapshot["node_specs"]["retrieve"]["retry_class"]


def test_front_half_recovers_proposal_review_and_single_queue_item(tmp_path):
    front_half = _front_half(tmp_path)
    proposal = front_half.record_proposal(_proposal(), author="analyst")

    pending = front_half.settle_proposal(proposal["proposal_id"], discard_score=0.4)
    assert pending["status"] == "discussion_pending"
    assert pending["queue_item"]["status"] == "pending_review"

    with pytest.raises(ValueError, match="non-author"):
        front_half.record_review(
            proposal["proposal_id"], reviewer="analyst", score=0.9, verdict="KEEP", content="self"
        )

    review = front_half.record_review(
        proposal["proposal_id"], reviewer="critic-1", score=0.9, verdict="KEEP", content="sound"
    )
    accepted = front_half.settle_proposal(proposal["proposal_id"], discard_score=0.4)
    repeated = front_half.settle_proposal(proposal["proposal_id"], discard_score=0.4)

    assert review["review_id"].startswith("rev-")
    assert accepted["status"] == "critiqued"
    assert accepted["queue_item"]["status"] == "ready"
    assert repeated["queue_item"]["queue_item_id"] == accepted["queue_item"]["queue_item_id"]

    recovered = DurableFrontHalf(front_half.transitions, front_half.run_id)
    snapshot = recovered.snapshot()
    assert snapshot["proposals"][0]["proposal_id"] == proposal["proposal_id"]
    assert snapshot["proposals"][0]["reviews"] == [review]
    assert snapshot["queue_items"] == [accepted["queue_item"]]
