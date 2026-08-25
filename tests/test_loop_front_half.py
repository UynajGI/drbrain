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


def test_front_half_rejects_conflicting_proposal_or_review_replays(tmp_path):
    front_half = _front_half(tmp_path)
    proposal = front_half.record_proposal(_proposal(), author="analyst")

    changed = _proposal()
    changed["prediction"] = "a different experiment contract"
    with pytest.raises(ValueError, match="conflicts"):
        front_half.record_proposal(changed, author="analyst")

    review = front_half.record_review(
        proposal["proposal_id"], reviewer="critic-1", score=0.9, verdict="KEEP", content="sound"
    )
    with pytest.raises(ValueError, match="different review_id"):
        front_half.transitions.record_front_half_review(
            front_half.run_id,
            proposal_id=proposal["proposal_id"],
            review_id=f"{review['review_id']}-conflict",
            reviewer="critic-1",
            score=0.9,
            verdict="KEEP",
            content="sound",
        )


def test_front_half_normalizes_contracts_and_rejects_node_contract_drift(tmp_path):
    front_half = _front_half(tmp_path)
    proposal = _proposal()
    proposal["conditions"] = {"labels": ("a", "b")}
    front_half.record_proposal(proposal, author="analyst")
    # JSON persistence turns tuples into lists; the same live contract remains a replay.
    front_half.record_proposal(proposal, author="analyst")

    changed = FRONT_HALF_NODE_SPECS[-1].to_dict()
    changed["max_attempts"] = int(changed["max_attempts"]) + 1
    with pytest.raises(ValueError, match="contract conflicts"):
        front_half.transitions.register_front_half_node_contracts(
            front_half.run_id, {"critique": changed}
        )


def test_front_half_does_not_revive_a_discarded_proposal(tmp_path):
    front_half = _front_half(tmp_path)
    proposal = front_half.record_proposal(_proposal(), author="analyst")
    front_half.record_review(
        proposal["proposal_id"], reviewer="critic-1", score=0.1, verdict="DISCARD", content="weak"
    )
    first = front_half.settle_proposal(proposal["proposal_id"], discard_score=0.4)
    front_half.record_review(
        proposal["proposal_id"], reviewer="critic-2", score=0.9, verdict="KEEP", content="later"
    )
    replay = front_half.settle_proposal(proposal["proposal_id"], discard_score=0.4)

    assert first["status"] == replay["status"] == "discarded"
    assert replay["queue_item"]["status"] == "discarded"


def test_front_half_audits_a_conflicting_canonical_review_replay(tmp_path):
    front_half = _front_half(tmp_path)
    proposal = front_half.record_proposal(_proposal(), author="analyst")
    review = front_half.record_review(
        proposal["proposal_id"], reviewer="critic-1", score=0.9, verdict="KEEP", content="sound"
    )

    canonical = front_half.transitions.record_front_half_review(
        front_half.run_id,
        proposal_id=proposal["proposal_id"],
        review_id=review["review_id"],
        reviewer="critic-1",
        score=0.1,
        verdict="DISCARD",
        content="changed after retry",
    )

    assert canonical["score"] == 0.9
    ledger = front_half.transitions._ledger  # noqa: SLF001 - inspect durable audit boundary
    assert ledger.events(front_half.run_id)[-1].event_type == "critic_review_replay_ignored"
