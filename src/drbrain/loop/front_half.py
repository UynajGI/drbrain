"""Durable proposal-to-queue records for the autoresearch workflow front half.

The workflow's message board and queue are intentionally still lightweight
per-run views.  This module gives those views stable identities and delegates
every canonical write and gate decision to :class:`TransitionService`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from drbrain.loop.transitions import TransitionService


@dataclass(frozen=True)
class FrontHalfNodeSpec:
    """The durable contract for one migrated workflow node."""

    name: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    allowed_tools: tuple[str, ...]
    max_attempts: int
    retry_class: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "allowed_tools": list(self.allowed_tools),
            "max_attempts": self.max_attempts,
            "retry_class": self.retry_class,
        }


def _node(
    name: str, input_name: str, output_name: str, *, allowed_tools: tuple[str, ...] = ()
) -> FrontHalfNodeSpec:
    return FrontHalfNodeSpec(
        name=name,
        input_schema={"event": input_name},
        output_schema={"event": output_name},
        allowed_tools=allowed_tools,
        max_attempts=3,
        retry_class="transient",
    )


# The migration deliberately stops at critique.  Compute, verification, settlement
# and reporting gain durable artifact semantics in PR6 rather than here.
FRONT_HALF_NODE_SPECS = (
    _node("plan_task", "StartEvent", "TaskPlanned"),
    _node("retrieve", "TaskPlanned|RetrieveAgain", "Retrieved", allowed_tools=("rag_retrieve",)),
    _node("filter", "Retrieved", "Filtered"),
    _node("parse_pdf", "Filtered", "Parsed", allowed_tools=("parse_pdf",)),
    _node("extract", "Parsed", "Extracted", allowed_tools=("extract",)),
    _node("normalize", "Extracted", "Normalized"),
    _node("fuse", "Normalized", "Fused"),
    _node("identify_gaps", "Fused", "GapsIdentified"),
    _node("critique", "GapsIdentified", "Critiqued"),
)


def _stable_id(prefix: str, *parts: str) -> str:
    value = "\x00".join(parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(value).hexdigest()[:24]}"


def proposal_id(run_id: str, claim_id: str) -> str:
    return _stable_id("prp", run_id, claim_id)


def review_id(proposal_id_value: str, reviewer: str) -> str:
    return _stable_id("rev", proposal_id_value, reviewer)


def queue_item_id(proposal_id_value: str) -> str:
    return _stable_id("que", proposal_id_value)


class DurableFrontHalf:
    """Run-bound facade over the transaction-owned proposal records."""

    def __init__(self, transitions: TransitionService, run_id: str) -> None:
        self.transitions = transitions
        self.run_id = run_id

    def ensure_node_contracts(self) -> None:
        self.transitions.register_front_half_node_contracts(
            self.run_id, {spec.name: spec.to_dict() for spec in FRONT_HALF_NODE_SPECS}
        )

    def record_proposal(self, hypothesis: dict[str, Any], *, author: str) -> dict[str, Any]:
        payload = dict(hypothesis)
        claim_id_value = str(payload.get("claim_id") or "").strip()
        if not claim_id_value:
            raise ValueError("durable proposal requires a stable claim_id")
        return self.transitions.record_front_half_proposal(
            self.run_id,
            proposal_id=proposal_id(self.run_id, claim_id_value),
            claim_id=claim_id_value,
            author=author,
            payload=payload,
        )

    def record_review(
        self,
        proposal_id_value: str,
        *,
        reviewer: str,
        score: float,
        verdict: str,
        content: str,
    ) -> dict[str, Any]:
        return self.transitions.record_front_half_review(
            self.run_id,
            proposal_id=proposal_id_value,
            review_id=review_id(proposal_id_value, reviewer),
            reviewer=reviewer,
            score=score,
            verdict=verdict,
            content=content,
        )

    def settle_proposal(self, proposal_id_value: str, *, discard_score: float) -> dict[str, Any]:
        return self.transitions.settle_front_half_proposal(
            self.run_id,
            proposal_id=proposal_id_value,
            queue_item_id=queue_item_id(proposal_id_value),
            discard_score=discard_score,
        )

    def snapshot(self) -> dict[str, Any]:
        return self.transitions.front_half_snapshot(self.run_id)
