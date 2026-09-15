"""Unified summary service and cache (plan T26).

Every accepted region summary goes through this one service.  Reuse is
decided by *identity*, never by text similarity: the cache key covers the
ordered member set (id, revision, content hash), the prompt id and text, the
model, the tokenizer and the output budget.  A different scope, order, prompt
or model revision therefore misses.

Failures are recorded, never reused as success: empty responses, truncated
finish reasons, over-budget outputs and model exceptions all leave a
``failed`` cache entry (which a retry overwrites), and callers must treat the
returned outcome as unusable.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

DEFAULT_TEMPLATE = (
    "Summarize the source passages below for retrieval routing.\n"
    "Keep factual details, numbers, negation and open questions. Do not add "
    "facts that are not present. Answer with the summary only.\n\n"
    "{members}\n"
)

MEMBER_TEMPLATE = (
    "[member id={node_id} paper={local_id} revision={revision} heading={heading}]\n"
    "{text}\n[/member]\n"
)


class SummaryModelError(RuntimeError):
    """The index model could not produce a usable summary."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class SummaryMember:
    node_id: str
    node_revision: int
    content_hash: str
    text: str
    local_id: str = ""
    heading_path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("summary member requires a node id")
        if self.node_revision < 1:
            raise ValueError("summary member revision must be >= 1")
        if not self.content_hash:
            raise ValueError("summary member requires a content hash")


@dataclass(frozen=True)
class SummaryContract:
    prompt_id: str = "tree-summarize-v1"
    template: str = DEFAULT_TEMPLATE
    model: str = ""
    tokenizer: str = ""
    max_output_tokens: int = 512
    input_budget: int = 3500
    member_order: str = "given"  # "given" | "sorted"

    def __post_init__(self) -> None:
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.input_budget <= 0:
            raise ValueError("input_budget must be positive")
        if self.member_order not in {"given", "sorted"}:
            raise ValueError("member_order must be 'given' or 'sorted'")
        if "{members}" not in self.template:
            raise ValueError("summary template must contain a {members} slot")

    def canonical(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "template": self.template,
            "model": self.model,
            "tokenizer": self.tokenizer,
            "max_output_tokens": self.max_output_tokens,
            "input_budget": self.input_budget,
            "member_order": self.member_order,
        }

    def digest(self) -> str:
        payload = json.dumps(self.canonical(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


#: Outcome reason prefix for a model call that failed (retryable
#: infrastructure), as opposed to a candidate group the acceptance gate
#: rejected.  Only the former may hold a stage back from "complete".
EXECUTION_FAILURE_PREFIX = "model_error"


@dataclass(frozen=True)
class SummaryResponse:
    text: str
    finish_reason: str = "stop"
    usage: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SummaryOutcome:
    ok: bool
    summary: str
    summary_tokens: int
    from_cache: bool
    prompt_tokens: int = 0
    reason: str = ""

    @property
    def execution_failure(self) -> bool:
        """True when the model call itself failed, not when a group was rejected."""
        return self.reason.startswith(EXECUTION_FAILURE_PREFIX)


class SummaryModel(Protocol):  # pragma: no cover - structural typing
    def complete(self, prompt: str, *, max_tokens: int) -> SummaryResponse: ...


def order_members(
    members: Sequence[SummaryMember], contract: SummaryContract
) -> list[SummaryMember]:
    items = list(members)
    if contract.member_order == "sorted":
        items.sort(key=lambda member: (member.local_id, member.node_id))
    return items


def members_signature(members: Sequence[SummaryMember]) -> str:
    """Ordered member identity: id, revision and content hash."""
    return "|".join(
        f"{member.node_id}@{member.node_revision}:{member.content_hash}" for member in members
    )


def cache_key(members: Sequence[SummaryMember], contract: SummaryContract) -> str:
    payload = f"{contract.digest()}|{members_signature(members)}"
    return "sum-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def build_prompt(members: Sequence[SummaryMember], contract: SummaryContract) -> str:
    blocks = [
        MEMBER_TEMPLATE.format(
            node_id=member.node_id,
            local_id=member.local_id or "-",
            revision=member.node_revision,
            heading=" > ".join(member.heading_path) if member.heading_path else "-",
            text=member.text,
        )
        for member in members
    ]
    return contract.template.format(members="\n".join(blocks))


def _default_count_tokens(text: str) -> int:
    from drbrain.services.tokens import count_tokens

    return int(count_tokens(text))


class SummaryService:
    """One summary service for structural and semantic candidates alike."""

    def __init__(self, db, *, count_tokens: Callable[[str], int] | None = None) -> None:
        self.db = db
        self._count_tokens = count_tokens or _default_count_tokens
        self.model_calls = 0

    def summarize(
        self,
        members: Sequence[SummaryMember],
        contract: SummaryContract,
        model: SummaryModel,
    ) -> SummaryOutcome:
        ordered = order_members(members, contract)
        if not ordered:
            raise ValueError("summarize requires at least one member")
        key = cache_key(ordered, contract)
        cached = self.db.get_summary_cache(key)
        if cached is not None and cached.get("state") == "ready" and cached.get("summary"):
            return SummaryOutcome(
                ok=True,
                summary=str(cached["summary"]),
                summary_tokens=int(cached.get("summary_tokens") or 0),
                prompt_tokens=int(cached.get("prompt_tokens") or 0),
                from_cache=True,
            )

        prompt = build_prompt(ordered, contract)
        prompt_tokens = self._count_tokens(prompt)
        if prompt_tokens > contract.input_budget:
            reason = "input_over_budget"
            self._record_failure(key, ordered, contract, reason, prompt_tokens)
            return SummaryOutcome(
                ok=False,
                summary="",
                summary_tokens=0,
                prompt_tokens=prompt_tokens,
                from_cache=False,
                reason=reason,
            )

        self.model_calls += 1
        try:
            response = model.complete(prompt, max_tokens=contract.max_output_tokens)
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised as outcome
            reason = f"{EXECUTION_FAILURE_PREFIX}: {type(exc).__name__}"
            self._record_failure(key, ordered, contract, reason, prompt_tokens)
            return SummaryOutcome(
                ok=False,
                summary="",
                summary_tokens=0,
                prompt_tokens=prompt_tokens,
                from_cache=False,
                reason=reason,
            )

        summary = str(getattr(response, "text", "") or "").strip()
        finish_reason = str(getattr(response, "finish_reason", "") or "").strip().lower()
        if not summary:
            reason = "empty_summary"
        elif finish_reason not in {"stop", "end_turn", "completed", ""}:
            reason = "summary_truncated"
        else:
            summary_tokens = self._count_tokens(summary)
            if summary_tokens > contract.max_output_tokens:
                reason = "summary_over_budget"
            else:
                self.db.put_summary_cache(
                    key,
                    state="ready",
                    summary=summary,
                    prompt_tokens=prompt_tokens,
                    summary_tokens=summary_tokens,
                    model=contract.model,
                    contract_json=json.dumps(contract.canonical(), sort_keys=True),
                    members_json=json.dumps(
                        [member.node_id for member in ordered], ensure_ascii=False
                    ),
                )
                return SummaryOutcome(
                    ok=True,
                    summary=summary,
                    summary_tokens=summary_tokens,
                    prompt_tokens=prompt_tokens,
                    from_cache=False,
                )
            self._record_failure(key, ordered, contract, reason, prompt_tokens)
            return SummaryOutcome(
                ok=False,
                summary="",
                summary_tokens=summary_tokens,
                prompt_tokens=prompt_tokens,
                from_cache=False,
                reason=reason,
            )

        self._record_failure(key, ordered, contract, reason, prompt_tokens)
        return SummaryOutcome(
            ok=False,
            summary="",
            summary_tokens=0,
            prompt_tokens=prompt_tokens,
            from_cache=False,
            reason=reason,
        )

    def _record_failure(
        self,
        key: str,
        members: Sequence[SummaryMember],
        contract: SummaryContract,
        reason: str,
        prompt_tokens: int,
    ) -> None:
        self.db.put_summary_cache(
            key,
            state="failed",
            reason=reason,
            prompt_tokens=prompt_tokens,
            model=contract.model,
            contract_json=json.dumps(contract.canonical(), sort_keys=True),
            members_json=json.dumps([member.node_id for member in members], ensure_ascii=False),
        )
