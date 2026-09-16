"""One candidate set for structural and semantic groupings (plan T31).

Structure hints (T20/T21) and structure-conditioned assignment candidates
(T30) enter the *same* candidate pool here.  A proposal is only a
specification — normalized member ids, one summary contract and an audit
origin — never a second children tree: nothing in this module writes nodes or
membership.  Two proposals are the same work when their member set *and*
contract identity match, so a structural hint and a semantic candidate over
the same members are summarised once; equal members under different contracts
stay distinct instead of silently reusing the wrong summary.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from drbrain.tree.assign import AssignmentCandidate
from drbrain.tree.summary import SummaryContract


@dataclass(frozen=True)
class CandidateProposal:
    """A normalized candidate parent group."""

    member_ids: tuple[str, ...]
    contract: dict[str, Any]
    origin: str = "semantic"
    weights: dict[str, float] = field(default_factory=dict)
    structure: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if len(set(self.member_ids)) != len(self.member_ids):
            raise ValueError("proposal members must be unique")
        if len(self.member_ids) < 1:
            raise ValueError("proposal requires at least one member")
        if not self.contract:
            raise ValueError("proposal requires a summary contract")
        if self.origin not in {"structure", "semantic", "mixed", "manual"}:
            raise ValueError(f"unsupported proposal origin {self.origin!r}")

    @property
    def key(self) -> str:
        return proposal_key(self.member_ids, self.contract)

    def merged_with(self, other: CandidateProposal) -> CandidateProposal:
        """Merge two identical-key proposals into one (origin becomes mixed)."""
        if self.key != other.key:
            raise ValueError("cannot merge proposals with different keys")
        origins = {self.origin, other.origin}
        origin = "mixed" if len(origins) > 1 else self.origin
        weights = {**other.weights, **self.weights}
        structure = tuple(dict.fromkeys([*self.structure, *other.structure]))
        return CandidateProposal(
            member_ids=self.member_ids,
            contract=self.contract,
            origin=origin,
            weights=weights,
            structure=structure,
        )


def proposal_key(member_ids: Sequence[str], contract: Mapping[str, Any]) -> str:
    members = "|".join(sorted(dict.fromkeys(str(member_id) for member_id in member_ids)))
    payload = json.dumps(dict(contract), sort_keys=True, ensure_ascii=False)
    return "prop-" + hashlib.sha256(f"{members}#{payload}".encode()).hexdigest()[:32]


def normalize_contract(contract: SummaryContract | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(contract, SummaryContract):
        return contract.canonical()
    required = {"prompt_id", "template", "model", "tokenizer", "max_output_tokens"}
    missing = required - set(contract)
    if missing:
        raise ValueError(f"summary contract missing fields: {sorted(missing)}")
    return dict(contract)


def proposal_from_candidate(
    candidate: AssignmentCandidate,
    contract: SummaryContract | Mapping[str, Any],
    *,
    min_members: int = 2,
) -> CandidateProposal | None:
    """Semantic candidate -> proposal; groups below the minimum are dropped."""
    member_ids = [node_id for node_id, _weight in candidate.members]
    if len(member_ids) < min_members:
        return None
    normalized = normalize_contract(contract)
    return CandidateProposal(
        member_ids=tuple(sorted(dict.fromkeys(member_ids))),
        contract=normalized,
        origin="semantic",
        weights={node_id: float(weight) for node_id, weight in candidate.members},
        structure=(("component", candidate.component_id),),
    )


def proposals_from_assignment(
    candidates: Iterable[AssignmentCandidate],
    contract: SummaryContract | Mapping[str, Any],
    *,
    min_members: int = 2,
) -> list[CandidateProposal]:
    proposals = []
    for candidate in candidates:
        proposal = proposal_from_candidate(candidate, contract, min_members=min_members)
        if proposal is not None:
            proposals.append(proposal)
    return proposals


def proposals_from_structure(
    hints: Sequence[Mapping[str, Any]],
    members_by_scope: Mapping[tuple[str, str], Sequence[str]],
    contract: SummaryContract | Mapping[str, Any],
    *,
    min_members: int = 2,
) -> list[CandidateProposal]:
    """Structural hints -> proposals over the *same* canonical member ids.

    ``members_by_scope`` maps a hint scope (``(local_id, scope_key)`` where
    scope key is the joined heading path or the page range) to canonical
    member ids; hints whose scope has fewer than ``min_members`` members are
    skipped (a single section is not worth a parent node).
    """
    normalized = normalize_contract(contract)
    proposals: list[CandidateProposal] = []
    for hint in hints:
        local_id = str(hint.get("local_id") or "")
        if "heading_path" in hint:
            scope_key = " > ".join(hint.get("heading_path") or ())
        elif "page_start" in hint:
            scope_key = f"{hint.get('page_start')}-{hint.get('page_end')}"
        else:
            continue
        member_ids = list(members_by_scope.get((local_id, scope_key), ()))
        if len(member_ids) < min_members:
            continue
        proposals.append(
            CandidateProposal(
                member_ids=tuple(sorted(dict.fromkeys(member_ids))),
                contract=normalized,
                origin="structure",
                structure=(("section", scope_key),),
            )
        )
    return proposals


def merge_proposals(*groups: Iterable[CandidateProposal]) -> list[CandidateProposal]:
    """Union proposals, de-duplicating by member set + contract identity."""
    merged: dict[str, CandidateProposal] = {}
    for group in groups:
        for proposal in group:
            existing = merged.get(proposal.key)
            merged[proposal.key] = proposal if existing is None else existing.merged_with(proposal)
    return list(merged.values())


def seen_keys(proposals: Iterable[CandidateProposal]) -> set[str]:
    return {proposal.key for proposal in proposals}
