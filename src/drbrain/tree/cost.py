"""Parent-acceptance and cost protocol (plan T05).

Frozen units and decisions
--------------------------
* Cost unit is a *token*, measured by the pipeline tokenizer; routing overhead
  is a configurable token constant per tool call (``tool_overhead_tokens``,
  default 60) so a region is only worth publishing when reading its summary
  plus one routed child is cheaper than reading the unique member text.
* Coverage is the union of *unique source ranges*; identical or overlapping
  leaf ranges count once, and a soft multi-parent path never inflates cost.
* The gate is a single-target estimate (the PageIndex routing assumption).
  Multi-evidence questions read several branches; that cost is estimated
  separately (``estimate_multi_target_cost``) and measured in T57/T58 — it is
  never presented as a worst-case guarantee.
* Two decisions per candidate group: a cheap pre-screen before calling the
  model, and a post-check on the actual summary.  A shorter summary alone
  never proves factual quality; acceptance only proves coverage/budget/gain.
* Rejection reasons come from the frozen ``REJECTION_REASONS`` set, and a
  rejected group leaves every member in place (enforced by the builder).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from drbrain.tree.affinity import SourceSpan
from drbrain.tree.contracts import REJECTION_REASONS

#: Token cost attributed to one navigation/tool step (read or expand).
DEFAULT_TOOL_OVERHEAD_TOKENS: int = 60


@dataclass(frozen=True)
class Coverage:
    """Unique source coverage of a candidate group."""

    spans: tuple[SourceSpan, ...]
    unique_tokens: int
    unique_chars: int
    documents: tuple[str, ...]

    def is_empty(self) -> bool:
        return not self.spans


def unique_coverage(spans: Iterable[SourceSpan]) -> Coverage:
    """Merge ranges per block; identical paths count once."""
    groups: dict[tuple[str, int, str], dict[tuple[int, int], SourceSpan]] = {}
    for span in spans:
        groups.setdefault((span.local_id, span.revision, span.block_id), {})[
            (span.char_start, span.char_end)
        ] = span
    total_tokens = 0.0
    total_chars = 0
    documents: list[str] = []
    unique: list[SourceSpan] = []
    for key, by_range in groups.items():
        documents.append(key[0])
        items = list(by_range.values())
        char_total = sum(item.char_end - item.char_start for item in items)
        token_total = sum(item.tokens for item in items)
        ratio = token_total / char_total if char_total > 0 else 0.0
        intervals = sorted((item.char_start, item.char_end) for item in items)
        union_len = 0
        cur_start, cur_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= cur_end:
                cur_end = max(cur_end, end)
            else:
                union_len += cur_end - cur_start
                cur_start, cur_end = start, end
        union_len += cur_end - cur_start
        total_chars += union_len
        total_tokens += union_len * ratio
        unique.extend(items)
    return Coverage(
        spans=tuple(unique),
        unique_tokens=int(round(total_tokens)),
        unique_chars=total_chars,
        documents=tuple(dict.fromkeys(documents)),
    )


@dataclass(frozen=True)
class Decision:
    """One gate decision with a frozen rejection reason."""

    accepted: bool
    reason: str = ""
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.accepted and self.reason:
            raise ValueError("accepted decision carries no rejection reason")
        if not self.accepted and self.reason not in REJECTION_REASONS:
            raise ValueError(f"unknown rejection reason {self.reason!r}")


@dataclass(frozen=True)
class CostParams:
    """Tunable cost parameters; defaults are provisional pending T57."""

    tool_overhead_tokens: int = DEFAULT_TOOL_OVERHEAD_TOKENS
    #: Pre-screen refuses candidates whose member text does not fit the
    #: summary input budget; over-budget groups are re-split or re-clustered
    #: by the builder instead of being truncated.
    summary_input_budget: int = 3500
    #: Maximum accepted summary output tokens.
    summary_output_budget: int = 512
    #: Minimum members for a parent; single members never merge upward.
    min_members: int = 2

    def __post_init__(self) -> None:
        if self.min_members < 2:
            raise ValueError("min_members must be >= 2")
        if self.summary_output_budget <= 0 or self.summary_input_budget <= 0:
            raise ValueError("budgets must be positive")


def read_cost_tokens(coverage: Coverage) -> int:
    """Cost of reading the unique member text instead of routing."""
    return coverage.unique_tokens


def estimated_route_cost(summary_budget_tokens: int, params: CostParams) -> int:
    """Summary plus one routed child read; the single-target assumption."""
    return int(summary_budget_tokens) + params.tool_overhead_tokens


def estimate_multi_target_cost(
    coverage: Coverage, *, branches: int, summary_tokens: int, params: CostParams
) -> dict[str, int]:
    """Explicit multi-evidence estimate; measured, never hidden."""
    if branches < 1:
        raise ValueError("branches must be >= 1")
    return {
        "branches": branches,
        "read_all_unique": coverage.unique_tokens,
        "route_then_read": summary_tokens
        + branches * params.tool_overhead_tokens
        + max(0, coverage.unique_tokens - summary_tokens),
        "route_only_lower_bound": summary_tokens + branches * params.tool_overhead_tokens,
    }


def pre_screen(
    *,
    member_ids: Sequence[str],
    coverage: Coverage,
    member_tokens: int,
    seen_member_keys: frozenset[str] | set[str],
    member_key: str,
    params: CostParams | None = None,
) -> Decision:
    """Cheap gate before any model call."""
    params = params or CostParams()
    if len(member_ids) < params.min_members:
        return Decision(False, "single_member", {"members": len(member_ids)})
    if coverage.is_empty():
        return Decision(False, "invalid_members", {"members": len(member_ids)})
    if member_key in seen_member_keys:
        return Decision(False, "duplicate_group", {"member_key": member_key})
    if member_tokens > params.summary_input_budget:
        return Decision(
            False,
            "input_over_budget",
            {"member_tokens": member_tokens, "budget": params.summary_input_budget},
        )
    route = estimated_route_cost(params.summary_output_budget, params)
    if read_cost_tokens(coverage) <= route:
        return Decision(
            False,
            "no_compression_gain",
            {"read_cost": read_cost_tokens(coverage), "route_cost": route, "phase": "estimate"},
        )
    return Decision(True, detail={"estimated_gain": read_cost_tokens(coverage) - route})


def post_check(
    *,
    summary_text: str,
    summary_tokens: int,
    finish_reason: str,
    coverage: Coverage,
    referenced_spans: Sequence[SourceSpan],
    params: CostParams | None = None,
) -> Decision:
    """Gate on the actual summary; only coverage/budget/gain are proven here."""
    params = params or CostParams()
    if not summary_text or not summary_text.strip():
        return Decision(False, "empty_summary", {})
    if finish_reason not in {"stop", "end_turn", "completed", ""}:
        return Decision(False, "summary_truncated", {"finish_reason": finish_reason})
    if summary_tokens > params.summary_output_budget:
        return Decision(
            False,
            "summary_over_budget",
            {"summary_tokens": summary_tokens, "budget": params.summary_output_budget},
        )
    expected = {span.key for span in coverage.spans}
    actual = {span.key for span in referenced_spans}
    if not actual or not expected.issubset(actual):
        return Decision(
            False,
            "coverage_incomplete",
            {"missing": sorted(str(key) for key in expected - actual)[:8]},
        )
    route = estimated_route_cost(summary_tokens, params)
    read = read_cost_tokens(coverage)
    if read <= route:
        return Decision(
            False,
            "no_compression_gain",
            {"read_cost": read, "route_cost": route, "phase": "actual"},
        )
    return Decision(True, detail={"read_cost": read, "route_cost": route, "gain": read - route})


def coverage_from_leaves(leaves: Mapping[str, SourceSpan]) -> Coverage:
    return unique_coverage(leaves.values())


# ── Proposal-level gates (T32/T33) ────────────────────────────────


def coverage_for_members(db, member_ids: Sequence[str], *, count_tokens=None) -> Coverage:
    """Exact unique coverage over the members' *leaf* spans (shared origins once)."""
    from drbrain.tree.assign import leaf_spans_of_nodes

    grouped = leaf_spans_of_nodes(db, member_ids, count_tokens=count_tokens)
    spans: list[SourceSpan] = []
    for node_id in member_ids:
        spans.extend(grouped.get(str(node_id), ()))
    return unique_coverage(spans)


def _profile_parts(db, member_ids: Sequence[str], count_tokens) -> list[tuple[str, tuple, int]]:
    from drbrain.tree.assign import source_profiles_for_nodes

    profiles = source_profiles_for_nodes(db, member_ids, count_tokens=count_tokens)
    parts: list[tuple[str, tuple, int]] = []
    for node_id in member_ids:
        profile = profiles.get(str(node_id))
        if profile is not None:
            parts.extend(profile.parts)
    return parts


def proposal_pre_screen(
    db,
    proposal,
    *,
    params: CostParams | None = None,
    seen_member_keys: frozenset[str] | set[str] | None = None,
    count_tokens=None,
) -> Decision:
    """Cheap pre-screen for one candidate proposal (no model call)."""
    params = params or CostParams()
    coverage = coverage_for_members(db, proposal.member_ids, count_tokens=count_tokens)
    member_tokens = sum(
        tokens for _local, _path, tokens in _profile_parts(db, proposal.member_ids, count_tokens)
    )
    return pre_screen(
        member_ids=proposal.member_ids,
        coverage=coverage,
        member_tokens=member_tokens,
        seen_member_keys=seen_member_keys or set(),
        member_key=proposal.key,
        params=params,
    )


def proposal_post_check(
    db,
    proposal,
    *,
    summary_text: str,
    summary_tokens: int,
    finish_reason: str,
    referenced_spans: Sequence[SourceSpan],
    params: CostParams | None = None,
    count_tokens=None,
) -> Decision:
    """Post-generation gate over the actual summary.

    ``referenced_spans`` must be the sources the generation contract actually
    put in front of the model (not the proposal's wish list): the gate rejects
    any accepted group whose summary was built from an incomplete member set.
    """
    coverage = coverage_for_members(db, proposal.member_ids, count_tokens=count_tokens)
    return post_check(
        summary_text=summary_text,
        summary_tokens=summary_tokens,
        finish_reason=finish_reason,
        coverage=coverage,
        referenced_spans=referenced_spans,
        params=params,
    )
