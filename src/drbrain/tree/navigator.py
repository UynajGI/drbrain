"""Stateful tree navigation: one walker, one ranked result, read receipts (T40).

The navigator never re-runs retrieval per step and never counts the same
canonical evidence twice: it keeps a frontier with the unread candidates, a
read set keyed by exact source spans, and explicit budgets for tool calls,
read nodes and tokens.  Reading is what turns a candidate into evidence —
summaries may be routed through, but content that was never read is reported
as such, and an exhausted budget produces an auditable ``partial`` result
instead of a quiet truncation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from drbrain.tree.contracts import ReadReceipt
from drbrain.tree.search import TreeCandidate, TreeSearch
from drbrain.tree.tools import ToolBudget, ToolError, ToolState, TreeTools

DEFAULT_TOP_K = 20


@dataclass
class NavigationStep:
    action: str
    node_id: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"action": self.action, "node_id": self.node_id, "detail": dict(self.detail)}


@dataclass
class NavigationResult:
    query: str
    candidates: list[TreeCandidate] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    receipts: list[ReadReceipt] = field(default_factory=list)
    trace: list[NavigationStep] = field(default_factory=list)
    status: str = "empty"  # ok | partial | empty | unavailable
    reason: str = ""
    budget: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "status": self.status,
            "reason": self.reason,
            "candidates": [candidate.to_json() for candidate in self.candidates],
            "evidence": list(self.evidence),
            "trace": [step.to_json() for step in self.trace],
            "budget": dict(self.budget),
            "read_spans": len({receipt.span_key for receipt in self.receipts}),
        }


class TreeNavigator:
    """One walker over an already-searched candidate set."""

    def __init__(
        self,
        db,
        *,
        tools: TreeTools | None = None,
        budget: ToolBudget | None = None,
        count_tokens: Callable[[str], int] | None = None,
    ) -> None:
        self.db = db
        self.tools = tools or TreeTools(db, budget=budget, count_tokens=count_tokens)
        self.budget = budget or self.tools.budget
        self.count_tokens = count_tokens or self.tools.count_tokens

    def navigate(
        self,
        query: str,
        candidates: Sequence[TreeCandidate],
        *,
        expand_regions: bool = True,
        max_expansions: int = 6,
    ) -> NavigationResult:
        result = NavigationResult(query=query, candidates=list(candidates))
        state = ToolState()
        read_spans: set[tuple[str, str, int, int]] = set()
        if not candidates:
            result.status = "empty"
            result.reason = "no_candidates"
            result.budget = state.to_json()
            return result

        expansions = 0
        for candidate in candidates:
            if state.truncated:
                result.status = "partial"
                result.reason = "budget_exhausted"
                break
            try:
                text, receipt = self.tools.read(
                    candidate.node_id, state, request_id=f"nav-{state.calls + 1}"
                )
            except ToolError as exc:
                result.trace.append(
                    NavigationStep("read_failed", candidate.node_id, {"error": str(exc)})
                )
                continue
            result.trace.append(
                NavigationStep(
                    "read",
                    candidate.node_id,
                    {"kind": candidate.kind, "score": round(candidate.score, 6)},
                )
            )
            if receipt.span_key not in read_spans:
                read_spans.add(receipt.span_key)
                result.receipts.append(receipt)
            evidence = {
                "node_id": candidate.node_id,
                "kind": candidate.kind,
                "score": candidate.score,
                "local_id": receipt.local_id,
                "text": text,
                "receipt": {
                    "block_id": receipt.block_id,
                    "char_start": receipt.char_start,
                    "char_end": receipt.char_end,
                    "content_hash": receipt.content_hash,
                    "tokens": receipt.tokens,
                },
            }
            if candidate.kind == "region":
                # A summary alone is a routing hint, not source evidence.
                evidence["source"] = "summary"
                if expand_regions and expansions < max_expansions:
                    expansions += 1
                    self._expand_for_reads(result, candidate.node_id, state, read_spans)
            else:
                evidence["source"] = "leaf"
            result.evidence.append(evidence)

        result.status = "partial" if state.truncated else "ok"
        if state.truncated and not result.reason:
            result.reason = "budget_exhausted"
        if not result.evidence:
            result.status = "empty"
            result.reason = result.reason or "nothing_readable"
        result.budget = state.to_json()
        return result

    def _expand_for_reads(
        self,
        result: NavigationResult,
        node_id: str,
        state: ToolState,
        read_spans: set[tuple[str, str, int, int]],
    ) -> None:
        """Read the children of a region so summary hits are backed by text."""
        try:
            children = self.tools.expand(node_id, state)
        except ToolError as exc:
            result.trace.append(NavigationStep("expand_failed", node_id, {"error": str(exc)}))
            return
        result.trace.append(NavigationStep("expand", node_id, {"children": len(children)}))
        for child in children:
            if state.truncated:
                return
            if child["kind"] != "leaf":
                continue
            try:
                text, receipt = self.tools.read(
                    child["node_id"], state, request_id=f"nav-{state.calls + 1}"
                )
            except ToolError:
                continue
            result.trace.append(
                NavigationStep("read", child["node_id"], {"parent": node_id, "kind": "leaf"})
            )
            if receipt.span_key in read_spans:
                continue
            read_spans.add(receipt.span_key)
            result.receipts.append(receipt)
            result.evidence.append(
                {
                    "node_id": child["node_id"],
                    "kind": "leaf",
                    "score": 0.0,
                    "local_id": receipt.local_id,
                    "text": text,
                    "source": "leaf",
                    "via": node_id,
                    "receipt": {
                        "block_id": receipt.block_id,
                        "char_start": receipt.char_start,
                        "char_end": receipt.char_end,
                        "content_hash": receipt.content_hash,
                        "tokens": receipt.tokens,
                    },
                }
            )


def navigate_query(
    db,
    search: TreeSearch,
    embed_query: Callable[[Sequence[str]], list[list[float]]],
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    budget: ToolBudget | None = None,
) -> NavigationResult:
    """End-to-end walking helper used by the CLI and the acceptance tests."""
    vectors = embed_query([query])
    if not vectors or not vectors[0]:
        return NavigationResult(query=query, status="unavailable", reason="query_embedding_failed")
    candidates = search.search(vectors[0], top_k=top_k)
    navigator = TreeNavigator(db, budget=budget)
    return navigator.navigate(query, candidates)
