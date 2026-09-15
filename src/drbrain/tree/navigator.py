"""Stateful tree navigation: one walker, one ranked result, read receipts (T40).

Design: ``docs/unified-tree-rag-design.md`` §5 — within ONE request the
navigator runs a tool loop over the unified node surface
(``search_nodes`` / ``expand`` / ``read`` / ``parents`` / ``read_scope``).  The
next action is chosen by a *planner*: the production planner asks the bound
``chat_model`` (an agent/Runner), while tests inject a deterministic substitute
that scripts which action is picked.  There is no second query state machine
and no fixed SDK chat.

The executor keeps the read set (keyed by exact source span), the expanded
nodes, the node revisions, the explicit budgets and the unresolved branches.
Reading is what turns a candidate into evidence: the original-text candidates
returned to the caller always come from an actual :class:`ReadReceipt`, so a
model-generated node id, page number or summary reference is never evidence.
A region read is a routing hint; a leaf read is source text.  The same leaf
reached along several paths is reported once with all its origins recorded,
exhausted budgets return an auditable ``partial`` result, and a walk that ends
without any leaf evidence says so instead of reporting success.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from loguru import logger

from drbrain.tree.contracts import ReadReceipt
from drbrain.tree.search import TreeCandidate, TreeSearch
from drbrain.tree.tools import ToolBudget, ToolError, ToolState, TreeTools

DEFAULT_TOP_K = 20

#: Tools the planner may choose inside one request.
ACTIONS: tuple[str, ...] = ("read", "expand", "parents", "read_scope", "search_nodes", "finish")

#: How much of one read the model-driven planner sees back in its observation.
_PLANNER_TEXT_LIMIT = 2000


@dataclass(frozen=True)
class NavAction:
    """One tool decision (chosen by a planner, validated by the executor)."""

    action: str
    node_id: str = ""
    char_start: int | None = None
    char_end: int | None = None
    query: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"unknown navigation action {self.action!r}")
        if self.action in {"read", "expand", "parents", "read_scope"} and not self.node_id.strip():
            raise ValueError(f"action {self.action!r} requires a node_id")
        if self.action == "search_nodes" and not self.query.strip():
            raise ValueError("search_nodes requires a query")

    def to_json(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "node_id": self.node_id,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "query": self.query,
            "reason": self.reason,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> NavAction:
        """Tolerant constructor for model tool-call arguments."""
        if not isinstance(payload, Mapping):
            raise ValueError("navigation action must be a mapping")
        name = str(payload.get("action") or "").strip()
        return cls(
            action=name,
            node_id=str(payload.get("node_id") or payload.get("nodeId") or "").strip(),
            char_start=_optional_int(payload.get("char_start")),
            char_end=_optional_int(payload.get("char_end")),
            query=str(payload.get("query") or "").strip(),
            reason=str(payload.get("reason") or "").strip(),
        )


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class ActionPlanner(Protocol):
    """The injected decision-maker; ``observe`` is optional."""

    def choose(self, context: Mapping[str, Any]) -> NavAction | None:  # pragma: no cover
        ...


class ScriptedPlanner:
    """Deterministic substitute: a fixed list of actions, no model calls.

    Tests script exactly which action the "model" picks next and can inspect
    every context the executor built.
    """

    def __init__(
        self,
        actions: Sequence[NavAction | Mapping[str, Any] | None],
        *,
        on_exhausted: NavAction | None = None,
    ) -> None:
        self.actions = [
            action if isinstance(action, NavAction) else _payload_action(action)
            for action in actions
        ]
        self.on_exhausted = on_exhausted or NavAction("finish", reason="script-exhausted")
        self.contexts: list[dict[str, Any]] = []
        self.observations: list[tuple[NavAction, dict[str, Any]]] = []

    def choose(self, context: Mapping[str, Any]) -> NavAction | None:
        self.contexts.append(dict(context))
        if not self.actions:
            return self.on_exhausted
        action = self.actions.pop(0)
        return action

    def observe(self, action: NavAction, outcome: Mapping[str, Any]) -> None:
        self.observations.append((action, dict(outcome)))


def _payload_action(payload: NavAction | Mapping[str, Any] | None) -> NavAction | None:
    if payload is None:
        return None
    if isinstance(payload, NavAction):
        return payload
    return NavAction.from_payload(payload)


class HeuristicPlanner:
    """Deterministic default: read the frontier, expand summaries it hits.

    This is the no-model policy of the same loop: a region is read for its
    routing value and expanded so the walk continues down to real leaves; a
    failed or refused read is dropped instead of retried forever.
    """

    def __init__(self) -> None:
        self._resolved: set[str] = set()
        self._pending_expand: str = ""

    def choose(self, context: Mapping[str, Any]) -> NavAction | None:
        if self._pending_expand:
            node_id, self._pending_expand = self._pending_expand, ""
            return NavAction("expand", node_id=node_id)
        read = set(context.get("read_nodes") or ())
        for item in context.get("frontier") or ():
            node_id = str(item.get("node_id") or "")
            if not node_id or node_id in read or node_id in self._resolved:
                continue
            return NavAction("read", node_id=node_id)
        return NavAction("finish", reason="frontier-exhausted")

    def observe(self, action: NavAction, outcome: Mapping[str, Any]) -> None:
        if not outcome.get("ok"):
            if action.action == "read":
                self._resolved.add(action.node_id)
            return
        if action.action == "read" and str(outcome.get("kind")) == "region":
            if not outcome.get("expand_refused"):
                self._pending_expand = action.node_id


#: Tool schemas handed to the chat model (Chat Completions ``tools`` shape).
NAVIGATION_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_nodes",
            "description": (
                "Search all published layers for another entry point. Use it when the "
                "question has a part the current entry set does not cover."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand",
            "description": "List the direct children of a published region node.",
            "parameters": {
                "type": "object",
                "properties": {"node_id": {"type": "string"}},
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": (
                "Read the exact text of one published node. Leaf bodies are source "
                "evidence; region summaries are routing hints only. Optional "
                "char_start/char_end bound the read inside the node."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "char_start": {"type": "integer"},
                    "char_end": {"type": "integer"},
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parents",
            "description": (
                "Reverse membership lookup: every published parent of a node. Use it "
                "to move from one text to the other members of the same topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {"node_id": {"type": "string"}},
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_scope",
            "description": (
                "Read every unique canonical span under a node (regions included) "
                "when the whole neighbourhood is needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {"node_id": {"type": "string"}},
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Stop the walk and return the evidence gathered so far.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
            },
        },
    },
]

DEFAULT_NAVIGATOR_PROMPT = (
    "You navigate a hierarchical document tree for one question. Choose ONE tool "
    "per round from the provided tools. Entry candidates are ranked node ids; "
    "region nodes carry summaries that route you to the leaves underneath — read "
    "a region, expand it, then read the leaf text you need. Use parents to move "
    "from one text to related members, read_scope for a whole neighbourhood, and "
    "search_nodes when the question has a part the current candidates miss. "
    "Source evidence must come from leaf reads; call finish once you have enough "
    "text (or when nothing else is reachable). Never invent node ids."
)


class ChatActionPlanner:
    """Production planner: the bound chat model chooses the next tool action.

    One request keeps one conversation: every executor observation is appended
    as a tool result, so the model sees exactly what was actually read.  The
    model never supplies evidence — its actions are validated and executed, and
    only executed reads produce receipts.
    """

    def __init__(
        self,
        model: Any,
        *,
        system_prompt: str = DEFAULT_NAVIGATOR_PROMPT,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.max_tokens = max(64, int(max_tokens))
        self.temperature = temperature
        self._messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        self._pending_tool_ids: list[str] = []

    def choose(self, context: Mapping[str, Any]) -> NavAction | None:
        messages = [
            *self._messages,
            {
                "role": "user",
                "content": "Current navigation state:\n"
                + json.dumps(context, ensure_ascii=False, sort_keys=True, default=str),
            },
        ]
        result = self.model.chat(
            messages,
            tools=NAVIGATION_TOOLS,
            tool_choice="auto",
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        self._messages.append(result.as_assistant_message())
        if not result.tool_calls:
            # Text-only answer: the model declares itself done.
            self._pending_tool_ids = []
            return NavAction("finish", reason="model-finished")
        call = result.tool_calls[0]
        self._pending_tool_ids = [call.id or ""]
        payload = dict(call.arguments_dict())
        payload["action"] = str(call.name or "").strip()
        try:
            return NavAction.from_payload(payload)
        except ValueError as exc:
            raise _PlannerActionError(str(exc), payload) from exc

    def observe(self, action: NavAction, outcome: Mapping[str, Any]) -> None:
        ids = self._pending_tool_ids or [""]
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": ids[0],
                "content": json.dumps(_planner_observation(action, outcome), ensure_ascii=False),
            }
        )
        for extra in ids[1:]:
            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": extra,
                    "content": json.dumps({"ok": False, "error": "one action per round"}),
                }
            )
        self._pending_tool_ids = []


class _PlannerActionError(RuntimeError):
    """The planner produced an action the protocol refuses."""

    def __init__(self, message: str, payload: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.payload = dict(payload)


def _planner_observation(action: NavAction, outcome: Mapping[str, Any]) -> dict[str, Any]:
    """Compact, bounded tool result for the model conversation."""
    keep = {
        key: outcome.get(key)
        for key in ("ok", "kind", "node_id", "layer", "tokens", "spans", "queued", "error")
        if outcome.get(key) is not None
    }
    if outcome.get("kind") == "region" and outcome.get("text"):
        keep["summary"] = str(outcome["text"])[:600]
    elif outcome.get("text"):
        keep["text"] = str(outcome["text"])[:_PLANNER_TEXT_LIMIT]
    for key in ("children", "parents", "hits"):
        if outcome.get(key):
            keep[key] = list(outcome[key])[:12]
    return keep


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
    unresolved: list[dict[str, Any]] = field(default_factory=list)
    planner: str = "heuristic"

    def evidence_counts(self) -> dict[str, int]:
        leaf = sum(1 for item in self.evidence if item.get("source") == "leaf")
        return {"leaf": leaf, "summary": len(self.evidence) - leaf}

    def to_json(self) -> dict[str, Any]:
        counts = self.evidence_counts()
        return {
            "query": self.query,
            "status": self.status,
            "reason": self.reason,
            "planner": self.planner,
            "candidates": [candidate.to_json() for candidate in self.candidates],
            "evidence": list(self.evidence),
            "trace": [step.to_json() for step in self.trace],
            "budget": dict(self.budget),
            "unresolved": [dict(item) for item in self.unresolved],
            "leaf_evidence": counts["leaf"],
            "summary_evidence": counts["summary"],
            "read_spans": len({receipt.span_key for receipt in self.receipts}),
        }


@dataclass
class _FrontierItem:
    node_id: str
    kind: str = ""
    title: str = ""
    layer: int = 0
    revision: int = 0
    via: str = ""


class TreeNavigator:
    """One walker over an already-searched candidate set."""

    def __init__(
        self,
        db,
        *,
        tools: TreeTools | None = None,
        budget: ToolBudget | None = None,
        count_tokens: Callable[[str], int] | None = None,
        search_nodes: Callable[[str], Sequence[TreeCandidate]] | None = None,
    ) -> None:
        self.db = db
        self.tools = tools or TreeTools(db, budget=budget, count_tokens=count_tokens)
        self.budget = budget or self.tools.budget
        self.count_tokens = count_tokens or self.tools.count_tokens
        self.search_nodes = search_nodes

    def navigate(
        self,
        query: str,
        candidates: Sequence[TreeCandidate],
        *,
        expand_regions: bool = True,
        max_expansions: int = 6,
        planner: Any = None,
    ) -> NavigationResult:
        """Run one request-scoped tool loop and return receipt-backed evidence."""
        return _Walk(
            self,
            query,
            list(candidates),
            expand_regions=expand_regions,
            max_expansions=max_expansions,
            planner=planner,
        ).run()


class _Walk:
    """The per-request executor: state, budgets and the tool loop."""

    def __init__(
        self,
        navigator: TreeNavigator,
        query: str,
        candidates: list[TreeCandidate],
        *,
        expand_regions: bool,
        max_expansions: int,
        planner: Any,
    ) -> None:
        self.tools = navigator.tools
        self.search_nodes = navigator.search_nodes
        self.expand_regions = bool(expand_regions)
        self.max_expansions = max(0, int(max_expansions))
        self.query = query
        self.planner = planner or HeuristicPlanner()
        self.result = NavigationResult(query=query, candidates=list(candidates))
        self.result.planner = type(self.planner).__name__
        self.state = ToolState()
        self.frontier: list[_FrontierItem] = [
            _FrontierItem(
                node_id=candidate.node_id,
                kind=candidate.kind,
                layer=candidate.layer,
                revision=candidate.node_revision,
            )
            for candidate in candidates
        ]
        self.read_nodes: set[str] = set()
        self.expanded: set[str] = set()
        self.revisions: dict[str, int] = {}
        self.expansions = 0
        self.steps = 0
        self.stop_reason = ""
        self._evidence_by_span: dict[tuple[str, str, int, int], dict[str, Any]] = {}
        self._unresolved: dict[str, dict[str, Any]] = {}
        #: Every path that reached a node (ordered, deduplicated).  The frontier
        #: keeps one entry per node, so a leaf shared by two parents accumulates
        #: both origins here instead of losing the second one.
        self._origins: dict[str, list[str]] = {}

    # ── loop ────────────────────────────────────────────────────────────
    def run(self) -> NavigationResult:
        if not self.result.candidates:
            self.result.status = "empty"
            self.result.reason = "no_candidates"
            self.result.budget = self.state.to_json()
            return self.result
        max_steps = max(1, int(self.tools.budget.max_calls))
        while True:
            if self.state.truncated:
                self.stop_reason = "budget_exhausted"
                break
            if self.steps >= max_steps:
                self.state.truncated = True
                self.stop_reason = "budget_exhausted"
                break
            self.steps += 1
            context = self._context()
            try:
                action = self.planner.choose(context)
            except _PlannerActionError as exc:
                self.result.trace.append(NavigationStep("invalid_action", "", {"error": str(exc)}))
                continue
            except Exception as exc:  # noqa: BLE001 - a planner outage is a reported state
                logger.warning("[tree] navigation planner failed: {}", exc)
                self.result.trace.append(
                    NavigationStep("planner_failed", "", {"error": _safe_error(exc)})
                )
                self.stop_reason = "planner_failed"
                break
            if action is None or action.action == "finish":
                self.stop_reason = (action.reason if action is not None else "") or "model-finished"
                if self.stop_reason == "model-finished":
                    self.stop_reason = ""
                break
            outcome = self._execute(action)
            try:
                observe = getattr(self.planner, "observe", None)
                if callable(observe):
                    observe(action, outcome)
            except Exception as exc:  # noqa: BLE001 - bookkeeping errors must not lose the walk
                logger.debug("[tree] planner observe failed: {}", exc)
        self._finalize()
        return self.result

    def _finalize(self) -> None:
        result = self.result
        for item in self.frontier:
            if item.node_id not in self.read_nodes:
                self._unresolved.setdefault(
                    item.node_id,
                    {
                        "node_id": item.node_id,
                        "kind": item.kind,
                        "reason": "not_read",
                        "via": item.via,
                    },
                )
        result.unresolved = list(self._unresolved.values())
        result.budget = self.state.to_json()
        counts = result.evidence_counts()
        if self.stop_reason in {"planner_failed", "budget_exhausted"} or self.state.truncated:
            result.status = "partial" if result.evidence else "empty"
            result.reason = self.stop_reason or "budget_exhausted"
            if not result.evidence:
                result.status = "empty"
        elif counts["leaf"]:
            result.status = "ok"
            result.reason = ""
        elif result.evidence:
            # Summary hits are routing hints; without leaf reads the walk has
            # no source evidence and must say so instead of reporting success.
            result.status = "partial"
            result.reason = "no_leaf_evidence"
        else:
            result.status = "empty"
            result.reason = self.stop_reason or "nothing_readable"

    # ── actions ─────────────────────────────────────────────────────────
    def _execute(self, action: NavAction) -> dict[str, Any]:
        try:
            if action.action == "read":
                return self._read(action)
            if action.action == "expand":
                return self._expand(action)
            if action.action == "parents":
                return self._parents(action)
            if action.action == "read_scope":
                return self._read_scope(action)
            if action.action == "search_nodes":
                return self._search(action)
        except ToolError as exc:
            self._record_failure(action, exc)
            return {
                "ok": False,
                "action": action.action,
                "node_id": action.node_id,
                "error": str(exc),
            }
        except Exception as exc:  # noqa: BLE001 - a tool bug is auditable, not a crash
            logger.warning("[tree] navigation tool {} failed: {}", action.action, exc)
            self._record_failure(action, exc)
            return {
                "ok": False,
                "action": action.action,
                "node_id": action.node_id,
                "error": _safe_error(exc),
            }
        self._record_failure(action, ToolError(f"unsupported action {action.action!r}"))
        return {
            "ok": False,
            "action": action.action,
            "error": f"unsupported action {action.action!r}",
        }

    def _read(self, action: NavAction) -> dict[str, Any]:
        text, receipt = self.tools.read(
            action.node_id,
            self.state,
            char_start=action.char_start,
            char_end=action.char_end,
            request_id=f"nav-{self.state.calls + 1}",
        )
        row = self.tools.db.get_tree_node(action.node_id) or {}
        kind = str(row.get("kind") or "leaf")
        layer = int(row.get("layer") or 0)
        via = self._vias_of(action.node_id)
        self.read_nodes.add(action.node_id)
        self.revisions[action.node_id] = receipt.node_revision
        self._unresolved.pop(action.node_id, None)
        source = "summary" if kind == "region" else "leaf"
        self.result.trace.append(
            NavigationStep(
                "read", action.node_id, {"kind": kind, "source": source, "tokens": receipt.tokens}
            )
        )
        self._record_evidence(
            node_id=action.node_id,
            kind=kind,
            layer=layer,
            source=source,
            text=text,
            receipt=receipt,
            via=via,
        )
        return {
            "ok": True,
            "action": "read",
            "kind": kind,
            "node_id": action.node_id,
            "layer": layer,
            "tokens": receipt.tokens,
            "text": text,
            "expand_refused": False,
        }

    def _expand(self, action: NavAction) -> dict[str, Any]:
        if not self.expand_regions:
            self._add_unresolved(action.node_id, "expansion_disabled")
            self.result.trace.append(
                NavigationStep("expand_refused", action.node_id, {"reason": "expansion_disabled"})
            )
            return {
                "ok": False,
                "refused": True,
                "expand_refused": True,
                "action": "expand",
                "node_id": action.node_id,
                "error": "summary expansion is disabled for this request",
            }
        if self.expansions >= self.max_expansions:
            self._add_unresolved(action.node_id, "expansion_limit")
            self.result.trace.append(
                NavigationStep("expand_refused", action.node_id, {"reason": "expansion_limit"})
            )
            return {
                "ok": False,
                "refused": True,
                "expand_refused": True,
                "action": "expand",
                "node_id": action.node_id,
                "error": f"expansion budget exhausted ({self.max_expansions})",
            }
        children = self.tools.expand(action.node_id, self.state)
        self.expansions += 1
        self.expanded.add(action.node_id)
        queued = 0
        known = {item.node_id for item in self.frontier} | self.read_nodes
        for child in children:
            child_id = str(child["node_id"])
            # A child already queued or read through another parent still gains
            # this origin: the frontier keeps one entry per node, the evidence
            # keeps every path that reached it.
            self._note_origin(child_id, action.node_id)
            if child_id in known:
                continue
            known.add(child_id)
            self.frontier.append(
                _FrontierItem(
                    node_id=child_id,
                    kind=str(child.get("kind") or ""),
                    title=str(child.get("title") or ""),
                    layer=int(child.get("layer") or 0),
                    revision=int(child.get("revision") or 0),
                    via=action.node_id,
                )
            )
            queued += 1
        if action.node_id in self.read_nodes:
            self._unresolved.pop(action.node_id, None)
        for child in children:
            child_id = str(child["node_id"])
            if child_id not in self.read_nodes:
                self._unresolved.setdefault(
                    child_id,
                    {
                        "node_id": child_id,
                        "kind": str(child.get("kind") or ""),
                        "reason": "queued",
                        "via": action.node_id,
                    },
                )
        self.result.trace.append(
            NavigationStep("expand", action.node_id, {"children": len(children), "queued": queued})
        )
        return {
            "ok": True,
            "action": "expand",
            "node_id": action.node_id,
            "children": [
                {"node_id": str(child["node_id"]), "kind": str(child.get("kind") or "")}
                for child in children
            ],
            "queued": queued,
        }

    def _parents(self, action: NavAction) -> dict[str, Any]:
        parents = self.tools.parents(action.node_id, self.state)
        self.result.trace.append(
            NavigationStep("parents", action.node_id, {"parents": [p["node_id"] for p in parents]})
        )
        return {
            "ok": True,
            "action": "parents",
            "node_id": action.node_id,
            "parents": [
                {
                    "node_id": str(parent["node_id"]),
                    "layer": int(parent["layer"]),
                    "title": str(parent.get("title") or ""),
                }
                for parent in parents
            ],
        }

    def _read_scope(self, action: NavAction) -> dict[str, Any]:
        spans = self.tools.read_scope(
            action.node_id, self.state, request_id=f"nav-{self.state.calls + 1}"
        )
        via = action.node_id
        self.read_nodes.add(action.node_id)
        if spans:
            self.revisions[action.node_id] = spans[0][1].node_revision
        self._unresolved.pop(action.node_id, None)
        texts: list[str] = []
        for text, receipt in spans:
            leaf = self._leaf_for_span(action.node_id, receipt)
            self._record_evidence(
                node_id=str(leaf["node_id"]) if leaf else receipt.node_id,
                revision=int(leaf["revision"]) if leaf else None,
                kind="leaf",
                layer=int(leaf["layer"]) if leaf else 0,
                source="leaf",
                text=text,
                receipt=receipt,
                via=via,
            )
            texts.append(text)
        self.result.trace.append(
            NavigationStep("read_scope", action.node_id, {"spans": len(spans)})
        )
        return {
            "ok": True,
            "action": "read_scope",
            "node_id": action.node_id,
            "spans": len(spans),
            "text": "\n".join(texts),
        }

    def _search(self, action: NavAction) -> dict[str, Any]:
        if self.search_nodes is None:
            self._add_unresolved(action.query or action.node_id, "search_unavailable")
            return {
                "ok": False,
                "refused": True,
                "action": "search_nodes",
                "query": action.query,
                "error": "search_nodes is unavailable for this request",
            }
        found = list(self.search_nodes(action.query) or [])
        known = {item.node_id for item in self.frontier} | self.read_nodes
        queued = 0
        for candidate in found:
            self._note_origin(candidate.node_id, f"search:{action.query}")
            if candidate.node_id in known:
                continue
            known.add(candidate.node_id)
            self.frontier.append(
                _FrontierItem(
                    node_id=candidate.node_id,
                    kind=candidate.kind,
                    layer=candidate.layer,
                    revision=candidate.node_revision,
                    via=f"search:{action.query}",
                )
            )
            queued += 1
        self.result.trace.append(
            NavigationStep(
                "search_nodes", "", {"query": action.query, "hits": len(found), "queued": queued}
            )
        )
        return {
            "ok": True,
            "action": "search_nodes",
            "query": action.query,
            "hits": [
                {"node_id": candidate.node_id, "kind": candidate.kind, "layer": candidate.layer}
                for candidate in found[:12]
            ],
            "queued": queued,
        }

    # ── shared bookkeeping ──────────────────────────────────────────────
    def _context(self) -> dict[str, Any]:
        last = self.result.trace[-1].to_json() if self.result.trace else None
        return {
            "query": self.query,
            "step": self.steps,
            "candidates": [candidate.to_json() for candidate in self.result.candidates[:20]],
            "frontier": [
                {
                    "node_id": item.node_id,
                    "kind": item.kind,
                    "title": item.title[:120],
                    "layer": item.layer,
                    "via": item.via,
                }
                for item in self.frontier[:50]
            ],
            "read_nodes": sorted(self.read_nodes),
            "expanded": sorted(self.expanded),
            "revisions": dict(sorted(self.revisions.items())),
            "unresolved": list(self._unresolved.values())[:50],
            "last": last,
            "budget": self.state.to_json(),
            "limits": {
                "expand_regions": self.expand_regions,
                "max_expansions": self.max_expansions,
                "expansions": self.expansions,
                "max_steps": max(1, int(self.tools.budget.max_calls)),
            },
        }

    def _via_of(self, node_id: str) -> str:
        for item in self.frontier:
            if item.node_id == node_id:
                return item.via
        return ""

    def _note_origin(self, node_id: str, via: str) -> None:
        """Remember one more path that reached a node (soft multi-parent leaves)."""
        if not node_id or not via:
            return
        origins = self._origins.setdefault(node_id, [])
        if via not in origins:
            origins.append(via)

    def _vias_of(self, node_id: str) -> list[str]:
        """Every distinct origin that reached this node, in recording order.

        A child already queued or read through another parent is not queued a
        second time, so the frontier alone would drop its second origin; the
        per-node origin list keeps all of them for the evidence record.
        """
        origins = list(self._origins.get(node_id) or [])
        for item in self.frontier:
            if item.node_id == node_id and item.via and item.via not in origins:
                origins.append(item.via)
        return origins

    def _add_unresolved(self, node_id: str, reason: str, *, drop_if_read: bool = False) -> None:
        if not node_id:
            return
        if drop_if_read and node_id in self.read_nodes:
            self._unresolved.pop(node_id, None)
            return
        existing = self._unresolved.get(node_id)
        if existing is None or str(existing.get("reason")) in {"queued", "not_read"}:
            self._unresolved[node_id] = {
                "node_id": node_id,
                "kind": str((existing or {}).get("kind") or ""),
                "reason": reason,
                "via": str((existing or {}).get("via") or ""),
            }

    def _leaf_for_span(self, scope_node_id: str, receipt: ReadReceipt) -> dict | None:
        """The exact leaf that owns a ``read_scope`` span (if reachable)."""
        stack = [scope_node_id]
        seen: set[str] = set()
        while stack:
            node_id = stack.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            row = self.tools.db.get_tree_node(node_id)
            if row is None or str(row.get("state")) != "ready":
                continue
            if str(row.get("kind")) == "leaf":
                if (
                    str(row.get("local_id")) == receipt.local_id
                    and str(row.get("block_id")) == receipt.block_id
                    and int(row.get("char_start") or 0) <= receipt.char_start
                    and int(row.get("char_end") or 0) >= receipt.char_end
                ):
                    return row
                continue
            stack.extend(child["child_id"] for child in self.tools.db.get_tree_children(node_id))
        return None

    def _record_failure(self, action: NavAction, exc: Exception) -> None:
        self.result.trace.append(
            NavigationStep(
                f"{action.action}_failed",
                action.node_id,
                {"error": _safe_error(exc)},
            )
        )
        if action.node_id:
            # A failed read must not silently drop the branch: it stays visible
            # as unresolved unless a later successful read clears it.
            self._unresolved.setdefault(
                action.node_id,
                {
                    "node_id": action.node_id,
                    "kind": "",
                    "reason": "action_failed",
                    "via": self._via_of(action.node_id),
                },
            )

    def _record_evidence(
        self,
        *,
        node_id: str,
        kind: str,
        layer: int,
        source: str,
        text: str,
        receipt: ReadReceipt,
        via: str | Sequence[str],
        revision: int | None = None,
    ) -> dict[str, Any]:
        """Record one span once, with every origin that reached it.

        ``via`` may be a single origin (region/scope readers) or the full list
        of paths a soft multi-parent leaf was reached through; re-reading the
        same span appends the new origins to the existing row instead of
        creating a second one.
        """
        origins = [via] if isinstance(via, str) else [str(item) for item in via]
        key = receipt.span_key
        existing = self._evidence_by_span.get(key)
        if existing is not None:
            recorded = existing.setdefault("via", [])
            for origin in origins:
                if origin and origin not in recorded:
                    recorded.append(origin)
            return existing
        item = {
            "node_id": node_id,
            "node_revision": int(revision if revision is not None else receipt.node_revision),
            "kind": kind,
            "layer": layer,
            "local_id": receipt.local_id,
            "score": self._score_of(node_id),
            "source": source,
            "text": text,
            "via": [origin for origin in origins if origin],
            "receipt": {
                "block_id": receipt.block_id,
                "char_start": receipt.char_start,
                "char_end": receipt.char_end,
                "content_hash": receipt.content_hash,
                "tokens": receipt.tokens,
            },
        }
        self._evidence_by_span[key] = item
        self.result.evidence.append(item)
        self.result.receipts.append(receipt)
        return item

    def _score_of(self, node_id: str) -> float:
        for candidate in self.result.candidates:
            if candidate.node_id == node_id:
                return float(candidate.score)
        return 0.0


def _safe_error(exc: Exception) -> str:
    try:
        from drbrain.security import safe_error

        return safe_error(exc)
    except Exception:  # noqa: BLE001 - never fail while reporting a failure
        return f"{type(exc).__name__}: {exc}"


def navigate_query(
    db,
    search: TreeSearch,
    embed_query: Callable[[Sequence[str]], list[list[float]]],
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    budget: ToolBudget | None = None,
    planner: Any = None,
) -> NavigationResult:
    """End-to-end walking helper used by the CLI and the acceptance tests.

    ``search_nodes`` inside the request re-enters through the same searcher, so
    a model-driven walk can look for another part of the question without a
    second retrieval stack.
    """

    def _search(text: str) -> list[TreeCandidate]:
        return search.search_from_text(embed_query, text, top_k=top_k)

    vectors = embed_query([query])
    if not vectors or not vectors[0]:
        return NavigationResult(query=query, status="unavailable", reason="query_embedding_failed")
    candidates = search.search(vectors[0], top_k=top_k)
    navigator = TreeNavigator(db, budget=budget, search_nodes=_search)
    return navigator.navigate(query, candidates, planner=planner)
