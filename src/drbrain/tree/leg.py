"""The production tree retrieval leg over the published unified generation (T43).

Design: ``docs/unified-tree-rag-design.md`` §5/§6 — the tree leg resolves the
*active* unified generation, searches every published layer from the shared
ANN, walks the tree with the stateful navigator, validates the read receipts
and returns **leaf text**.  It never falls back to the retired per-paper
PageIndex ANN (``tree_vectors.tree_layer='pageindex'``) and never accepts an
implicit second entry pool: a missing generation is fail-closed.

Revision agreement: the returned leaf hits carry the canonical node id and
content hash that the walk actually read.  Callers that fuse the tree leg with
the BM25/vector legs (which read the SQL node projection) pass a ``verify``
callable — the production call site checks the same node id maps to the same
content revision there, so the three legs cannot silently mix revisions.

Model role: the walk is driven by the ``chat_model`` role when one resolves;
without it the deterministic in-process policy runs and the outcome records
that choice (``planner``), so a tree-only query stays read-only and offline
instead of calling an unrelated model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from drbrain.rag.config import get_llamaindex_config
from drbrain.runtime import runtime_scoped_path
from drbrain.tree.embedding_identity import profile_from_config
from drbrain.tree.navigator import (
    ChatActionPlanner,
    HeuristicPlanner,
    NavigationResult,
    TreeNavigator,
)
from drbrain.tree.publish import get_active_tree_generation, resolve_tree_generation
from drbrain.tree.reading import ReadOnlyTreeStore
from drbrain.tree.search import TreeCandidate, TreeSearch
from drbrain.tree.tools import ToolBudget
from drbrain.tree.vector_store import UnifiedVectorStore

DEFAULT_TREE_STORAGE = "data/tree"
DEFAULT_TOP_K = 20


class TreeLegUnavailableError(RuntimeError):
    """The unified tree leg cannot serve this query (fail-closed)."""


@dataclass(frozen=True)
class TreeLegHit:
    """One receipt-backed leaf read from the published generation."""

    node_id: str
    local_id: str
    node_revision: int
    content_hash: str
    block_id: str
    char_start: int
    char_end: int
    tokens: int
    score: float
    text: str
    via: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        """SQL projection key (``<paper_id>:<node_id>``) used by the fusion rows."""
        return f"{self.local_id}:{self.node_id}"

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "node_id": self.node_id,
            "local_id": self.local_id,
            "node_revision": self.node_revision,
            "content_hash": self.content_hash,
            "block_id": self.block_id,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "tokens": self.tokens,
            "score": round(float(self.score), 6),
            "via": list(self.via),
        }


@dataclass
class TreeLegOutcome:
    """Auditable result of one tree-leg query."""

    status: str  # ok | empty | unavailable
    reason: str = ""
    generation: str = ""
    profile_id: str = ""
    planner: str = ""
    navigation_status: str = ""
    navigation_reason: str = ""
    hits: list[TreeLegHit] = field(default_factory=list)
    summary_hits: int = 0
    read_spans: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[dict[str, Any]] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok" and bool(self.hits)

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "generation": self.generation,
            "profile_id": self.profile_id,
            "planner": self.planner,
            "navigation_status": self.navigation_status,
            "navigation_reason": self.navigation_reason,
            "hits": len(self.hits),
            "summary_hits": self.summary_hits,
            "read_spans": self.read_spans,
            "entries": [
                {
                    "action": step.get("action"),
                    "node_id": step.get("node_id"),
                    "detail": {
                        key: value
                        for key, value in dict(step.get("detail") or {}).items()
                        if key in {"kind", "source", "children", "queued", "reason", "spans"}
                    },
                }
                for step in self.trace
            ],
            "unresolved": [dict(item) for item in self.unresolved],
            "budget": dict(self.budget),
        }


def active_tree_generation(cfg: Any, storage_dir: str | Path | None = None) -> str | None:
    """The published unified generation for this config, or ``None``."""
    return get_active_tree_generation(_tree_storage_root(cfg, storage_dir))


def resolve_navigation_planner(cfg: Any) -> tuple[Any, str]:
    """The chat-model planner when the chat role resolves, else deterministic.

    A missing/unusable chat endpoint is not a silent model fallback: no model
    is called at all, and the label records why the deterministic walk ran.
    """
    try:
        from drbrain.services.chat_model import ChatModel

        model = ChatModel(cfg=cfg)
    except Exception as exc:  # noqa: BLE001 - any resolution failure picks the deterministic walk
        reason = f"{type(exc).__name__}: {exc}"
        logger.info("[tree] chat navigation model unavailable; deterministic walk ({})", reason)
        return HeuristicPlanner(), f"heuristic:{type(exc).__name__}"
    return ChatActionPlanner(model), "chat_model"


def run_tree_leg(
    cfg: Any,
    *,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    storage_dir: str | Path | None = None,
    planner: Any = None,
    planner_label: str = "",
    view: str = "all",
    budget: ToolBudget | None = None,
    expand_regions: bool = True,
    max_expansions: int = 6,
    verify: Callable[[TreeLegHit], bool] | None = None,
    embed: Callable[[Sequence[str]], list[list[float]]] | None = None,
    local_ids: Sequence[str] | None = None,
) -> TreeLegOutcome:
    """Resolve → search → navigate → validate → leaf text (one call, no writes).

    ``local_ids`` scopes the ANN entry search (and every in-walk re-search) to
    the given papers, so a paper-scoped query cannot silently read another
    paper's leaves; ``None`` keeps the corpus-wide search.
    """
    root = _tree_storage_root(cfg, storage_dir)
    generation = get_active_tree_generation(root)
    if not generation:
        raise TreeLegUnavailableError(f"no active unified tree generation under {root}")
    resolved = resolve_tree_generation(root, generation)
    profile = profile_from_config(getattr(cfg, "embed", None))
    if profile.dimension is None:
        raise TreeLegUnavailableError("embedding profile has no dimension; cannot read the ANN")
    embed_query = embed or (lambda texts: _embed_batch(list(texts), getattr(cfg, "embed", None)))
    vectors = embed_query([query])
    if not vectors or not vectors[0]:
        raise TreeLegUnavailableError("query embedding failed")
    with UnifiedVectorStore(Path(resolved["vectors"]), dimension=int(profile.dimension)) as store:
        searcher = TreeSearch(store, profile_id=profile.profile_id(), top_k=top_k)
        candidates = searcher.search(vectors[0], top_k=top_k, view=view, local_ids=local_ids)
        outcome = TreeLegOutcome(
            status="empty",
            reason="no_candidates",
            generation=generation,
            profile_id=profile.profile_id(),
            navigation_status="empty",
            navigation_reason="no_candidates",
        )
        if not candidates:
            return outcome
        if planner is None:
            planner, resolved_label = resolve_navigation_planner(cfg)
            planner_label = planner_label or resolved_label
        label = planner_label or type(planner).__name__
        with ReadOnlyTreeStore(resolved["snapshot"]) as db:
            navigator = TreeNavigator(
                db,
                budget=budget,
                search_nodes=lambda text: searcher.search_from_text(
                    embed_query, text, top_k=top_k, view=view, local_ids=local_ids
                ),
            )
            result = navigator.navigate(
                query,
                candidates,
                expand_regions=expand_regions,
                max_expansions=max_expansions,
                planner=planner,
            )
        _apply_navigation(outcome, result, candidates, verify=verify, planner_label=label)
    return outcome


def _apply_navigation(
    outcome: TreeLegOutcome,
    result: NavigationResult,
    candidates: Sequence[TreeCandidate],
    *,
    verify: Callable[[TreeLegHit], bool] | None,
    planner_label: str = "",
) -> None:
    outcome.planner = planner_label or result.planner
    outcome.navigation_status = result.status
    outcome.navigation_reason = result.reason
    outcome.summary_hits = result.evidence_counts()["summary"]
    outcome.read_spans = len({receipt.span_key for receipt in result.receipts})
    outcome.trace = [step.to_json() for step in result.trace]
    outcome.unresolved = [dict(item) for item in result.unresolved]
    outcome.budget = dict(result.budget)
    entry_scores = {candidate.node_id: float(candidate.score) for candidate in candidates}
    hits: list[TreeLegHit] = []
    dropped = 0
    for item in result.evidence:
        if item.get("source") != "leaf":
            continue
        hit = TreeLegHit(
            node_id=str(item["node_id"]),
            local_id=str(item["local_id"]),
            node_revision=int(item.get("node_revision") or 0),
            content_hash=str((item.get("receipt") or {}).get("content_hash") or ""),
            block_id=str((item.get("receipt") or {}).get("block_id") or ""),
            char_start=int((item.get("receipt") or {}).get("char_start") or 0),
            char_end=int((item.get("receipt") or {}).get("char_end") or 0),
            tokens=int((item.get("receipt") or {}).get("tokens") or 0),
            score=_path_score(item, entry_scores),
            text=str(item.get("text") or ""),
            via=tuple(str(origin) for origin in (item.get("via") or ())),
        )
        if verify is not None and not verify(hit):
            dropped += 1
            continue
        hits.append(hit)
    hits.sort(key=lambda hit: (-hit.score, hit.node_id))
    outcome.hits = hits
    if hits:
        outcome.status = "ok"
        outcome.reason = ""
    elif result.status == "empty":
        outcome.status = "empty"
        outcome.reason = result.reason or "nothing_readable"
    else:
        # Summary hints without leaf text are not source evidence; a revision
        # mismatch drops what was read.  Report that instead of presenting the
        # leg as successful.
        outcome.status = "unavailable"
        outcome.reason = "revision_mismatch" if dropped else (result.reason or "no_leaf_evidence")


def _path_score(item: dict[str, Any], entry_scores: dict[str, float]) -> float:
    """Best entry score along the path that produced this leaf (region → leaf)."""
    best = float(entry_scores.get(str(item.get("node_id")), 0.0))
    for origin in item.get("via") or ():
        best = max(best, float(entry_scores.get(str(origin), 0.0)))
    return best


def _tree_storage_root(cfg: Any, storage_dir: str | Path | None) -> Path:
    if storage_dir is not None:
        return Path(storage_dir)
    configured = str(getattr(get_llamaindex_config(cfg), "tree_storage", "") or "")
    return runtime_scoped_path(configured or DEFAULT_TREE_STORAGE, label="tree storage")


def _embed_batch(texts: list[str], embed_cfg: Any) -> list[list[float]]:
    from drbrain.services.embedding import _embed_batch as embed_batch

    return embed_batch(texts, embed_cfg)


__all__ = [
    "DEFAULT_TOP_K",
    "TreeLegHit",
    "TreeLegOutcome",
    "TreeLegUnavailableError",
    "active_tree_generation",
    "resolve_navigation_planner",
    "run_tree_leg",
]
