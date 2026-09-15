"""Unified tree builder: one builder, one node model, bounded rounds (T34/T35).

One round takes the current frontier, fits the two-stage posterior (T27/T28),
reweights it with structural affinity (T29/T30), merges structural and
semantic candidates (T31), runs the cost gate (T32/T33) and — only for
accepted groups — calls the shared summary service (T26), embeds the *new*
summary text (T23) and publishes one staged region node (T11).  Leaves never
get re-embedded: their vectors come from the shared cache/store.

Termination is bounded and recorded:

* a round that produces no accepted parent stops the build (``no_parent``);
* a frontier that does not shrink stops it (``no_shrink``);
* a layer budget or a per-round node budget stops it (``layer_budget`` /
  ``budget_exhausted``);
* everything not promoted stays in the frontier as a root, so every origin
  leaf remains reachable (multi-root DAG, never a fabricated single tree).
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from loguru import logger

from drbrain.tree.assign import (
    node_source_profile,
    reweighted_stage,
    soft_assignment,
)
from drbrain.tree.clustering import ClusteringParams, global_stage, local_stages
from drbrain.tree.contracts import ChildRef, NodeRecord, region_node_id
from drbrain.tree.cost import CostParams, proposal_post_check, proposal_pre_screen
from drbrain.tree.proposals import (
    CandidateProposal,
    merge_proposals,
    proposals_from_assignment,
    proposals_from_structure,
)
from drbrain.tree.summary import SummaryContract, SummaryMember, SummaryService
from drbrain.tree.vector_store import needs_write_meta


class BuilderError(RuntimeError):
    """The builder was given an inconsistent tree or configuration."""


@dataclass(frozen=True)
class BuilderConfig:
    clustering: ClusteringParams = field(default_factory=ClusteringParams)
    cost: CostParams = field(default_factory=CostParams)
    contract: SummaryContract = field(default_factory=SummaryContract)
    lam: float = 2.0
    max_layers: int = 5
    min_frontier: int = 3
    max_new_nodes_per_round: int = 200
    use_structure_hints: bool = True
    structure_hints_per_document: int = 32
    #: Per-run cap on the seed frontier (T59 scheduling): only the first N
    #: ready seeds enter this run; the rest stay roots for later runs.
    #: 0 keeps the unbounded whole-frontier fit.
    frontier_limit: int = 0
    #: Summary calls allowed to be in flight per round (T59 scheduling).
    summary_workers: int = 1


@dataclass
class RoundMetrics:
    round_index: int
    frontier_size: int
    components: int = 0
    proposals: int = 0
    accepted: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    #: Execution failures (model/clustering), kept apart from rejections so a
    #: broken endpoint can never be reported as a valid stop (T06/T36).
    failed: dict[str, int] = field(default_factory=dict)
    created_nodes: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    summary_tokens: int = 0
    duration_ms: float = 0.0
    stop_reason: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "round": self.round_index,
            "frontier_size": self.frontier_size,
            "components": self.components,
            "proposals": self.proposals,
            "accepted": self.accepted,
            "rejected": dict(self.rejected),
            "failed": dict(self.failed),
            "created_nodes": list(self.created_nodes),
            "prompt_tokens": self.prompt_tokens,
            "summary_tokens": self.summary_tokens,
            "duration_ms": round(self.duration_ms, 3),
            "stop_reason": self.stop_reason,
        }


@dataclass
class BuildResult:
    rounds: list[RoundMetrics] = field(default_factory=list)
    roots: list[str] = field(default_factory=list)
    stop_reason: str = ""
    ready_layers: int = 0
    frontier_total: int = 0
    frontier_processed: int = 0

    @property
    def created_nodes(self) -> list[str]:
        return [node for metrics in self.rounds for node in metrics.created_nodes]

    @property
    def failed(self) -> dict[str, int]:
        """Aggregated execution failures across rounds (empty when clean)."""
        totals: dict[str, int] = {}
        for metrics in self.rounds:
            for reason, count in metrics.failed.items():
                totals[reason] = totals.get(reason, 0) + count
        return totals

    def to_json(self) -> dict[str, Any]:
        return {
            "rounds": [metrics.to_json() for metrics in self.rounds],
            "roots": list(self.roots),
            "layers": self.ready_layers,
            "stop_reason": self.stop_reason,
            "created": len(self.created_nodes),
            "failed": self.failed,
            "frontier": {
                "total": self.frontier_total,
                "processed": self.frontier_processed,
            },
        }


def _default_count(text: str) -> int:
    from drbrain.services.tokens import count_tokens

    return int(count_tokens(text))


class TreeBuilder:
    """One builder for the whole corpus; no second semantic tree engine."""

    def __init__(
        self,
        db,
        *,
        vectors=None,
        summary_service: SummaryService | None = None,
        embed: Callable[[Sequence[str]], list[list[float]]] | None = None,
        config: BuilderConfig | None = None,
        count_tokens: Callable[[str], int] | None = None,
        model: Any | None = None,
        profile_id: str = "",
    ) -> None:
        self.db = db
        self.vectors = vectors
        self.count_tokens = count_tokens or _default_count
        self.summary = summary_service or SummaryService(db, count_tokens=self.count_tokens)
        self.embed = embed
        self.config = config or BuilderConfig()
        self.profile_id = str(profile_id or "unknown")
        if model is not None:
            self.set_model(model)

    def set_model(self, model: Any) -> TreeBuilder:
        """Bind the summary/index model (fail-closed before any work)."""
        self.model = model
        self._model_impl = model
        return self

    # ── public API ───────────────────────────────────────────────
    def build(self, seed_nodes: Sequence[str]) -> BuildResult:
        frontier = [node_id for node_id in seed_nodes if self._ready(node_id)]
        if not frontier:
            raise BuilderError("build requires at least one ready seed node")
        self._model()  # fail closed before any embedding/clustering work
        result = BuildResult()
        result.frontier_total = len(frontier)
        if self.config.frontier_limit > 0 and len(frontier) > self.config.frontier_limit:
            # Bounded batches (T59): process a slice per run; the remaining
            # seeds stay roots and re-enter the next run's frontier.  The
            # result records both counts so a capped build is never mistaken
            # for full coverage.
            frontier = frontier[: self.config.frontier_limit]
        result.frontier_processed = len(frontier)
        layer = max((self._node_layer(node_id) for node_id in frontier), default=0) + 1
        for round_index in range(self.config.max_layers):
            if len(frontier) <= self.config.min_frontier:
                result.stop_reason = "frontier_below_minimum"
                break
            metrics, created = self._round(round_index, frontier, layer)
            result.rounds.append(metrics)
            if metrics.stop_reason:
                result.stop_reason = metrics.stop_reason
                break
            if not created:
                result.stop_reason = "no_parent"
                break
            if len(created) >= len(frontier):
                frontier = created
                result.stop_reason = "no_shrink"
                break
            frontier = created
            layer += 1
        if not result.stop_reason:
            result.stop_reason = "layer_budget"
        result.roots = self._roots(seed_nodes, result.created_nodes)
        result.ready_layers = layer - 1
        return result

    # ── one round ────────────────────────────────────────────────
    def _round(self, round_index: int, frontier: Sequence[str], layer: int):
        started = time.monotonic()
        metrics = RoundMetrics(round_index=round_index, frontier_size=len(frontier))
        timings: dict[str, float] = {}

        def mark(name: str, mark_from: float) -> float:
            now = time.monotonic()
            timings[name] = timings.get(name, 0.0) + (now - mark_from)
            return now

        t0 = time.monotonic()
        profiles = {
            node_id: node_source_profile(self.db, node_id, count_tokens=self.count_tokens)
            for node_id in frontier
        }
        t0 = mark("profiles", t0)
        embeddings = self._embeddings(frontier)
        t0 = mark("embeddings", t0)
        proposals: list[CandidateProposal] = []
        if self.embed is not None and len(frontier) >= 3:
            try:
                fitted_global = global_stage(embeddings, frontier, self.config.clustering)
                local_map = local_stages(
                    embeddings,
                    frontier,
                    self._conditioned_global(fitted_global, profiles),
                    self.config.clustering,
                )
            except Exception as exc:  # noqa: BLE001 - clustering failure is a state
                logger.warning("[tree] clustering failed in round {}: {}", round_index, exc)
                metrics.stop_reason = "clustering_failed"
                metrics.duration_ms = (time.monotonic() - started) * 1000
                return metrics, []
            metrics.components = len(local_map)
            t0 = mark("clustering", t0)
            candidates = self._assignment_candidates(local_map, profiles)
            t0 = mark("assignment", t0)
            proposals.extend(
                proposals_from_assignment(
                    candidates,
                    self.config.contract,
                    min_members=self.config.cost.min_members,
                )
            )
            t0 = mark("assignment-proposals", t0)
        if self.config.use_structure_hints:
            proposals.extend(self._structure_proposals(frontier))
        t0 = mark("structure", t0)
        merged = merge_proposals(proposals)
        metrics.proposals = len(merged)
        t0 = mark("merge", t0)
        accepted, created = self._materialize(merged, metrics, layer)
        mark("gates+summaries", t0)
        metrics.accepted = accepted
        metrics.created_nodes = created
        metrics.duration_ms = (time.monotonic() - started) * 1000
        if not merged:
            metrics.stop_reason = "no_candidates"
        logger.info(
            "[tree] round {} (frontier={}): {} | total={:.0f}s",
            round_index,
            len(frontier),
            " ".join(f"{name}={value:.0f}s" for name, value in timings.items()),
            metrics.duration_ms / 1000,
        )
        return metrics, created

    def _assignment_candidates(self, local_map, profiles):
        """Semantic candidates: reweight each local stage with the affinity matrix."""
        candidates = []
        for _component_id, stage in local_map.items():
            raw = stage.posterior_stage()  # lam=0 keeps the upstream posterior intact
            candidates.extend(
                soft_assignment(raw, profiles, lam=self.config.lam, with_affinity=True)
            )
        return candidates

    def _conditioned_global(self, fitted_global, profiles):
        """Global stage with the structural prior already applied (T04/T30).

        Each stage reweights its own posterior before the strict threshold; at
        the global level the corrected membership is what decides which rows
        share a local fit, so ``local_stages`` must see it rather than the raw
        upstream labels.
        """
        raw = fitted_global.posterior_stage()
        corrected = reweighted_stage(raw, profiles, lam=self.config.lam)
        if corrected is raw:
            return fitted_global
        indices = corrected.label_indices()
        labels = tuple(indices.get(row_id, ()) for row_id in fitted_global.row_ids)
        return replace(fitted_global, labels=labels)

    def _structure_proposals(self, frontier: Sequence[str]) -> list[CandidateProposal]:
        """Structure hints over the frontier's documents -> candidate proposals."""
        from drbrain.tree.outline import extract_md_outline

        by_doc: dict[str, list[str]] = {}
        for node_id in frontier:
            row = self.db.get_tree_node(node_id)
            if row is None or row["kind"] != "leaf":
                continue
            by_doc.setdefault(str(row["local_id"]), []).append(node_id)
        members_by_scope: dict[tuple[str, str], list[str]] = {}
        hints: list[dict[str, Any]] = []
        for local_id, node_ids in by_doc.items():
            if len(node_ids) < self.config.cost.min_members:
                continue
            text = self._document_text(local_id)
            if not text:
                continue
            try:
                outline = extract_md_outline(
                    text, max_nodes=self.config.structure_hints_per_document
                )
            except Exception as exc:  # noqa: BLE001 - hints are optional
                logger.debug("[tree] outline extraction failed for {}: {}", local_id, exc)
                continue
            for hint in outline:
                path = tuple(getattr(hint, "heading_path", ()) or ())
                scope_key = " > ".join(path)
                members = [
                    node_id
                    for node_id in node_ids
                    if self._leaf_heading(node_id)[: len(path)] == path
                ]
                if len(members) < self.config.cost.min_members:
                    continue
                hints.append({"local_id": local_id, "heading_path": list(path)})
                members_by_scope[(local_id, scope_key)] = members
        return proposals_from_structure(
            hints, members_by_scope, self.config.contract, min_members=self.config.cost.min_members
        )

    def _materialize(
        self, proposals: Sequence[CandidateProposal], metrics: RoundMetrics, layer: int
    ) -> tuple[int, list[str]]:
        """Prescreen serially; keep up to ``summary_workers`` calls in flight.

        The prescreen order, the ``seen_member_keys`` coverage filter and the
        publish order stay strictly sequential — only the model call itself
        overlaps (T59 scheduling).  A failed call still stops the round, at the
        cost of at most ``summary_workers - 1`` calls already in flight.
        """
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        seen_keys: set[str] = set()
        accepted = 0
        created: list[str] = []
        stop = False
        workers = max(1, int(self.config.summary_workers))
        pending: deque[tuple[CandidateProposal, Any]] = deque()

        def drain() -> None:
            nonlocal accepted, stop
            proposal, future = pending.popleft()
            node_id, execution_failed = self._publish_summary(
                proposal, future.result(), layer, metrics
            )
            if node_id is None:
                if execution_failed:
                    # A broken endpoint is a state, not a rejected group: stop
                    # instead of spending one failing call per candidate.  The
                    # members stay in the frontier (nothing was published).
                    metrics.stop_reason = "summary_failed"
                    stop = True
                    return
                metrics.rejected["summary_rejected"] = (
                    metrics.rejected.get("summary_rejected", 0) + 1
                )
                return
            seen_keys.add(proposal.key)
            accepted += 1
            created.append(node_id)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for proposal in sorted(proposals, key=lambda item: -len(item.member_ids)):
                if stop:
                    break
                while len(pending) >= workers:
                    drain()
                    if stop:
                        break
                if stop:
                    break
                if len(created) >= self.config.max_new_nodes_per_round:
                    metrics.rejected["budget_exhausted"] = (
                        metrics.rejected.get("budget_exhausted", 0) + 1
                    )
                    continue
                decision = proposal_pre_screen(
                    self.db,
                    proposal,
                    params=self.config.cost,
                    seen_member_keys=seen_keys,
                    count_tokens=self.count_tokens,
                )
                if not decision.accepted:
                    metrics.rejected[decision.reason] = metrics.rejected.get(decision.reason, 0) + 1
                    continue
                members = [self._summary_member(node_id) for node_id in proposal.member_ids]
                pending.append(
                    (
                        proposal,
                        pool.submit(
                            self.summary.summarize,
                            members,
                            self.config.contract,
                            self._model(),
                        ),
                    )
                )
            while pending and not stop:
                drain()
        return accepted, created

    def _publish_summary(
        self, proposal: CandidateProposal, outcome: Any, layer: int, metrics: RoundMetrics
    ) -> tuple[str | None, bool]:
        """Post-check and publish one summarised candidate; ``(node_id, failed)``."""
        metrics.prompt_tokens += outcome.prompt_tokens
        if not outcome.ok:
            if outcome.execution_failure:
                metrics.failed[outcome.reason] = metrics.failed.get(outcome.reason, 0) + 1
                return None, True
            return None, False
        metrics.summary_tokens += outcome.summary_tokens
        from drbrain.tree.cost import coverage_for_members

        referenced = coverage_for_members(
            self.db, proposal.member_ids, count_tokens=self.count_tokens
        )
        decision = proposal_post_check(
            self.db,
            proposal,
            summary_text=outcome.summary,
            summary_tokens=outcome.summary_tokens,
            finish_reason="stop",
            referenced_spans=referenced.spans,
            params=self.config.cost,
            count_tokens=self.count_tokens,
        )
        if not decision.accepted:
            return None, False
        children = tuple(
            ChildRef(child_id=node_id, child_revision=self._node_revision(node_id), ordinal=index)
            for index, node_id in enumerate(proposal.member_ids)
        )
        record = NodeRecord(
            node_id=region_node_id(children, self.config.contract.canonical()),
            revision=1,
            kind="region",
            state="staging",
            layer=layer,
            content_hash=hashlib.sha256(outcome.summary.encode("utf-8")).hexdigest(),
            summary=outcome.summary,
            children=children,
            contract=self.config.contract.canonical(),
            origin=proposal.origin,
        )
        self.db.insert_tree_node(record)
        self.db.publish_tree_node(record.node_id)
        self._embed_new_summary(record)
        return record.node_id, False

    # ── helpers ──────────────────────────────────────────────────
    def _model(self):
        model = getattr(self, "_model_impl", None)
        if model is None:
            raise BuilderError("build requires an index model (set builder.model)")
        return model

    def _embed_new_summary(self, record: NodeRecord) -> None:
        if self.embed is None or self.vectors is None:
            return
        vector = self.embed([record.summary])[0]
        from drbrain.tree.vector_store import VectorEntry

        profile_id = getattr(self, "profile_id", "unknown")
        entry = VectorEntry(
            node_id=record.node_id,
            node_revision=record.revision,
            kind="region",
            local_id="",
            layer=record.layer,
            content_hash=record.content_hash,
            profile_id=profile_id,
            vector=tuple(float(value) for value in vector),
        )
        self.db.upsert_node_vector(
            entry.node_id,
            node_revision=entry.node_revision,
            kind=entry.kind,
            profile_id=entry.profile_id,
            content_hash=entry.content_hash,
            dimension=len(entry.vector),
            local_id=entry.local_id,
            layer=entry.layer,
            state="staging",
        )
        self.vectors.upsert([entry])
        self.db.upsert_node_vector(
            entry.node_id,
            node_revision=entry.node_revision,
            kind=entry.kind,
            profile_id=entry.profile_id,
            content_hash=entry.content_hash,
            dimension=len(entry.vector),
            local_id=entry.local_id,
            layer=entry.layer,
            state="ready",
        )

    def _embeddings(self, frontier: Sequence[str]) -> list[list[float]]:
        if self.embed is None:
            return []
        reused, missing = self._stored_vectors(frontier)
        if missing:
            texts = [self._node_text(node_id) for node_id in missing]
            fresh = [list(vector) for vector in self.embed(texts)]
            if len(fresh) != len(missing):
                raise BuilderError(
                    f"embedder returned {len(fresh)} vectors for {len(missing)} nodes"
                )
            for node_id, vector in zip(missing, fresh):
                reused[node_id] = vector
        return [list(reused[node_id]) for node_id in frontier]

    def _stored_vectors(self, frontier: Sequence[str]) -> tuple[dict[str, list[float]], list[str]]:
        """Vectors already stored for exactly this revision/hash/profile.

        T34: only new summaries get embedded.  A frontier node whose vector the
        prepare stage (leaves) or an earlier round (regions) already computed is
        read back instead of embedded again; anything with missing or stale
        metadata falls through to the embedder.
        """
        reuse: dict[str, list[float]] = {}
        missing: list[str] = []
        if self.vectors is None:
            return reuse, list(frontier)
        candidates: list[str] = []
        for node_id in frontier:
            row = self.db.get_tree_node(node_id)
            if row is None:
                raise BuilderError(f"unknown node {node_id!r}")
            if needs_write_meta(
                self.db.get_node_vector(node_id),
                node_revision=int(row["revision"]),
                content_hash=str(row["content_hash"]),
                profile_id=self.profile_id,
            ):
                missing.append(node_id)
            else:
                candidates.append(node_id)
        if candidates:
            try:
                stored = self.vectors.get(candidates)
            except Exception as exc:  # noqa: BLE001 - recomputing beats failing the round
                logger.warning("[tree] stored vectors unavailable: {}", exc)
                stored = {}
            for node_id in candidates:
                entry = stored.get(node_id)
                if entry is None:
                    missing.append(node_id)
                else:
                    reuse[node_id] = [float(value) for value in entry.vector]
        return reuse, missing

    def _summary_member(self, node_id: str) -> SummaryMember:
        row = self.db.get_tree_node(node_id)
        if row is None:
            raise BuilderError(f"unknown node {node_id!r}")
        return SummaryMember(
            node_id=node_id,
            node_revision=int(row["revision"]),
            content_hash=str(row["content_hash"]),
            text=self._node_text(node_id),
            local_id=str(row["local_id"] or ""),
            heading_path=self._leaf_heading(node_id),
        )

    def _node_text(self, node_id: str) -> str:
        row = self.db.get_tree_node(node_id)
        if row is None:
            raise BuilderError(f"unknown node {node_id!r}")
        if row["kind"] == "region":
            return str(row["summary"])
        blocks = self.db.get_content_blocks(row["local_id"], int(row["doc_revision"]))
        block = next((item for item in blocks if item["block_id"] == row["block_id"]), None)
        if block is None:
            raise BuilderError(f"leaf {node_id!r} references a missing block")
        text = str(block["text"])
        char_start = int(row["char_start"] or 0)
        char_end = int(row["char_end"] or len(text))
        return text[char_start:char_end]

    def _leaf_heading(self, node_id: str) -> tuple[str, ...]:
        row = self.db.get_tree_node(node_id)
        if row is None:
            raise BuilderError(f"unknown node {node_id!r}")
        if row["kind"] == "region":
            import json

            return ("",)
        import json

        return tuple(json.loads(row.get("heading_path") or "[]")) or ()

    def _document_text(self, local_id: str) -> str:
        from drbrain.storage.content import read_text

        try:
            return read_text(self.db, local_id)
        except Exception:  # noqa: BLE001 - hints are optional for legacy docs
            return ""

    def _node_layer(self, node_id: str) -> int:
        row = self.db.get_tree_node(node_id)
        return int(row["layer"]) if row else 0

    def _node_revision(self, node_id: str) -> int:
        row = self.db.get_tree_node(node_id)
        if row is None:
            raise BuilderError(f"unknown node {node_id!r}")
        return int(row["revision"])

    def _ready(self, node_id: str) -> bool:
        row = self.db.get_tree_node(node_id)
        return bool(row) and str(row["state"]) == "ready"

    def _roots(self, seed_nodes: Sequence[str], created_nodes: Sequence[str]) -> list[str]:
        """Top-level nodes of this build: no parent edge, still present.

        Leaves that were never promoted stay as roots, so the result is a
        multi-root DAG and every origin leaf remains reachable.
        """
        scope = list(dict.fromkeys([*seed_nodes, *created_nodes]))
        roots: list[str] = []
        for node_id in scope:
            if self.db.get_tree_node(node_id) is None:
                continue
            if not self.db.get_tree_parents(node_id):
                roots.append(node_id)
        return roots
