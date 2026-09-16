"""Stage observability for the unified store (plan T42).

One snapshot answers "what exists, what is in progress, what failed" for the
whole tree pipeline: document revisions, canonical blocks, nodes by kind and
state, node vectors, summary-cache outcomes, build jobs and the published
generation.  The aggregate stage state follows the frozen worst-wins rules
(T06), so an unfinished or damaged store can never be reported as ready, and
the numbers are recorded as JSON (queryable later) instead of being smuggled
into log prose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from drbrain.tree.contracts import StageReport, StageState, combine_stage_states

#: Metadata key prefix for the latest recorded stage snapshot.
STAGE_KEY_PREFIX = "tree_stage:"


def _counts(db, sql: str, params: tuple = ()) -> dict[str, int]:
    return {str(key): int(value) for key, value in db.conn.execute(sql, params).fetchall()}


@dataclass
class TreeSnapshot:
    documents: dict[str, int] = field(default_factory=dict)
    blocks: int = 0
    nodes: dict[str, int] = field(default_factory=dict)
    node_kinds: dict[str, int] = field(default_factory=dict)
    vectors: dict[str, int] = field(default_factory=dict)
    summaries: dict[str, int] = field(default_factory=dict)
    jobs: dict[str, int] = field(default_factory=dict)
    generation: str = ""
    errors: dict[str, int] = field(default_factory=dict)

    def stage_state(self) -> StageState:
        parts: list[StageState] = []
        if not self.documents and not self.nodes:
            return StageState.ABSENT
        if self.documents.get("failed"):
            parts.append(StageState.FAILED)
        if self.documents.get("stale"):
            parts.append(StageState.STALE)
        if self.jobs.get("running") or self.nodes.get("staging"):
            parts.append(StageState.RUNNING)
        if self.nodes.get("failed") or self.vectors.get("failed"):
            parts.append(StageState.FAILED)
        if self.errors.get("summaries_failed"):
            parts.append(StageState.DEGRADED)
        if not self.documents.get("ready") and self.documents:
            parts.append(StageState.PARTIAL)
        if not parts:
            parts.append(StageState.READY)
        return combine_stage_states(parts)

    def to_json(self) -> dict[str, Any]:
        return {
            "state": self.stage_state().value,
            "documents": dict(self.documents),
            "blocks": self.blocks,
            "nodes": dict(self.nodes),
            "node_kinds": dict(self.node_kinds),
            "vectors": dict(self.vectors),
            "summaries": dict(self.summaries),
            "jobs": dict(self.jobs),
            "generation": self.generation,
            "errors": dict(self.errors),
        }


def tree_snapshot(db, *, storage_dir: str | Path | None = None) -> TreeSnapshot:
    """Collect the current pipeline state from the live database."""
    snapshot = TreeSnapshot()
    snapshot.documents = _counts(
        db, "SELECT COALESCE(state, 'unknown'), COUNT(*) FROM document_revisions GROUP BY state"
    )
    row = db.conn.execute("SELECT COUNT(*) FROM content_blocks").fetchone()
    snapshot.blocks = int(row[0]) if row else 0
    snapshot.nodes = _counts(
        db, "SELECT COALESCE(state, 'unknown'), COUNT(*) FROM tree_nodes GROUP BY state"
    )
    snapshot.node_kinds = _counts(
        db, "SELECT COALESCE(kind, 'unknown'), COUNT(*) FROM tree_nodes GROUP BY kind"
    )
    snapshot.vectors = _counts(
        db, "SELECT COALESCE(state, 'unknown'), COUNT(*) FROM node_vectors GROUP BY state"
    )
    snapshot.summaries = _counts(
        db,
        "SELECT COALESCE(state, 'unknown'), COUNT(*) FROM tree_summary_cache GROUP BY state",
    )
    snapshot.jobs = _counts(
        db, "SELECT COALESCE(state, 'unknown'), COUNT(*) FROM tree_build_jobs GROUP BY state"
    )
    if snapshot.summaries.get("failed"):
        snapshot.errors["summaries_failed"] = int(snapshot.summaries["failed"])
    if storage_dir is not None:
        from drbrain.tree.publish import get_active_tree_generation

        snapshot.generation = get_active_tree_generation(storage_dir) or ""
    return snapshot


def record_stage(
    db,
    report: StageReport,
    *,
    stage_key: str | None = None,
) -> dict[str, Any]:
    """Persist one stage report as queryable metadata (no body text)."""
    key = f"{STAGE_KEY_PREFIX}{stage_key or report.stage}"
    payload = report.to_json()
    from loguru import logger

    logger.info(
        "[tree] stage {}: {} (created={} reused={} failed={})",
        payload["stage"],
        payload["state"],
        payload["created"],
        payload["reused"],
        payload["failed"],
    )
    db.conn.execute(
        "INSERT OR REPLACE INTO vector_metadata (key, value) VALUES (?, ?)",
        (key, json.dumps(payload, ensure_ascii=False, sort_keys=True)),
    )
    db.conn.commit()
    return payload


def read_stage(db, stage: str) -> dict[str, Any] | None:
    """Read back the last recorded report for a stage (or ``None``)."""
    row = db.conn.execute(
        "SELECT value FROM vector_metadata WHERE key = ?", (f"{STAGE_KEY_PREFIX}{stage}",)
    ).fetchone()
    if row is None:
        return None
    try:
        payload = json.loads(row[0])
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def stage_reports(db) -> dict[str, dict[str, Any]]:
    """Every recorded stage report, keyed by stage name."""
    rows = db.conn.execute(
        "SELECT key, value FROM vector_metadata WHERE key LIKE ?", (f"{STAGE_KEY_PREFIX}%",)
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for key, value in rows:
        try:
            payload = json.loads(value)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict):
            out[str(key)[len(STAGE_KEY_PREFIX) :]] = payload
    return out
