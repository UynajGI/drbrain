"""One durable, single-flight ``index build`` job (04-arch A1).

The build used to be a plain synchronous CLI body: two callers (a terminal and
the WebUI) could start the same corpus build at the same moment, and a long
hierarchy run had no durable progress record.  This module adds the missing job
around the existing work — nothing else changes:

* the scope slot is **deterministic** (``job-<sha256(scope)[:20]>``), so every
  caller converges on the same row and the database is the arbiter: the slot is
  claimed by exactly one worker (``claim_tree_job`` for a live lease, the lease
  defends it afterwards, ``reopen_tree_job`` restarts a finished one);
* progress is a checkpoint per finished stage (``fts`` → ``vectors`` →
  ``hierarchy`` → ``publication``) plus the existing live counters, so the UI
  polls facts instead of inventing an ETA;
* the build itself (``prepare_unified_index``) only reports stage completion
  through its optional ``on_stage`` callback — it knows nothing about jobs.

Callers keep their own runtime/symlink path policy: the CLI passes the handle it
opened via ``cli._common.open_db``, the WebUI service passes its own.  This
module never resolves configuration paths for them.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from drbrain.security import safe_error

#: ``tree_build_jobs.kind`` for the corpus index build.
KIND = "index_build"
#: States that mean "a run is in flight or resumable" (never ours to steal while leased).
ACTIVE_STATES: tuple[str, ...] = ("pending", "running", "paused")
#: States a finished run can be restarted from.
TERMINAL_STATES: tuple[str, ...] = ("done", "failed")
#: The stages a build runs, in order; a checkpoint names the *next* stage.
#: ``lexical`` is the BM25 stage the bare ``drbrain index`` also runs.
STAGES: tuple[str, ...] = ("lexical", "fts", "vectors", "hierarchy", "publication")
#: Lease for the build slot. Long enough for a stage, renewed at every boundary.
DEFAULT_LEASE_SECONDS = 900


class IndexBuildError(RuntimeError):
    """Base class for index-build job failures."""


class IndexBuildBusyError(IndexBuildError):
    """Another worker holds the scope's slot (a live lease)."""

    def __init__(self, job_id: str, *, state: str = "", owner: str = "") -> None:
        detail = f"state={state or 'unknown'}"
        if owner:
            detail += f", owner={owner}"
        super().__init__(f"index build job {job_id!r} is already running ({detail})")
        self.job_id = str(job_id)
        self.state = str(state)
        self.owner = str(owner)


class IndexBuildProfileUnavailableError(IndexBuildError):
    """The embedding profile cannot be built, so no stage can run."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"embedding profile unavailable: {reason}")
        self.reason = str(reason)


# ── scope / slot ─────────────────────────────────────────────────────────────


def tree_storage_root(cfg: Any, tree_storage: str | Path | None = None) -> Path:
    """The unified tree root, resolved without a ``typer.Context``.

    Mirrors the CLI's default chain (``llamaindex.tree_storage`` → ``data/tree``)
    and honors an exported runtime selector, so an embedded caller cannot
    silently write a tree under the repository instead of the selected root.
    """
    if tree_storage:
        return Path(str(tree_storage)).expanduser()
    from drbrain.rag.config import get_llamaindex_config
    from drbrain.services.index_report import DEFAULT_TREE_STORAGE

    configured = str(getattr(get_llamaindex_config(cfg), "tree_storage", "") or "").strip()
    raw = Path(configured or DEFAULT_TREE_STORAGE).expanduser()
    if raw.is_absolute():
        return raw
    if "DRBRAIN_ROOT" in os.environ or "DRBRAIN_RUNTIME_ROOT" in os.environ:
        from drbrain.runtime import RuntimeContext

        return Path(RuntimeContext.create().assert_within_root(raw, label="tree storage"))
    return raw.resolve()


def index_build_scope_key(cfg: Any, db: Any, *, tree_storage: str | Path | None = None) -> str:
    """One slot per (database, embedding profile, tree storage).

    A different embedding model or tree root is a different deployment identity
    (the same rule the published generation follows), so it gets its own slot
    instead of blocking an unrelated build.
    """
    from drbrain.services.index_report import embedding_profile

    profile, _ = embedding_profile(cfg)
    profile_id = profile.profile_id() if profile is not None else "no-profile"
    db_path = str(getattr(db, "path", "") or "")
    return f"index-build|{db_path}|{profile_id}|{tree_storage_root(cfg, tree_storage)}"


def index_build_job_id(scope_key: str) -> str:
    """Deterministic slot id: every caller computes the same job for a scope."""
    digest = hashlib.sha256(str(scope_key).encode("utf-8")).hexdigest()[:20]
    return f"job-{digest}"


def worker_owner(prefix: str = "index-build") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ── job row access ───────────────────────────────────────────────────────────


def slot_row(db: Any, scope_key: str) -> dict | None:
    """The scope's job row, if the slot was ever created."""
    return db.get_tree_job(index_build_job_id(scope_key))


def lease_expired(db: Any, job_id: str) -> bool:
    """Whether a ``running`` job's lease has lapsed (a crashed worker).

    Read-only and evaluated by SQLite, so it agrees with the exact comparison
    ``claim_tree_job`` uses instead of relying on Python clock arithmetic.
    """
    row = db.conn.execute(
        """SELECT claim_expires_at IS NULL OR claim_expires_at < CURRENT_TIMESTAMP
           FROM tree_build_jobs WHERE job_id = ?""",
        (str(job_id),),
    ).fetchone()
    return bool(row[0]) if row is not None else False


def job_is_active(db: Any, row: dict | None) -> bool:
    """True when the run is in flight (or resumable) **and** still leased.

    A ``running`` row whose lease lapsed is not "active": it is a crashed or
    killed run, and the next caller may take it over.
    """
    if not row:
        return False
    state = str(row.get("state") or "")
    if state not in ACTIVE_STATES:
        return False
    if state == "running":
        return not lease_expired(db, str(row.get("job_id") or ""))
    return True


def ensure_index_build_job(
    db: Any,
    scope_key: str,
    *,
    owner: str,
    job_id: str = "",
    ttl_seconds: int = DEFAULT_LEASE_SECONDS,
) -> tuple[dict, str]:
    """Resolve the scope slot and try to take it.

    Returns ``(row, disposition)`` where disposition is one of:

    * ``"claimed"`` — a fresh or finished slot is now ours;
    * ``"taken_over"`` — a stale (lease-expired) run is now ours;
    * ``"reused"`` — somebody else holds a live lease; the caller must not work.

    ``job_id`` lets a caller name the slot explicitly (the WebUI hands the same
    id to its worker); it must belong to ``scope_key`` or the call fails closed.
    """
    slot = str(job_id or index_build_job_id(scope_key))
    row = db.get_tree_job(slot)
    if row is None:
        try:
            db.insert_tree_job(slot, scope_key, kind=KIND)
        except sqlite3.IntegrityError:
            pass  # another process created the slot first; reuse its row
        row = db.get_tree_job(slot)
    if row is None:  # pragma: no cover - only reachable on a broken database
        raise IndexBuildError(f"cannot open the index-build job slot {slot}")
    if str(row.get("scope_key") or "") != str(scope_key):
        raise IndexBuildError(f"job {slot!r} belongs to another scope; refusing to reuse it")

    state = str(row.get("state") or "")
    if state in TERMINAL_STATES:
        if db.reopen_tree_job(slot, owner, ttl_seconds=ttl_seconds):
            return db.get_tree_job(slot) or row, "claimed"
        return db.get_tree_job(slot) or row, "reused"
    if db.claim_tree_job(slot, owner, ttl_seconds=ttl_seconds):
        return db.get_tree_job(slot) or row, (
            "taken_over" if state in ("running", "paused") else "claimed"
        )
    return db.get_tree_job(slot) or row, "reused"


# ── progress ─────────────────────────────────────────────────────────────────


def _loads(raw: Any, default: Any) -> Any:
    try:
        value = json.loads(raw or "")
    except (TypeError, ValueError):
        return default
    return value if isinstance(value, type(default)) else default


def rebuild_lexical(
    db: Any, *, rebuild: bool = False, notify: Callable[[str], None] | None = None
) -> dict[str, Any]:
    """The incremental lexical BM25 stage, on a caller-owned handle.

    Identical to the historical bare-``drbrain index`` behavior (skip when no
    paper changed since the last run; ``--force`` rebuilds) — it lives here so
    the CLI and the WebUI job run *one* implementation of the first stage.
    """
    from drbrain.query.bm25 import build_bm25_index

    if not rebuild:
        last_run = db.get_last_run("index")
        max_ts = db.get_max_paper_timestamp()
        if last_run is not None and (max_ts is None or max_ts <= last_run):
            count = db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            return {"documents": count, "indexed": False, "up_to_date": True}

    if notify is not None:
        notify("Building BM25 index...")
    index = build_bm25_index(db)
    doc_count = len(index._documents)
    db.set_last_run("index")
    db.commit()
    return {"documents": doc_count, "indexed": True}


def vector_pending(db: Any, profile: Any) -> int | None:
    """Ready leaves/regions whose vector is missing or stale.

    Uses the same enumeration the vector stage and ``index status`` use
    (``index_report._vector_backlog``), so the number cannot drift; returns
    ``None`` when the profile cannot produce one.
    """
    if profile is None:
        return None
    from drbrain.services import index_report

    helper = getattr(index_report, "vector_backlog", None) or getattr(
        index_report, "_vector_backlog", None
    )
    if helper is None:  # pragma: no cover - defensive against a rename
        return None
    try:
        return int((helper(db, profile) or {}).get("pending") or 0)
    except Exception:  # noqa: BLE001 - progress is best-effort, never fatal
        return None


def live_index_counts(db: Any) -> dict[str, int]:
    """Cheap live counters a poller may read on every tick (all indexed counts)."""
    return {
        "leaves_ready": int(db.count_tree_nodes(kind="leaf", state="ready")),
        "regions_ready": int(db.count_tree_nodes(kind="region", state="ready")),
        "vectors_ready": int(db.count_node_vectors(state="ready")),
        "leaves_missing_parent": int(db.count_leaves_missing_parent()),
    }


def _failed_stages(metrics: dict[str, Any]) -> set[str]:
    """Stage names that did not succeed, in the panel's own vocabulary.

    ``PrepareOutcome`` calls the publication stage ``publish``; the panel lists
    it as ``publication``.
    """
    out: set[str] = set()
    for item in metrics.get("failed_stages") or []:
        name = str(item)
        out.add("publication" if name == "publish" else name)
    return out


def _stage_errors(record: dict[str, Any]) -> dict[str, str]:
    """Why each failed stage failed — the copyable reason the panel shows.

    Read from the recorded last build (the stage payloads carry the stage's own
    error text); a stage that failed without a message keeps the stage name as
    its only report, which is still better than a bare "failed".
    """
    out: dict[str, str] = {}
    for name in ("fts", "vectors", "hierarchy", "publication"):
        payload = record.get(name)
        if not isinstance(payload, dict):
            continue
        if str(payload.get("status")) not in {"failed", "partial"}:
            continue
        detail = str(payload.get("error") or payload.get("reason") or "").strip()
        if detail:
            out[name] = detail[:500]
    return out


def _stage_states(
    stages_done: list[str], payload_stage: str, state: str, failed: set[str] | None = None
) -> dict[str, str]:
    """Per-stage status words for the panel (labels stay in the template layer)."""
    failed = failed or set()
    out: dict[str, str] = {}
    for name in STAGES:
        if name in failed:
            out[name] = "failed"
        elif name in stages_done:
            out[name] = "done"
        elif name == payload_stage and state in ("running", "pending"):
            out[name] = "running"
        else:
            out[name] = "pending"
    return out


def index_job_payload(db: Any, row: dict | None, *, cfg: Any = None) -> dict[str, Any]:
    """The job, as the UI reads it (04-arch A1 contract)."""
    if row is None:
        return {}
    from drbrain.tree.prepare import LAST_BUILD_KEY

    checkpoint = _loads(row.get("checkpoint_json"), {})
    metrics = _loads(row.get("metrics_json"), {})
    state = str(row.get("state") or "")
    stages_done = [str(item) for item in (checkpoint.get("stages_done") or [])]
    stage = str(checkpoint.get("stage") or "")
    failed = _failed_stages(metrics if isinstance(metrics, dict) else {})
    if state in TERMINAL_STATES:
        stage = ""
        stages_done = [name for name in STAGES if name in stages_done] or list(STAGES)
    payload: dict[str, Any] = {
        "job_id": str(row.get("job_id") or ""),
        "kind": str(row.get("kind") or KIND),
        "state": state,
        "owner": str(row.get("owner") or ""),
        "reason": str(row.get("reason") or ""),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "stage": stage,
        "stages_done": stages_done,
        "stage_states": _stage_states(stages_done, stage, state, failed),
        "failed_stages": sorted(failed),
        "pending": dict(checkpoint.get("pending") or {}),
        "counts": dict(checkpoint.get("counts") or {}),
        "metrics": {
            key: metrics.get(key)
            for key in ("ok", "changed", "published", "failed_stages", "duration_ms")
            if key in metrics
        },
        "lease_expired": state == "running" and lease_expired(db, str(row.get("job_id") or "")),
        "stale": state == "running" and lease_expired(db, str(row.get("job_id") or "")),
    }
    try:
        payload["live"] = live_index_counts(db)
    except Exception as exc:  # noqa: BLE001 - a counter must never break the panel
        payload["live"] = {}
        payload["live_error"] = safe_error(exc)
    # The vectors still owed for this run (known from the run's own start), so a
    # poller can show "done / pending" without re-enumerating the corpus.
    payload["live"]["vectors_pending"] = (checkpoint.get("pending") or {}).get("vectors")
    try:
        last = _loads(db.get_vector_metadata(LAST_BUILD_KEY), {})
    except Exception:  # noqa: BLE001
        last = {}
    if isinstance(last, dict) and last:
        payload["last_build"] = {
            "ok": bool(last.get("ok")),
            "failed_stages": list(last.get("failed_stages") or []),
            "published": str(last.get("published") or ""),
            "changed": bool(last.get("changed")),
            # The stage's own error text: the UI shows it as the copyable
            # reason (FR-I5 "失败能自助定位"), never as a bare stage name.
            "stage_errors": _stage_errors(last),
        }
    else:
        payload["last_build"] = {
            "ok": None,
            "failed_stages": [],
            "published": "",
            "changed": False,
            "stage_errors": {},
        }
    return payload


def index_jobs(
    db: Any,
    *,
    cfg: Any = None,
    kind: str = "",
    states: tuple[str, ...] = (),
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Job rows for the generic job list (``GET /api/jobs``)."""
    wanted = {str(item) for item in states if str(item)}
    rows = db.list_tree_jobs(limit=max(1, int(limit)))
    if wanted:
        rows = [row for row in rows if str(row.get("state") or "") in wanted]
    if kind:
        rows = [row for row in rows if str(row.get("kind") or "") == str(kind)]
    return [index_job_payload(db, row, cfg=cfg) for row in rows]


# ── the build itself ─────────────────────────────────────────────────────────


def _checkpoint(
    db: Any,
    job_id: str,
    owner: str,
    *,
    stage: str,
    stages_done: list[str],
    pending: dict[str, Any],
    counts: dict[str, Any],
    started_at: float,
) -> bool:
    """Persist progress and renew the lease.

    Returns whether we still hold the lease.  A checkpoint write is best-effort
    (progress must never fail a build), but a *lost* lease is not: another
    worker may be running the same slot, so the caller stops.
    """
    checkpoint = {
        "stage": stage,
        "stages_done": list(stages_done),
        "started_at": float(started_at),
        "pending": dict(pending),
        "counts": dict(counts),
    }
    try:
        db.checkpoint_tree_job(
            job_id,
            checkpoint=json.dumps(checkpoint, ensure_ascii=False, sort_keys=True),
            owner=owner,
        )
    except Exception as exc:  # noqa: BLE001 - progress must not fail the build
        from loguru import logger

        logger.warning("[index] could not checkpoint job {}: {}", job_id, exc)
    return bool(db.renew_tree_job(job_id, owner, ttl_seconds=DEFAULT_LEASE_SECONDS))


def run_index_build(
    cfg: Any,
    *,
    db: Any,
    force: bool = False,
    tree_storage: str | Path | None = None,
    notify: Callable[[str], None] | None = None,
    job_id: str = "",
    owner: str = "",
    ttl_seconds: int = DEFAULT_LEASE_SECONDS,
) -> dict[str, Any]:
    """Run the corpus index build under one claimed job slot.

    Raises :class:`IndexBuildBusyError` when a live lease already holds the slot and
    :class:`IndexBuildProfileUnavailableError` when no embedding profile exists.
    Any other failure is recorded on the job row and re-raised unchanged, so the
    CLI keeps its traceback/exit-code contract.
    """
    from drbrain.rag.config import get_llamaindex_config
    from drbrain.services.index_report import embed_section, embedding_profile
    from drbrain.tree.prepare import prepare_unified_index

    started = time.monotonic()
    scope_key = index_build_scope_key(cfg, db, tree_storage=tree_storage)
    slot = str(job_id or index_build_job_id(scope_key))
    owner = str(owner or worker_owner())

    profile, profile_error = embedding_profile(cfg)

    row, disposition = ensure_index_build_job(
        db, scope_key, owner=owner, job_id=slot, ttl_seconds=ttl_seconds
    )
    if disposition == "reused":
        raise IndexBuildBusyError(
            slot, state=str(row.get("state") or ""), owner=str(row.get("owner") or "")
        )
    if profile is None:
        # Claimed the slot first so the failure is visible on the job row instead
        # of leaving the caller with an error and nothing to poll.
        reason = f"embedding profile unavailable: {profile_error or 'unknown reason'}"
        try:
            db.finish_tree_job(slot, "failed", reason=reason[:500])
        except Exception:  # noqa: BLE001 - the raised error is the primary signal
            pass
        raise IndexBuildProfileUnavailableError(profile_error or "unknown reason")

    li = get_llamaindex_config(cfg)
    root = tree_storage_root(cfg, tree_storage)
    stages_done: list[str] = []
    counts: dict[str, Any] = {}
    pending = {
        "vectors": vector_pending(db, profile),
        "leaves": int(db.count_tree_nodes(kind="leaf", state="ready")),
        "leaves_missing_parent": int(db.count_leaves_missing_parent()),
    }
    # Publish the first checkpoint before any work: a page that polls one
    # millisecond after the start already sees which stage is running.
    _checkpoint(
        db,
        slot,
        owner,
        stage=STAGES[0],
        stages_done=[],
        pending=pending,
        counts={},
        started_at=started,
    )

    lease_lost = False

    def on_stage(stage: str, payload: dict[str, Any]) -> None:
        nonlocal lease_lost
        stages_done.append(stage)
        counts[f"{stage}_status"] = str(payload.get("status") or "")
        for key in (
            "nodes",
            "embedded",
            "pending",
            "created",
            "frontier_remaining",
            "vector_count",
        ):
            if payload.get(key) is not None:
                counts[f"{stage}_{key}"] = payload[key]
        remaining = [name for name in STAGES if name not in stages_done]
        if not _checkpoint(
            db,
            slot,
            owner,
            stage=remaining[0] if remaining else "",
            stages_done=stages_done,
            pending=pending,
            counts=counts,
            started_at=started,
        ):
            # ``prepare_unified_index`` deliberately swallows callback errors,
            # so the loss is recorded and acted on after the call; the stages
            # are idempotent, and rewriting the winner's job row would be worse.
            lease_lost = True

    try:
        # Stage 0 is the lexical BM25 rebuild; it is not part of the unified
        # prepare call but it is the same job's work.
        lexical = rebuild_lexical(db, rebuild=bool(force), notify=notify)
        stages_done.append("lexical")
        counts["lexical_documents"] = int(lexical.get("documents") or 0)
        if not _checkpoint(
            db,
            slot,
            owner,
            stage=STAGES[1],
            stages_done=stages_done,
            pending=pending,
            counts=counts,
            started_at=started,
        ):
            raise IndexBuildBusyError(slot, state="running", owner="another worker")

        outcome = prepare_unified_index(
            db,
            storage_dir=root,
            profile=profile,
            embed_cfg=embed_section(cfg),
            config=cfg,
            summary_max_tokens=li.summary_max_tokens,
            summary_input_budget=li.summary_input_budget,
            hierarchy_frontier_limit=int(getattr(li, "hierarchy_frontier_limit", 0) or 0),
            hierarchy_summary_workers=int(getattr(li, "hierarchy_summary_workers", 1) or 1),
            force=bool(force),
            on_stage=on_stage,
        )
    except IndexBuildBusyError:
        raise  # the slot belongs to someone else: never rewrite their row
    except Exception as exc:  # noqa: BLE001 - recorded on the job, then re-raised
        reason = safe_error(exc) or type(exc).__name__
        try:
            db.finish_tree_job(slot, "failed", reason=reason[:500])
        except Exception:  # noqa: BLE001 - the original failure is the one to raise
            pass
        raise
    if lease_lost:
        raise IndexBuildBusyError(slot, state="running", owner="another worker")

    payload = outcome.to_json()
    payload["lexical"] = lexical
    payload["job_id"] = slot
    payload["duration_ms"] = round((time.monotonic() - started) * 1000, 3)
    metrics = {
        "ok": bool(outcome.ok),
        "changed": bool(outcome.changed),
        "published": str(outcome.published or ""),
        "failed_stages": list(outcome.failed_stages or []),
        "duration_ms": payload["duration_ms"],
        "stages": {name: str((payload.get(name) or {}).get("status") or "") for name in STAGES},
    }
    reason = ""
    if not outcome.ok:
        failed = ", ".join(str(item) for item in (outcome.failed_stages or [])) or "unknown stage"
        reason = f"failed stages: {failed}"
    try:
        db.checkpoint_tree_job(
            slot,
            checkpoint=json.dumps(
                {
                    "stage": "",
                    "stages_done": list(STAGES),
                    "started_at": float(started),
                    "pending": pending,
                    "counts": counts,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            metrics=json.dumps(metrics, ensure_ascii=False, sort_keys=True),
            owner=owner,
        )
    except Exception:  # noqa: BLE001 - the outcome itself is already recorded in LAST_BUILD_KEY
        pass
    db.finish_tree_job(slot, "done" if outcome.ok else "failed", reason=reason)
    return payload


__all__ = [
    "ACTIVE_STATES",
    "DEFAULT_LEASE_SECONDS",
    "KIND",
    "STAGES",
    "TERMINAL_STATES",
    "IndexBuildBusyError",
    "IndexBuildError",
    "IndexBuildProfileUnavailableError",
    "ensure_index_build_job",
    "index_build_job_id",
    "index_build_scope_key",
    "index_job_payload",
    "index_jobs",
    "job_is_active",
    "lease_expired",
    "live_index_counts",
    "run_index_build",
    "slot_row",
    "tree_storage_root",
    "vector_pending",
    "worker_owner",
]
