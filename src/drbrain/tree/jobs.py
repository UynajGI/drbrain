"""Build-job claims, checkpoints and re-run idempotency (plan T36).

Recovery rests on three properties that already hold elsewhere:

* summaries are cached by members+contract, so an interrupted build never
  re-summarises what it already produced (T26);
* nodes are inserted idempotently with content-addressed identities, so
  re-running a round cannot duplicate a parent (T11);
* vectors are keyed by node revision + content hash + profile, so only new
  summaries are embedded (T23/T24).

This module adds the job ledger around those: a claim with a lease (two
workers cannot both run one job), a checkpoint payload the resumed worker
reads back, and an explicit non-promise: an external LLM call may have
succeeded without its result being recorded, so a retry is allowed — there is
no exactly-once guarantee for network calls.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from loguru import logger


class JobError(RuntimeError):
    """The job ledger rejected the requested transition."""


@dataclass(frozen=True)
class JobClaim:
    job_id: str
    owner: str
    granted: bool
    checkpoint: dict[str, Any]


class TreeJobStore:
    """Thin, explicit wrapper over the ``tree_build_jobs`` table."""

    def __init__(self, db) -> None:
        self.db = db

    def create(self, scope_key: str, *, kind: str = "build") -> str:
        job_id = f"job-{uuid.uuid4().hex[:20]}"
        self.db.insert_tree_job(job_id, scope_key, kind=kind)
        return job_id

    def claim(self, job_id: str, owner: str, *, ttl_seconds: int = 900) -> JobClaim:
        granted = self.db.claim_tree_job(job_id, owner, ttl_seconds=ttl_seconds)
        row = self.db.get_tree_job(job_id)
        if row is None:
            raise JobError(f"unknown job {job_id!r}")
        import json

        try:
            checkpoint = json.loads(row.get("checkpoint_json") or "{}")
        except (TypeError, ValueError):
            checkpoint = {}
            logger.warning("[tree] job {} has an unreadable checkpoint", job_id)
        return JobClaim(job_id=job_id, owner=owner, granted=bool(granted), checkpoint=checkpoint)

    def save_checkpoint(
        self,
        job_id: str,
        *,
        owner: str,
        checkpoint: dict[str, Any],
        metrics: dict[str, Any] | None = None,
    ) -> None:
        import json

        self.db.checkpoint_tree_job(
            job_id,
            checkpoint=json.dumps(checkpoint, ensure_ascii=False, sort_keys=True),
            metrics=json.dumps(metrics, ensure_ascii=False, sort_keys=True) if metrics else None,
            owner=owner,
        )

    def finish(self, job_id: str, *, done: bool, reason: str = "") -> None:
        self.db.finish_tree_job(job_id, "done" if done else "failed", reason=reason)

    def pause(self, job_id: str, *, reason: str = "") -> None:
        self.db.finish_tree_job(job_id, "paused", reason=reason)

    def state(self, job_id: str) -> dict | None:
        return self.db.get_tree_job(job_id)
