"""Canonical per-paper artifact stages and status values.

The paper row describes the lifecycle of the bibliographic record.  This
module describes derived artifacts, which may be built independently and may
legitimately be degraded (for example, a paper can have ``raw`` text while
PageIndex is unavailable).
"""

from __future__ import annotations

from typing import Final

ARTIFACT_STAGES: Final[tuple[str, ...]] = (
    "raw",
    "tree",
    "kg",
    "pageindex",
    "raptor",
    "rag_text",
    "rag_snapshot",
)

ARTIFACT_STATUSES: Final[tuple[str, ...]] = (
    "pending",
    "running",
    "ready",
    "degraded",
    "failed",
    "skipped",
)


def validate_artifact_stage(stage: str) -> str:
    value = str(stage).strip()
    if value not in ARTIFACT_STAGES:
        raise ValueError(f"unknown artifact stage: {stage!r}")
    return value


def validate_artifact_status(status: str) -> str:
    value = str(status).strip()
    if value not in ARTIFACT_STATUSES:
        raise ValueError(f"unknown artifact status: {status!r}")
    return value
