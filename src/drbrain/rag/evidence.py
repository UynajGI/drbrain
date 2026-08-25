"""Shared, fail-closed evidence contract for RAG answer surfaces.

An answer may only be emitted when it can be tied to at least one stable
``paper_id`` and/or ``node_id`` that came from an allowed retrieval path.  The
same identifier shape is persisted by :meth:`Database.record_answer`, making
the runtime gate and audit trail agree.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

INSUFFICIENT_EVIDENCE_STATUS = "insufficient_evidence"
INSUFFICIENT_EVIDENCE_MESSAGE = "证据不足，无法基于当前检索结果回答"


def evidence_ids_from_records(records: Iterable[Mapping[str, Any]] | None) -> list[str]:
    """Return unique stable evidence ids from retrieved-record metadata.

    A legacy index may contain either ``paper_id`` or ``node_id`` but not both,
    so either is accepted.  Arbitrary answer text never qualifies as evidence.
    """
    ids: list[str] = []
    for record in records or ():
        paper_id = str(record.get("paper_id") or "").strip()
        node_id = str(record.get("node_id") or "").strip()
        if paper_id or node_id:
            ids.append(f"{paper_id}:{node_id}" if paper_id and node_id else (paper_id or node_id))
    return list(dict.fromkeys(ids))


def has_retrieved_evidence(records: Iterable[Mapping[str, Any]] | None) -> bool:
    """Whether retrieval produced at least one auditable evidence identifier."""
    return bool(evidence_ids_from_records(records))
