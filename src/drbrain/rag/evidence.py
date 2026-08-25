"""Shared, fail-closed evidence contract for RAG answer surfaces.

An answer may only be emitted when it can be tied to at least one stable
``paper_id`` and/or ``node_id`` that came from an allowed retrieval path.  The
same identifier shape is persisted by :meth:`Database.record_answer`, making
the runtime gate and audit trail agree.
"""

from __future__ import annotations

import hashlib
import json
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


def build_evidence_record(
    *,
    generation: str,
    query: str,
    retriever: str,
    rank: int,
    score: float,
    source: Mapping[str, Any],
    filters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build additive, stable provenance fields for one retrieved passage.

    ``source`` remains caller-owned and is never mutated.  The identifier binds
    a logical index snapshot, the document/chunk locator, and the exact content
    returned to the model; a published later generation consequently cannot
    masquerade as evidence from an in-flight run.
    """
    paper_id = str(source.get("paper_id") or "").strip()
    node_id = str(source.get("node_id") or "").strip()
    title = str(source.get("title") or "").strip()
    text = str(source.get("text") or "")
    resolved_generation = str(generation or "legacy").strip() or "legacy"
    content_checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
    document_locator = {"paper_id": paper_id, "title": title}
    chunk_locator = {"node_id": node_id}
    identity = {
        "generation": resolved_generation,
        "document_locator": document_locator,
        "chunk_locator": chunk_locator,
        "content_checksum": content_checksum,
    }
    canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "evidence_id": "ev-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24],
        "generation": resolved_generation,
        "document_locator": document_locator,
        "chunk_locator": chunk_locator,
        "content_checksum": content_checksum,
        "query": str(query),
        "filters": dict(filters or {}),
        "retriever": str(retriever),
        "rank": max(1, int(rank)),
        "score": float(score),
    }
