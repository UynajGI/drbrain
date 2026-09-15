"""Reading pre-migration evidence without rewriting the main store (plan T53).

Evidence recorded before the unified store keeps its own identifiers —
``{paper_id}:{local_node_id}`` from the PageIndex era, or a canonical
``nl-...``/``nr-...``/``nb-...`` id.  Readers resolve, in order: the exact
canonical revision when the id is canonical; the read-only legacy adapter for
old ids whose files still exist; a bounded alias lookup when the stored
snippet still matches a canonical leaf (mapping the old id onto today's node);
and finally the evidence row's own stored snippet.

Nothing here writes, and no second retrieval copy is kept for new papers: the
default facts stay in the main store, and compatibility is resolved on demand.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drbrain.storage.node_projection import read_node_text

CANONICAL_NODE_PREFIXES = ("nl-", "nr-", "nb-")


@dataclass(frozen=True)
class EvidenceView:
    evidence_id: str
    paper_id: str
    node_id: str
    source: str  # "canonical" | "legacy" | "alias" | "snippet" | ""
    text: str = ""
    revision: int | None = None
    alias_node_id: str = ""
    warnings: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return bool(self.text)


def _node_revision(db, node_id: str) -> int | None:
    try:
        row = db.conn.execute(
            "SELECT doc_revision FROM tree_nodes WHERE node_id = ?", (str(node_id),)
        ).fetchone()
    except Exception:  # noqa: BLE001 - un-migrated stores have no node table
        return None
    return int(row[0]) if row and row[0] is not None else None


def _alias_by_snippet(db, paper_id: str, snippet: str) -> tuple[str | None, int | None]:
    """Best-effort mapping of an old node id onto a canonical leaf (T53).

    The stored snippet is usually a fragment of the original node text, so the
    lookup is a bounded substring probe over the paper's ready leaves.
    """
    text = str(snippet or "").strip()
    if len(text) < 24:
        return None, None
    needle = text[:160]
    try:
        row = db.conn.execute(
            "SELECT n.node_id, n.doc_revision FROM tree_nodes n "
            "JOIN content_blocks b ON b.block_id = n.block_id "
            "WHERE n.local_id = ? AND n.state = 'ready' AND n.kind = 'leaf' "
            "AND instr(b.text, ?) > 0 ORDER BY b.ordinal LIMIT 1",
            (str(paper_id), needle),
        ).fetchone()
    except Exception:  # noqa: BLE001 - best-effort alias mapping
        return None, None
    if row is None:
        return None, None
    return str(row[0]), int(row[1] or 1)


def read_evidence_text(
    db,
    evidence: dict[str, Any],
    *,
    papers_root: str | Path | None = None,
) -> EvidenceView:
    """Resolve one evidence row to readable text without writing anything."""
    evidence_id = str(evidence.get("evidence_id") or "")
    paper_id = str(evidence.get("paper_id") or "")
    node_id = str(evidence.get("node_id") or "")
    snippet = str(evidence.get("snippet") or "")
    warnings: list[str] = []

    if node_id.startswith(CANONICAL_NODE_PREFIXES):
        text = read_node_text(db.conn, node_id)
        if text:
            return EvidenceView(
                evidence_id,
                paper_id,
                node_id,
                "canonical",
                text,
                _node_revision(db, node_id),
            )
        warnings.append("canonical node has no ready text")

    if papers_root is not None and paper_id and node_id:
        try:
            from drbrain.storage.legacy_content import discover, load, section_text

            refs = discover(papers_root, paper_id)
            if refs:
                document = load(refs[0])
                text = section_text(document, node_id)
                if text:
                    return EvidenceView(evidence_id, paper_id, node_id, "legacy", text, None)
        except Exception as exc:  # noqa: BLE001 - compatibility stays best-effort
            warnings.append(f"legacy read failed: {type(exc).__name__}")

    if paper_id and snippet:
        alias_id, revision = _alias_by_snippet(db, paper_id, snippet)
        if alias_id:
            text = read_node_text(db.conn, alias_id)
            if text:
                return EvidenceView(
                    evidence_id,
                    paper_id,
                    node_id,
                    "alias",
                    text,
                    revision,
                    alias_node_id=alias_id,
                    warnings=tuple(warnings),
                )

    if snippet:
        return EvidenceView(
            evidence_id, paper_id, node_id, "snippet", snippet, None, warnings=tuple(warnings)
        )
    return EvidenceView(
        evidence_id,
        paper_id,
        node_id,
        "",
        "",
        None,
        warnings=tuple(warnings + ["evidence has no readable text"]),
    )


def resolve_evidence(
    db,
    evidence_id: str,
    *,
    papers_root: str | Path | None = None,
) -> EvidenceView | None:
    """Fetch one evidence row by id and resolve its text (``None`` if absent)."""
    row = db.get_evidence(str(evidence_id))
    if row is None:
        return None
    return read_evidence_text(db, row, papers_root=papers_root)


__all__ = [
    "CANONICAL_NODE_PREFIXES",
    "EvidenceView",
    "read_evidence_text",
    "resolve_evidence",
]
