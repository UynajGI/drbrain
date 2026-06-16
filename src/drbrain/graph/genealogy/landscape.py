"""Domain landscape workspace analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from drbrain.storage.database import Database

if TYPE_CHECKING:
    pass

from drbrain.graph.genealogy.paradigm import (
    _format_provenance,
    _get_concept_provenance,
)


def landscape_workspace(
    db: Database,
    workspace_path: str | None = None,
    paper_ids: list[str] | None = None,
) -> dict:
    """Generate a domain landscape: timeline, gaps, debates.

    Returns dict with keys: timeline, gaps, debates.
    """
    result: dict[str, list] = {"timeline": [], "gaps": [], "debates": []}

    # Determine paper IDs
    if paper_ids is None:
        if workspace_path:
            from drbrain.storage.workspace import load_workspace_papers

            try:
                paper_ids = load_workspace_papers(workspace_path)
            except (FileNotFoundError, OSError):
                paper_ids = []
        else:
            return {"error": "No workspace or paper_ids provided"}

    if not paper_ids:
        return result

    # Build timeline: papers ordered by year
    placeholders = ",".join("?" for _ in paper_ids)
    rows = db.conn.execute(
        f"SELECT local_id, title, year FROM papers "
        f"WHERE local_id IN ({placeholders}) AND year IS NOT NULL "
        f"ORDER BY year, local_id",
        paper_ids,
    ).fetchall()

    # Batch-preload concepts per paper (eliminates N+1: was 1 query per paper).
    # Load all matching concepts, then slice top-5 per paper in Python.
    concept_rows = db.conn.execute(
        f"SELECT local_id, label, type FROM concepts WHERE local_id IN ({placeholders})",
        paper_ids,
    ).fetchall()
    paper_concepts: dict[str, list[tuple[str, str]]] = {}
    for local_id, label, ctype in concept_rows:
        paper_concepts.setdefault(local_id, []).append((label, ctype))

    for row in rows:
        concepts = paper_concepts.get(row[0], [])[:5]

        result["timeline"].append(
            {
                "local_id": row[0],
                "title": row[1][:80] if row[1] else "",
                "year": row[2],
                "key_concepts": [{"label": c[0], "type": c[1]} for c in concepts],
            }
        )

    # Detect persistent gaps and debate zones via seed analysis
    try:
        from drbrain.graph.engine import GraphEngine

        graph = GraphEngine()
        graph.load_from_db(db)
        seeds = graph.detect_research_seeds(db)

        for s in seeds:
            seed_type = s.get("type", "")
            if seed_type == "unaddressed_gap":
                concept_label = s.get("concept", "")
                section, node_id, paper_id = _get_concept_provenance(db, concept_label, "Gap")
                provenance = _format_provenance(section, node_id, paper_id)
                result["gaps"].append(
                    {
                        "description": s.get("description", ""),
                        "concept": concept_label,
                        "section": section,
                        "node_id": node_id,
                        "paper_id": paper_id,
                        "provenance": provenance,
                    }
                )
            elif seed_type == "debate_zone":
                concept_label = s.get("concept", "")
                # Look up the debate target's own provenance
                section, node_id, paper_id = _get_concept_provenance(db, concept_label, None)
                provenance = _format_provenance(section, node_id, paper_id)
                result["debates"].append(
                    {
                        "description": s.get("description", ""),
                        "concept": concept_label,
                        "section": section,
                        "node_id": node_id,
                        "paper_id": paper_id,
                        "provenance": provenance,
                    }
                )
    except Exception:
        pass  # Seed detection requires data; skip if insufficient

    return result


def _get_concepts_by_type(db: Database, paper_ids: list[str], ctype: str) -> list[str]:
    """Get distinct concept labels of a given type from paper IDs."""
    if not paper_ids:
        return []
    placeholders = ",".join("?" for _ in paper_ids)
    rows = db.conn.execute(
        f"SELECT DISTINCT label FROM concepts WHERE local_id IN ({placeholders}) AND type = ?",
        (*paper_ids, ctype),
    ).fetchall()
    return [r[0] for r in rows]
