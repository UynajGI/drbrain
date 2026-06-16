"""Paradigm shift detection, difficulty analysis, frontier analysis."""

from __future__ import annotations

import time as _time
from typing import TYPE_CHECKING

from loguru import logger

from drbrain.graph.engine import GraphEngine
from drbrain.storage.database import Database

if TYPE_CHECKING:
    pass


def detect_paradigm_shifts(
    graph: GraphEngine,
    db: Database,
    concept: str | None = None,
    paper_ids: list[str] | None = None,
    decline_threshold: float = 0.5,
    growth_threshold: int = 3,
    explosion_threshold: int = 8,
    descendant_threshold: int = 3,
    cascade_threshold: int = 1,
) -> list[dict]:
    """Detect paradigm shifts: replacement, explosion, cross-domain.

    Returns list of shift dicts, each with: type, description, concepts involved.

    Args:
        concept: Optional concept label to filter explosion detection.
        paper_ids: Optional list of paper IDs to scope detection to a workspace.
        decline_threshold: Fraction decline to flag replacement (default 0.5 = 50%).
        growth_threshold: Min new-method papers for replacement.
        explosion_threshold: Min total papers for explosion detection.
        descendant_threshold: Min descendant concepts for explosion (PRD: 3+).
        cascade_threshold: Min cascaded concepts for cross-domain detection.
    """
    _t0_genealogy = _time.monotonic()
    logger.info(
        "[genealogy] paradigm shift detection — concept=%s papers=%d",
        concept,
        len(paper_ids) if paper_ids else 0,
    )
    results: list[dict] = []

    # Type 1: Replacement -- find challenges edges where old is declining
    challenge_edges = db.conn.execute(
        "SELECT src_id, dst_id, weight FROM edges WHERE relation = 'challenges'"
    ).fetchall()

    # Batch query: year counts for all challenge-edge concepts at once
    challenge_concepts: set[str] = set()
    for src_id, dst_id, _ in challenge_edges:
        challenge_concepts.add(src_id)
        challenge_concepts.add(dst_id)

    year_map: dict[str, list[tuple[int, int]]] = {}
    if challenge_concepts:
        placeholders = ",".join("?" for _ in challenge_concepts)
        year_rows = db.conn.execute(
            f"SELECT c.label, p.year, COUNT(*) FROM concepts c "
            f"JOIN papers p ON c.local_id = p.local_id "
            f"WHERE c.label IN ({placeholders}) AND p.year IS NOT NULL "
            f"GROUP BY c.label, p.year ORDER BY c.label, p.year",
            tuple(challenge_concepts),
        ).fetchall()
        for label, year, cnt in year_rows:
            year_map.setdefault(label, []).append((year, cnt))

    for src_id, dst_id, conf in challenge_edges:
        old_years = year_map.get(dst_id, [])
        new_years = year_map.get(src_id, [])

        if len(old_years) >= 2 and len(new_years) >= 2:
            max_old_year = max(y for y, _ in old_years)
            old_recent = sum(c for y, c in old_years if y >= max_old_year - 2)
            old_old = sum(c for y, c in old_years if y < max_old_year - 2)

            if old_old > 0 and old_recent / old_old <= (1 - decline_threshold):
                new_total = sum(c for _, c in new_years)
                if new_total >= growth_threshold:
                    old_sec, old_nid, old_pid = _get_concept_provenance(db, dst_id)
                    new_sec, new_nid, new_pid = _get_concept_provenance(db, src_id)
                    results.append(
                        {
                            "type": "replacement",
                            "old_concept": dst_id,
                            "new_concept": src_id,
                            "old_provenance": _format_provenance(old_sec, old_nid, old_pid),
                            "new_provenance": _format_provenance(new_sec, new_nid, new_pid),
                            "description": f"{src_id} is replacing {dst_id} (decline: {old_recent}/{old_old}, new: {new_total})",
                            "confidence": conf,
                        }
                    )

    # Type 2: Explosion -- concept with rapid growth + descendants
    if concept:
        concept_labels = [concept]
    elif paper_ids:
        # Filter concepts to only those appearing in workspace papers
        placeholders = ",".join("?" for _ in paper_ids)
        concept_labels = [
            r[0]
            for r in db.conn.execute(
                f"SELECT DISTINCT c.label FROM concepts c "
                f"WHERE c.local_id IN ({placeholders}) AND c.type IN ('Method', 'Problem')",
                paper_ids,
            ).fetchall()
        ]
    else:
        concept_labels = [
            r[0]
            for r in db.conn.execute(
                "SELECT DISTINCT label FROM concepts WHERE type IN ('Method', 'Problem')"
            ).fetchall()
        ]

    # Batch-load year counts for all candidate labels (avoids N+1 per-label queries)
    explosion_year_map: dict[str, list[tuple[int, int]]] = {}
    if concept_labels:
        ph = ",".join("?" for _ in concept_labels)
        explosion_rows = db.conn.execute(
            f"SELECT c.label, p.year, COUNT(*) FROM concepts c "
            f"JOIN papers p ON c.local_id = p.local_id "
            f"WHERE c.label IN ({ph}) AND p.year IS NOT NULL "
            f"GROUP BY c.label, p.year",
            tuple(concept_labels),
        ).fetchall()
        for label, year, cnt in explosion_rows:
            explosion_year_map.setdefault(label, []).append((year, cnt))

    for clabel in concept_labels:
        year_counts = explosion_year_map.get(clabel, [])

        total = sum(c for _, c in year_counts)
        if total >= explosion_threshold and len(year_counts) <= 2:
            # Check for descendant concepts via graph
            descendants = []
            if clabel in graph.graph:
                for n in graph.graph.neighbors(clabel):
                    for edge_data in graph.graph[clabel][n].values():
                        rel = edge_data.get("relation", "")
                        if rel in ("extends", "refines", "applies"):
                            descendants.append(n)

            if len(descendants) >= descendant_threshold:
                section, node_id, paper_id = _get_concept_provenance(db, clabel)
                results.append(
                    {
                        "type": "explosion",
                        "concept": clabel,
                        "paper_count": total,
                        "descendants": descendants[:10],
                        "section": section,
                        "node_id": node_id,
                        "paper_id": paper_id,
                        "provenance": _format_provenance(section, node_id, paper_id),
                        "description": f"{clabel} exploded to {total} papers with {len(descendants)} descendant concepts",
                    }
                )

    # Type 3: Cross-domain invasion -- applies edges with cascading
    applies_edges = db.conn.execute(
        "SELECT src_id, dst_id, weight FROM edges WHERE relation = 'applies'"
    ).fetchall()

    for src_id, dst_id, conf in applies_edges:
        # Check if dst has further descendants (cascade in new domain)
        cascade = []
        visited = {src_id, dst_id}
        queue = [dst_id]
        while queue and len(cascade) <= 5:
            node = queue.pop(0)
            if node not in graph.graph:
                continue
            for n in graph.graph.neighbors(node):
                if n in visited:
                    continue
                visited.add(n)
                for edge_data in graph.graph[node][n].values():
                    rel = edge_data.get("relation", "")
                    if rel in ("extends", "refines", "applies"):
                        cascade.append(n)
                        queue.append(n)

        if len(cascade) >= cascade_threshold:
            src_sec, src_nid, src_pid = _get_concept_provenance(db, src_id)
            results.append(
                {
                    "type": "cross_domain",
                    "source_concept": src_id,
                    "target_concept": dst_id,
                    "cascade": cascade[:10],
                    "source_provenance": _format_provenance(src_sec, src_nid, src_pid),
                    "description": f"{src_id} crossed domains via {dst_id}, spawned {len(cascade)} concepts",
                    "confidence": conf,
                }
            )

    return results


def _get_concept_provenance(
    db: Database,
    label: str,
    ctype: str | None = None,
) -> tuple[str, str, str]:
    """Look up (section, node_id, paper_id) for a concept by label+type.

    When ctype is None, matches any type (preferring higher confidence).
    Returns the highest-confidence match. Returns ("", "", "") if not found.
    """
    if ctype:
        row = db.conn.execute(
            "SELECT section, node_id, local_id FROM concepts "
            "WHERE LOWER(label) = LOWER(?) AND type = ? "
            "ORDER BY confidence DESC LIMIT 1",
            (label, ctype),
        ).fetchone()
    else:
        row = db.conn.execute(
            "SELECT section, node_id, local_id FROM concepts "
            "WHERE LOWER(label) = LOWER(?) "
            "ORDER BY confidence DESC LIMIT 1",
            (label,),
        ).fetchone()
    if row is None:
        return "", "", ""
    return row[0] or "", row[1] or "", row[2] or ""


def _format_provenance(section: str, node_id: str, paper_id: str) -> str:
    """Format provenance fields into a human-readable string.

    Returns '[source: <section> of <paper_id>]' or '[source: unknown]'.
    """
    if section and paper_id:
        return f"[source: {section} of {paper_id}]"
    if section:
        return f"[source: {section}]"
    if paper_id:
        return f"[source: {paper_id}]"
    return "[source: unknown]"


def analyze_difficulty(db: Database) -> dict:
    """Classify gaps by source section semantics to build a difficulty map.

    Groups Gap concepts into:
      - limitation: from sections with "limitation"/"weakness"/"shortcoming"
      - future_work: from sections with "future"/"direction"/"open problem"
      - discussion: from sections with "discussion"/"conclusion"
      - uncategorized: everything else

    Returns dict with keys: limitation, future_work, discussion, uncategorized.
    Each value is a list of {label, section, paper_id, provenance}.
    """
    result: dict[str, list[dict]] = {
        "limitation": [],
        "future_work": [],
        "discussion": [],
        "uncategorized": [],
    }

    rows = db.conn.execute(
        "SELECT label, section, node_id, local_id FROM concepts WHERE type = 'Gap'"
    ).fetchall()

    for label, section, node_id, paper_id in rows:
        section_lower = (section or "").lower()
        if any(kw in section_lower for kw in ("limitation", "weakness", "shortcoming")):
            cat = "limitation"
        elif any(
            kw in section_lower for kw in ("future", "direction", "open problem", "open question")
        ):
            cat = "future_work"
        elif any(kw in section_lower for kw in ("discussion", "conclusion")):
            cat = "discussion"
        else:
            cat = "uncategorized"

        result[cat].append(
            {
                "label": label,
                "section": section or "",
                "node_id": node_id or "",
                "paper_id": paper_id or "",
                "provenance": _format_provenance(section or "", node_id or "", paper_id or ""),
            }
        )

    return result


def analyze_frontier(db: Database) -> dict:
    """Composite knowledge frontier: gaps, debates, recency, paradigm shifts.

    Synthesizes existing analysis functions into a single frontier report.
    Uses gap classification, debate detection, paper recency to identify
    active research fronts.

    Returns dict with keys: active_gaps, debates, paradigm_shifts, summary.
    """
    from drbrain.graph.engine import GraphEngine

    graph = GraphEngine()
    graph.load_from_db(db)

    # 1. Gap recency — group gaps by paper year
    gap_rows = db.conn.execute(
        "SELECT c.label, c.section, c.node_id, c.local_id, p.year "
        "FROM concepts c JOIN papers p ON c.local_id = p.local_id "
        "WHERE c.type = 'Gap' AND p.year IS NOT NULL "
        "ORDER BY p.year DESC"
    ).fetchall()

    # 2. Difficulty classification
    difficulty = analyze_difficulty(db)

    # 3. Active debates
    seeds = graph.detect_research_seeds(db)
    debates = [s for s in seeds if s["type"] == "debate_zone"]

    # 4. Paradigm shifts (top-level concepts only)
    shifts = detect_paradigm_shifts(graph, db, decline_threshold=0.3, growth_threshold=2)

    # Compute recency-aware gaps: group by recency bucket
    recent_year = max((r[4] or 0) for r in gap_rows) if gap_rows else 0
    active_gaps: list[dict] = []
    stale_gaps: list[dict] = []
    for label, section, node_id, paper_id, year in gap_rows:
        gap_info = {
            "label": label,
            "section": section or "",
            "paper_id": paper_id or "",
            "year": year or 0,
            "provenance": _format_provenance(section or "", node_id or "", paper_id or ""),
        }
        if year and year >= recent_year - 3:
            active_gaps.append(gap_info)
        else:
            stale_gaps.append(gap_info)

    summary_parts = []
    if active_gaps:
        summary_parts.append(f"{len(active_gaps)} active gaps (last 3 years)")
    if stale_gaps:
        summary_parts.append(f"{len(stale_gaps)} stale gaps (older)")
    if debates:
        summary_parts.append(f"{len(debates)} active debates")
    summary_parts.append(
        f"limitation={len(difficulty['limitation'])}, "
        f"future_work={len(difficulty['future_work'])}, "
        f"discussion={len(difficulty['discussion'])}"
    )

    return {
        "active_gaps": active_gaps,
        "stale_gaps": stale_gaps,
        "difficulty": difficulty,
        "debates": debates,
        "paradigm_shifts": shifts,
        "summary": ", ".join(summary_parts),
    }
