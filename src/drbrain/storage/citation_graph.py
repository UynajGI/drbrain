"""Citation graph analysis: shared references, co-citation, frontier signals."""

from __future__ import annotations

import sqlite3


def find_shared_refs(local_id: str, conn: sqlite3.Connection) -> list[dict]:
    """Find papers that share references with the given paper.

    Returns list of {shared_with, shared_with_title, shared_count, status, shared_papers}.
    status is 'unlinked' when no direct citation edge exists between the papers.
    """
    rows = conn.execute(
        "SELECT cc2.source_paper, p.title, COUNT(*) as shared_count "
        "FROM citation_cache cc1 "
        "JOIN citation_cache cc2 ON cc1.target_title = cc2.target_title "
        "  AND cc1.source_paper != cc2.source_paper "
        "JOIN papers p ON p.local_id = cc2.source_paper "
        "WHERE cc1.source_paper = ? "
        "  AND cc1.relation = 'references' "
        "  AND cc2.relation = 'references' "
        "GROUP BY cc2.source_paper "
        "ORDER BY shared_count DESC",
        (local_id,),
    ).fetchall()

    results = []
    for shared_with, title, count in rows:
        direct = conn.execute(
            "SELECT COUNT(*) FROM edges "
            "WHERE (src_id = ? AND dst_id = ? AND relation = 'references') "
            "   OR (src_id = ? AND dst_id = ? AND relation = 'references')",
            (local_id, shared_with, shared_with, local_id),
        ).fetchone()[0]

        shared_papers = conn.execute(
            "SELECT cc1.target_title, cc1.target_year "
            "FROM citation_cache cc1 "
            "JOIN citation_cache cc2 ON cc1.target_title = cc2.target_title "
            "  AND cc2.source_paper = ? AND cc2.relation = 'references' "
            "WHERE cc1.source_paper = ? AND cc1.relation = 'references'",
            (shared_with, local_id),
        ).fetchall()

        results.append(
            {
                "shared_with": shared_with,
                "shared_with_title": title,
                "shared_count": count,
                "status": "unlinked" if direct == 0 else "linked",
                "shared_papers": [{"title": sp[0], "year": sp[1]} for sp in shared_papers],
            }
        )

    return results


def get_citation_counts(local_id: str, conn: sqlite3.Connection) -> dict:
    """Return reference and citation counts for a paper."""
    refs = conn.execute(
        "SELECT COUNT(*) FROM citation_cache WHERE source_paper = ? AND relation = 'references'",
        (local_id,),
    ).fetchone()[0]

    citing = conn.execute(
        "SELECT COUNT(*) FROM citation_cache WHERE source_paper = ? AND relation = 'citing'",
        (local_id,),
    ).fetchone()[0]

    return {"references": refs, "citing": citing}


def query_citation_graph(
    local_id: str,
    conn: sqlite3.Connection,
    ctype: str = "all",
) -> dict:
    """Query citation graph for a paper.

    Args:
        local_id: Paper ID.
        conn: SQLite connection.
        ctype: One of 'refs', 'citing', 'shared-refs', 'all'.

    Returns:
        Dict with keys: paper, refs[], citing[], shared_refs[].
    """
    paper = conn.execute(
        "SELECT local_id, title, year FROM papers WHERE local_id = ?",
        (local_id,),
    ).fetchone()

    if not paper:
        return {"paper": None, "refs": [], "citing": [], "shared_refs": []}

    result: dict = {
        "paper": {"local_id": paper[0], "title": paper[1], "year": paper[2]},
        "refs": [],
        "citing": [],
        "shared_refs": [],
    }

    if ctype in ("refs", "all"):
        refs = conn.execute(
            "SELECT target_title, target_year, target_doi FROM citation_cache "
            "WHERE source_paper = ? AND relation = 'references' "
            "ORDER BY target_year DESC",
            (local_id,),
        ).fetchall()
        result["refs"] = [{"title": r[0], "year": r[1], "doi": r[2]} for r in refs]

    if ctype in ("citing", "all"):
        citing = conn.execute(
            "SELECT target_title, target_year, target_doi FROM citation_cache "
            "WHERE source_paper = ? AND relation = 'citing' "
            "ORDER BY target_year DESC",
            (local_id,),
        ).fetchall()
        result["citing"] = [{"title": c[0], "year": c[1], "doi": c[2]} for c in citing]

    if ctype in ("shared-refs", "all"):
        result["shared_refs"] = find_shared_refs(local_id, conn)

    counts = get_citation_counts(local_id, conn)
    result["counts"] = counts

    return result
