"""Federated search — query local library + external sources (arXiv)."""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
from typing import Any

ARXIV_API_URL = "http://export.arxiv.org/api/query"


def _normalize_arxiv_ref(arxiv_id: str | None) -> str:
    """Normalize an arXiv ID for dedup matching.

    Strips "arXiv:" prefix and version suffix (v1, v2, etc.).
    """
    if not arxiv_id:
        return ""
    aid = str(arxiv_id).strip()
    aid = re.sub(r"^arxiv:", "", aid, flags=re.IGNORECASE)
    aid = re.sub(r"v\d+$", "", aid)
    return aid.strip()


def _build_arxiv_query_url(query: str, max_results: int = 10) -> str:
    """Build the arXiv Atom API URL for a search query."""
    encoded = urllib.parse.quote(query)
    return f"{ARXIV_API_URL}?search_query=all:{encoded}&start=0&max_results={max_results}"


def search_arxiv(query: str, max_results: int = 10) -> list[dict]:
    """Search arXiv Atom API and return simplified paper dicts.

    Args:
        query: Search query string.
        max_results: Maximum results (default 10).

    Returns:
        List of dicts with keys: ``title``, ``authors`` (list), ``year``,
        ``doi``, ``arxiv_id``, ``summary``, ``published``.
    """
    url = _build_arxiv_query_url(query, max_results)
    req = urllib.request.Request(url, headers={"User-Agent": "DrBrain/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            from defusedxml import ElementTree as DefusedXML

            tree = DefusedXML.parse(resp)
    except Exception:
        return []

    ns = "{http://www.w3.org/2005/Atom}"
    results: list[dict] = []
    for entry in tree.iter(f"{ns}entry"):
        title_el = entry.find(f"{ns}title")
        title = title_el.text.strip() if title_el is not None and title_el.text else ""

        authors: list[str] = []
        for author in entry.iter(f"{ns}author"):
            name_el = author.find(f"{ns}name")
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        summary_el = entry.find(f"{ns}summary")
        summary = summary_el.text.strip() if summary_el is not None and summary_el.text else ""

        published_el = entry.find(f"{ns}published")
        published = (
            published_el.text.strip() if published_el is not None and published_el.text else ""
        )
        year = int(published[:4]) if len(published) >= 4 else None

        arxiv_id = ""
        doi = ""
        for link in entry.iter(f"{ns}link"):
            href = link.attrib.get("href", "")
            if "arxiv.org/abs/" in href:
                arxiv_id = href.split("/abs/")[-1]
            if "doi.org" in href:
                doi = href.split("doi.org/")[-1]

        results.append(
            {
                "title": title,
                "authors": authors,
                "year": year,
                "doi": doi,
                "arxiv_id": arxiv_id,
                "summary": summary,
                "published": published,
            }
        )

    return results


def _merge_with_local_status(
    arxiv_results: list[dict],
    in_lib_dois: set[str],
    in_lib_arxiv_ids: set[str],
) -> list[dict]:
    """Annotate arXiv results with ``ingested`` flag based on local library matches.

    Args:
        arxiv_results: Results from ``search_arxiv()``.
        in_lib_dois: Set of lowercased DOIs present in local library.
        in_lib_arxiv_ids: Set of normalized arXiv IDs in local library.

    Returns:
        Same results with ``ingested: bool`` key added.
    """
    merged: list[dict] = []
    for r in arxiv_results:
        doi_lower = (r.get("doi") or "").lower()
        normalized_arxiv = _normalize_arxiv_ref(r.get("arxiv_id", ""))
        ingested = bool(
            (doi_lower and doi_lower in in_lib_dois)
            or (normalized_arxiv and normalized_arxiv in in_lib_arxiv_ids)
        )
        merged.append({**r, "ingested": ingested})
    return merged


def search_local(
    db_path: str,
    query: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search the local DrBrain library via BM25 over concepts/arguments.

    Args:
        db_path: Path to drbrain.db.
        query: Search query string.
        limit: Max results.

    Returns:
        List of paper dicts with ``title``, ``local_id``, ``year``,
        ``authors``, ``score``.
    """
    from drbrain.storage.connection import connect_wal

    if not __import__("pathlib").Path(db_path).exists():
        return []

    try:
        conn = connect_wal(db_path)
        # Simple BM25-like search over concepts and arguments
        search_term = f"%{query}%"
        rows = conn.execute(
            """SELECT DISTINCT p.local_id, p.title, p.year, p.authors
               FROM papers p
               LEFT JOIN concepts c ON c.local_id = p.local_id
               LEFT JOIN arguments a ON a.local_id = p.local_id
               WHERE p.title LIKE ?
                  OR c.label LIKE ?
                  OR a.label LIKE ?
               ORDER BY p.year DESC
               LIMIT ?""",
            (search_term, search_term, search_term, limit),
        ).fetchall()
        conn.close()

        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "local_id": row[0],
                    "title": row[1] or "Untitled",
                    "year": row[2],
                    "authors": row[3] or "",
                    "score": 1.0,
                }
            )
        return results
    except Exception:
        return []
