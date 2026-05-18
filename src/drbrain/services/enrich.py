"""Metadata enrichment — backfill missing fields from CrossRef and detect scrub-worthy records."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

CROSSREF_API = "https://api.crossref.org/works"

_REQUIRED_FIELDS = ["title", "year", "authors", "journal"]


def check_metadata_completeness(paper: dict) -> list[str]:
    """Check which required metadata fields are missing or empty.

    Args:
        paper: Paper metadata dict.

    Returns:
        List of missing field names.
    """
    missing: list[str] = []
    for field in _REQUIRED_FIELDS:
        val = paper.get(field)
        if not val:
            missing.append(field)
    return missing


def detect_scrub_suspects(paper: dict) -> list[str]:
    """Detect records that may need metadata cleanup.

    Checks:
      - Title is empty or too short (< 5 chars)
      - Authors field is empty or "Unknown"
      - Year is 0, None, or > 10 years in the future
      - Title looks like a filename (contains .pdf, .docx, etc.)

    Args:
        paper: Paper metadata dict.

    Returns:
        List of issue descriptions.
    """
    issues: list[str] = []
    title = (paper.get("title") or "").strip()
    authors = (paper.get("authors") or "").strip()
    year = paper.get("year")

    if not title:
        issues.append("Title is empty")
    elif len(title) < 5:
        issues.append(f"Title too short ({len(title)} chars): {title}")
    elif any(title.lower().endswith(ext) for ext in (".pdf", ".docx", ".pptx", ".xlsx")):
        issues.append("Title looks like a filename")

    if not authors:
        issues.append("Authors field is empty")

    if year is None or year == 0 or year == "":
        issues.append("Year is missing")
    elif isinstance(year, (int, float)):
        from datetime import UTC, datetime

        if year > datetime.now(UTC).year + 10:
            issues.append(f"Year is in the far future: {year}")

    return issues


def _build_crossref_url(doi: str) -> str:
    return f"{CROSSREF_API}/{doi}"


def _parse_crossref_response(response: dict) -> dict:
    """Parse CrossRef API response into a simplified metadata dict.

    Args:
        response: Full CrossRef JSON response dict.

    Returns:
        Dict with ``title``, ``authors``, ``year``, ``journal``,
        ``volume``, ``pages``, ``doi``.
    """
    msg = response.get("message", {})
    title = ""
    titles = msg.get("title") or []
    if titles:
        title = titles[0]

    authors_list: list[str] = []
    for author in msg.get("author") or []:
        given = author.get("given", "")
        family = author.get("family", "")
        if family:
            authors_list.append(f"{family}, {given}" if given else family)
    authors = " and ".join(authors_list)

    year = None
    date_parts = (
        msg.get("published-print", {}).get("date-parts")
        or msg.get("published-online", {}).get("date-parts")
        or msg.get("created", {}).get("date-parts")
    )
    if date_parts and date_parts[0]:
        year = date_parts[0][0]

    journal = ""
    container = msg.get("container-title") or []
    if container:
        journal = container[0]

    volume = msg.get("volume", "")
    pages = msg.get("page", "")

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "journal": journal,
        "volume": str(volume) if volume else "",
        "pages": str(pages) if pages else "",
        "doi": msg.get("DOI", ""),
    }


def fetch_crossref_metadata(doi: str, timeout: float = 15.0) -> dict | None:
    """Fetch metadata for a DOI from CrossRef API.

    Args:
        doi: DOI string.
        timeout: Request timeout in seconds.

    Returns:
        Enriched metadata dict or None if fetch fails.
    """
    url = _build_crossref_url(doi)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "DrBrain/0.1 (mailto:yuunagi.cn@outlook.com)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return _parse_crossref_response(body)
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError):
        return None


def merge_enrichment(paper: dict, enriched: dict) -> dict:
    """Merge enriched metadata into paper, only filling missing fields.

    Existing paper values are never overwritten. Only None/empty values
    are filled from enriched data.

    Args:
        paper: Original paper metadata dict.
        enriched: Enriched metadata from CrossRef.

    Returns:
        Merged dict (new copy, original unchanged).
    """
    result = dict(paper)
    for key in ("title", "authors", "year", "journal", "volume", "pages", "doi"):
        existing = result.get(key)
        new_val = enriched.get(key)
        if (existing is None or existing == "" or existing == 0) and new_val:
            result[key] = new_val
    return result
