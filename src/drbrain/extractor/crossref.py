"""CrossRef API client for DOI enrichment."""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from typing import Any

CROSSREF_API = "https://api.crossref.org/works"


def _clean_title(title: str) -> str:
    """Normalize title for CrossRef query."""
    title = re.sub(r"[^a-zA-Z0-9\s\-]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def fetch_doi_by_title(
    title: str, email: str | None = None, max_retries: int = 2, retry_delay: float = 1.0
) -> dict[str, Any] | None:
    """Search CrossRef by title and return DOI + metadata if found."""
    clean = _clean_title(title)
    if not clean:
        return None

    url = (
        f"{CROSSREF_API}?query.title={urllib.parse.quote(clean)}&select=DOI,title,year,link&rows=1"
    )
    headers: dict[str, str] = {"Accept": "application/json"}
    if email:
        headers["mailto"] = email

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req)
            data = json.loads(resp.read())
            items = data.get("message", {}).get("items", [])
            if not items:
                return None
            best = items[0]
            # Check similarity — require title starts match or contains key words
            cross_title = best.get("title", [""])[0]
            if _titles_match(clean, _clean_title(cross_title)):
                return {
                    "doi": best.get("DOI"),
                    "title": cross_title,
                    "year": best.get("published-print", {}).get("date-parts", [[None]])[0][0]
                    or best.get("published-online", {}).get("date-parts", [[None]])[0][0],
                }
            return None
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            continue
    return None


def _titles_match(a: str, b: str) -> bool:
    """Check if two cleaned titles are similar enough."""
    a_lower = a.lower()
    b_lower = b.lower()
    # Exact match after cleaning
    if a_lower == b_lower:
        return True
    # One is prefix of the other
    if a_lower.startswith(b_lower) or b_lower.startswith(a_lower):
        return True
    # Shared word overlap (require 70%+ of words in common)
    words_a = set(a_lower.split())
    words_b = set(b_lower.split())
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b)
    min_len = min(len(words_a), len(words_b))
    return overlap / min_len >= 0.7


def fetch_doi_by_doi(
    doi: str, email: str | None = None, max_retries: int = 2, retry_delay: float = 1.0
) -> dict[str, Any] | None:
    """Direct DOI resolution - bypasses title search entirely."""
    if not doi:
        return None

    url = f"{CROSSREF_API}/{urllib.parse.quote(doi)}"
    headers: dict[str, str] = {"Accept": "application/json"}
    if email:
        headers["mailto"] = email

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req)
            data = json.loads(resp.read())
            msg = data.get("message", {})
            doi_val = msg.get("DOI")
            title = msg.get("title", [""])[0]
            year = (
                msg.get("published-print", {}).get("date-parts", [[None]])[0][0]
                or msg.get("published-online", {}).get("date-parts", [[None]])[0][0]
            )
            return {"doi": doi_val, "title": title, "year": year}
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            continue
    return None


def fetch_doi_by_arxiv(
    arxiv_id: str, email: str | None = None, max_retries: int = 2, retry_delay: float = 1.0
) -> dict[str, Any] | None:
    """Look up DOI via CrossRef using arXiv ID with container-title filter."""
    clean_arxiv = re.sub(r"v\d+$", "", arxiv_id).strip()
    if not clean_arxiv:
        return None

    # Use query.bibcode or query.container-title for arXiv
    url = f"{CROSSREF_API}?query.bibliographic_info={urllib.parse.quote(clean_arxiv)}&select=DOI,title,year,arxivid&rows=10"
    headers: dict[str, str] = {"Accept": "application/json"}
    if email:
        headers["mailto"] = email

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req)
            data = json.loads(resp.read())
            items = data.get("message", {}).get("items", [])
            for item in items:
                # Check if this is actually our arXiv paper
                item_arxiv = item.get("arxivid", "")
                if item_arxiv and re.sub(r"v\d+$", "", item_arxiv) == clean_arxiv:
                    doi = item.get("DOI")
                    year = (
                        item.get("published-print", {}).get("date-parts", [[None]])[0][0]
                        or item.get("published-online", {}).get("date-parts", [[None]])[0][0]
                    )
                    title = item.get("title", [""])[0]
                    return {"doi": doi, "title": title, "year": year}
            # Fall back: check DOIs for physical review papers
            for item in items:
                doi = item.get("DOI", "")
                if doi and "10.1103" in doi.lower():
                    year = (
                        item.get("published-print", {}).get("date-parts", [[None]])[0][0]
                        or item.get("published-online", {}).get("date-parts", [[None]])[0][0]
                    )
                    title = item.get("title", [""])[0]
                    return {"doi": doi, "title": title, "year": year}
            return None
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            continue
    return None
