"""CrossRef API client for DOI enrichment."""

from __future__ import annotations

import re
import urllib.parse
from typing import Any

import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

CROSSREF_API = "https://api.crossref.org/works"


def _http_session() -> requests.Session:
    """Create a requests Session with retry/backoff for academic APIs."""
    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


_CROSSREF_SESSION: requests.Session | None = None


def _get_session() -> requests.Session:
    global _CROSSREF_SESSION
    if _CROSSREF_SESSION is None:
        _CROSSREF_SESSION = _http_session()
    return _CROSSREF_SESSION


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

    try:
        resp = _get_session().get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
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
    except requests.RequestException:
        logger.exception("CrossRef API error")
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

    try:
        resp = _get_session().get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message", {})
        doi_val = msg.get("DOI")
        title = msg.get("title", [""])[0]
        year = (
            msg.get("published-print", {}).get("date-parts", [[None]])[0][0]
            or msg.get("published-online", {}).get("date-parts", [[None]])[0][0]
        )
        return {"doi": doi_val, "title": title, "year": year}
    except requests.RequestException:
        logger.exception("CrossRef API error")
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

    try:
        resp = _get_session().get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
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
    except requests.RequestException:
        logger.exception("CrossRef API error")
        return None
