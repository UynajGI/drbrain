"""CrossRef API client for DOI enrichment."""
from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
import json
from typing import Any


CROSSREF_API = "https://api.crossref.org/works"


def _clean_title(title: str) -> str:
    """Normalize title for CrossRef query."""
    title = re.sub(r"[^a-zA-Z0-9\s\-]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def fetch_doi_by_title(title: str, email: str | None = None,
                       max_retries: int = 2, retry_delay: float = 1.0) -> dict[str, Any] | None:
    """Search CrossRef by title and return DOI + metadata if found."""
    clean = _clean_title(title)
    if not clean:
        return None

    url = f"{CROSSREF_API}?query.title={urllib.parse.quote(clean)}&select=DOI,title,year,link&rows=1"
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
