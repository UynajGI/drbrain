"""OpenAlex API client for DOI enrichment and citation expansion."""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from typing import Any

OPENALEX_BASE = "https://api.openalex.org"


def _select_fields(fields: list[str]) -> str:
    """Build OpenAlex select parameter."""
    return ",".join(fields)


def search_work_by_title(
    title: str, token: str | None = None, max_retries: int = 2, retry_delay: float = 1.0
) -> dict[str, Any] | None:
    """Search OpenAlex by title and return DOI + metadata if found."""
    if not title:
        return None

    fields = _select_fields(["id", "doi", "title", "publication_year", "ids"])
    url = f"{OPENALEX_BASE}/works?search={urllib.parse.quote(title)}&per_page=1&select={fields}"
    headers: dict[str, str] = {"Accept": "application/json"}
    if token:
        headers.setdefault("User-Agent", f"DrBrain (mailto:{token})")

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req)
            data = json.loads(resp.read())
            results = data.get("results", [])
            if not results:
                return None
            best = results[0]
            doi = best.get("doi", "")
            if doi:
                doi = re.sub(r"^https?://doi\.org/", "", doi)
            return {
                "doi": doi or None,
                "title": best.get("title", ""),
                "year": best.get("publication_year"),
                "openalex_id": best.get("id", ""),
            }
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            continue
    return None


def search_work_by_arxiv(
    arxiv_id: str, token: str | None = None, max_retries: int = 2, retry_delay: float = 1.0
) -> dict[str, Any] | None:
    """Search OpenAlex by arXiv ID."""
    clean = re.sub(r"v\d+$", "", arxiv_id).strip()
    if not clean:
        return None

    fields = _select_fields(["id", "doi", "title", "publication_year", "ids"])
    url = f"{OPENALEX_BASE}/works?filter=arxiv_id:{urllib.parse.quote(clean)}&per_page=1&select={fields}"
    headers: dict[str, str] = {"Accept": "application/json"}

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req)
            data = json.loads(resp.read())
            results = data.get("results", [])
            if not results:
                return None
            best = results[0]
            doi = best.get("doi", "")
            if doi:
                doi = re.sub(r"^https?://doi\.org/", "", doi)
            return {
                "doi": doi or None,
                "title": best.get("title", ""),
                "year": best.get("publication_year"),
                "openalex_id": best.get("id", ""),
            }
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            continue
    return None


def get_work_by_doi(
    doi: str, token: str | None = None, max_retries: int = 2, retry_delay: float = 1.0
) -> dict[str, Any] | None:
    """Fetch work by DOI from OpenAlex."""
    if not doi:
        return None

    clean_doi = re.sub(r"^https?://doi\.org/", "", doi)
    fields = _select_fields(
        ["id", "doi", "title", "publication_year", "ids", "referenced_works", "cited_by_api_url"]
    )
    url = f"{OPENALEX_BASE}/works/doi:{urllib.parse.quote(clean_doi)}?select={fields}"
    headers: dict[str, str] = {"Accept": "application/json"}

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req)
            data = json.loads(resp.read())
            if not data or "error" in data:
                return None
            doi_val = data.get("doi", "")
            if doi_val:
                doi_val = re.sub(r"^https?://doi\.org/", "", doi_val)
            return {
                "doi": doi_val or None,
                "title": data.get("title", ""),
                "year": data.get("publication_year"),
                "openalex_id": data.get("id", ""),
                "referenced_works": data.get("referenced_works", []),
            }
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            continue
    return None


def get_work_references(
    openalex_id: str, token: str | None = None, max_retries: int = 2, retry_delay: float = 1.0
) -> list[dict[str, Any]]:
    """Fetch referenced works from OpenAlex. Returns list of basic paper info."""
    if not openalex_id:
        return []

    refs: list[dict[str, Any]] = []

    for attempt in range(max_retries):
        try:
            headers: dict[str, str] = {"Accept": "application/json"}
            req = urllib.request.Request(openalex_id, headers=headers)
            resp = urllib.request.urlopen(req)
            data = json.loads(resp.read())
            ref_ids = data.get("referenced_works", [])
            for ref_id in ref_ids[:50]:
                ref_info = get_work_by_openalex_id(ref_id, token=token)
                if ref_info:
                    refs.append(ref_info)
            return refs
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            continue
    return []


def get_work_by_openalex_id(
    openalex_id: str, token: str | None = None, max_retries: int = 2, retry_delay: float = 0.5
) -> dict[str, Any] | None:
    """Fetch a single work by its OpenAlex ID."""
    fields = _select_fields(["id", "doi", "title", "publication_year", "ids"])
    url = f"{openalex_id}?select={fields}"
    headers: dict[str, str] = {"Accept": "application/json"}

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req)
            data = json.loads(resp.read())
            if not data:
                return None
            doi = data.get("doi", "")
            if doi:
                doi = re.sub(r"^https?://doi\.org/", "", doi)
            return {
                "doi": doi or None,
                "title": data.get("title", ""),
                "year": data.get("publication_year"),
                "openalex_id": data.get("id", ""),
            }
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            continue
    return None


def _extract_author_short_id(url: str) -> str | None:
    """Extract short author ID from OpenAlex URL like 'https://openalex.org/A5023806754'."""
    if not url:
        return None
    m = re.search(r"/(A\d{10})$", url)
    return m.group(1) if m else None


def search_authors_by_work(
    doi: str | None = None,
    title: str | None = None,
    token: str | None = None,
) -> list[dict[str, Any]] | None:
    """Fetch authorships from OpenAlex for a given work.

    Prefer DOI lookup for accuracy; fallback to title search.
    Returns list of dicts with: author_id (short), display_name, orcid, raw_affiliation.
    Returns None if the work cannot be found.
    """
    if not doi and not title:
        return None

    fields = _select_fields(["id", "authorships"])

    work: dict[str, Any] | None = None

    # Try DOI first
    if doi:
        clean_doi = re.sub(r"^https?://doi\.org/", "", doi)
        url = f"{OPENALEX_BASE}/works/doi:{urllib.parse.quote(clean_doi)}?select={fields}"
        headers: dict[str, str] = {"Accept": "application/json"}
        if token:
            headers["User-Agent"] = f"DrBrain (mailto:{token})"
        try:
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            if data and "error" not in data:
                work = data
        except Exception:
            pass

    # Fallback to title search
    if work is None and title:
        url = f"{OPENALEX_BASE}/works?search={urllib.parse.quote(title)}&per_page=1&select={fields}"
        headers = {"Accept": "application/json"}
        if token:
            headers["User-Agent"] = f"DrBrain (mailto:{token})"
        try:
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            results = data.get("results", [])
            if results:
                work = results[0]
        except Exception:
            pass

    if work is None:
        return None

    authorships = work.get("authorships") or []
    authors: list[dict[str, Any]] = []
    for ship in authorships:
        author = ship.get("author")
        if not author:
            continue
        author_id = _extract_author_short_id(author.get("id", ""))
        if not author_id:
            continue
        authors.append(
            {
                "author_id": author_id,
                "display_name": author.get("display_name", ""),
                "orcid": author.get("orcid"),
                "raw_affiliation": ship.get("raw_affiliation_strings") or [],
            }
        )

    return authors if authors else None


def batch_fetch_works(
    work_ids: list[str], token: str | None = None, max_retries: int = 2, retry_delay: float = 0.5
) -> list[dict[str, Any]]:
    """Batch fetch multiple works from OpenAlex using the bulk endpoint."""
    if not work_ids:
        return []

    fields = _select_fields(["id", "doi", "title", "publication_year", "ids"])
    # Use filter endpoint with multiple IDs
    id_filter = "|".join(work_ids[:50])
    url = f"{OPENALEX_BASE}/works?filter=openalex_id:{id_filter}&per_page=50&select={fields}"
    headers: dict[str, str] = {"Accept": "application/json"}

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req)
            data = json.loads(resp.read())
            results = data.get("results", [])
            works = []
            for r in results:
                doi = r.get("doi", "")
                if doi:
                    doi = re.sub(r"^https?://doi\.org/", "", doi)
                works.append(
                    {
                        "doi": doi or None,
                        "title": r.get("title", ""),
                        "year": r.get("publication_year"),
                        "openalex_id": r.get("id", ""),
                    }
                )
            return works
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            continue
    return []
