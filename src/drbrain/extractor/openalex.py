"""OpenAlex API client for DOI enrichment and citation expansion."""

from __future__ import annotations

import re
import urllib.parse
from typing import Any

import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

OPENALEX_BASE = "https://api.openalex.org"


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


_OPENALEX_SESSION: requests.Session | None = None


def _get_session() -> requests.Session:
    global _OPENALEX_SESSION
    if _OPENALEX_SESSION is None:
        _OPENALEX_SESSION = _http_session()
    return _OPENALEX_SESSION


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

    try:
        resp = _get_session().get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
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
    except requests.RequestException:
        logger.exception("OpenAlex API error")
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

    try:
        resp = _get_session().get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
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
    except requests.RequestException:
        logger.exception("OpenAlex API error")
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

    try:
        resp = _get_session().get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
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
    except requests.RequestException:
        logger.exception("OpenAlex API error")
        return None


def get_work_references(
    openalex_id: str, token: str | None = None, max_retries: int = 2, retry_delay: float = 1.0
) -> list[dict[str, Any]]:
    """Fetch referenced works from OpenAlex. Returns list of basic paper info."""
    if not openalex_id:
        return []

    refs: list[dict[str, Any]] = []

    try:
        headers: dict[str, str] = {"Accept": "application/json"}
        resp = _get_session().get(openalex_id, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        ref_ids = data.get("referenced_works", [])
        for ref_id in ref_ids[:50]:
            ref_info = get_work_by_openalex_id(ref_id, token=token)
            if ref_info:
                refs.append(ref_info)
        return refs
    except requests.RequestException:
        logger.exception("OpenAlex API error")
        return []


def get_work_by_openalex_id(
    openalex_id: str, token: str | None = None, max_retries: int = 2, retry_delay: float = 0.5
) -> dict[str, Any] | None:
    """Fetch a single work by its OpenAlex ID."""
    fields = _select_fields(["id", "doi", "title", "publication_year", "ids"])
    url = f"{openalex_id}?select={fields}"
    headers: dict[str, str] = {"Accept": "application/json"}

    try:
        resp = _get_session().get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
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
    except requests.RequestException:
        logger.exception("OpenAlex API error")
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
            resp = _get_session().get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data and "error" not in data:
                work = data
        except requests.RequestException:
            logger.exception("OpenAlex API error")

    # Fallback to title search
    if work is None and title:
        url = f"{OPENALEX_BASE}/works?search={urllib.parse.quote(title)}&per_page=1&select={fields}"
        headers = {"Accept": "application/json"}
        if token:
            headers["User-Agent"] = f"DrBrain (mailto:{token})"
        try:
            resp = _get_session().get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if results:
                work = results[0]
        except requests.RequestException:
            logger.exception("OpenAlex API error")

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

    try:
        resp = _get_session().get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
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
    except requests.RequestException:
        logger.exception("OpenAlex API error")
        return []
