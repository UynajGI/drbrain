"""OpenAlex API client for DOI enrichment and citation expansion."""
from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
import json
from typing import Any

OPENALEX_BASE = "https://api.openalex.org"


def _select_fields(fields: list[str]) -> str:
    """Build OpenAlex select parameter."""
    return ",".join(fields)


def search_work_by_title(title: str, token: str | None = None,
                         max_retries: int = 2, retry_delay: float = 1.0) -> dict[str, Any] | None:
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


def search_work_by_arxiv(arxiv_id: str, token: str | None = None,
                         max_retries: int = 2, retry_delay: float = 1.0) -> dict[str, Any] | None:
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


def get_work_by_doi(doi: str, token: str | None = None,
                    max_retries: int = 2, retry_delay: float = 1.0) -> dict[str, Any] | None:
    """Fetch work by DOI from OpenAlex."""
    if not doi:
        return None

    clean_doi = re.sub(r"^https?://doi\.org/", "", doi)
    fields = _select_fields(["id", "doi", "title", "publication_year", "ids",
                              "referenced_works", "cited_by_api_url"])
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


def get_work_references(openalex_id: str, token: str | None = None,
                        max_retries: int = 2, retry_delay: float = 1.0) -> list[dict[str, Any]]:
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
                ref_info = _fetch_work_by_id(ref_id, token=token)
                if ref_info:
                    refs.append(ref_info)
            return refs
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            continue
    return []


def _fetch_work_by_id(openalex_id: str, token: str | None = None,
                      max_retries: int = 2, retry_delay: float = 1.0) -> dict[str, Any] | None:
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
