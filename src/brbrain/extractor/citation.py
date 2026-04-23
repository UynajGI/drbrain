"""Semantic Scholar API client for citation expansion."""
from __future__ import annotations

import time
import logging
import re

import requests

from brbrain.report.generator import RefEntry

log = logging.getLogger(__name__)
S2_BASE = "https://api.semanticscholar.org/graph/v1/paper"
S2_FIELDS = "title,year,externalIds,authors,citationCount,references,citations"

DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF = 2.0  # exponential backoff multiplier in seconds


def fetch_s2_paper(paper_id: str, api_key: str | None = None) -> dict | None:
    """Fetch paper details from Semantic Scholar API."""
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    url = f"{S2_BASE}/{paper_id}?fields={S2_FIELDS}"
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning(f"S2 API error for {paper_id}: {e}")
        return None


def search_s2(query: str, limit: int = 50, api_key: str | None = None) -> list[dict]:
    """Search Semantic Scholar."""
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    url = f"{S2_BASE}/search?query={requests.utils.quote(query)}&limit={limit}&fields={S2_FIELDS}"
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])
    except Exception as e:
        log.warning(f"S2 search error: {e}")
        return []


def _s2_retry(fn, url: str, headers: dict, max_retries: int) -> dict | None:
    """Retry on 429 with exponential backoff. Non-429 errors fail immediately."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            resp = getattr(e, "response", None)
            status = resp.status_code if resp is not None else None
            if status == 429:
                delay = DEFAULT_BACKOFF * (2 ** attempt)
                log.warning(f"S2 rate limit (429), retry {attempt+1}/{max_retries} in {delay}s")
                time.sleep(delay)
            else:
                log.warning(f"S2 API error (status={status}): {e}")
                return None
        except Exception as e:
            log.warning(f"S2 API error: {e}")
            return None
    return None


def fetch_s2_with_retry(
    paper_id: str, api_key: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict | None:
    """Fetch paper details from S2 API with retry on 429."""
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    url = f"{S2_BASE}/{paper_id}?fields={S2_FIELDS}"
    return _s2_retry(fetch_s2_with_retry, url, headers, max_retries)


def search_s2_with_retry(
    query: str, limit: int = 50, api_key: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> list[dict]:
    """Search S2 with retry on 429."""
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    url = f"{S2_BASE}/search?query={requests.utils.quote(query)}&limit={limit}&fields={S2_FIELDS}"
    data = _s2_retry(search_s2_with_retry, url, headers, max_retries)
    if data is None:
        return []
    return data.get("data", [])


def parse_s2_response(s2_data: dict) -> dict:
    """Parse S2 API response into standardized dict."""
    ext_ids = s2_data.get("externalIds") or {}
    return {
        "title": s2_data.get("title", ""),
        "year": s2_data.get("year"),
        "s2_id": s2_data.get("paperId"),
        "doi": ext_ids.get("DOI"),
        "arxiv": ext_ids.get("ArXiv"),
        "openalex_id": ext_ids.get("OpenAlex"),
        "citation_count": s2_data.get("citationCount", 0),
    }


def match_to_local(db, ref: dict) -> RefEntry:
    """Match a reference to local DB. Create RefEntry with in_graph flag."""
    # Try DOI
    if ref.get("doi"):
        local_id = db.get_paper_by_external_id("doi", ref["doi"])
        if local_id:
            return RefEntry(
                title=ref["title"], year=ref["year"],
                ids={"doi": ref["doi"]},
                in_graph=True, local_id=local_id,
            )
    # Try arXiv
    if ref.get("arxiv"):
        local_id = db.get_paper_by_external_id("arxiv", ref["arxiv"])
        if local_id:
            return RefEntry(
                title=ref["title"], year=ref["year"],
                ids={"arxiv": ref["arxiv"]},
                in_graph=True, local_id=local_id,
            )
    # Try S2 ID
    if ref.get("s2_id"):
        local_id = db.get_paper_by_external_id("s2_id", ref["s2_id"])
        if local_id:
            return RefEntry(
                title=ref["title"], year=ref["year"],
                ids={"s2_id": ref["s2_id"]},
                in_graph=True, local_id=local_id,
            )
    # Try title+year
    if ref.get("title") and ref.get("year"):
        local_id = db.fuzzy_match_title_year(ref["title"], ref["year"])
        if local_id:
            return RefEntry(
                title=ref["title"], year=ref["year"],
                in_graph=True, local_id=local_id,
            )

    # Not found
    return RefEntry(
        title=ref.get("title", ""), year=ref.get("year"),
        ids={k: v for k, v in ref.items() if k in ("doi", "arxiv", "s2_id", "openalex_id") and v},
        in_graph=False, local_id=None,
    )


def expand_citations(db, local_id: str, config: dict) -> tuple[list[RefEntry], list[RefEntry]]:
    """Expand a paper's citation network. Returns (references, citations)."""
    paper = db.get_paper(local_id)
    if not paper:
        return [], []

    # Try to find S2 ID
    s2_id = paper.get("s2_id")
    if not s2_id:
        title = paper.get("title", "")
        if not title:
            return [], []
        results = search_s2(title, limit=1)
        if not results:
            return [], []
        parsed = parse_s2_response(results[0])
        s2_id = parsed.get("s2_id")
        if s2_id:
            db.conn.execute(
                "UPDATE paper_ids SET s2_id = ? WHERE local_id = ?",
                (s2_id, local_id),
            )
            db.commit()

    if not s2_id:
        return [], []

    data = fetch_s2_paper(s2_id)
    if not data:
        return [], []

    # Backfill missing external IDs from S2
    ext_ids = data.get("externalIds") or {}
    s2_doi = ext_ids.get("DOI")
    s2_arxiv = ext_ids.get("ArXiv")
    if s2_arxiv:
        s2_arxiv = re.sub(r"v\d+$", "", s2_arxiv)
    if s2_doi or s2_arxiv:
        existing_doi = paper.get("doi")
        existing_arxiv = paper.get("arxiv")
        if not existing_doi and s2_doi:
            db.conn.execute(
                "UPDATE paper_ids SET doi = ? WHERE local_id = ?", (s2_doi, local_id)
            )
            db.commit()
        if not existing_arxiv and s2_arxiv:
            db.conn.execute(
                "UPDATE paper_ids SET arxiv = ? WHERE local_id = ?", (s2_arxiv, local_id)
            )
            db.commit()

    rate_limit = config.get("api", {}).get("s2_rate_limit", 100)
    delay = 60.0 / rate_limit

    # Process references
    references: list[RefEntry] = []
    for ref in data.get("references", [])[:50]:
        parsed = parse_s2_response(ref)
        entry = match_to_local(db, parsed)
        references.append(entry)
        if not entry.in_graph:
            pid = f"p{local_id[1:]}_ref_{len(references)}"
            db.insert_paper(pid, entry.title or "Unknown", entry.year, "placeholder")
            db.insert_paper_ids(pid, doi=entry.ids.get("doi"), arxiv=entry.ids.get("arxiv"),
                               s2_id=entry.ids.get("s2_id"), openalex_id=entry.ids.get("openalex_id"))
            db.conn.execute(
                "INSERT OR IGNORE INTO edges (src_id, dst_id, relation, source_paper, weight) VALUES (?, ?, ?, ?, ?)",
                (local_id, pid, "cites", local_id, 1.0),
            )
            db.commit()
        time.sleep(delay)

    # Process citations (papers citing this one)
    citations: list[RefEntry] = []
    for cit in data.get("citations", [])[:50]:
        parsed = parse_s2_response(cit)
        entry = match_to_local(db, parsed)
        citations.append(entry)
        if not entry.in_graph:
            pid = f"p{local_id[1:]}_cit_{len(citations)}"
            db.insert_paper(pid, entry.title or "Unknown", entry.year, "placeholder")
            db.insert_paper_ids(pid, doi=entry.ids.get("doi"), arxiv=entry.ids.get("arxiv"),
                               s2_id=entry.ids.get("s2_id"), openalex_id=entry.ids.get("openalex_id"))
            db.conn.execute(
                "INSERT OR IGNORE INTO edges (src_id, dst_id, relation, source_paper, weight) VALUES (?, ?, ?, ?, ?)",
                (pid, local_id, "cited_by", local_id, 1.0),
            )
            db.commit()
        time.sleep(delay)

    return references, citations
