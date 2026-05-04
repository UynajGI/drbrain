"""Semantic Scholar API client for citation expansion."""

from __future__ import annotations

import json
import re
import time
import uuid

import requests
from loguru import logger as _cit_log

from drbrain.extractor.cache import ApiCache
from drbrain.extractor.crossref import fetch_doi_by_arxiv, fetch_doi_by_title
from drbrain.report.generator import RefEntry

S2_BASE = "https://api.semanticscholar.org/graph/v1/paper"
S2_FIELDS = "title,year,externalIds,authors,citationCount,references,citations"

DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF = 2.0  # exponential backoff multiplier in seconds

_cache: ApiCache | None = None


def _get_cache(config: dict) -> ApiCache | None:
    """Get or create the API cache from config."""
    global _cache
    cache_ttl = config.get("api", {}).get("cache_ttl")
    if cache_ttl and cache_ttl > 0:
        if _cache is None:
            cache_dir = config.get("dirs", {}).get("cache", "data/cache")
            _cache = ApiCache(cache_dir, ttl=cache_ttl)
        return _cache
    return None


def fetch_s2_paper(
    paper_id: str, api_key: str | None = None, cache: ApiCache | None = None
) -> dict | None:
    """Fetch paper details from Semantic Scholar API."""
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    url = f"{S2_BASE}/{paper_id}?fields={S2_FIELDS}"

    if cache:
        cached = cache.get(f"s2_paper:{paper_id}")
        if cached is not None:
            return cached

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if cache:
            cache.set(f"s2_paper:{paper_id}", data)
        return data
    except Exception as e:
        _cit_log.warning(f"S2 API error for {paper_id}: {e}")
        return None


def search_s2(
    query: str, limit: int = 50, api_key: str | None = None, cache: ApiCache | None = None
) -> list[dict]:
    """Search Semantic Scholar."""
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    url = f"{S2_BASE}/search?query={requests.utils.quote(query)}&limit={limit}&fields={S2_FIELDS}"
    cache_key = f"s2_search:{query}:{limit}"

    if cache:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        result = data.get("data", [])
        if cache:
            cache.set(cache_key, result)
        return result
    except Exception as e:
        _cit_log.warning(f"S2 search error: {e}")
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
                delay = DEFAULT_BACKOFF * (2**attempt)
                _cit_log.warning(
                    f"S2 rate limit (429), retry {attempt + 1}/{max_retries} in {delay}s"
                )
                time.sleep(delay)
            else:
                _cit_log.warning(f"S2 API error (status={status}): {e}")
                return None
        except Exception as e:
            _cit_log.warning(f"S2 API error: {e}")
            return None
    return None


def fetch_s2_with_retry(
    paper_id: str,
    api_key: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict | None:
    """Fetch paper details from S2 API with retry on 429."""
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    url = f"{S2_BASE}/{paper_id}?fields={S2_FIELDS}"
    return _s2_retry(fetch_s2_with_retry, url, headers, max_retries)


def search_s2_with_retry(
    query: str,
    limit: int = 50,
    api_key: str | None = None,
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
                title=ref["title"],
                year=ref["year"],
                ids={"doi": ref["doi"]},
                in_graph=True,
                local_id=local_id,
            )
    # Try arXiv
    if ref.get("arxiv"):
        local_id = db.get_paper_by_external_id("arxiv", ref["arxiv"])
        if local_id:
            return RefEntry(
                title=ref["title"],
                year=ref["year"],
                ids={"arxiv": ref["arxiv"]},
                in_graph=True,
                local_id=local_id,
            )
    # Try S2 ID
    if ref.get("s2_id"):
        local_id = db.get_paper_by_external_id("s2_id", ref["s2_id"])
        if local_id:
            return RefEntry(
                title=ref["title"],
                year=ref["year"],
                ids={"s2_id": ref["s2_id"]},
                in_graph=True,
                local_id=local_id,
            )
    # Try OpenAlex ID
    if ref.get("openalex_id"):
        local_id = db.get_paper_by_external_id("openalex_id", ref["openalex_id"])
        if local_id:
            return RefEntry(
                title=ref["title"],
                year=ref["year"],
                ids={"openalex_id": ref["openalex_id"]},
                in_graph=True,
                local_id=local_id,
            )
    # Try title+year
    if ref.get("title") and ref.get("year"):
        local_id = db.fuzzy_match_title_year(ref["title"], ref["year"])
        if local_id:
            return RefEntry(
                title=ref["title"],
                year=ref["year"],
                in_graph=True,
                local_id=local_id,
            )

    # Not found
    return RefEntry(
        title=ref.get("title", ""),
        year=ref.get("year"),
        ids={k: v for k, v in ref.items() if k in ("doi", "arxiv", "s2_id", "openalex_id") and v},
        in_graph=False,
        local_id=None,
    )


def expand_citations(db, local_id: str, config: dict) -> tuple[list[RefEntry], list[RefEntry]]:
    """Expand a paper's citation network. Returns (references, citations)."""
    paper = db.get_paper(local_id)
    if not paper:
        return [], []

    cache = _get_cache(config)
    s2_api_key = config.get("api", {}).get("s2_api_key") or None

    # Try S2 first
    s2_id = paper.get("s2_id")
    s2_data = None
    if not s2_id:
        title = paper.get("title", "")
        if title:
            results = search_s2(title, limit=1, api_key=s2_api_key, cache=cache)
            if results:
                parsed = parse_s2_response(results[0])
                s2_id = parsed.get("s2_id")
                if s2_id:
                    db.conn.execute(
                        "UPDATE paper_ids SET s2_id = ? WHERE local_id = ?",
                        (s2_id, local_id),
                    )
                    db.commit()

    if s2_id:
        s2_data = fetch_s2_paper(s2_id, api_key=s2_api_key, cache=cache)

    if s2_data:
        refs, cits = _process_citations_from_s2(db, local_id, s2_data, paper, config, cache)
        # If S2 returned data but no DOI, try CrossRef as enrichment (Spec §11 Stage 6.5)
        if not paper.get("doi") and not ext_ids_from_s2(s2_data).get("doi"):
            crossref_email = config.get("api", {}).get("crossref_email")
            crossref_result = _crossref_doi_enrich(paper, crossref_email)
            if crossref_result and crossref_result.get("doi"):
                db.conn.execute(
                    "UPDATE paper_ids SET doi = ? WHERE local_id = ?",
                    (crossref_result["doi"], local_id),
                )
                db.commit()
        return refs, cits

    # Fallback: CrossRef DOI enrichment when S2 returns 429/no data (Spec §11 Stage 6.5)
    if not paper.get("doi"):
        crossref_email = config.get("api", {}).get("crossref_email")
        crossref_result = _crossref_doi_enrich(paper, crossref_email)
        if crossref_result and crossref_result.get("doi"):
            db.conn.execute(
                "UPDATE paper_ids SET doi = ? WHERE local_id = ?",
                (crossref_result["doi"], local_id),
            )
            db.commit()

    # Fallback: OpenAlex
    openalex_token = config.get("api", {}).get("openalex_token")
    return _expand_with_openalex(db, local_id, paper, openalex_token)


def _process_citations_from_s2(
    db, local_id: str, data: dict, paper: dict, config: dict, cache: ApiCache | None = None
) -> tuple[list[RefEntry], list[RefEntry]]:
    """Process citation data from S2 API."""
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
            db.conn.execute("UPDATE paper_ids SET doi = ? WHERE local_id = ?", (s2_doi, local_id))
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
    new_ref_placeholders: list[tuple[str, str, int | None, dict]] = []
    new_ref_edges: list[tuple[str, str, str, str, float]] = []

    for ref in (data.get("references") or [])[:50]:
        parsed = parse_s2_response(ref)
        entry = match_to_local(db, parsed)
        _cache_citation(
            db,
            local_id,
            parsed["title"],
            parsed["year"],
            "references",
            target_doi=parsed["doi"],
            target_s2_id=parsed["s2_id"],
        )
        references.append(entry)
        if not entry.in_graph:
            pid = f"p{local_id[1:]}_ref_{len(references)}"
            new_ref_placeholders.append((pid, entry.title or "Unknown", entry.year, entry.ids))
            new_ref_edges.append((local_id, pid, "cites", local_id, 1.0))
        time.sleep(delay)

    # Batch insert reference placeholders
    if new_ref_placeholders:
        for pid, title, year, ids in new_ref_placeholders:
            db.insert_paper(pid, title, year, "placeholder")
            db.insert_paper_ids(
                pid,
                doi=ids.get("doi"),
                arxiv=ids.get("arxiv"),
                s2_id=ids.get("s2_id"),
                openalex_id=ids.get("openalex_id"),
            )
        for src_id, dst_id, relation, source_paper, weight in new_ref_edges:
            db.conn.execute(
                "INSERT OR IGNORE INTO edges (src_id, dst_id, relation, source_paper, weight) VALUES (?, ?, ?, ?, ?)",
                (src_id, dst_id, relation, source_paper, weight),
            )
        db.commit()

    # Process citations (papers citing this one)
    citations: list[RefEntry] = []
    new_cit_placeholders: list[tuple[str, str, int | None, dict]] = []
    new_cit_edges: list[tuple[str, str, str, str, float]] = []

    for cit in (data.get("citations") or [])[:50]:
        parsed = parse_s2_response(cit)
        entry = match_to_local(db, parsed)
        _cache_citation(
            db,
            local_id,
            parsed["title"],
            parsed["year"],
            "citing",
            target_doi=parsed["doi"],
            target_s2_id=parsed["s2_id"],
        )
        citations.append(entry)
        if not entry.in_graph:
            pid = f"p{local_id[1:]}_cit_{len(citations)}"
            new_cit_placeholders.append((pid, entry.title or "Unknown", entry.year, entry.ids))
            new_cit_edges.append((pid, local_id, "cited_by", local_id, 1.0))
        time.sleep(delay)

    # Batch insert citation placeholders
    if new_cit_placeholders:
        for pid, title, year, ids in new_cit_placeholders:
            db.insert_paper(pid, title, year, "placeholder")
            db.insert_paper_ids(
                pid,
                doi=ids.get("doi"),
                arxiv=ids.get("arxiv"),
                s2_id=ids.get("s2_id"),
                openalex_id=ids.get("openalex_id"),
            )
        for src_id, dst_id, relation, source_paper, weight in new_cit_edges:
            db.conn.execute(
                "INSERT OR IGNORE INTO edges (src_id, dst_id, relation, source_paper, weight) VALUES (?, ?, ?, ?, ?)",
                (src_id, dst_id, relation, source_paper, weight),
            )
        db.commit()

    return references, citations


def expand_citations_oa(db, local_id: str) -> int:
    """Expand citations using OpenAlex (via pyalex). Returns number of references added."""
    import pyalex
    from pyalex import Works

    # Get OpenAlex ID from DB
    row = db.conn.execute(
        "SELECT openalex_id FROM paper_ids WHERE local_id = ?", (local_id,)
    ).fetchone()
    oa_id = row[0] if row and row[0] else None
    if not oa_id:
        return 0

    works = Works()
    try:
        w = works[oa_id]
    except Exception:
        return 0

    ref_ids = w.get("referenced_works", [])
    added = 0
    for rid in ref_ids[:50]:
        try:
            r = works[rid]
            doi = r.get("doi", "")
            if doi:
                doi = re.sub(r"^https?://doi\.org/", "", doi)
            title = r.get("title", "") or "Untitled"
            year = r.get("publication_year")

            # Check if already in local DB
            existing = None
            if doi:
                existing = db.get_paper_by_external_id("doi", doi)
            if not existing:
                existing = db.get_paper_by_external_id("openalex_id", rid)

            if existing:
                local_ref = existing
            else:
                local_ref = f"p{uuid.uuid4().hex[:6]}"
                db.insert_paper(local_ref, title, year, "placeholder")
                ids = {"doi": doi or None, "openalex_id": rid.replace('https://openalex.org/', '')}
                db.insert_paper_ids(local_ref, **ids)
                db.commit()

            # Record citation relationship
            db.conn.execute(
                "INSERT OR IGNORE INTO citation_cache "
                "(source_paper, target_title, target_year, relation, target_doi) "
                "VALUES (?, ?, ?, 'references', ?)",
                (local_id, title, year, doi or None),
            )
            added += 1
        except Exception:
            continue

    # Also fetch citing papers
    citing_added = 0
    try:
        import urllib.request as _ureq
        cite_url = f"https://api.openalex.org/works?filter=cites:{oa_id}&per_page=50&sort=cited_by_count:desc"
        req = _ureq.Request(cite_url, headers={"Accept": "application/json"})
        resp = _ureq.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        for r in data.get("results", [])[:50]:
            try:
                ctitle = r.get("title", "") or "Untitled"
                cyear = r.get("publication_year")
                cdoi = r.get("doi", "")
                if cdoi:
                    cdoi = re.sub(r"^https?://doi\.org/", "", cdoi)
                db.conn.execute(
                    "INSERT OR IGNORE INTO citation_cache "
                    "(source_paper, target_title, target_year, relation, target_doi) "
                    "VALUES (?, ?, ?, 'citing', ?)",
                    (local_id, ctitle, cyear, cdoi or None),
                )
                citing_added += 1
            except Exception:
                continue
        db.commit()
    except Exception:
        pass

    db.commit()
    return added, citing_added


def _crossref_doi_enrich(paper: dict, email: str | None = None) -> dict | None:
    """CrossRef DOI enrichment fallback (Spec §11 Stage 6.5).

    Try arXiv ID first if available, then title search.
    """
    arxiv = paper.get("arxiv")
    title = paper.get("title", "")

    if arxiv:
        result = fetch_doi_by_arxiv(arxiv, email=email)
        if result and result.get("doi"):
            _cit_log.debug("CrossRef DOI via arXiv: %s -> %s", arxiv, result["doi"])
            return result

    if title:
        result = fetch_doi_by_title(title, email=email)
        if result and result.get("doi"):
            _cit_log.debug("CrossRef DOI via title: %s -> %s", title, result["doi"])
            return result

    return None


def ext_ids_from_s2(data: dict) -> dict:
    """Extract parsed external IDs from S2 response."""
    ext_ids = data.get("externalIds") or {}
    doi = ext_ids.get("DOI")
    arxiv = ext_ids.get("ArXiv")
    if arxiv:
        arxiv = re.sub(r"v\d+$", "", arxiv)
    return {
        "doi": doi,
        "arxiv": arxiv,
        "s2_id": data.get("paperId"),
        "openalex_id": ext_ids.get("OpenAlex"),
    }


def _expand_with_openalex(
    db, local_id: str, paper: dict, token: str | None = None
) -> tuple[list[RefEntry], list[RefEntry]]:
    """Fallback: expand citations using OpenAlex when S2 fails."""
    from drbrain.extractor.openalex import (
        batch_fetch_works,
        get_work_by_doi,
        search_work_by_arxiv,
        search_work_by_title,
    )

    oa_token = token
    title = paper.get("title", "")
    arxiv = paper.get("arxiv")
    doi = paper.get("doi")

    # Find the paper in OpenAlex
    oa_work = None
    if doi:
        oa_work = get_work_by_doi(doi, token=oa_token)
    if not oa_work and arxiv:
        oa_work = search_work_by_arxiv(arxiv, token=oa_token)
    if not oa_work and title:
        oa_work = search_work_by_title(title, token=oa_token)

    # If found but missing referenced_works, fetch them via DOI
    if oa_work and not oa_work.get("referenced_works") and oa_work.get("doi"):
        full_work = get_work_by_doi(oa_work["doi"], token=oa_token)
        if full_work and full_work.get("referenced_works"):
            oa_work["referenced_works"] = full_work["referenced_works"]

    if not oa_work:
        return [], []

    if not oa_work:
        return [], []

    # Backfill DOI from OpenAlex
    if not paper.get("doi") and oa_work.get("doi"):
        db.conn.execute(
            "UPDATE paper_ids SET doi = ? WHERE local_id = ?",
            (oa_work["doi"], local_id),
        )
        db.commit()

    # Update openalex_id if we have it
    if oa_work.get("openalex_id"):
        db.conn.execute(
            "UPDATE paper_ids SET openalex_id = ? WHERE local_id = ?",
            (oa_work["openalex_id"], local_id),
        )
        db.commit()

    # Fetch references using batch endpoint
    references: list[RefEntry] = []
    ref_ids = oa_work.get("referenced_works", [])[:50]
    if ref_ids:
        oa_refs = batch_fetch_works(ref_ids, token=oa_token)
        for ref_info in oa_refs:
            entry = match_to_local(db, ref_info)
            references.append(entry)
            if not entry.in_graph:
                pid = f"p{local_id[1:]}_ref_{len(references)}"
                db.insert_paper(pid, entry.title or "Unknown", entry.year, "placeholder")
                db.insert_paper_ids(
                    pid,
                    doi=entry.ids.get("doi"),
                    arxiv=entry.ids.get("arxiv"),
                    openalex_id=entry.ids.get("openalex_id"),
                )
                db.conn.execute(
                    "INSERT OR IGNORE INTO edges (src_id, dst_id, relation, source_paper, weight) VALUES (?, ?, ?, ?, ?)",
                    (local_id, pid, "cites", local_id, 1.0),
                )
                db.commit()

    return references, []


def _cache_citation(
    db,
    source_paper: str,
    target_title: str,
    target_year: int | None,
    relation: str,
    target_doi: str | None = None,
    target_s2_id: str | None = None,
) -> None:
    db.conn.execute(
        "INSERT OR IGNORE INTO citation_cache "
        "(source_paper, target_title, target_year, relation, target_doi, target_s2_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (source_paper, target_title, target_year, relation, target_doi, target_s2_id),
    )
