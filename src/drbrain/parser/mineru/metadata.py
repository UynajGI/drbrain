"""Metadata resolution from arXiv, CrossRef, OpenAlex, S2, and DeepXiv."""

from __future__ import annotations

import re

from loguru import logger as _parse_log


def _titles_match(a: str, b: str) -> bool:
    """Check if two titles likely refer to the same paper. Uses word overlap ratio."""
    a_words = set(a.strip().lower().rstrip(".").split())
    b_words = set(b.strip().lower().rstrip(".").split())
    if not a_words or not b_words:
        return False
    smaller = a_words if len(a_words) <= len(b_words) else b_words
    larger = b_words if smaller == a_words else a_words
    overlap = len(smaller & larger)
    return overlap >= len(smaller) * 0.6  # 60% word overlap


def _resolve_metadata(
    arxiv: str | None = None,
    raw_title: str | None = None,
    raw_year: int | None = None,
    raw_doi: str | None = None,
    deepxiv_token: str = "",
    s2_api_key: str = "",
) -> dict:
    """Cross-validate metadata from arXiv, CrossRef, S2, and OpenAlex.

    Strategy:
    1. If arXiv ID: fetch from arXiv API (authoritative for arXiv papers)
    2. If title: search CrossRef, OpenAlex, S2
    3. DOI from CrossRef with title+year consistency check
    4. Multiple source consensus → high confidence
    5. Returns {title, year, doi, s2_id, openalex_id}
    """
    sources: dict[str, dict] = {}

    from concurrent.futures import ThreadPoolExecutor

    # Determine the search title: prefer arxiv result, fall back to raw_title
    # We run arxiv and deepxiv (which only needs the arxiv ID string) in
    # parallel with the title-based lookups.  If arxiv returns a title we
    # use it for the title-based sources; otherwise we fall back to raw_title.
    _fetched_arxiv_title: str | None = None
    _fetched_arxiv_year: int | None = None

    def _run_arxiv():
        nonlocal _fetched_arxiv_title, _fetched_arxiv_year
        if arxiv:
            _fetched_arxiv_title, _fetched_arxiv_year = _fetch_arxiv_metadata(arxiv)

    def _run_deepxiv():
        nonlocal sources
        if arxiv:
            dx = _fetch_deepxiv_metadata(arxiv, token=deepxiv_token)
            if dx and dx.get("title"):
                sources["deepxiv"] = {
                    "title": dx["title"],
                    "year": dx["year"],
                    "doi": None,
                    "s2_id": None,
                    "openalex_id": None,
                    "journal": "",
                    "publisher": "",
                    "citation_count": dx.get("citations") or 0,
                }

    def _run_title_sources():
        nonlocal _fetched_arxiv_title, _fetched_arxiv_year, sources
        search_title = _fetched_arxiv_title or raw_title or ""
        if not search_title:
            return

        # CrossRef
        cr_title, cr_year, cr_doi, cr_journal, cr_publisher = _fetch_crossref_metadata(search_title)
        if cr_doi or cr_year:
            sources["crossref"] = {
                "title": cr_title,
                "year": cr_year,
                "doi": cr_doi,
                "s2_id": None,
                "openalex_id": None,
                "journal": cr_journal,
                "publisher": cr_publisher,
                "citation_count": 0,
            }

        # OpenAlex
        oa_title, oa_year, oa_id, oa_journal, oa_cited = _fetch_openalex_metadata(search_title)
        if oa_title or oa_year:
            sources["openalex"] = {
                "title": oa_title,
                "year": oa_year,
                "doi": None,
                "s2_id": None,
                "openalex_id": oa_id,
                "journal": oa_journal,
                "publisher": "",
                "citation_count": oa_cited,
            }

        # Semantic Scholar
        s2_title, s2_year, s2_id, s2_journal, s2_cited = _fetch_s2_metadata(search_title)
        if s2_title or s2_year:
            sources["s2"] = {
                "title": s2_title,
                "year": s2_year,
                "doi": None,
                "s2_id": s2_id,
                "openalex_id": None,
                "journal": s2_journal,
                "publisher": "",
                "citation_count": s2_cited,
            }

    # ── Parallel execution ──
    # Phase 1: arxiv + deepxiv (both only need arxiv ID string)
    # Phase 2: title-based sources (need arxiv title or raw_title)
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_arxiv = pool.submit(_run_arxiv)
        pool.submit(_run_deepxiv)
        f_arxiv.result()  # wait for arxiv to set _fetched_arxiv_title

    with ThreadPoolExecutor(max_workers=3) as pool:
        pool.submit(_run_title_sources)
        # deepxiv already done in phase 1; no additional work here
        # We use max_workers=3 but only submit 1 task to keep it simple

    # Store arxiv source if it returned data
    if _fetched_arxiv_title or _fetched_arxiv_year:
        sources["arxiv"] = {
            "title": _fetched_arxiv_title,
            "year": _fetched_arxiv_year,
            "doi": None,
            "s2_id": None,
            "openalex_id": None,
        }

    # ── Resolution ──
    final_doi = raw_doi
    final_title = raw_title
    final_year = raw_year

    # Use text-extracted year as anchor to filter API results
    _text_year = raw_year  # from PDF text parsing

    def _year_consistent(api_year, anchor):
        if not api_year or not anchor:
            return True
        return abs(api_year - anchor) <= 5

    # Only trust CrossRef's DOI if title matches AND year is consistent
    cr_data = sources.get("crossref", {})
    cr_doi = cr_data.get("doi")
    cr_year = cr_data.get("year")
    if cr_doi and not final_doi:
        cr_title = cr_data.get("title") or ""
        ref_title = final_title or ""
        if _titles_match(ref_title, cr_title):
            anchor = _text_year or sources.get("arxiv", {}).get("year")
            if _year_consistent(cr_year, anchor):
                final_doi = cr_doi

    if final_doi:
        if cr_data.get("year"):
            final_year = cr_data["year"]
        if cr_data.get("title"):
            final_title = cr_data["title"]

    # Filter API years by text-year consistency
    filtered_sources = {
        k: v
        for k, v in sources.items()
        if not _text_year or not v.get("year") or _year_consistent(v["year"], _text_year)
    }

    if not final_year:
        # Use filtered sources (year-consistent with text anchor)
        years = [(k, v["year"]) for k, v in filtered_sources.items() if v.get("year")]
        if len(years) >= 2 and len(set(y for _, y in years)) == 1:
            final_year = years[0][1]
        elif sources.get("arxiv", {}).get("year"):
            final_year = sources["arxiv"]["year"]
        elif years:
            final_year = years[0][1]

    if not final_title or final_title == raw_title:
        for src in ["arxiv", "crossref", "s2", "openalex"]:
            if sources.get(src, {}).get("title"):
                final_title = sources[src]["title"]
                break

    # Collect external IDs from sources
    final_s2_id = None
    final_openalex_id = None
    for src_name in ["crossref", "s2", "openalex"]:
        s = sources.get(src_name, {})
        if s.get("s2_id") and not final_s2_id:
            final_s2_id = s["s2_id"]
        if s.get("openalex_id") and not final_openalex_id:
            final_openalex_id = s["openalex_id"]

    # Collect venue metadata: prefer CrossRef for journal/publisher, then OpenAlex, then S2
    final_journal = ""
    final_publisher = ""
    final_citation_count = 0
    for src_name in ["crossref", "openalex", "s2", "deepxiv"]:
        s = sources.get(src_name, {})
        if s.get("journal") and not final_journal:
            final_journal = s["journal"]
        if s.get("publisher") and not final_publisher:
            final_publisher = s["publisher"]
        if s.get("citation_count") and not final_citation_count:
            final_citation_count = s["citation_count"]

    return {
        "title": final_title,
        "year": final_year,
        "doi": final_doi,
        "s2_id": final_s2_id,
        "openalex_id": final_openalex_id,
        "journal": final_journal,
        "publisher": final_publisher,
        "citation_count": final_citation_count,
    }


def _fetch_arxiv_metadata(arxiv_id: str) -> tuple[str | None, int | None]:
    """Fetch title and year from arXiv API via arxiv library."""
    try:
        import arxiv

        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(client.results(search))
        year = paper.published.year if paper.published else None
        return paper.title, year
    except Exception as e:
        _parse_log.debug("arXiv metadata fetch failed, inferring year from ID: {}", e)
        # Fallback: infer year from arXiv ID (1706.03762 → 2017)
        m = re.match(r"(\d{2})(\d{2})\.\d{4,5}", arxiv_id)
        if m:
            yy = int(m.group(1))
            year = 2000 + yy if yy <= 50 else 1900 + yy
            return None, year
        return None, None


def _fetch_openalex_metadata(
    title: str,
) -> tuple[str | None, int | None, str | None, str | None, int]:
    """Fetch title, year, OpenAlex ID, journal, and citation count via pyalex."""
    if not title:
        return None, None, None, None, 0
    try:
        from pyalex import Works as _Works

        works = _Works()
        results = list(works.search(title).get(per_page=1))
        if results:
            w = results[0]
            oa_id = w.get("id", "").replace("https://openalex.org/", "")
            journal = ""
            loc = w.get("primary_location") or {}
            source = loc.get("source") or {}
            if source:
                journal = source.get("display_name") or ""
            cited = w.get("cited_by_count") or 0
            return w.get("title"), w.get("publication_year"), oa_id or None, journal, cited
    except Exception as e:
        _parse_log.debug("OpenAlex metadata fetch failed: {}", e)
    return None, None, None, None, 0


def _fetch_s2_metadata(
    title: str, api_key: str = ""
) -> tuple[str | None, int | None, str | None, str | None, int]:
    """Fetch title, year, paperId, journal, and citationCount from Semantic Scholar API."""
    if not title:
        return None, None, None, None, 0
    try:
        import json as _json
        import urllib.parse as _uparse
        import urllib.request as _ureq

        fields = "title,year,paperId,journal,citationCount"
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={_uparse.quote(title)}&limit=1&fields={fields}"
        headers = {"Accept": "application/json"}
        if api_key:
            headers["x-api-key"] = api_key
        req = _ureq.Request(url, headers=headers)
        resp = _ureq.urlopen(req, timeout=10)
        data = _json.loads(resp.read())
        papers = data.get("data", [])
        if papers:
            p = papers[0]
            journal = ""
            j = p.get("journal") or {}
            if j:
                journal = j.get("name", "")
            return (
                p.get("title"),
                p.get("year"),
                p.get("paperId"),
                journal,
                p.get("citationCount") or 0,
            )
    except Exception as e:
        _parse_log.debug("S2 metadata fetch failed: {}", e)
    return None, None, None, None, 0


def _fetch_deepxiv_metadata(arxiv_id: str, token: str = "") -> dict | None:
    """Fetch metadata from DeepXiv API (title, year, TLDR, keywords, citations)."""
    if not arxiv_id:
        return None
    try:
        import os as _os

        from deepxiv_sdk import Reader as _Reader

        _token = token or _os.environ.get("DEEPXIV_TOKEN", "")
        r = _Reader(token=_token) if _token else _Reader()
        data = r.brief(arxiv_id)
        year = None
        if data.get("publish_at"):
            year = int(data["publish_at"][:4])
        return {
            "title": data.get("title"),
            "year": year,
            "tldr": data.get("tldr"),
            "keywords": data.get("keywords", []),
            "citations": data.get("citations"),
        }
    except Exception as e:
        _parse_log.debug("DeepXiv metadata fetch failed: {}", e)
        return None


def _fetch_crossref_metadata(
    title: str,
) -> tuple[str | None, int | None, str | None, str | None, str | None]:
    """Fetch title, year, DOI, journal, and publisher from CrossRef by title search."""
    if not title:
        return None, None, None, None, None
    try:
        import json as _json
        import urllib.parse as _uparse
        import urllib.request as _ureq

        clean = title.strip()[:200]
        url = f"https://api.crossref.org/works?query.bibliographic={_uparse.quote(clean)}&rows=1"
        req = _ureq.Request(url, headers={"Accept": "application/json"})
        resp = _ureq.urlopen(req, timeout=10)
        data = _json.loads(resp.read())
        items = data.get("message", {}).get("items", [])
        if items:
            item = items[0]
            doi = item.get("DOI")
            cr_title = item.get("title", [None])[0]
            year = item.get("published-print", {}).get("date-parts", [[None]])[0][0]
            if not year:
                year = item.get("created", {}).get("date-parts", [[None]])[0][0]
            journal = item.get("container-title", [None])[0] or ""
            publisher = item.get("publisher", "")
            return cr_title, year, doi, journal, publisher
    except Exception as e:
        _parse_log.debug("CrossRef metadata fetch failed: {}", e)
    return None, None, None, None, None
