"""Metadata repair via external API enrichment."""

from __future__ import annotations

import re

from loguru import logger

REPAIR_SOURCES = {
    "doi": ["title", "authors", "year", "journal", "volume", "pages"],
    "arxiv": ["title", "authors", "year"],
    "title_year": ["doi", "journal"],
}


def normalize_title(title: str) -> str:
    """Normalize a paper title: fix all-caps, strip arXiv IDs, trim."""
    title = title.strip()
    title = re.sub(
        r"^arxiv:\s*\d{4}\.\d{4,5}v?\d*\s*[-–—]*\s*",
        "",
        title,
        flags=re.IGNORECASE,
    ).strip()
    alpha = [c for c in title if c.isalpha()]
    if alpha and sum(1 for c in alpha if c.isupper()) / len(alpha) > 0.8:
        small = {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
        }
        words = title.split()
        title = " ".join(
            w.capitalize() if i == 0 or i == len(words) - 1 or w.lower() not in small else w.lower()
            for i, w in enumerate(words)
        )
    return title


def _repair_via_crossref(db, paper: dict) -> list[dict]:
    doi = paper.get("doi")
    if not doi:
        return []
    try:
        from drbrain.extractor.crossref import fetch_work_by_doi

        data = fetch_work_by_doi(doi)
    except Exception:
        try:
            logger.exception("Repair failed")
        except Exception:
            pass
        return []

    repairs = []
    if data and isinstance(data, dict):
        crossref_title = (data.get("title", [""]) or [""])[0]
        if crossref_title and normalize_title(paper.get("title", "")) != crossref_title:
            repairs.append(
                {
                    "field": "title",
                    "old": paper["title"],
                    "new": crossref_title,
                    "source": "CrossRef",
                }
            )

        crossref_year = (data.get("created") or {}).get("date-parts", [[None]])[0][0]
        if crossref_year and str(paper.get("year")) != str(crossref_year):
            repairs.append(
                {
                    "field": "year",
                    "old": paper.get("year"),
                    "new": crossref_year,
                    "source": "CrossRef",
                }
            )

        authors = data.get("author", [])
        if authors:
            author_str = " and ".join(
                f"{a.get('given', '')} {a.get('family', '')}".strip() for a in authors
            )
            if author_str.strip():
                repairs.append(
                    {"field": "authors", "old": "", "new": author_str, "source": "CrossRef"}
                )

        journal = (data.get("container-title", [""]) or [""])[0]
        if journal:
            repairs.append({"field": "journal", "old": "", "new": journal, "source": "CrossRef"})

        abstract = (data.get("abstract", "") or "").strip()
        if abstract and not paper.get("abstract"):
            repairs.append(
                {"field": "abstract", "old": "", "new": abstract[:2000], "source": "CrossRef"}
            )

        cited = data.get("is-referenced-by-count") or 0
        if cited and not paper.get("citation_count"):
            repairs.append(
                {"field": "citation_count", "old": 0, "new": cited, "source": "CrossRef"}
            )
    return repairs


def _repair_via_arxiv(db, paper: dict) -> list[dict]:
    arxiv = paper.get("arxiv")
    if not arxiv:
        return []
    try:
        from drbrain.parser.mineru_parser import _fetch_arxiv_metadata

        title, year = _fetch_arxiv_metadata(arxiv)
    except Exception:
        try:
            logger.exception("Repair failed")
        except Exception:
            pass
        return []

    repairs = []
    if title and normalize_title(paper.get("title", "")) != title:
        repairs.append({"field": "title", "old": paper["title"], "new": title, "source": "arXiv"})
    if year and paper.get("year") != year:
        repairs.append({"field": "year", "old": paper.get("year"), "new": year, "source": "arXiv"})
    return repairs


def _enrich_via_openalex(db, paper: dict) -> list[dict]:
    """Fetch abstract, citation_count, authors, volume, pages from OpenAlex."""
    title = paper.get("title", "")
    doi = paper.get("doi")
    if not title and not doi:
        return []

    try:
        from drbrain.extractor.openalex import get_work_enriched, search_authors_by_work

        enriched = None
        if doi:
            enriched = get_work_enriched(doi)
        if not enriched and title:
            from drbrain.extractor.openalex import search_work_by_title

            work = search_work_by_title(title)
            if work and work.get("doi"):
                enriched = get_work_enriched(work["doi"])

        # Fetch authors separately in case enriched didn't include them
        authors_str = enriched.get("authors", "") if enriched else ""
        if not authors_str and (doi or title):
            authors_list = search_authors_by_work(doi=doi, title=title)
            if authors_list:
                authors_str = " and ".join(
                    a.get("display_name", "") for a in authors_list if a.get("display_name")
                )
    except Exception:
        try:
            logger.exception("OpenAlex enrichment failed")
        except Exception:
            pass
        return []

    repairs = []
    if enriched and isinstance(enriched, dict):
        if not paper.get("abstract") and enriched.get("abstract"):
            repairs.append(
                {
                    "field": "abstract",
                    "old": "",
                    "new": enriched["abstract"][:2000],
                    "source": "OpenAlex",
                }
            )
        if not paper.get("citation_count") and enriched.get("cited_by_count"):
            repairs.append(
                {
                    "field": "citation_count",
                    "old": 0,
                    "new": enriched["cited_by_count"],
                    "source": "OpenAlex",
                }
            )
        if not paper.get("journal") and enriched.get("journal"):
            repairs.append(
                {
                    "field": "journal",
                    "old": "",
                    "new": enriched["journal"],
                    "source": "OpenAlex",
                }
            )
        if not paper.get("volume") and enriched.get("volume"):
            repairs.append(
                {
                    "field": "volume",
                    "old": "",
                    "new": enriched["volume"],
                    "source": "OpenAlex",
                }
            )
        if not paper.get("pages") and enriched.get("pages"):
            repairs.append(
                {
                    "field": "pages",
                    "old": "",
                    "new": enriched["pages"],
                    "source": "OpenAlex",
                }
            )

    if authors_str and not paper.get("authors"):
        repairs.append(
            {
                "field": "authors",
                "old": "",
                "new": authors_str,
                "source": "OpenAlex",
            }
        )

    return repairs


def _repair_via_title_year(db, paper: dict) -> list[dict]:
    if paper.get("doi"):
        return []
    title = paper.get("title", "")
    if not title:
        return []
    try:
        from drbrain.extractor.crossref import fetch_doi_by_title

        doi_info = fetch_doi_by_title(title)
    except Exception:
        try:
            logger.exception("Repair failed")
        except Exception:
            pass
        return []
    if doi_info and doi_info.get("doi"):
        return [{"field": "doi", "old": None, "new": doi_info["doi"], "source": "CrossRef"}]
    return []


def repair_paper(db, local_id: str, *, dry_run: bool = False) -> list[dict]:
    """Run all repair sources on a single paper."""
    logger.info("[repair] %s — checking metadata", local_id)
    paper = db.get_paper(local_id)
    if not paper:
        logger.warning("[repair] %s not found", local_id)
        return [{"field": "error", "reason": f"Paper not found: {local_id}"}]

    repairs = []

    normalized = normalize_title(paper.get("title", ""))
    if normalized != paper.get("title", ""):
        repairs.append(
            {"field": "title", "old": paper["title"], "new": normalized, "source": "normalization"}
        )

    for repair_fn in (
        _repair_via_crossref,
        _repair_via_arxiv,
        _repair_via_title_year,
        _enrich_via_openalex,
    ):
        try:
            repairs.extend(repair_fn(db, paper))
        except Exception:
            try:
                logger.exception("Repair failed")
            except Exception:
                pass

    if repairs and not dry_run:
        for r in repairs:
            if r["field"] == "title":
                db.conn.execute(
                    "UPDATE papers SET title = ? WHERE local_id = ?", (r["new"], local_id)
                )
            elif r["field"] == "year":
                db.conn.execute(
                    "UPDATE papers SET year = ? WHERE local_id = ?", (r["new"], local_id)
                )
            elif r["field"] == "doi":
                db.conn.execute(
                    "UPDATE paper_ids SET doi = ? WHERE local_id = ?", (r["new"], local_id)
                )
            elif r["field"] == "journal":
                db.conn.execute(
                    "UPDATE papers SET journal = ? WHERE local_id = ?", (r["new"], local_id)
                )
            elif r["field"] == "abstract":
                db.conn.execute(
                    "UPDATE papers SET abstract = ? WHERE local_id = ?", (r["new"][:2000], local_id)
                )
            elif r["field"] == "citation_count":
                db.conn.execute(
                    "UPDATE papers SET citation_count = ? WHERE local_id = ?", (r["new"], local_id)
                )
            elif r["field"] == "authors":
                db.conn.execute(
                    "UPDATE papers SET authors = ? WHERE local_id = ?", (r["new"], local_id)
                )
            elif r["field"] == "volume":
                db.conn.execute(
                    "UPDATE papers SET volume = ? WHERE local_id = ?", (r["new"], local_id)
                )
            elif r["field"] == "pages":
                db.conn.execute(
                    "UPDATE papers SET pages = ? WHERE local_id = ?", (r["new"], local_id)
                )
        db.commit()

    logger.info("[repair] %s — %d fields repaired (dry_run=%s)", local_id, len(repairs), dry_run)
    return repairs
