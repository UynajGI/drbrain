"""Shared helper functions for CLI commands."""

from __future__ import annotations


def _enrich_doi_from_crossref(title: str, email: str | None = None) -> dict | None:
    """Try to find DOI for a paper title via CrossRef API."""
    try:
        from drbrain.extractor.crossref import fetch_doi_by_title

        return fetch_doi_by_title(title, email=email)
    except Exception:
        return None


def _enrich_doi_from_crossref_arxiv(arxiv_id: str, email: str | None = None) -> dict | None:
    """Fallback: find DOI via arXiv ID in CrossRef."""
    try:
        from drbrain.extractor.crossref import fetch_doi_by_arxiv

        return fetch_doi_by_arxiv(arxiv_id, email=email)
    except Exception:
        return None


def _enrich_doi_from_crossref_doi(doi: str, email: str | None = None) -> dict | None:
    """Fallback: resolve DOI directly via CrossRef."""
    try:
        from drbrain.extractor.crossref import fetch_doi_by_doi

        return fetch_doi_by_doi(doi, email=email)
    except Exception:
        return None


def _enrich_doi_from_openalex(
    title: str, arxiv: str | None = None, token: str | None = None
) -> dict | None:
    """Try OpenAlex title search, then arXiv fallback."""
    try:
        from drbrain.extractor.openalex import search_work_by_arxiv, search_work_by_title

        result = search_work_by_title(title, token=token)
        if result and result.get("doi"):
            return result
        if arxiv:
            result = search_work_by_arxiv(arxiv, token=token)
            if result and result.get("doi"):
                return result
        return None
    except Exception:
        return None
