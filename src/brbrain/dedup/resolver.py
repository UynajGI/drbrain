"""Triple-ID resolution: DOI → arXiv → S2 → OpenAlex → title fuzzy match."""

from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass


@dataclass
class PaperIDs:
    """Standardized external identifiers for a paper."""

    doi: str | None = None
    arxiv: str | None = None
    s2_id: str | None = None
    openalex_id: str | None = None


def normalize_doi(raw: str) -> str:
    """Strip URL prefix, lowercase."""
    doi = raw.strip().lower()
    doi = re.sub(r"^https?://doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi.strip()


def normalize_arxiv(raw: str) -> str:
    """Strip version suffix, standardize format."""
    raw = raw.strip()
    raw = re.sub(r"v\d+$", "", raw)
    m = re.search(r"(\d{4}\.\d{4,5})", raw)
    return m.group(1) if m else raw


def title_key(title: str) -> str:
    """Canonical title for fuzzy matching."""
    t = title.lower().strip()
    t = re.sub(r"\b(the|a|an)\b", "", t)
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def title_hash(title: str) -> str:
    """Short hash for API cache key."""
    return hashlib.md5(title_key(title).encode()).hexdigest()[:12]


class DedupEngine:
    """Resolve paper identity via triple-ID priority matching."""

    PRIORITY = ["doi", "arxiv", "s2_id", "openalex_id"]

    def __init__(self, db):
        """db must implement get_paper_by_id methods."""
        self.db = db

    def resolve(
        self,
        ids: PaperIDs,
        title: str = "",
        year: int | None = None,
    ) -> str | None:
        """Return existing local_id if matched, else None.

        Priority: DOI > arXiv > S2 > OpenAlex > title+year fuzzy.
        """
        for key in self.PRIORITY:
            val = getattr(ids, key)
            if val:
                local_id = self.db.get_paper_by_external_id(key, val)
                if local_id:
                    return local_id

        if title and year:
            local_id = self.db.fuzzy_match_title_year(title, year)
            if local_id:
                return local_id

        return None
