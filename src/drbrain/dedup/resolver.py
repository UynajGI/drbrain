"""Triple-ID resolution: DOI → arXiv → S2 → OpenAlex → title fuzzy match."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass
class PaperIDs:
    """Standardized external identifiers for a paper."""

    doi: str | None = None
    arxiv: str | None = None
    s2_id: str | None = None
    openalex_id: str | None = None

    def normalized(self) -> PaperIDs:
        """Return identifiers in their canonical comparison form."""
        return PaperIDs(
            doi=_normalize_optional("doi", self.doi),
            arxiv=_normalize_optional("arxiv", self.arxiv),
            s2_id=_normalize_optional("s2_id", self.s2_id),
            openalex_id=_normalize_optional("openalex_id", self.openalex_id),
        )


def normalize_doi(raw: str) -> str:
    """Strip URL prefix, lowercase."""
    doi = raw.strip().lower()
    doi = re.sub(r"^doi:\s*", "", doi)
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    doi = doi.strip()
    if not re.fullmatch(r"10\.\d{4,9}/\S+", doi):
        return ""
    return doi


def normalize_arxiv(raw: str) -> str:
    """Strip version suffix, standardize format."""
    raw = raw.strip()
    raw = re.sub(r"v\d+$", "", raw)
    m = re.search(r"(\d{4}\.\d{4,5})", raw)
    return m.group(1) if m else raw


def normalize_openalex(raw: str) -> str:
    """Normalize an OpenAlex work identifier to its bare ``W...`` form."""
    value = raw.strip().rstrip("/")
    if "/" in value:
        value = value.rsplit("/", 1)[-1]
    # A bare host (or URL with an empty suffix) is not an external identity.
    if value.lower() in {"openalex.org", "https:", "http:"}:
        return ""
    return value if re.fullmatch(r"W\d+", value, re.IGNORECASE) else ""


def _normalize_optional(kind: str, value: str | None) -> str | None:
    """Normalize one optional identifier while preserving missing values."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{kind} must be a string or null")
    value = value.strip()
    if not value:
        return None
    if "\x00" in value:
        raise ValueError(f"{kind} must not contain NUL bytes")
    if kind == "doi":
        normalized = normalize_doi(value)
    elif kind == "arxiv":
        normalized = normalize_arxiv(value)
    elif kind == "openalex_id":
        normalized = normalize_openalex(value)
    else:
        normalized = value if re.fullmatch(r"[A-Za-z0-9_-]{6,128}", value) else ""
    if kind == "arxiv" and not re.fullmatch(r"(?:\d{4}\.\d{4,5})(?:v\d+)?", normalized):
        normalized = ""
    return normalized.strip() or None


def canonical_paper_id(
    ids: PaperIDs | None = None,
    *,
    title: str = "",
    year: int | str | None = None,
    source_key: str | None = None,
) -> str:
    """Return a deterministic, filesystem-safe local paper identity.

    The strongest available external identifier is used as the identity seed
    (DOI, arXiv, Semantic Scholar, then OpenAlex).  Records without an
    external identifier fall back to canonical title/year, and finally an
    explicit source key.  A 128-bit SHA-256 prefix avoids the short random-ID
    collision that previously affected large corpus imports while keeping the
    database ID a single portable path component. If no external identifier
    exists, an explicit source key takes precedence over title/year.

    ``source_key`` is deliberately explicit: callers with no metadata must
    provide a stable input (for example a content digest) instead of silently
    generating an ID from process state or a temporary path.
    """
    normalized = (ids or PaperIDs()).normalized()
    identity: str | None = None
    for kind in DedupEngine.PRIORITY:
        value = getattr(normalized, kind)
        if value:
            identity = f"{kind}:{value}"
            break
    if identity is None and source_key is not None and str(source_key).strip():
        # An explicit source key (normally a content digest or stable source
        # identifier) disambiguates records that happen to share title/year.
        identity = f"source:{str(source_key).strip()}"
    if identity is None:
        canonical_title = title_key(title) if isinstance(title, str) else ""
        if canonical_title:
            normalized_year = "" if year is None else str(year).strip()
            identity = f"title:{canonical_title}|year:{normalized_year}"
    if identity is None:
        raise ValueError("paper identity requires an external ID, title, or stable source key")
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
    return f"p{digest}"


def title_key(title: str) -> str:
    """Canonical title for fuzzy matching."""
    if not isinstance(title, str):
        raise ValueError("paper title must be a string")
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
        normalized = ids.normalized()
        for key in self.PRIORITY:
            val = getattr(normalized, key)
            if val:
                local_id = self.db.get_paper_by_external_id(key, val)
                if local_id:
                    return local_id

        if title and year:
            local_id = self.db.fuzzy_match_title_year(title, year)
            if local_id:
                return local_id

        return None
