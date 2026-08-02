"""Source-agnostic data models for corpus ingestion.

Defines the canonical :class:`PaperRecord` / :class:`PaperRelations` records that
every :class:`CorpusSource` adapter yields, plus a helper for the unified Sciverse
response envelope (``code`` / ``message`` / ``biz_code``).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Author:
    """A paper author with an optional ORCID identifier."""

    name: str
    orcid: str | None = None


@dataclass
class PaperRecord:
    """A source-agnostic paper metadata record.

    ``unique_id`` is the stable primary key used for deduplication and
    cross-service linking (Sciverse ``unique_id`` / OpenAlex id). ``doi`` may be
    absent and is only a secondary dedup key. ``keywords`` / ``topics`` provided
    by the source can serve as a zero-LLM, zero-fulltext concept source.
    """

    unique_id: str
    title: str
    abstract: str = ""
    year: int | None = None
    doi: str | None = None
    venue: str = ""
    authors: list[Author] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    citation_count: int = 0
    reference_count: int = 0
    source: str = ""
    has_fulltext: bool = False


@dataclass
class PaperRelations:
    """Citation / reference / related-work relations for a single paper.

    Each entry is a raw relation item (e.g. ``{"id", "id_type", "title"}`` for
    Sciverse) left as a dict so adapters need not normalise prematurely.
    """

    unique_id: str
    citations: list[dict] = field(default_factory=list)
    references: list[dict] = field(default_factory=list)
    related_works: list[dict] = field(default_factory=list)


@runtime_checkable
class CorpusSource(Protocol):
    """Protocol every corpus adapter implements.

    Adapters yield :class:`PaperRecord` items from :meth:`search` and optionally
    expose citation relations (:meth:`fetch_relations`) and field discovery
    (:meth:`catalog`).
    """

    name: str

    def search(
        self,
        query: str | None = None,
        *,
        year_from: int | None = None,
        year_to: int | None = None,
        venues: list[str] | None = None,
        sort: list[dict] | None = None,
        limit: int = 100,
    ) -> Iterator[PaperRecord]:
        """Yield paper records matching the query / filters."""
        ...

    def fetch_relations(self, unique_id: str) -> PaperRelations | None:
        """Return citation/reference/related-work relations for a paper."""
        ...

    def catalog(self) -> dict:
        """Return discoverable field metadata (filterable/sortable/projectable)."""
        ...


def is_success_envelope(payload: dict[str, Any]) -> bool:
    """Check a unified Sciverse response envelope for success.

    Sciverse wraps responses with ``code`` / ``message`` / ``biz_code``. A call is
    successful when ``code == "SUCCESS"`` (case-insensitive) or ``biz_code == 0``.
    Payloads without an envelope (e.g. plain result dicts) are treated as success.

    Args:
        payload: Decoded JSON response body.

    Returns:
        True if the envelope indicates success (or is absent).
    """
    if not isinstance(payload, dict):
        return False
    code = payload.get("code")
    biz_code = payload.get("biz_code")
    if code is None and biz_code is None:
        return True
    if isinstance(code, str) and code.upper() == "SUCCESS":
        return True
    if biz_code == 0:
        return True
    return False
