"""Sciverse corpus source adapter.

Implements :class:`CorpusSource` on top of the Sciverse REST API:

* ``catalog``  → ``GET /meta-catalog`` (field discovery, cached)
* ``search``   → ``POST /meta-search`` (filters / sort / page + cursor paging)
* ``fetch_relations`` → ``POST /meta-paper-relations`` (CITATIONS / REFERENCES /
  RELATED_WORKS, keyed by ``unique_id``)

Field names follow the official Sciverse schema and should be validated against
``meta-catalog`` rather than hard-coded elsewhere.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from loguru import logger

from drbrain.concept_graph.sources._http import SciverseClient
from drbrain.concept_graph.sources.base import Author, PaperRecord, PaperRelations

# Fields projected from meta-search (must be projectable per meta-catalog).
_SEARCH_FIELDS = [
    "unique_id",
    "doc_id",
    "title",
    "abstract",
    "doi",
    "language",
    "author",
    "publication_published_year",
    "publication_venue_name_unified",
    "keywords",
    "topics",
    "primary_topic",
    "citation_count",
    "reference_count",
]

_PAGE_SIZE = 100
_SHALLOW_LIMIT = 10000  # page * page_size must stay <= this for shallow paging


class SciverseSource:
    """Corpus source backed by the Sciverse academic literature API."""

    name = "sciverse"

    def __init__(
        self,
        token: str,
        base_url: str = "https://api.sciverse.space",
        *,
        rate_limit: int = 30,
        client: SciverseClient | None = None,
    ):
        self.client = client or SciverseClient(token, base_url, rate_limit=rate_limit)
        self._catalog_cache: dict[str, dict] = {}

    # ── catalog ──────────────────────────────────────────────────────────

    def catalog(self, collection: str = "papers") -> dict:
        """Fetch and cache the meta-catalog field table for ``collection``."""
        if collection not in self._catalog_cache:
            payload = self.client.get("meta-catalog", params={"collection": collection})
            self._catalog_cache[collection] = payload
        return self._catalog_cache[collection]

    # ── search ───────────────────────────────────────────────────────────

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
        """Yield up to ``limit`` paper records matching the query / filters."""
        filters = self._build_filters(year_from=year_from, year_to=year_to, venues=venues)
        emitted = 0
        cursor: str | None = None
        page = 1

        while emitted < limit:
            page_size = min(_PAGE_SIZE, limit - emitted)
            body: dict[str, Any] = {"fields": _SEARCH_FIELDS, "page_size": page_size}
            if query:
                body["query"] = query
            if filters:
                body["filters"] = filters
            if sort:
                body["sort"] = sort
            if cursor:
                body["cursor"] = cursor
            else:
                body["page"] = page

            payload = self.client.post("meta-search", json=body)
            results = payload.get("results", [])
            if not results:
                break

            for item in results:
                yield self._to_record(item)
                emitted += 1
                if emitted >= limit:
                    return

            cursor = payload.get("next_cursor")
            # Stop when there are no more pages, or shallow paging would overflow.
            if page * page_size >= _SHALLOW_LIMIT and not cursor:
                break
            if not cursor and page >= payload.get("total_pages", page):
                break
            if cursor:
                cursor = cursor  # advance via cursor once provided
            page += 1

    @staticmethod
    def _build_filters(
        *,
        year_from: int | None,
        year_to: int | None,
        venues: list[str] | None,
    ) -> list[dict]:
        filters: list[dict] = []
        if year_from is not None:
            filters.append(
                {
                    "field": "publication_published_year",
                    "operator": "FILTER_OP_GTE",
                    "value": year_from,
                }
            )
        if year_to is not None:
            filters.append(
                {
                    "field": "publication_published_year",
                    "operator": "FILTER_OP_LTE",
                    "value": year_to,
                }
            )
        if venues:
            filters.append(
                {
                    "field": "publication_venue_name_unified",
                    "operator": "FILTER_OP_IN",
                    "value": venues,
                }
            )
        return filters

    def _to_record(self, item: dict) -> PaperRecord:
        authors = [
            Author(name=a.get("name", ""), orcid=a.get("orcid"))
            for a in item.get("author", [])
            if isinstance(a, dict)
        ]
        topics = [
            t.get("display_name", "")
            for t in item.get("topics", [])
            if isinstance(t, dict) and t.get("display_name")
        ]
        primary = item.get("primary_topic")
        if isinstance(primary, dict) and primary.get("display_name"):
            name = primary["display_name"]
            if name not in topics:
                topics.insert(0, name)
        return PaperRecord(
            unique_id=item.get("unique_id", ""),
            title=item.get("title", ""),
            abstract=item.get("abstract", "") or "",
            year=item.get("publication_published_year"),
            doi=item.get("doi"),
            venue=item.get("publication_venue_name_unified", "") or "",
            authors=authors,
            keywords=list(item.get("keywords", []) or []),
            topics=topics,
            citation_count=int(item.get("citation_count", 0) or 0),
            reference_count=int(item.get("reference_count", 0) or 0),
            source=self.name,
            has_fulltext=bool(item.get("doc_id")),
        )

    # ── relations ────────────────────────────────────────────────────────

    def fetch_relations(self, unique_id: str) -> PaperRelations | None:
        """Fetch citation / reference / related-work lists for ``unique_id``."""
        if not unique_id:
            return None
        relations = PaperRelations(unique_id=unique_id)
        mapping = {
            "CITATIONS": "citations",
            "REFERENCES": "references",
            "RELATED_WORKS": "related_works",
        }
        for relation, attr in mapping.items():
            items = self._fetch_relation_page(unique_id, relation)
            setattr(relations, attr, items)
        logger.debug(
            "[sciverse] relations for {}: {} citations, {} references, {} related",
            unique_id,
            len(relations.citations),
            len(relations.references),
            len(relations.related_works),
        )
        return relations

    def _fetch_relation_page(self, unique_id: str, relation: str) -> list[dict]:
        items: list[dict] = []
        page = 1
        while True:
            payload = self.client.post(
                "meta-paper-relations",
                json={"unique_id": unique_id, "relation": relation, "page": page, "page_size": 200},
            )
            batch = payload.get("items", [])
            items.extend(batch)
            if page >= payload.get("total_pages", page) or not batch:
                break
            page += 1
        return items
