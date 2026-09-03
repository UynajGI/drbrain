"""CrossRef corpus source adapter (works API, forward references).

Wraps the CrossRef ``/works`` REST API into the :class:`CorpusSource` protocol.
``unique_id`` is the DOI. Citation relations are forward-only: CrossRef exposes
each work's reference list (publisher-submitted), not a cited-by index, so
:meth:`fetch_relations` returns ``references`` and no ``citations``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from urllib.parse import quote

import requests

from drbrain.concept_graph.sources.base import Author, PaperRecord, PaperRelations

CROSSREF_BASE = "https://api.crossref.org"


class CrossRefSource:
    """Corpus source backed by the CrossRef ``/works`` API."""

    name = "crossref"

    def __init__(self, mailto: str | None = None, base_url: str = CROSSREF_BASE):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": "DrBrain/0.1 (mailto:{}; concept-graph research)".format(mailto or "")}
        )

    def search(
        self,
        query: str | None = None,
        *,
        year_from: int | None = None,
        year_to: int | None = None,
        venues: list[str] | None = None,
        sort: list[dict] | None = None,  # accepted for protocol parity; unused
        limit: int = 100,
    ) -> Iterator[PaperRecord]:
        """Yield up to ``limit`` CrossRef works matching the query / filters."""
        params: dict[str, Any] = {"rows": min(1000, limit)}
        if query:
            params["query.bibliographic"] = query
        filters: list[str] = []
        if year_from is not None:
            filters.append(f"from-pub-date:{year_from}-01-01")
        if year_to is not None:
            filters.append(f"until-pub-date:{year_to}-12-31")
        if venues:
            filters.append("container-title:" + ",".join(quote(v) for v in venues))
        if filters:
            params["filter"] = ",".join(filters)

        emitted = 0
        cursor = "*"
        while emitted < limit:
            params["cursor"] = cursor
            resp = self._session.get(f"{self.base_url}/works", params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("message", {}).get("items", [])
            if not items:
                break
            for item in items:
                yield self._to_record(item)
                emitted += 1
                if emitted >= limit:
                    return
            cursor = data.get("message", {}).get("next-cursor")
            if not cursor:
                break

    @staticmethod
    def _first_author(item: dict) -> list[Author]:
        authors: list[Author] = []
        for a in item.get("author", []) or []:
            name = " ".join(x for x in (a.get("given", ""), a.get("family", "")) if x)
            orcid = a.get("ORCID", "") or None
            if name:
                authors.append(Author(name=name, orcid=orcid))
        return authors

    def _to_record(self, item: dict) -> PaperRecord:
        issued = item.get("issued", {}).get("date-parts", [[None]])[0]
        year = issued[0] if issued and issued[0] else None
        doi = item.get("DOI", "")
        refs = item.get("reference", []) or []
        container = (item.get("container-title") or [""])[0]
        return PaperRecord(
            unique_id=doi,
            title=item.get("title", [""])[0] or "",
            abstract="",
            year=year,
            doi=doi or None,
            venue=container,
            authors=self._first_author(item),
            keywords=[],
            topics=[],
            citation_count=int(item.get("is-referenced-by-count", 0) or 0),
            reference_count=len(refs),
            source=self.name,
            has_fulltext=False,
        )

    def fetch_relations(self, unique_id: str) -> PaperRelations | None:
        """Fetch the reference list for a DOI (forward citations only)."""
        if not unique_id:
            return None
        resp = self._session.get(f"{self.base_url}/works/{quote(unique_id, safe='')}", timeout=60)
        if resp.status_code >= 400:
            return None
        message = resp.json().get("message", {})
        references = []
        for ref in message.get("reference", []) or []:
            ref_doi = ref.get("DOI", "")
            entry = {"id": ref_doi, "id_type": "doi", "title": ref.get("article-title", "")}
            if ref_doi:
                references.append(entry)
        return PaperRelations(unique_id=unique_id, references=references)

    def catalog(self) -> dict:
        """CrossRef has no catalog endpoint; return an empty capability map."""
        return {}
