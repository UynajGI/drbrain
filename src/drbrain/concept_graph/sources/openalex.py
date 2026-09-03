"""OpenAlex corpus source adapter.

Wraps the OpenAlex ``/works`` endpoint into the :class:`CorpusSource` protocol,
reusing the existing HTTP session and abstract-reconstruction helpers from
:mod:`drbrain.extractor.openalex`. Output is shaped identically to
:class:`SciverseSource` (``unique_id`` = OpenAlex work id).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from drbrain.concept_graph.sources.base import Author, PaperRecord, PaperRelations
from drbrain.extractor.openalex import _get_session, _reconstruct_abstract

OPENALEX_BASE = "https://api.openalex.org"
_PER_PAGE = 100


class OpenAlexSource:
    """Corpus source backed by the OpenAlex ``/works`` API."""

    name = "openalex"

    def __init__(
        self,
        base_url: str = OPENALEX_BASE,
        token: str | None = None,
        mailto: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        # OpenAlex raises the anonymous rate cap from ~10 to ~100 req/s for
        # requests that carry a polite-pool mailto.
        self.mailto = mailto
        self._session = _get_session()
        if self.token:
            self._session.headers.update({"Authorization": f"Bearer {self.token}"})

    def _params(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = dict(params or {})
        if self.mailto:
            params.setdefault("mailto", self.mailto)
        return params

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
        """Yield up to ``limit`` OpenAlex works matching the query / filters."""
        filters: list[str] = []
        if year_from is not None:
            filters.append(f"from_publication_date:{year_from}-01-01")
        if year_to is not None:
            filters.append(f"to_publication_date:{year_to}-12-31")
        if venues:
            filters.append("primary_location.source.display_name.search:" + " OR ".join(venues))

        params: dict[str, Any] = {"per_page": min(_PER_PAGE, limit)}
        if query:
            params["search"] = query
        if filters:
            params["filter"] = ",".join(filters)

        emitted = 0
        page = 1
        while emitted < limit:
            params["page"] = page
            resp = self._session.get(
                f"{self.base_url}/works", params=self._params(params), timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                break
            for work in results:
                yield self._to_record(work)
                emitted += 1
                if emitted >= limit:
                    return
            page += 1

    def _to_record(self, work: dict) -> PaperRecord:
        authors = []
        for authorship in work.get("authorships", []):
            author = authorship.get("author", {})
            authors.append(Author(name=author.get("display_name", ""), orcid=author.get("orcid")))
        keywords = [k.get("keyword", "") for k in work.get("keywords", []) if k.get("keyword")]
        topics = [
            t.get("display_name", "") for t in work.get("topics", []) if t.get("display_name")
        ]
        source = (work.get("primary_location") or {}).get("source") or {}
        doi = work.get("doi")
        if doi and doi.startswith("https://doi.org/"):
            doi = doi[len("https://doi.org/") :]
        return PaperRecord(
            unique_id=work.get("id", ""),
            title=work.get("title", "") or "",
            abstract=_reconstruct_abstract(work.get("abstract_inverted_index")),
            year=work.get("publication_year"),
            doi=doi,
            venue=source.get("display_name", "") or "",
            authors=authors,
            keywords=keywords,
            topics=topics,
            citation_count=int(work.get("cited_by_count", 0) or 0),
            reference_count=int(work.get("referenced_works_count", 0) or 0),
            source=self.name,
            has_fulltext=bool(work.get("open_access", {}).get("is_oa")),
        )

    def fetch_relations(self, unique_id: str) -> PaperRelations | None:
        """Fetch referenced works for an OpenAlex work id (citations omitted)."""
        if not unique_id:
            return None
        resp = self._session.get(
            f"{self.base_url}/works/{unique_id}", params=self._params(), timeout=60
        )
        if resp.status_code >= 400:
            return None
        work = resp.json()
        references = [{"id": ref} for ref in work.get("referenced_works", [])]
        return PaperRelations(unique_id=unique_id, references=references)

    def catalog(self) -> dict:
        """OpenAlex has no catalog endpoint; return an empty capability map."""
        return {}
