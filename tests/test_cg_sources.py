"""Tests for corpus source data models, the Sciverse adapter and the registry."""

from __future__ import annotations

from typing import Any

import pytest

from drbrain.concept_graph.sources.base import (
    Author,
    CorpusSource,
    PaperRecord,
    is_success_envelope,
)
from drbrain.concept_graph.sources.registry import available_sources, get_source
from drbrain.concept_graph.sources.sciverse import SciverseSource

# ── base models ──────────────────────────────────────────────────────────────


def test_paper_record_defaults() -> None:
    rec = PaperRecord(unique_id="u1", title="T")
    assert rec.abstract == ""
    assert rec.year is None
    assert rec.doi is None
    assert rec.authors == []
    assert rec.keywords == []
    assert rec.has_fulltext is False


def test_author_optional_orcid() -> None:
    assert Author(name="A").orcid is None
    assert Author(name="B", orcid="0000").orcid == "0000"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"code": "SUCCESS", "biz_code": 0}, True),
        ({"code": "success"}, True),
        ({"biz_code": 0}, True),
        ({"code": "ERROR", "biz_code": 1}, False),
        ({}, True),  # no envelope → treated as success
        ({"results": []}, True),
    ],
)
def test_is_success_envelope(payload: dict, expected: bool) -> None:
    assert is_success_envelope(payload) is expected


# ── Sciverse adapter (fake client) ───────────────────────────────────────────


class FakeClient:
    """Records calls and returns canned payloads keyed by path."""

    def __init__(self, responses: dict[str, list[dict]]):
        self._responses = responses
        self.posts: list[tuple[str, dict]] = []
        self.gets: list[tuple[str, dict]] = []

    def post(self, path: str, json: dict[str, Any] | None = None) -> dict:
        self.posts.append((path, json or {}))
        queue = self._responses[path]
        return queue.pop(0) if len(queue) == 1 else queue.pop(0)

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        self.gets.append((path, params or {}))
        return self._responses[path][0]


def _search_result(**overrides: Any) -> dict:
    base = {
        "unique_id": "paper:10.1/x",
        "doc_id": "d_1",
        "title": "Graphene cathodes",
        "abstract": "We study graphene.",
        "doi": "10.1/x",
        "author": [{"name": "Alice", "orcid": "0001"}],
        "publication_published_year": 2024,
        "publication_venue_name_unified": "Adv. Energy Mater.",
        "keywords": ["graphene", "battery"],
        "topics": [{"display_name": "Materials science"}],
        "primary_topic": {"display_name": "Energy"},
        "citation_count": 42,
        "reference_count": 7,
    }
    base.update(overrides)
    return base


def test_sciverse_search_maps_record() -> None:
    client = FakeClient(
        {"meta-search": [{"results": [_search_result()], "total_pages": 1, "code": "SUCCESS"}]}
    )
    src = SciverseSource("tok", client=client)  # type: ignore[arg-type]
    records = list(src.search("graphene", year_from=2022, limit=10))
    assert len(records) == 1
    rec = records[0]
    assert rec.unique_id == "paper:10.1/x"
    assert rec.year == 2024
    assert rec.authors == [Author(name="Alice", orcid="0001")]
    assert rec.keywords == ["graphene", "battery"]
    assert "Energy" in rec.topics and "Materials science" in rec.topics
    assert rec.citation_count == 42
    assert rec.has_fulltext is True
    # filters propagated
    body = client.posts[0][1]
    assert {
        "field": "publication_published_year",
        "operator": "FILTER_OP_GTE",
        "value": 2022,
    } in body["filters"]


def test_sciverse_search_respects_limit_and_empty() -> None:
    client = FakeClient({"meta-search": [{"results": [], "total_pages": 0, "code": "SUCCESS"}]})
    src = SciverseSource("tok", client=client)  # type: ignore[arg-type]
    assert list(src.search(limit=5)) == []


def test_sciverse_fetch_relations_paginates() -> None:
    client = FakeClient(
        {
            "meta-paper-relations": [
                {"items": [{"id": "a"}], "total_pages": 2, "code": "SUCCESS"},
                {"items": [{"id": "b"}], "total_pages": 2, "code": "SUCCESS"},
            ]
            * 3
        }
    )
    src = SciverseSource("tok", client=client)  # type: ignore[arg-type]
    rel = src.fetch_relations("paper:1")
    assert rel is not None
    # each of the 3 relations paginated across 2 pages → 2 items each
    assert len(rel.citations) == 2
    assert len(rel.references) == 2
    assert len(rel.related_works) == 2


def test_sciverse_catalog_caches() -> None:
    client = FakeClient({"meta-catalog": [{"fields": [], "code": "SUCCESS"}]})
    src = SciverseSource("tok", client=client)  # type: ignore[arg-type]
    src.catalog()
    src.catalog()
    assert len(client.gets) == 1  # cached on second call


def test_sciverse_satisfies_protocol() -> None:
    src = SciverseSource("tok", client=FakeClient({}))  # type: ignore[arg-type]
    assert isinstance(src, CorpusSource)


# ── registry ─────────────────────────────────────────────────────────────────


def test_registry_lists_builtin_sources() -> None:
    names = available_sources()
    assert "sciverse" in names
    assert "openalex" in names


def test_registry_unknown_source_raises() -> None:
    with pytest.raises(KeyError):
        get_source("does-not-exist")
