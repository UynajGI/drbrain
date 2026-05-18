"""Tests for federated search service.

TDD: tests written before implementation.
"""

from __future__ import annotations


class TestFederatedSearchModule:
    """Test that the fsearch module exists and exports expected functions."""

    def test_module_imports(self):
        from drbrain.services import fsearch

        assert fsearch is not None

    def test_search_local_exists(self):
        from drbrain.services.fsearch import search_local

        assert callable(search_local)

    def test_build_arxiv_query_url(self):
        from drbrain.services.fsearch import _build_arxiv_query_url

        url = _build_arxiv_query_url("machine learning", max_results=5)
        assert "search_query" in url
        assert "machine+learning" in url or "machine%20learning" in url
        assert "max_results=5" in url

    def test_build_arxiv_query_url_default_max(self):
        from drbrain.services.fsearch import _build_arxiv_query_url

        url = _build_arxiv_query_url("transformers")
        assert "max_results=10" in url


class TestNormalizeArxivRef:
    """Test arXiv ID normalization for dedup matching."""

    def test_normalize_simple(self):
        from drbrain.services.fsearch import _normalize_arxiv_ref

        assert _normalize_arxiv_ref("2301.12345") == "2301.12345"

    def test_normalize_with_prefix(self):
        from drbrain.services.fsearch import _normalize_arxiv_ref

        assert _normalize_arxiv_ref("arXiv:2301.12345") == "2301.12345"

    def test_normalize_with_version(self):
        from drbrain.services.fsearch import _normalize_arxiv_ref

        assert _normalize_arxiv_ref("2301.12345v2") == "2301.12345"

    def test_normalize_none(self):
        from drbrain.services.fsearch import _normalize_arxiv_ref

        assert _normalize_arxiv_ref("") == ""
        assert _normalize_arxiv_ref(None) == ""


class TestMergeResults:
    """Test result merging with ingested annotation."""

    def test_merge_no_local(self):
        from drbrain.services.fsearch import _merge_with_local_status

        arxiv_results = [
            {"title": "Paper A", "arxiv_id": "2301.00001"},
            {"title": "Paper B", "arxiv_id": "2301.00002"},
        ]
        merged = _merge_with_local_status(arxiv_results, set(), set())
        assert len(merged) == 2
        assert not merged[0]["ingested"]
        assert not merged[1]["ingested"]

    def test_merge_with_doi_match(self):
        from drbrain.services.fsearch import _merge_with_local_status

        arxiv_results = [
            {"title": "Paper A", "arxiv_id": "2301.1", "doi": "10.1234/a"},
        ]
        merged = _merge_with_local_status(arxiv_results, {"10.1234/a"}, set())
        assert merged[0]["ingested"] is True

    def test_merge_with_arxiv_id_match(self):
        from drbrain.services.fsearch import _merge_with_local_status

        arxiv_results = [
            {"title": "Paper A", "arxiv_id": "2301.00001"},
        ]
        merged = _merge_with_local_status(arxiv_results, set(), {"2301.00001"})
        assert merged[0]["ingested"] is True

    def test_merge_empty_arxiv_results(self):
        from drbrain.services.fsearch import _merge_with_local_status

        merged = _merge_with_local_status([], set(), set())
        assert merged == []
