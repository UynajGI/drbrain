"""Tests for federated search service (src/drbrain/services/fsearch.py).

All HTTP and DB calls are mocked.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest

from drbrain.services import fsearch
from drbrain.services.fsearch import (
    _build_arxiv_query_url,
    _merge_with_local_status,
    _normalize_arxiv_ref,
    search_arxiv,
    search_local,
)

# ── module surface ─────────────────────────────────────────────────────────


class TestModuleSurface:
    def test_module_imports(self):
        assert fsearch is not None

    def test_search_local_exists(self):
        assert callable(search_local)

    def test_search_arxiv_exists(self):
        assert callable(search_arxiv)


# ── _build_arxiv_query_url ─────────────────────────────────────────────────


class TestBuildArxivQueryUrl:
    def test_query_encoded(self):
        url = _build_arxiv_query_url("machine learning", max_results=5)
        assert "search_query" in url
        assert "machine+learning" in url or "machine%20learning" in url
        assert "max_results=5" in url

    def test_default_max_results(self):
        url = _build_arxiv_query_url("transformers")
        assert "max_results=10" in url

    def test_special_characters_encoded(self):
        url = _build_arxiv_query_url("graph & tree", max_results=3)
        assert "max_results=3" in url
        # & must be encoded in the query term, not split as a param
        assert "graph" in url


# ── _normalize_arxiv_ref ───────────────────────────────────────────────────


class TestNormalizeArxivRef:
    def test_simple(self):
        assert _normalize_arxiv_ref("2301.12345") == "2301.12345"

    def test_with_prefix(self):
        assert _normalize_arxiv_ref("arXiv:2301.12345") == "2301.12345"

    def test_prefix_case_insensitive(self):
        assert _normalize_arxiv_ref("ARXIV:2301.12345") == "2301.12345"

    def test_with_version(self):
        assert _normalize_arxiv_ref("2301.12345v2") == "2301.12345"

    def test_prefix_and_version(self):
        assert _normalize_arxiv_ref("arXiv:2301.12345v3") == "2301.12345"

    def test_none_and_empty(self):
        assert _normalize_arxiv_ref("") == ""
        assert _normalize_arxiv_ref(None) == ""


# ── _merge_with_local_status ───────────────────────────────────────────────


class TestMergeResults:
    def test_no_local(self):
        results = [
            {"title": "A", "arxiv_id": "2301.00001"},
            {"title": "B", "arxiv_id": "2301.00002"},
        ]
        merged = _merge_with_local_status(results, set(), set())
        assert len(merged) == 2
        assert not merged[0]["ingested"]
        assert not merged[1]["ingested"]
        # Original keys preserved
        assert merged[0]["title"] == "A"

    def test_doi_match_case_insensitive(self):
        results = [{"title": "A", "arxiv_id": "2301.1", "doi": "10.1234/A"}]
        merged = _merge_with_local_status(results, {"10.1234/a"}, set())
        assert merged[0]["ingested"] is True

    def test_arxiv_id_match_uses_normalization(self):
        # local stores normalized (versionless) form
        results = [{"title": "A", "arxiv_id": "2301.00001v2"}]
        merged = _merge_with_local_status(results, set(), {"2301.00001"})
        assert merged[0]["ingested"] is True

    def test_empty(self):
        assert _merge_with_local_status([], set(), set()) == []

    def test_missing_keys_safe(self):
        # Result dict lacking doi/arxiv_id entirely
        merged = _merge_with_local_status([{"title": "X"}], {"10.1/x"}, {"2301.1"})
        assert merged[0]["ingested"] is False
        assert merged[0]["title"] == "X"


# ── search_arxiv ───────────────────────────────────────────────────────────


def _atom_xml(entries_xml: str) -> str:
    """Build an Atom feed string using the default XML namespace."""
    return f'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">{entries_xml}</feed>'


def _entry(
    title: str = "Paper",
    authors: list[str] | None = None,
    summary: str = "Abstract.",
    published: str = "2023-01-01T00:00:00Z",
    arxiv_id: str = "2301.00001",
    doi: str | None = None,
) -> str:
    author_xml = "".join(f"<author><name>{a}</name></author>" for a in (authors or ["Author A"]))
    links = f'<link href="http://arxiv.org/abs/{arxiv_id}"/>'
    if doi:
        links += f'<link href="http://dx.doi.org/{doi}"/>'
    return (
        "<entry>"
        f"<title>{title}</title>"
        f"{author_xml}"
        f"<summary>{summary}</summary>"
        f"<published>{published}</published>"
        f"{links}"
        "</entry>"
    )


@pytest.fixture(autouse=False)
def fake_defusedxml(monkeypatch):
    """Install a minimal fake defusedxml so the deferred import in search_arxiv resolves."""
    import sys
    import types
    import xml.etree.ElementTree as ET

    mod = types.ModuleType("defusedxml")
    sub = types.ModuleType("defusedxml.ElementTree")
    sub.parse = ET.parse
    sub.fromstring = ET.fromstring
    mod.ElementTree = sub
    monkeypatch.setitem(sys.modules, "defusedxml", mod)
    monkeypatch.setitem(sys.modules, "defusedxml.ElementTree", sub)
    return sub


def _patch_arxiv_with_xml(xml: str):
    """Patch urlopen + defusedxml to yield a real ElementTree parsed from xml."""
    import xml.etree.ElementTree as ET

    real_root = ET.fromstring(xml)
    fake_resp = MagicMock()
    fake_resp.__enter__ = MagicMock(return_value=io.BytesIO(xml.encode()))
    fake_resp.__exit__ = MagicMock(return_value=False)
    return (
        patch("urllib.request.urlopen", return_value=fake_resp),
        patch("defusedxml.ElementTree.parse", return_value=real_root),
    )


class TestSearchArxiv:
    def test_parses_entries(self, fake_defusedxml):
        xml = _atom_xml(_entry(title="My Paper", arxiv_id="2301.1", doi="10.1/x"))
        p1, p2 = _patch_arxiv_with_xml(xml)
        with p1, p2:
            results = search_arxiv("query")
        assert len(results) == 1
        assert results[0]["title"] == "My Paper"
        assert results[0]["arxiv_id"] == "2301.1"
        assert results[0]["doi"] == "10.1/x"
        assert results[0]["year"] == 2023

    def test_authors_parsed(self, fake_defusedxml):
        xml = _atom_xml(_entry(title="T", authors=["Alice", "Bob"], arxiv_id="2301.2"))
        p1, p2 = _patch_arxiv_with_xml(xml)
        with p1, p2:
            results = search_arxiv("q")
        assert results[0]["authors"] == ["Alice", "Bob"]

    def test_multiple_entries(self, fake_defusedxml):
        xml = _atom_xml(_entry(title="A") + _entry(title="B", arxiv_id="2301.2"))
        p1, p2 = _patch_arxiv_with_xml(xml)
        with p1, p2:
            results = search_arxiv("q")
        assert len(results) == 2
        titles = {r["title"] for r in results}
        assert titles == {"A", "B"}

    def test_network_error_returns_empty(self, fake_defusedxml):
        with patch("urllib.request.urlopen", side_effect=Exception("network down")):
            assert search_arxiv("q") == []

    def test_request_uses_user_agent_header(self, fake_defusedxml):
        fake_resp = MagicMock()
        fake_resp.__enter__ = MagicMock(return_value=io.BytesIO(b"<x/>"))
        fake_resp.__exit__ = MagicMock(return_value=False)
        with (
            patch("urllib.request.urlopen", return_value=fake_resp) as uo,
            patch(
                "defusedxml.ElementTree.parse",
                return_value=MagicMock(iter=lambda tag: iter([])),
            ),
        ):
            search_arxiv("q")
        req = uo.call_args[0][0]
        assert req.headers.get("User-agent") == "DrBrain/0.1"

    def test_empty_year_when_published_missing(self, fake_defusedxml):
        xml = _atom_xml("<entry><title>T</title><link href='http://arxiv.org/abs/1.2'/></entry>")
        p1, p2 = _patch_arxiv_with_xml(xml)
        with p1, p2:
            results = search_arxiv("q")
        assert len(results) == 1
        assert results[0]["year"] is None
        assert results[0]["arxiv_id"] == "1.2"

    def test_published_too_short_for_year(self, fake_defusedxml):
        xml = _atom_xml(_entry(published="20", arxiv_id="9.9"))
        p1, p2 = _patch_arxiv_with_xml(xml)
        with p1, p2:
            results = search_arxiv("q")
        assert results[0]["year"] is None


# ── search_local ───────────────────────────────────────────────────────────


class TestSearchLocal:
    def test_missing_db_returns_empty(self, tmp_path):
        nonexistent = str(tmp_path / "does_not_exist.db")
        assert search_local(nonexistent, "anything") == []

    def test_returns_rows_from_connection(self, tmp_path):
        db_path = tmp_path / "db.sqlite"
        db_path.write_text("")  # exists so the existence check passes
        fake_conn = MagicMock()
        fake_conn.execute.return_value.fetchall.return_value = [
            ("p1", "Title One", 2023, "Author A"),
            ("p2", "Title Two", 2022, "Author B"),
        ]
        with (
            patch("drbrain.storage.connection.connect_wal", return_value=fake_conn),
        ):
            results = search_local(str(db_path), "Title", limit=5)
        assert len(results) == 2
        assert results[0]["local_id"] == "p1"
        assert results[0]["title"] == "Title One"
        assert results[0]["year"] == 2023
        assert results[0]["score"] == 1.0
        fake_conn.close.assert_called_once()

    def test_query_error_returns_empty(self, tmp_path):
        db_path = tmp_path / "db.sqlite"
        db_path.write_text("")
        fake_conn = MagicMock()
        fake_conn.execute.side_effect = Exception("sql error")
        with patch("drbrain.storage.connection.connect_wal", return_value=fake_conn):
            assert search_local(str(db_path), "x") == []

    def test_untitled_fallback_when_title_null(self, tmp_path):
        db_path = tmp_path / "db.sqlite"
        db_path.write_text("")
        fake_conn = MagicMock()
        fake_conn.execute.return_value.fetchall.return_value = [("p3", None, 2020, "")]
        with patch("drbrain.storage.connection.connect_wal", return_value=fake_conn):
            results = search_local(str(db_path), "x")
        assert results[0]["title"] == "Untitled"
        assert results[0]["authors"] == ""

    def test_search_term_wrapped_with_wildcards(self, tmp_path):
        """Verifies the LIKE pattern passed to SQL."""
        db_path = tmp_path / "db.sqlite"
        db_path.write_text("")
        fake_conn = MagicMock()
        fake_conn.execute.return_value.fetchall.return_value = []
        with patch("drbrain.storage.connection.connect_wal", return_value=fake_conn):
            search_local(str(db_path), "neural", limit=10)
        sql_args = fake_conn.execute.call_args[0][1]
        # First three params are the LIKE search_term
        assert sql_args[0] == "%neural%"
        assert sql_args[1] == "%neural%"
        assert sql_args[2] == "%neural%"
        assert sql_args[3] == 10
