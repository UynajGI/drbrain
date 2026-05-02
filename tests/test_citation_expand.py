"""Tests for citation.py: expand_citations, match_to_local, parse_s2_response."""

import tempfile
import unittest.mock
from pathlib import Path

from drbrain.extractor.cache import ApiCache
from drbrain.extractor.citation import (
    _crossref_doi_enrich,
    _expand_with_openalex,
    expand_citations,
    ext_ids_from_s2,
    fetch_s2_paper,
    match_to_local,
    parse_s2_response,
    search_s2,
)
from drbrain.storage.database import Database


def _make_db_with_paper(
    local_id: str,
    title: str,
    year: int,
    doi: str = None,
    arxiv: str = None,
    s2_id: str = None,
    openalex_id: str = None,
) -> Database:
    """Create a temp DB with one paper."""
    td = tempfile.mkdtemp()
    db = Database(Path(td) / "test.db")
    db.insert_paper(local_id, title, year, "uploaded")
    db.insert_paper_ids(local_id, doi=doi, arxiv=arxiv, s2_id=s2_id, openalex_id=openalex_id)
    db.commit()
    return db


# -- parse_s2_response --


def test_parse_s2_response_minimal():
    """parse_s2_response handles minimal response."""
    data = {"paperId": "abc123", "title": "Test", "year": 2024}
    parsed = parse_s2_response(data)
    assert parsed["s2_id"] == "abc123"
    assert parsed["title"] == "Test"
    assert parsed["year"] == 2024
    assert parsed["citation_count"] == 0


def test_parse_s2_response_with_ids():
    """parse_s2_response extracts all external IDs."""
    data = {
        "paperId": "abc",
        "title": "Test",
        "year": 2024,
        "externalIds": {
            "DOI": "10.1234/test",
            "ArXiv": "2401.12345",
            "OpenAlex": "W123",
        },
        "citationCount": 42,
    }
    parsed = parse_s2_response(data)
    assert parsed["doi"] == "10.1234/test"
    assert parsed["arxiv"] == "2401.12345"
    assert parsed["openalex_id"] == "W123"
    assert parsed["citation_count"] == 42


def test_parse_s2_response_null_ext_ids():
    """parse_s2_response handles null externalIds."""
    data = {"paperId": "abc", "title": "Test", "year": 2024, "externalIds": None}
    parsed = parse_s2_response(data)
    assert parsed["doi"] is None


# -- match_to_local --


def test_match_to_local_by_doi():
    """match_to_local finds paper by DOI."""
    db = _make_db_with_paper("p1", "Test Paper", 2024, doi="10.1234/test")
    ref = {"title": "Test Paper", "year": 2024, "doi": "10.1234/test"}
    entry = match_to_local(db, ref)
    assert entry.in_graph is True
    assert entry.local_id == "p1"
    db.close()


def test_match_to_local_by_arxiv():
    """match_to_local finds paper by arXiv ID."""
    db = _make_db_with_paper("p1", "Test Paper", 2024, arxiv="2401.12345")
    ref = {"title": "Test Paper", "year": 2024, "arxiv": "2401.12345"}
    entry = match_to_local(db, ref)
    assert entry.in_graph is True
    assert entry.local_id == "p1"
    db.close()


def test_match_to_local_by_s2_id():
    """match_to_local finds paper by S2 ID."""
    db = _make_db_with_paper("p1", "Test Paper", 2024, s2_id="abc123")
    ref = {"title": "Test Paper", "year": 2024, "s2_id": "abc123"}
    entry = match_to_local(db, ref)
    assert entry.in_graph is True
    assert entry.local_id == "p1"
    db.close()


def test_match_to_local_by_title_year():
    """match_to_local finds paper by exact title+year."""
    db = _make_db_with_paper("p1", "Exact Title", 2024)
    ref = {"title": "Exact Title", "year": 2024}
    entry = match_to_local(db, ref)
    assert entry.in_graph is True
    assert entry.local_id == "p1"
    db.close()


def test_match_to_local_not_found():
    """match_to_local returns in_graph=False when no match."""
    db = _make_db_with_paper("p1", "Different Paper", 2024)
    ref = {"title": "Unknown Paper", "year": 2025, "doi": "10.9999/nope"}
    entry = match_to_local(db, ref)
    assert entry.in_graph is False
    assert entry.local_id is None
    db.close()


# -- expand_citations --


def test_expand_citations_creates_placeholder_neighbors():
    """expand_citations creates placeholder nodes for references not in graph."""
    s2_data = {
        "paperId": "s2_abc",
        "title": "Seed Paper",
        "year": 2024,
        "externalIds": {"DOI": "10.1234/seed"},
        "citationCount": 5,
        "references": [
            {
                "paperId": "ref1",
                "title": "Ref Paper 1",
                "year": 2020,
                "externalIds": None,
                "citationCount": 0,
            },
        ],
        "citations": [],
    }

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        db = Database(str(db_path))
        db.insert_paper("p1", "Seed Paper", 2024, "uploaded")
        db.insert_paper_ids("p1", doi="10.1234/seed", s2_id="s2_abc")
        db.commit()

        cfg = {"api": {"s2_rate_limit": 100}}

        with unittest.mock.patch("drbrain.extractor.citation.fetch_s2_paper", return_value=s2_data):
            refs, cits = expand_citations(db, "p1", cfg)

        # Reference should be created as placeholder
        papers = db.get_all_papers()
        placeholder_titles = {p["title"] for p in papers if p["status"] == "placeholder"}
        assert "Ref Paper 1" in placeholder_titles

        # Edge should exist
        edges = db.conn.execute(
            "SELECT src_id, dst_id, relation FROM edges WHERE relation='cites'"
        ).fetchall()
        assert len(edges) >= 1

        db.close()


def test_expand_citations_matches_existing_neighbor():
    """expand_citations marks existing papers as in_graph."""
    s2_data = {
        "paperId": "s2_abc",
        "title": "Seed Paper",
        "year": 2024,
        "externalIds": {"DOI": "10.1234/seed"},
        "citationCount": 0,
        "references": [
            {
                "paperId": "ref1",
                "title": "Existing Ref",
                "year": 2020,
                "externalIds": {"DOI": "10.5678/existing"},
                "citationCount": 0,
            },
        ],
        "citations": [],
    }

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        db = Database(str(db_path))
        db.insert_paper("p1", "Seed Paper", 2024, "uploaded")
        db.insert_paper_ids("p1", doi="10.1234/seed", s2_id="s2_abc")
        db.insert_paper("p2", "Existing Ref", 2020, "uploaded")
        db.insert_paper_ids("p2", doi="10.5678/existing")
        db.commit()

        cfg = {"api": {"s2_rate_limit": 100}}

        with unittest.mock.patch("drbrain.extractor.citation.fetch_s2_paper", return_value=s2_data):
            refs, cits = expand_citations(db, "p1", cfg)

        assert len(refs) == 1
        assert refs[0].in_graph is True
        assert refs[0].local_id == "p2"
        db.close()


def test_expand_citations_returns_empty_when_no_paper():
    """expand_citations returns empty when paper not found."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        refs, cits = expand_citations(db, "nonexistent", {})
        assert refs == []
        assert cits == []
        db.close()


def test_expand_citations_returns_empty_when_no_s2_id():
    """expand_citations returns empty when paper has no S2 ID and search fails."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "No ID Paper", 2024, "uploaded")
        db.commit()

        with (
            unittest.mock.patch("drbrain.extractor.citation.search_s2", return_value=[]),
            unittest.mock.patch(
                "drbrain.extractor.citation._expand_with_openalex", return_value=([], [])
            ),
        ):
            refs, cits = expand_citations(db, "p1", {})

        assert refs == []
        assert cits == []
        db.close()


def test_expand_citations_handles_citations_direction():
    """Citations create edges from citing paper TO seed paper (cited_by)."""
    s2_data = {
        "paperId": "s2_seed",
        "title": "Seed Paper",
        "year": 2024,
        "externalIds": None,
        "citationCount": 0,
        "references": [],
        "citations": [
            {
                "paperId": "cit1",
                "title": "Citing Paper",
                "year": 2025,
                "externalIds": None,
                "citationCount": 0,
            },
        ],
    }

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        db = Database(str(db_path))
        db.insert_paper("p1", "Seed Paper", 2024, "uploaded")
        db.insert_paper_ids("p1", s2_id="s2_seed")
        db.commit()

        cfg = {"api": {"s2_rate_limit": 100}}

        with unittest.mock.patch("drbrain.extractor.citation.fetch_s2_paper", return_value=s2_data):
            refs, cits = expand_citations(db, "p1", cfg)

        # cited_by edge: placeholder -> seed
        edges = db.conn.execute(
            "SELECT src_id, dst_id, relation FROM edges WHERE relation='cited_by'"
        ).fetchall()
        assert len(edges) >= 1
        # dst should be the seed paper
        assert any(e[1] == "p1" for e in edges)

        db.close()


def test_expand_citations_batches_placeholder_commits():
    """expand_citations uses batch commits for multiple placeholders, not one-per-item."""
    s2_data = {
        "paperId": "s2_batch",
        "title": "Seed Paper",
        "year": 2024,
        "externalIds": {"DOI": "10.1234/seed"},
        "citationCount": 10,
        "references": [
            {
                "paperId": f"ref{i}",
                "title": f"Ref Paper {i}",
                "year": 2020 + i,
                "externalIds": None,
                "citationCount": 0,
            }
            for i in range(1, 6)
        ],
        "citations": [
            {
                "paperId": f"cit{i}",
                "title": f"Citing Paper {i}",
                "year": 2025,
                "externalIds": None,
                "citationCount": 0,
            }
            for i in range(1, 4)
        ],
    }

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        db = Database(str(db_path))
        db.insert_paper("p1", "Seed Paper", 2024, "uploaded")
        db.insert_paper_ids("p1", doi="10.1234/seed", s2_id="s2_batch")
        db.commit()

        commit_count = [0]
        original_commit = db.commit

        def counting_commit():
            commit_count[0] += 1
            return original_commit()

        db.commit = counting_commit

        cfg = {"api": {"s2_rate_limit": 100}}

        with unittest.mock.patch("drbrain.extractor.citation.fetch_s2_paper", return_value=s2_data):
            refs, cits = expand_citations(db, "p1", cfg)

        assert len(refs) == 5
        assert len(cits) == 3

        # All 8 placeholders should be created with batch commits
        # Old code: 8 commits (one per item). New code: ~2 commits (one per batch).
        assert commit_count[0] <= 4, f"Expected <=4 batched commits, got {commit_count[0]}"

        db.close()


# -- original non-retry functions --


def test_fetch_s2_paper_success():
    """fetch_s2_paper returns parsed JSON on success."""
    mock_resp = unittest.mock.Mock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = unittest.mock.Mock()
    mock_resp.json.return_value = {"paperId": "abc"}

    with unittest.mock.patch("requests.get", return_value=mock_resp):
        result = fetch_s2_paper("abc")
        assert result is not None
        assert result["paperId"] == "abc"


def test_fetch_s2_paper_error():
    """fetch_s2_paper returns None on error."""
    with unittest.mock.patch("requests.get", side_effect=Exception("fail")):
        result = fetch_s2_paper("abc")
        assert result is None


def test_search_s2_success():
    """search_s2 returns data list on success."""
    mock_resp = unittest.mock.Mock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = unittest.mock.Mock()
    mock_resp.json.return_value = {"data": [{"paperId": "a"}, {"paperId": "b"}]}

    with unittest.mock.patch("requests.get", return_value=mock_resp):
        results = search_s2("test query")
        assert len(results) == 2


def test_search_s2_error():
    """search_s2 returns empty list on error."""
    with unittest.mock.patch("requests.get", side_effect=Exception("fail")):
        results = search_s2("test")
        assert results == []


def test_search_s2_uses_api_key():
    """search_s2 includes x-api-key header when provided."""
    captured = {}

    def mock_get(*args, **kwargs):
        captured["headers"] = kwargs.get("headers", {})
        resp = unittest.mock.Mock()
        resp.status_code = 200
        resp.raise_for_status = unittest.mock.Mock()
        resp.json.return_value = {"data": []}
        return resp

    with unittest.mock.patch("requests.get", side_effect=mock_get):
        search_s2("test", api_key="key-123")
        assert captured.get("headers", {}).get("x-api-key") == "key-123"


# -- fetch_s2_paper: cache hit ---


def test_fetch_s2_paper_cache_hit():
    """fetch_s2_paper returns cached data without HTTP request when cache hit."""
    cached_data = {"paperId": "cached_abc", "title": "Cached Paper", "year": 2023}

    with tempfile.TemporaryDirectory() as td:
        cache = ApiCache(td, ttl=3600)
        cache.set("s2_paper:cached_abc", cached_data)

        # requests.get should NOT be called
        with unittest.mock.patch("requests.get") as mock_get:
            result = fetch_s2_paper("cached_abc", cache=cache)
            mock_get.assert_not_called()

    assert result is not None
    assert result["paperId"] == "cached_abc"
    assert result["title"] == "Cached Paper"


def test_fetch_s2_paper_cache_expired():
    """fetch_s2_paper ignores expired cache and makes HTTP request."""
    cached_data = {"paperId": "expired_abc", "title": "Expired Paper"}

    with tempfile.TemporaryDirectory() as td:
        cache = ApiCache(td, ttl=0)  # TTL=0 means always expired
        cache.set("s2_paper:expired_abc", cached_data)

        fresh_data = {"paperId": "fresh_abc", "title": "Fresh Paper", "year": 2024}
        mock_resp = unittest.mock.Mock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = unittest.mock.Mock()
        mock_resp.json.return_value = fresh_data

        with unittest.mock.patch("requests.get", return_value=mock_resp):
            result = fetch_s2_paper("expired_abc", cache=cache)

    assert result is not None
    assert result["paperId"] == "fresh_abc"


# -- fetch_s2_paper: API key header ---


def test_fetch_s2_paper_sets_api_key_header():
    """fetch_s2_paper includes x-api-key header when api_key provided."""
    mock_resp = unittest.mock.Mock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = unittest.mock.Mock()
    mock_resp.json.return_value = {"paperId": "abc", "title": "Test"}

    captured = {}

    def _capture(url, **kwargs):
        captured["headers"] = kwargs.get("headers", {})
        return mock_resp

    with unittest.mock.patch("requests.get", side_effect=_capture):
        fetch_s2_paper("abc", api_key="my-secret-key")

    assert captured["headers"].get("x-api-key") == "my-secret-key"


# -- _crossref_doi_enrich ---


def test_crossref_doi_enrich_via_arxiv():
    """_crossref_doi_enrich resolves DOI via arXiv when available."""
    paper = {"arxiv": "2401.12345", "title": "Test Paper"}

    with unittest.mock.patch(
        "drbrain.extractor.citation.fetch_doi_by_arxiv",
        return_value={"doi": "10.1234/from_arxiv", "title": "Test Paper", "year": 2024},
    ):
        result = _crossref_doi_enrich(paper, email="test@example.com")

    assert result is not None
    assert result["doi"] == "10.1234/from_arxiv"


def test_crossref_doi_enrich_via_title():
    """_crossref_doi_enrich falls back to title search when arXiv fails."""
    paper = {"arxiv": "2401.12345", "title": "Unique Paper Title"}

    with (
        unittest.mock.patch("drbrain.extractor.citation.fetch_doi_by_arxiv", return_value=None),
        unittest.mock.patch(
            "drbrain.extractor.citation.fetch_doi_by_title",
            return_value={"doi": "10.5678/from_title", "title": "Unique Paper Title", "year": 2023},
        ),
    ):
        result = _crossref_doi_enrich(paper, email="test@example.com")

    assert result is not None
    assert result["doi"] == "10.5678/from_title"


def test_crossref_doi_enrich_no_title_match():
    """_crossref_doi_enrich returns None when title lookup also fails."""
    paper = {"arxiv": "2401.12345", "title": "Title With No Match"}

    with (
        unittest.mock.patch("drbrain.extractor.citation.fetch_doi_by_arxiv", return_value=None),
        unittest.mock.patch("drbrain.extractor.citation.fetch_doi_by_title", return_value=None),
    ):
        result = _crossref_doi_enrich(paper)

    assert result is None


def test_crossref_doi_enrich_no_arxiv_no_title():
    """_crossref_doi_enrich returns None when paper has no arXiv or title."""
    result = _crossref_doi_enrich({})
    assert result is None


# -- ext_ids_from_s2 ---


def test_ext_ids_from_s2_full():
    """ext_ids_from_s2 extracts all external IDs from S2 response."""
    data = {
        "paperId": "s2_12345",
        "externalIds": {
            "DOI": "10.1234/s2test",
            "ArXiv": "2401.12345v2",
            "OpenAlex": "W999",
        },
    }
    result = ext_ids_from_s2(data)
    assert result["doi"] == "10.1234/s2test"
    assert result["arxiv"] == "2401.12345"  # version stripped
    assert result["s2_id"] == "s2_12345"
    assert result["openalex_id"] == "W999"


def test_ext_ids_from_s2_no_external_ids():
    """ext_ids_from_s2 handles missing externalIds."""
    data = {"paperId": "s2_minimal", "title": "Minimal"}
    result = ext_ids_from_s2(data)
    assert result["s2_id"] == "s2_minimal"
    assert result["doi"] is None
    assert result["arxiv"] is None


def test_ext_ids_from_s2_null_external_ids():
    """ext_ids_from_s2 handles null externalIds."""
    data = {"paperId": "s2_null", "externalIds": None}
    result = ext_ids_from_s2(data)
    assert result["s2_id"] == "s2_null"
    assert result["doi"] is None


# -- expand_citations: OpenAlex fallback ---


def test_expand_citations_openalex_fallback():
    """expand_citations falls back to OpenAlex when S2 returns no data."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Fallback Paper", 2024, "uploaded")
        db.insert_paper_ids("p1", s2_id="s2_fallback")
        db.commit()

        cfg = {"api": {}}

        fake_refs = [
            type(
                "RefEntry",
                (),
                {
                    "title": "OARef",
                    "year": 2022,
                    "in_graph": False,
                    "local_id": None,
                    "ids": {"doi": "10.9999/oaref"},
                },
            )()
        ]

        with (
            unittest.mock.patch("drbrain.extractor.citation.fetch_s2_paper", return_value=None),
            unittest.mock.patch(
                "drbrain.extractor.citation._crossref_doi_enrich", return_value=None
            ),
            unittest.mock.patch(
                "drbrain.extractor.citation._expand_with_openalex",
                return_value=(fake_refs, []),
            ) as mock_oa,
        ):
            refs, cits = expand_citations(db, "p1", cfg)

        mock_oa.assert_called_once()
        assert len(refs) == 1
        assert refs[0].title == "OARef"
        assert cits == []

        db.close()


# -- expand_citations: finds paper via S2 title search ---


def test_expand_citations_finds_s2_id_via_search():
    """expand_citations finds paper via S2 title search when s2_id is missing."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Unique S2 Search Paper", 2024, "uploaded")
        db.insert_paper_ids("p1")  # create paper_ids row, no s2_id
        db.commit()

        # search_s2 finds the paper
        search_result = [
            {
                "paperId": "s2_discovered",
                "title": "Unique S2 Search Paper",
                "year": 2024,
                "externalIds": {"DOI": "10.1234/discovered"},
            }
        ]

        s2_data = {
            "paperId": "s2_discovered",
            "title": "Unique S2 Search Paper",
            "year": 2024,
            "externalIds": {"DOI": "10.1234/discovered"},
            "authors": [],
            "citationCount": 0,
            "references": [],
            "citations": [],
        }

        cfg = {"api": {"s2_rate_limit": 100}}

        with (
            unittest.mock.patch("drbrain.extractor.citation.search_s2", return_value=search_result),
            unittest.mock.patch("drbrain.extractor.citation.fetch_s2_paper", return_value=s2_data),
        ):
            refs, cits = expand_citations(db, "p1", cfg)

        # s2_id should have been backfilled
        s2_id = db.conn.execute("SELECT s2_id FROM paper_ids WHERE local_id = 'p1'").fetchone()[0]
        assert s2_id == "s2_discovered"

        db.close()


# -- expand_citations: S2 returns data without DOI, triggers crossref enrichment ---


def test_expand_citations_crossref_enrich_on_missing_doi():
    """expand_citations calls _crossref_doi_enrich when S2 has no DOI."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "No DOI Paper", 2024, "uploaded")
        db.insert_paper_ids("p1", s2_id="s2_nodoi")
        db.commit()

        s2_data = {
            "paperId": "s2_nodoi",
            "title": "No DOI Paper",
            "year": 2024,
            "externalIds": {},  # no DOI
            "authors": [],
            "citationCount": 0,
            "references": [],
            "citations": [],
        }

        cfg = {"api": {"s2_rate_limit": 100, "crossref_email": "test@ex.com"}}

        with (
            unittest.mock.patch("drbrain.extractor.citation.fetch_s2_paper", return_value=s2_data),
            unittest.mock.patch(
                "drbrain.extractor.citation._crossref_doi_enrich",
                return_value={"doi": "10.5678/enriched", "title": "No DOI Paper", "year": 2024},
            ) as mock_enrich,
        ):
            expand_citations(db, "p1", cfg)

        mock_enrich.assert_called_once()
        doi = db.conn.execute("SELECT doi FROM paper_ids WHERE local_id = 'p1'").fetchone()[0]
        assert doi == "10.5678/enriched"

        db.close()


# -- match_to_local: openalex_id ---


def test_match_to_local_by_openalex_id():
    """match_to_local finds paper by OpenAlex ID."""
    db = _make_db_with_paper("p1", "Test Paper", 2024, openalex_id="W12345")
    ref = {"title": "Test Paper", "year": 2024, "openalex_id": "W12345"}
    entry = match_to_local(db, ref)
    assert entry.in_graph is True
    assert entry.local_id == "p1"
    db.close()


# -- search_s2 with cache ---


def test_search_s2_cache_hit():
    """search_s2 returns cached results without HTTP request when cache hit."""
    cached_results = [{"paperId": "cached_search", "title": "Cached Search Result"}]

    with tempfile.TemporaryDirectory() as td:
        cache = ApiCache(td, ttl=3600)
        cache.set("s2_search:cache query:50", cached_results)

        with unittest.mock.patch("requests.get") as mock_get:
            results = search_s2("cache query", limit=50, cache=cache)
            mock_get.assert_not_called()

    assert len(results) == 1
    assert results[0]["paperId"] == "cached_search"


# -- expand_citations: S2 failure + crossref fallback success ---


def test_expand_citations_s2_fails_crossref_succeeds():
    """expand_citations sets DOI via CrossRef when S2 fails and paper has no DOI."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "S2 Fail Paper", 2024, "uploaded")
        db.insert_paper_ids("p1", s2_id="s2_dead")
        db.commit()

        cfg = {"api": {"crossref_email": "test@ex.com"}}

        with (
            unittest.mock.patch("drbrain.extractor.citation.fetch_s2_paper", return_value=None),
            unittest.mock.patch(
                "drbrain.extractor.citation._crossref_doi_enrich",
                return_value={
                    "doi": "10.9999/s2fail_fixed",
                    "title": "S2 Fail Paper",
                    "year": 2024,
                },
            ),
            unittest.mock.patch(
                "drbrain.extractor.citation._expand_with_openalex",
                return_value=([], []),
            ),
        ):
            expand_citations(db, "p1", cfg)

        doi = db.conn.execute("SELECT doi FROM paper_ids WHERE local_id = 'p1'").fetchone()[0]
        assert doi == "10.9999/s2fail_fixed"

        db.close()


# -- search_s2: cache population ---


def test_search_s2_populates_cache():
    """search_s2 stores results in cache after successful API call."""
    mock_resp = unittest.mock.Mock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = unittest.mock.Mock()
    mock_resp.json.return_value = {"data": [{"paperId": "fresh_x"}]}

    with tempfile.TemporaryDirectory() as td:
        cache = ApiCache(td, ttl=3600)

        with unittest.mock.patch("requests.get", return_value=mock_resp):
            results = search_s2("populate test", limit=10, cache=cache)

        assert len(results) == 1

        # Verify it was cached
        cached = cache.get("s2_search:populate test:10")
        assert cached is not None
        assert cached[0]["paperId"] == "fresh_x"


# -- _expand_with_openalex: title-only paper ---


def test_expand_with_openalex_title_only():
    """_expand_with_openalex finds paper by title and creates reference placeholders."""
    oa_work = {
        "doi": "10.1111/oa_title",
        "title": "OpenAlex Title Paper",
        "year": 2023,
        "openalex_id": "https://openalex.org/W_oa",
        "referenced_works": [
            "https://openalex.org/W_ref_a",
            "https://openalex.org/W_ref_b",
        ],
    }

    ref_works = [
        {
            "doi": "10.2222/ref_a",
            "title": "Ref A",
            "year": 2021,
            "openalex_id": "https://openalex.org/W_ref_a",
        },
        {
            "doi": "10.3333/ref_b",
            "title": "Ref B",
            "year": 2020,
            "openalex_id": "https://openalex.org/W_ref_b",
        },
    ]

    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "OpenAlex Title Paper", 2023, "uploaded")
        db.insert_paper_ids("p1")
        db.commit()

        with (
            unittest.mock.patch("drbrain.extractor.openalex.get_work_by_doi", return_value=None),
            unittest.mock.patch(
                "drbrain.extractor.openalex.search_work_by_arxiv", return_value=None
            ),
            unittest.mock.patch(
                "drbrain.extractor.openalex.search_work_by_title",
                return_value=oa_work,
            ),
            unittest.mock.patch(
                "drbrain.extractor.openalex.batch_fetch_works",
                return_value=ref_works,
            ),
        ):
            refs, cits = _expand_with_openalex(
                db, "p1", {"title": "OpenAlex Title Paper"}, token=None
            )

        assert len(refs) == 2
        assert refs[0].title == "Ref A"
        assert refs[1].title == "Ref B"
        assert cits == []

        # DOI should have been backfilled
        doi = db.conn.execute("SELECT doi FROM paper_ids WHERE local_id = 'p1'").fetchone()[0]
        assert doi == "10.1111/oa_title"

        db.close()
