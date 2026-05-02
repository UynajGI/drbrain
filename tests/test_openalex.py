"""Tests for OpenAlex API client."""

import json
from unittest import mock

from drbrain.extractor.openalex import (
    _extract_author_short_id,
    batch_fetch_works,
    get_work_by_doi,
    get_work_by_openalex_id,
    get_work_references,
    search_authors_by_work,
    search_work_by_arxiv,
    search_work_by_title,
)


def test_search_work_by_title_empty():
    """search_work_by_title returns None for empty title."""
    assert search_work_by_title("") is None


def test_search_work_by_title_success():
    """search_work_by_title finds work and strips DOI URL prefix."""
    mock_response = {
        "results": [
            {
                "id": "https://openalex.org/W123",
                "doi": "https://doi.org/10.1234/test",
                "title": ["Test Paper"],
                "publication_year": 2024,
            }
        ]
    }
    mock_resp = mock.Mock()
    mock_resp.read.return_value = json.dumps(mock_response)

    with mock.patch("urllib.request.urlopen", return_value=mock_resp):
        result = search_work_by_title("Test Paper")

    assert result is not None
    assert result["doi"] == "10.1234/test"
    assert result["year"] == 2024


def test_search_work_by_title_no_results():
    """search_work_by_title returns None when no results found."""
    mock_resp = mock.Mock()
    mock_resp.read.return_value = json.dumps({"results": []})

    with mock.patch("urllib.request.urlopen", return_value=mock_resp):
        result = search_work_by_title("Nonexistent Paper 12345")

    assert result is None


def test_search_work_by_arxiv_success():
    """search_work_by_arxiv finds work by arXiv ID."""
    mock_response = {
        "results": [
            {
                "id": "https://openalex.org/W456",
                "doi": "https://doi.org/10.1103/test",
                "title": ["Arxiv Paper"],
                "publication_year": 2025,
            }
        ]
    }
    mock_resp = mock.Mock()
    mock_resp.read.return_value = json.dumps(mock_response)

    with mock.patch("urllib.request.urlopen", return_value=mock_resp):
        result = search_work_by_arxiv("2501.12345")

    assert result is not None
    assert result["doi"] == "10.1103/test"


def test_search_work_by_arxiv_empty():
    """search_work_by_arxiv returns None for empty arXiv ID."""
    assert search_work_by_arxiv("") is None
    assert search_work_by_arxiv("  ") is None


def test_get_work_by_doi_success():
    """get_work_by_doi resolves DOI and returns metadata."""
    mock_response = {
        "id": "https://openalex.org/W789",
        "doi": "https://doi.org/10.5678/another",
        "title": "Another Paper",
        "publication_year": 2023,
        "referenced_works": ["https://openalex.org/W1", "https://openalex.org/W2"],
    }
    mock_resp = mock.Mock()
    mock_resp.read.return_value = json.dumps(mock_response)

    with mock.patch("urllib.request.urlopen", return_value=mock_resp):
        result = get_work_by_doi("10.5678/another")

    assert result is not None
    assert result["doi"] == "10.5678/another"
    assert len(result["referenced_works"]) == 2


def test_get_work_by_doi_empty():
    """get_work_by_doi returns None for empty DOI."""
    assert get_work_by_doi("") is None


def test_batch_fetch_works():
    """batch_fetch_works retrieves multiple works in one call."""
    mock_response = {
        "results": [
            {
                "id": "https://openalex.org/W1",
                "doi": "https://doi.org/10.1/one",
                "title": "Paper One",
                "publication_year": 2024,
            },
            {
                "id": "https://openalex.org/W2",
                "doi": "https://doi.org/10.2/two",
                "title": "Paper Two",
                "publication_year": 2025,
            },
        ]
    }
    mock_resp = mock.Mock()
    mock_resp.read.return_value = json.dumps(mock_response)

    with mock.patch("urllib.request.urlopen", return_value=mock_resp):
        results = batch_fetch_works(["https://openalex.org/W1", "https://openalex.org/W2"])

    assert len(results) == 2
    assert results[0]["doi"] == "10.1/one"
    assert results[1]["doi"] == "10.2/two"


# --- search_work_by_title: token header ---


def test_search_work_by_title_sets_user_agent_with_token():
    """search_work_by_title sets User-Agent header when token provided."""
    mock_resp = mock.Mock()
    mock_resp.read.return_value = json.dumps({"results": []})

    captured_req = []

    def _capture(req):
        captured_req.append(req)
        return mock_resp

    with mock.patch("urllib.request.urlopen", side_effect=_capture):
        search_work_by_title("test", token="user@example.com")

    # urllib.request.Request lowercases header keys
    assert captured_req[0].get_header("User-agent") == "DrBrain (mailto:user@example.com)"


# --- search_work_by_title: retry ---


def test_search_work_by_title_retries_on_error():
    """search_work_by_title retries after urllib error then succeeds."""
    mock_success = mock.Mock()
    mock_success.read.return_value = json.dumps(
        {
            "results": [
                {
                    "id": "W1",
                    "doi": "10.1234/retry",
                    "title": ["Retry Paper"],
                    "publication_year": 2024,
                }
            ]
        }
    )

    def _fail_then_succeed(req):
        _fail_then_succeed.calls += 1
        if _fail_then_succeed.calls == 1:
            raise OSError("network error")
        return mock_success

    _fail_then_succeed.calls = 0

    with mock.patch("urllib.request.urlopen", side_effect=_fail_then_succeed):
        result = search_work_by_title("retry test", max_retries=2, retry_delay=0)

    assert result is not None
    assert result["doi"] == "10.1234/retry"
    assert _fail_then_succeed.calls == 2


def test_search_work_by_title_gives_up_after_max_retries():
    """search_work_by_title returns None after exhausting retries."""
    with mock.patch("urllib.request.urlopen", side_effect=OSError("persistent error")):
        result = search_work_by_title("always fails", max_retries=3, retry_delay=0)

    assert result is None


# --- search_work_by_title: doi without http prefix ---


def test_search_work_by_title_doi_no_http_prefix():
    """search_work_by_title returns DOI as-is when no http prefix present."""
    mock_resp = mock.Mock()
    mock_resp.read.return_value = json.dumps(
        {
            "results": [
                {
                    "id": "W_clean",
                    "doi": "10.5555/clean",
                    "title": ["Clean DOI Paper"],
                    "publication_year": 2023,
                }
            ]
        }
    )

    with mock.patch("urllib.request.urlopen", return_value=mock_resp):
        result = search_work_by_title("clean doi test")

    assert result is not None
    assert result["doi"] == "10.5555/clean"


# --- search_work_by_arxiv: version stripping ---


def test_search_work_by_arxiv_strips_version_suffix():
    """search_work_by_arxiv strips v2, v3 etc from arXiv ID."""
    mock_resp = mock.Mock()
    mock_resp.read.return_value = json.dumps(
        {
            "results": [
                {
                    "id": "W_arxiv_v2",
                    "doi": "10.1103/arxivtest",
                    "title": ["ArXiv Versioned"],
                    "publication_year": 2025,
                }
            ]
        }
    )

    with mock.patch("urllib.request.urlopen", return_value=mock_resp):
        result = search_work_by_arxiv("2501.12345v2")

    assert result is not None
    assert result["doi"] == "10.1103/arxivtest"


# --- search_work_by_arxiv: retry ---


def test_search_work_by_arxiv_retries_on_error():
    """search_work_by_arxiv retries after error then succeeds."""
    mock_success = mock.Mock()
    mock_success.read.return_value = json.dumps(
        {
            "results": [
                {
                    "id": "W_arxiv_retry",
                    "doi": "10.1103/retried",
                    "title": ["Retried ArXiv"],
                    "publication_year": 2024,
                }
            ]
        }
    )

    def _fail_then_succeed(req):
        _fail_then_succeed.calls += 1
        if _fail_then_succeed.calls == 1:
            raise OSError("timeout")
        return mock_success

    _fail_then_succeed.calls = 0

    with mock.patch("urllib.request.urlopen", side_effect=_fail_then_succeed):
        result = search_work_by_arxiv("2501.99999", max_retries=2, retry_delay=0)

    assert result is not None
    assert result["doi"] == "10.1103/retried"


# --- get_work_by_doi: error response and retry ---


def test_get_work_by_doi_error_rejected():
    """get_work_by_doi returns None when response contains 'error' key."""
    mock_resp = mock.Mock()
    mock_resp.read.return_value = json.dumps({"error": "not found", "message": "nope"})

    with mock.patch("urllib.request.urlopen", return_value=mock_resp):
        result = get_work_by_doi("10.9999/notfound")

    assert result is None


def test_get_work_by_doi_retries_on_error():
    """get_work_by_doi retries then succeeds."""
    mock_success = mock.Mock()
    mock_success.read.return_value = json.dumps(
        {
            "id": "W_doi_retry",
            "doi": "10.1234/doi_retry",
            "title": "DOI Retry Paper",
            "publication_year": 2024,
        }
    )

    def _fail_then_succeed(req):
        _fail_then_succeed.calls += 1
        if _fail_then_succeed.calls == 1:
            raise OSError("boom")
        return mock_success

    _fail_then_succeed.calls = 0

    with mock.patch("urllib.request.urlopen", side_effect=_fail_then_succeed):
        result = get_work_by_doi("10.1234/doi_retry", max_retries=2, retry_delay=0)

    assert result is not None
    assert result["doi"] == "10.1234/doi_retry"


# --- get_work_by_doi: strip http://doi.org/ prefix ---


def test_get_work_by_doi_strips_http_doi_prefix():
    """get_work_by_doi strips http://doi.org/ from input DOI."""
    mock_resp = mock.Mock()
    mock_resp.read.return_value = json.dumps(
        {
            "id": "W_prefix",
            "doi": "https://doi.org/10.1234/prefixed",
            "title": "Prefixed DOI Paper",
            "publication_year": 2024,
        }
    )

    with mock.patch("urllib.request.urlopen", return_value=mock_resp):
        result = get_work_by_doi("https://doi.org/10.1234/prefixed")

    assert result is not None
    # returned DOI should have prefix stripped
    assert result["doi"] == "10.1234/prefixed"


# --- get_work_by_openalex_id ---


def test_get_work_by_openalex_id_success():
    """get_work_by_openalex_id fetches a single work by its OpenAlex ID."""
    mock_resp = mock.Mock()
    mock_resp.read.return_value = json.dumps(
        {
            "id": "https://openalex.org/W_core",
            "doi": "https://doi.org/10.1234/core",
            "title": "Core Paper",
            "publication_year": 2023,
        }
    )

    with mock.patch("urllib.request.urlopen", return_value=mock_resp):
        result = get_work_by_openalex_id("https://openalex.org/W_core")

    assert result is not None
    assert result["doi"] == "10.1234/core"
    assert result["title"] == "Core Paper"
    assert result["year"] == 2023
    assert result["openalex_id"] == "https://openalex.org/W_core"


def test_get_work_by_openalex_id_empty_data():
    """get_work_by_openalex_id returns None for empty response."""
    mock_resp = mock.Mock()
    mock_resp.read.return_value = json.dumps({})

    with mock.patch("urllib.request.urlopen", return_value=mock_resp):
        result = get_work_by_openalex_id("https://openalex.org/W_empty")

    assert result is None


def test_get_work_by_openalex_id_retry():
    """get_work_by_openalex_id retries on error."""
    mock_success = mock.Mock()
    mock_success.read.return_value = json.dumps(
        {
            "id": "W_oa_retry",
            "doi": "10.1234/oa_retry",
            "title": "OA Retry",
            "publication_year": 2024,
        }
    )

    def _fail_then_succeed(req):
        _fail_then_succeed.calls += 1
        if _fail_then_succeed.calls == 1:
            raise OSError("boom")
        return mock_success

    _fail_then_succeed.calls = 0

    with mock.patch("urllib.request.urlopen", side_effect=_fail_then_succeed):
        result = get_work_by_openalex_id(
            "https://openalex.org/W_oa_retry", max_retries=2, retry_delay=0
        )

    assert result is not None
    assert result["doi"] == "10.1234/oa_retry"


# --- get_work_references ---


def test_get_work_references_empty_id():
    """get_work_references returns empty list for empty ID."""
    assert get_work_references("") == []


def test_get_work_references_success():
    """get_work_references fetches referenced works."""
    work_resp = mock.Mock()
    work_resp.read.return_value = json.dumps(
        {
            "id": "https://openalex.org/W_parent",
            "referenced_works": [
                "https://openalex.org/W_ref1",
                "https://openalex.org/W_ref2",
            ],
        }
    )

    ref1_info = {
        "doi": "10.1111/ref1",
        "title": "Ref One",
        "year": 2020,
        "openalex_id": "https://openalex.org/W_ref1",
    }
    ref2_info = {
        "doi": "10.2222/ref2",
        "title": "Ref Two",
        "year": 2021,
        "openalex_id": "https://openalex.org/W_ref2",
    }

    def _urlopen_side(req):
        return work_resp

    with mock.patch("urllib.request.urlopen", side_effect=_urlopen_side):
        with mock.patch(
            "drbrain.extractor.openalex.get_work_by_openalex_id",
            side_effect=[ref1_info, ref2_info],
        ):
            results = get_work_references("https://openalex.org/W_parent")

    assert len(results) == 2
    assert results[0]["doi"] == "10.1111/ref1"
    assert results[1]["doi"] == "10.2222/ref2"


# --- _extract_author_short_id ---


def test_extract_author_short_id_valid():
    """_extract_author_short_id extracts A-prefixed ID from URL."""
    assert _extract_author_short_id("https://openalex.org/A5023806754") == "A5023806754"


def test_extract_author_short_id_no_match():
    """_extract_author_short_id returns None for non-matching URL."""
    assert _extract_author_short_id("https://openalex.org/W123") is None


def test_extract_author_short_id_empty():
    """_extract_author_short_id returns None for empty string."""
    assert _extract_author_short_id("") is None


# --- search_authors_by_work ---


def test_search_authors_by_work_empty_params():
    """search_authors_by_work returns None when no doi or title."""
    assert search_authors_by_work() is None
    assert search_authors_by_work(doi=None, title=None) is None


def test_search_authors_by_work_via_doi():
    """search_authors_by_work fetches authors via DOI lookup."""
    mock_resp = mock.Mock()
    mock_resp.read.return_value = json.dumps(
        {
            "id": "W_authors",
            "authorships": [
                {
                    "author": {
                        "id": "https://openalex.org/A5023806754",
                        "display_name": "Jane Researcher",
                        "orcid": "https://orcid.org/0000-0001-2345-6789",
                    },
                    "raw_affiliation_strings": ["University of Testing"],
                },
                {
                    "author": {
                        "id": "https://openalex.org/A5023806755",
                        "display_name": "John Colleague",
                        "orcid": None,
                    },
                    "raw_affiliation_strings": [],
                },
            ],
        }
    )

    with mock.patch("urllib.request.urlopen", return_value=mock_resp):
        authors = search_authors_by_work(doi="10.1234/with_authors")

    assert authors is not None
    assert len(authors) == 2
    assert authors[0]["author_id"] == "A5023806754"
    assert authors[0]["display_name"] == "Jane Researcher"
    assert authors[0]["orcid"] == "https://orcid.org/0000-0001-2345-6789"
    assert authors[0]["raw_affiliation"] == ["University of Testing"]
    assert authors[1]["author_id"] == "A5023806755"
    assert authors[1]["display_name"] == "John Colleague"


def test_search_authors_by_work_via_title_fallback():
    """search_authors_by_work falls back to title search when DOI fails."""
    title_mock_resp = mock.Mock()
    title_mock_resp.read.return_value = json.dumps(
        {
            "results": [
                {
                    "id": "W_title_found",
                    "authorships": [
                        {
                            "author": {
                                "id": "https://openalex.org/A5000000001",
                                "display_name": "Title Matcher",
                                "orcid": None,
                            },
                            "raw_affiliation_strings": ["Some Institute"],
                        }
                    ],
                }
            ]
        }
    )

    # DOI call raises OSError, title call succeeds
    with mock.patch(
        "urllib.request.urlopen",
        side_effect=[OSError("DOI lookup failed"), title_mock_resp],
    ):
        authors = search_authors_by_work(doi="10.1234/bad_doi", title="Some Title")

    assert authors is not None
    assert len(authors) == 1
    assert authors[0]["author_id"] == "A5000000001"
    assert authors[0]["display_name"] == "Title Matcher"


def test_search_authors_by_work_no_authorships():
    """search_authors_by_work returns None when work has no authorships."""
    mock_resp = mock.Mock()
    mock_resp.read.return_value = json.dumps(
        {
            "id": "W_no_authors",
            "authorships": [],
        }
    )

    with mock.patch("urllib.request.urlopen", return_value=mock_resp):
        authors = search_authors_by_work(doi="10.1234/no_authors")

    assert authors is None


def test_search_authors_by_work_skips_missing_author_id():
    """search_authors_by_work skips authors without extractable short ID."""
    mock_resp = mock.Mock()
    mock_resp.read.return_value = json.dumps(
        {
            "id": "W_partial",
            "authorships": [
                {
                    "author": {
                        "id": "https://openalex.org/A5000000001",
                        "display_name": "Valid Author",
                        "orcid": None,
                    },
                    "raw_affiliation_strings": [],
                },
                {
                    "author": {
                        "id": "",  # no ID
                        "display_name": "No ID Author",
                        "orcid": None,
                    },
                    "raw_affiliation_strings": [],
                },
                {
                    "author": {
                        "id": "https://openalex.org/W123",  # wrong format
                        "display_name": "Work ID Author",
                        "orcid": None,
                    },
                    "raw_affiliation_strings": [],
                },
            ],
        }
    )

    with mock.patch("urllib.request.urlopen", return_value=mock_resp):
        authors = search_authors_by_work(doi="10.1234/partial")

    assert authors is not None
    assert len(authors) == 1
    assert authors[0]["author_id"] == "A5000000001"


# --- batch_fetch_works edge cases ---


def test_batch_fetch_works_empty_list():
    """batch_fetch_works returns empty list for empty input."""
    assert batch_fetch_works([]) == []


def test_batch_fetch_works_retries_on_error():
    """batch_fetch_works retries on error then succeeds."""
    mock_success = mock.Mock()
    mock_success.read.return_value = json.dumps(
        {
            "results": [
                {
                    "id": "W_batch_retry",
                    "doi": "10.1234/batch_retry",
                    "title": "Batch Retry",
                    "publication_year": 2024,
                }
            ]
        }
    )

    def _fail_then_succeed(req):
        _fail_then_succeed.calls += 1
        if _fail_then_succeed.calls == 1:
            raise OSError("batch fail")
        return mock_success

    _fail_then_succeed.calls = 0

    with mock.patch("urllib.request.urlopen", side_effect=_fail_then_succeed):
        results = batch_fetch_works(["W1"], max_retries=2, retry_delay=0)

    assert len(results) == 1
    assert results[0]["doi"] == "10.1234/batch_retry"
