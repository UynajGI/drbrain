"""Tests for OpenAlex API client."""

from unittest import mock

import requests

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


def _mock_session(json_data):
    """Create a mock requests.Session that returns json_data on .get()."""
    mock_sess = mock.Mock()
    mock_resp = mock.Mock()
    mock_resp.json.return_value = json_data
    mock_resp.raise_for_status.return_value = None
    mock_sess.get.return_value = mock_resp
    return mock_sess


def _mock_session_error(exc):
    """Create a mock session whose .get() raises exc."""
    mock_sess = mock.Mock()
    mock_sess.get.side_effect = exc
    return mock_sess


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
    mock_sess = _mock_session(mock_response)

    with mock.patch("drbrain.extractor.openalex._get_session", return_value=mock_sess):
        result = search_work_by_title("Test Paper")

    assert result is not None
    assert result["doi"] == "10.1234/test"
    assert result["year"] == 2024


def test_search_work_by_title_no_results():
    """search_work_by_title returns None when no results found."""
    mock_sess = _mock_session({"results": []})

    with mock.patch("drbrain.extractor.openalex._get_session", return_value=mock_sess):
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
    mock_sess = _mock_session(mock_response)

    with mock.patch("drbrain.extractor.openalex._get_session", return_value=mock_sess):
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
    mock_sess = _mock_session(mock_response)

    with mock.patch("drbrain.extractor.openalex._get_session", return_value=mock_sess):
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
    mock_sess = _mock_session(mock_response)

    with mock.patch("drbrain.extractor.openalex._get_session", return_value=mock_sess):
        results = batch_fetch_works(["https://openalex.org/W1", "https://openalex.org/W2"])

    assert len(results) == 2
    assert results[0]["doi"] == "10.1/one"
    assert results[1]["doi"] == "10.2/two"


# --- search_work_by_title: token header ---


def test_search_work_by_title_sets_user_agent_with_token():
    """search_work_by_title sets User-Agent header when token provided."""
    mock_sess = mock.Mock()
    mock_resp = mock.Mock()
    mock_resp.json.return_value = {"results": []}
    mock_resp.raise_for_status.return_value = None
    captured_kwargs = []

    def _capture(url, **kwargs):
        captured_kwargs.append(kwargs)
        return mock_resp

    mock_sess.get.side_effect = _capture

    with mock.patch("drbrain.extractor.openalex._get_session", return_value=mock_sess):
        search_work_by_title("test", token="user@example.com")

    assert captured_kwargs[0]["headers"]["User-Agent"] == ("DrBrain (mailto:user@example.com)")


# --- retry behavior (delegated to Session/Retry adapter) ---


def test_search_work_by_title_handles_error():
    """search_work_by_title returns None when session raises RequestException."""
    mock_sess = _mock_session_error(requests.ConnectionError("network error"))

    with mock.patch("drbrain.extractor.openalex._get_session", return_value=mock_sess):
        result = search_work_by_title("network fails")

    assert result is None


def test_search_work_by_title_handles_error_with_retry_params():
    """max_retries/retry_delay kept in signature but retry handled by Session."""
    mock_sess = _mock_session_error(requests.HTTPError("persistent error"))

    with mock.patch("drbrain.extractor.openalex._get_session", return_value=mock_sess):
        result = search_work_by_title("always fails", max_retries=3, retry_delay=0)

    assert result is None


# --- search_work_by_title: doi without http prefix ---


def test_search_work_by_title_doi_no_http_prefix():
    """search_work_by_title returns DOI as-is when no http prefix present."""
    mock_response = {
        "results": [
            {
                "id": "W_clean",
                "doi": "10.5555/clean",
                "title": ["Clean DOI Paper"],
                "publication_year": 2023,
            }
        ]
    }
    mock_sess = _mock_session(mock_response)

    with mock.patch("drbrain.extractor.openalex._get_session", return_value=mock_sess):
        result = search_work_by_title("clean doi test")

    assert result is not None
    assert result["doi"] == "10.5555/clean"


# --- search_work_by_arxiv: version stripping ---


def test_search_work_by_arxiv_strips_version_suffix():
    """search_work_by_arxiv strips v2, v3 etc from arXiv ID."""
    mock_response = {
        "results": [
            {
                "id": "W_arxiv_v2",
                "doi": "10.1103/arxivtest",
                "title": ["ArXiv Versioned"],
                "publication_year": 2025,
            }
        ]
    }
    mock_sess = _mock_session(mock_response)

    with mock.patch("drbrain.extractor.openalex._get_session", return_value=mock_sess):
        result = search_work_by_arxiv("2501.12345v2")

    assert result is not None
    assert result["doi"] == "10.1103/arxivtest"


# --- search_work_by_arxiv: error handling ---


def test_search_work_by_arxiv_handles_error():
    """search_work_by_arxiv returns None on RequestException."""
    mock_sess = _mock_session_error(requests.Timeout("timeout"))

    with mock.patch("drbrain.extractor.openalex._get_session", return_value=mock_sess):
        result = search_work_by_arxiv("2501.99999")

    assert result is None


# --- get_work_by_doi: error response and error handling ---


def test_get_work_by_doi_error_rejected():
    """get_work_by_doi returns None when response contains 'error' key."""
    mock_sess = _mock_session({"error": "not found", "message": "nope"})

    with mock.patch("drbrain.extractor.openalex._get_session", return_value=mock_sess):
        result = get_work_by_doi("10.9999/notfound")

    assert result is None


def test_get_work_by_doi_handles_request_error():
    """get_work_by_doi returns None on RequestException."""
    mock_sess = _mock_session_error(requests.ConnectionError("boom"))

    with mock.patch("drbrain.extractor.openalex._get_session", return_value=mock_sess):
        result = get_work_by_doi("10.1234/doi_fail")

    assert result is None


# --- get_work_by_doi: strip http://doi.org/ prefix ---


def test_get_work_by_doi_strips_http_doi_prefix():
    """get_work_by_doi strips http://doi.org/ from input DOI."""
    mock_response = {
        "id": "W_prefix",
        "doi": "https://doi.org/10.1234/prefixed",
        "title": "Prefixed DOI Paper",
        "publication_year": 2024,
    }
    mock_sess = _mock_session(mock_response)

    with mock.patch("drbrain.extractor.openalex._get_session", return_value=mock_sess):
        result = get_work_by_doi("https://doi.org/10.1234/prefixed")

    assert result is not None
    # returned DOI should have prefix stripped
    assert result["doi"] == "10.1234/prefixed"


# --- get_work_by_openalex_id ---


def test_get_work_by_openalex_id_success():
    """get_work_by_openalex_id fetches a single work by its OpenAlex ID."""
    mock_response = {
        "id": "https://openalex.org/W_core",
        "doi": "https://doi.org/10.1234/core",
        "title": "Core Paper",
        "publication_year": 2023,
    }
    mock_sess = _mock_session(mock_response)

    with mock.patch("drbrain.extractor.openalex._get_session", return_value=mock_sess):
        result = get_work_by_openalex_id("https://openalex.org/W_core")

    assert result is not None
    assert result["doi"] == "10.1234/core"
    assert result["title"] == "Core Paper"
    assert result["year"] == 2023
    assert result["openalex_id"] == "https://openalex.org/W_core"


def test_get_work_by_openalex_id_empty_data():
    """get_work_by_openalex_id returns None for empty response."""
    mock_sess = _mock_session({})

    with mock.patch("drbrain.extractor.openalex._get_session", return_value=mock_sess):
        result = get_work_by_openalex_id("https://openalex.org/W_empty")

    assert result is None


def test_get_work_by_openalex_id_handles_error():
    """get_work_by_openalex_id returns None on RequestException."""
    mock_sess = _mock_session_error(requests.ConnectionError("boom"))

    with mock.patch("drbrain.extractor.openalex._get_session", return_value=mock_sess):
        result = get_work_by_openalex_id("https://openalex.org/W_fail")

    assert result is None


# --- get_work_references ---


def test_get_work_references_empty_id():
    """get_work_references returns empty list for empty ID."""
    assert get_work_references("") == []


def test_get_work_references_success():
    """get_work_references fetches referenced works."""
    mock_response = {
        "id": "https://openalex.org/W_parent",
        "referenced_works": [
            "https://openalex.org/W_ref1",
            "https://openalex.org/W_ref2",
        ],
    }
    mock_sess = _mock_session(mock_response)

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

    with mock.patch("drbrain.extractor.openalex._get_session", return_value=mock_sess):
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
    mock_response = {
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
    mock_sess = _mock_session(mock_response)

    with mock.patch("drbrain.extractor.openalex._get_session", return_value=mock_sess):
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
    fail_sess = _mock_session_error(requests.ConnectionError("DOI lookup failed"))
    title_mock_response = {
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
    title_sess = _mock_session(title_mock_response)

    # DOI call uses session (fails), title call uses session (succeeds)
    with mock.patch(
        "drbrain.extractor.openalex._get_session",
        side_effect=[fail_sess, title_sess],
    ):
        authors = search_authors_by_work(doi="10.1234/bad_doi", title="Some Title")

    assert authors is not None
    assert len(authors) == 1
    assert authors[0]["author_id"] == "A5000000001"
    assert authors[0]["display_name"] == "Title Matcher"


def test_search_authors_by_work_no_authorships():
    """search_authors_by_work returns None when work has no authorships."""
    mock_response = {
        "id": "W_no_authors",
        "authorships": [],
    }
    mock_sess = _mock_session(mock_response)

    with mock.patch("drbrain.extractor.openalex._get_session", return_value=mock_sess):
        authors = search_authors_by_work(doi="10.1234/no_authors")

    assert authors is None


def test_search_authors_by_work_skips_missing_author_id():
    """search_authors_by_work skips authors without extractable short ID."""
    mock_response = {
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
    mock_sess = _mock_session(mock_response)

    with mock.patch("drbrain.extractor.openalex._get_session", return_value=mock_sess):
        authors = search_authors_by_work(doi="10.1234/partial")

    assert authors is not None
    assert len(authors) == 1
    assert authors[0]["author_id"] == "A5000000001"


# --- batch_fetch_works edge cases ---


def test_batch_fetch_works_empty_list():
    """batch_fetch_works returns empty list for empty input."""
    assert batch_fetch_works([]) == []


def test_batch_fetch_works_handles_error():
    """batch_fetch_works returns empty list on RequestException."""
    mock_sess = _mock_session_error(requests.ConnectionError("batch fail"))

    with mock.patch("drbrain.extractor.openalex._get_session", return_value=mock_sess):
        results = batch_fetch_works(["W1"])

    assert results == []
