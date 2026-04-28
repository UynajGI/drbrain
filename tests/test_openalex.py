"""Tests for OpenAlex API client."""
from drbrain.extractor.openalex import (
    search_work_by_title, search_work_by_arxiv, get_work_by_doi,
    batch_fetch_works, get_work_by_openalex_id,
)
from unittest import mock
import json


def test_search_work_by_title_empty():
    """search_work_by_title returns None for empty title."""
    assert search_work_by_title("") is None


def test_search_work_by_title_success():
    """search_work_by_title finds work and strips DOI URL prefix."""
    mock_response = {
        "results": [{
            "id": "https://openalex.org/W123",
            "doi": "https://doi.org/10.1234/test",
            "title": ["Test Paper"],
            "publication_year": 2024,
        }]
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
        "results": [{
            "id": "https://openalex.org/W456",
            "doi": "https://doi.org/10.1103/test",
            "title": ["Arxiv Paper"],
            "publication_year": 2025,
        }]
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
            {"id": "https://openalex.org/W1", "doi": "https://doi.org/10.1/one", "title": "Paper One", "publication_year": 2024},
            {"id": "https://openalex.org/W2", "doi": "https://doi.org/10.2/two", "title": "Paper Two", "publication_year": 2025},
        ]
    }
    mock_resp = mock.Mock()
    mock_resp.read.return_value = json.dumps(mock_response)

    from drbrain.extractor.openalex import batch_fetch_works
    with mock.patch("urllib.request.urlopen", return_value=mock_resp):
        results = batch_fetch_works(["https://openalex.org/W1", "https://openalex.org/W2"])

    assert len(results) == 2
    assert results[0]["doi"] == "10.1/one"
    assert results[1]["doi"] == "10.2/two"
