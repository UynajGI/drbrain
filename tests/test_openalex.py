"""Tests for OpenAlex API client."""
from brbrain.extractor.openalex import (
    search_work_by_title, search_work_by_arxiv, get_work_by_doi,
    _fetch_work_by_id
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


def test_fetch_work_by_id():
    """_fetch_work_by_id retrieves a single work."""
    mock_response = {
        "id": "https://openalex.org/W999",
        "doi": "https://doi.org/10.9999/fetch",
        "title": "Fetch Paper",
        "publication_year": 2024,
    }
    mock_resp = mock.Mock()
    mock_resp.read.return_value = json.dumps(mock_response)

    with mock.patch("urllib.request.urlopen", return_value=mock_resp):
        result = _fetch_work_by_id("https://openalex.org/W999")

    assert result is not None
    assert result["doi"] == "10.9999/fetch"
    assert result["openalex_id"] == "https://openalex.org/W999"
