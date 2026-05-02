"""Tests for CrossRef DOI enrichment."""

import json
from unittest import mock

from drbrain.extractor.crossref import (
    _clean_title,
    _titles_match,
    fetch_doi_by_arxiv,
    fetch_doi_by_doi,
    fetch_doi_by_title,
)


def test_clean_title_removes_special_chars():
    """_clean_title strips punctuation and collapses spaces."""
    assert _clean_title("L-entropy: A new genuine measure!") == "L-entropy A new genuine measure"


def test_clean_title_lower():
    """_clean_title normalizes to single spaces."""
    assert _clean_title("  Attention   Is  All  You  Need  ") == "Attention Is All You Need"


def test_titles_match_exact():
    """_titles_match returns True for identical titles."""
    assert _titles_match("attention is all you need", "attention is all you need") is True


def test_titles_match_prefix():
    """_titles_match returns True when one title is prefix of other."""
    assert (
        _titles_match("attention is all you need", "attention is all you need in transformers")
        is True
    )


def test_titles_match_overlap():
    """_titles_match returns True for high word overlap."""
    assert (
        _titles_match("deep learning for NLP", "deep learning for natural language processing")
        is True
    )


def test_titles_match_different():
    """_titles_match returns False for unrelated titles."""
    assert _titles_match("deep learning for NLP", "quantum computing review") is False


def test_fetch_doi_by_title_empty():
    """fetch_doi_by_title returns None for empty title."""
    assert fetch_doi_by_title("") is None
    assert fetch_doi_by_title("   ") is None


def test_fetch_doi_by_title_success():
    """fetch_doi_by_title returns DOI when CrossRef finds a match."""
    mock_response = {
        "message": {
            "items": [
                {
                    "DOI": "10.1234/test",
                    "title": ["Test Paper Title"],
                    "published-print": {"date-parts": [[2024, 1, 1]]},
                }
            ]
        }
    }

    mock_resp = mock.Mock()
    mock_resp.read.return_value = json.dumps(mock_response)

    with mock.patch("urllib.request.urlopen", return_value=mock_resp):
        result = fetch_doi_by_title("Test Paper Title", email="test@test.com")

    assert result is not None
    assert result["doi"] == "10.1234/test"
    assert result["year"] == 2024


def test_fetch_doi_by_title_no_results():
    """fetch_doi_by_title returns None when CrossRef returns empty."""
    mock_response = {"message": {"items": []}}
    mock_resp = mock.Mock()
    mock_resp.read.return_value = json.dumps(mock_response)

    with mock.patch("urllib.request.urlopen", return_value=mock_resp):
        result = fetch_doi_by_title("Nonexistent Paper 12345")

    assert result is None


def test_fetch_doi_by_title_retries_on_error():
    """fetch_doi_by_title retries on network errors."""
    call_count = 0

    def mock_urlopen(*args):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError("timeout")
        mock_resp = mock.Mock()
        mock_resp.read.return_value = json.dumps(
            {
                "message": {
                    "items": [
                        {
                            "DOI": "10.9999/retry",
                            "title": ["Retry Paper"],
                            "published-online": {"date-parts": [[2025]]},
                        }
                    ]
                }
            }
        )
        return mock_resp

    with mock.patch("urllib.request.urlopen", side_effect=mock_urlopen):
        result = fetch_doi_by_title("Retry Paper", max_retries=2, retry_delay=0.01)

    assert call_count == 2
    assert result is not None
    assert result["doi"] == "10.9999/retry"


def test_fetch_doi_by_doi_success():
    """fetch_doi_by_doi resolves a known DOI directly."""
    mock_response = {
        "message": {
            "DOI": "10.1103/kmpl-mdbx",
            "title": ["L-entropy: A New Genuine Multipartite Entanglement Measure"],
            "published-print": {"date-parts": [[2025, 3, 15]]},
        }
    }
    mock_resp = mock.Mock()
    mock_resp.read.return_value = json.dumps(mock_response)

    with mock.patch("urllib.request.urlopen", return_value=mock_resp):
        result = fetch_doi_by_doi("10.1103/kmpl-mdbx", email="test@test.com")

    assert result is not None
    assert result["doi"] == "10.1103/kmpl-mdbx"
    assert result["year"] == 2025


def test_fetch_doi_by_doi_empty():
    """fetch_doi_by_doi returns None for empty DOI."""
    assert fetch_doi_by_doi("") is None


def test_fetch_doi_by_doi_retries_on_error():
    """fetch_doi_by_doi retries on network errors."""
    call_count = 0

    def mock_urlopen(*args):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError("timeout")
        mock_resp = mock.Mock()
        mock_resp.read.return_value = json.dumps(
            {
                "message": {
                    "DOI": "10.9999/direct",
                    "title": ["Direct DOI Paper"],
                    "published-online": {"date-parts": [[2025]]},
                }
            }
        )
        return mock_resp

    with mock.patch("urllib.request.urlopen", side_effect=mock_urlopen):
        result = fetch_doi_by_doi("10.9999/direct", max_retries=2, retry_delay=0.01)

    assert call_count == 2
    assert result is not None
    assert result["doi"] == "10.9999/direct"


# -- fetch_doi_by_arxiv --


def test_fetch_doi_by_arxiv_success():
    """fetch_doi_by_arxiv extracts DOI from CrossRef response with matching arxivid."""
    mock_response = {
        "message": {
            "items": [
                {
                    "arxivid": "1706.03762v1",
                    "DOI": "10.1103/PhysRevX.7.031045",
                    "title": ["Attention Is All You Need"],
                    "published-print": {"date-parts": [[2017, 6, 12]]},
                }
            ]
        }
    }
    mock_resp = mock.Mock()
    mock_resp.read.return_value = json.dumps(mock_response)

    with mock.patch("urllib.request.urlopen", return_value=mock_resp):
        result = fetch_doi_by_arxiv("1706.03762v1", email="test@test.com")

    assert result is not None
    assert result["doi"] == "10.1103/PhysRevX.7.031045"
    assert result["title"] == "Attention Is All You Need"
    assert result["year"] == 2017


def test_fetch_doi_by_arxiv_strips_version_suffix():
    """fetch_doi_by_arxiv strips v{N} suffix from arxiv ID before querying."""
    mock_response = {
        "message": {
            "items": [
                {
                    "arxivid": "1706.03762v3",
                    "DOI": "10.1103/PhysRevX.7.031045",
                    "title": ["Attention Is All You Need"],
                    "published-online": {"date-parts": [[2017]]},
                }
            ]
        }
    }
    mock_resp = mock.Mock()
    mock_resp.read.return_value = json.dumps(mock_response)

    with mock.patch("urllib.request.urlopen", return_value=mock_resp):
        result = fetch_doi_by_arxiv("1706.03762v5")  # different version, still matches

    assert result is not None
    assert result["doi"] == "10.1103/PhysRevX.7.031045"


def test_fetch_doi_by_arxiv_fallback_phys_rev():
    """fetch_doi_by_arxiv falls back to items with 10.1103 DOI when no arxivid match."""
    mock_response = {
        "message": {
            "items": [
                {
                    "arxivid": "",
                    "DOI": "",
                    "title": [""],
                },
                {
                    "arxivid": "",
                    "DOI": "10.1103/PhysRevLett.131.123456",
                    "title": ["Quantum Error Correction Review"],
                    "published-online": {"date-parts": [[2023]]},
                },
            ]
        }
    }
    mock_resp = mock.Mock()
    mock_resp.read.return_value = json.dumps(mock_response)

    with mock.patch("urllib.request.urlopen", return_value=mock_resp):
        result = fetch_doi_by_arxiv("2301.12345")

    assert result is not None
    assert result["doi"] == "10.1103/PhysRevLett.131.123456"
    assert result["title"] == "Quantum Error Correction Review"


def test_fetch_doi_by_arxiv_no_match():
    """fetch_doi_by_arxiv returns None when no arxivid or phys rev match."""
    mock_response = {
        "message": {
            "items": [
                {
                    "arxivid": "1901.00001",
                    "DOI": "10.1000/j.jmlr.2020.01",
                    "title": ["Some Other Paper"],
                }
            ]
        }
    }
    mock_resp = mock.Mock()
    mock_resp.read.return_value = json.dumps(mock_response)

    with mock.patch("urllib.request.urlopen", return_value=mock_resp):
        # "1706.03762" != "1901.00001" after stripping versions
        result = fetch_doi_by_arxiv("1706.03762")

    assert result is None


def test_fetch_doi_by_arxiv_empty():
    """fetch_doi_by_arxiv returns None for empty arxiv ID."""
    assert fetch_doi_by_arxiv("") is None
    assert fetch_doi_by_arxiv("   ") is None


# -- _titles_match punctuation --


def test_titles_match_different_punctuation():
    """_titles_match returns True for same words with different punctuation."""
    # These would be cleaned to the same text by _clean_title before matching
    assert _titles_match("deep learning a new approach", "deep learning a new approach") is True


def test_titles_match_special_chars_difference():
    """_titles_match handles titles where one has special chars removed."""
    assert (
        _titles_match("l entropy a new genuine measure", "l-entropy a new genuine measure") is True
    )


def test_titles_match_no_common_words():
    """_titles_match returns False when titles share zero words."""
    assert _titles_match("the", "quantum") is False
