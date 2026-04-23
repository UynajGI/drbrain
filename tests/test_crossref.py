"""Tests for CrossRef DOI enrichment."""
from brbrain.extractor.crossref import (
    fetch_doi_by_title, _clean_title, _titles_match
)
from unittest import mock
import json


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
    assert _titles_match("attention is all you need", "attention is all you need in transformers") is True


def test_titles_match_overlap():
    """_titles_match returns True for high word overlap."""
    assert _titles_match("deep learning for NLP", "deep learning for natural language processing") is True


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
        mock_resp.read.return_value = json.dumps({
            "message": {"items": [{
                "DOI": "10.9999/retry",
                "title": ["Retry Paper"],
                "published-online": {"date-parts": [[2025]]},
            }]}
        })
        return mock_resp

    with mock.patch("urllib.request.urlopen", side_effect=mock_urlopen):
        result = fetch_doi_by_title("Retry Paper", max_retries=2, retry_delay=0.01)

    assert call_count == 2
    assert result is not None
    assert result["doi"] == "10.9999/retry"
