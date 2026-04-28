"""Tests for S2 API retry logic."""
import unittest.mock
import requests
from drbrain.extractor.citation import fetch_s2_with_retry, search_s2_with_retry


def _make_429_response():
    """Create a mock response that raises 429 on raise_for_status."""
    resp = unittest.mock.Mock()
    resp.status_code = 429
    err = requests.HTTPError("429")
    err.response = resp
    resp.raise_for_status.side_effect = err
    return resp


def _make_success_response(data):
    """Create a mock response that returns JSON data."""
    resp = unittest.mock.Mock()
    resp.status_code = 200
    resp.raise_for_status = unittest.mock.Mock()
    resp.json.return_value = data
    return resp


def test_fetch_s2_retries_on_429():
    """fetch_s2_with_retry retries 3 times on 429 rate limit, then succeeds."""
    call_count = 0

    def mock_get(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return _make_429_response()
        return _make_success_response({"paperId": "abc123"})

    with unittest.mock.patch("requests.get", side_effect=mock_get):
        result = fetch_s2_with_retry("abc123")
        assert call_count == 3
        assert result is not None
        assert result["paperId"] == "abc123"


def test_fetch_s2_gives_up_after_max_retries():
    """fetch_s2_with_retry returns None after 3 consecutive 429s."""
    def mock_get(*args, **kwargs):
        return _make_429_response()

    with unittest.mock.patch("requests.get", side_effect=mock_get):
        result = fetch_s2_with_retry("abc123", max_retries=3)
        assert result is None


def test_fetch_s2_no_retry_on_non_429():
    """fetch_s2_with_retry does not retry on 500 errors (non-rate-limit)."""
    call_count = 0

    def mock_get(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        resp = unittest.mock.Mock()
        resp.status_code = 500
        err = requests.HTTPError("500")
        err.response = resp
        resp.raise_for_status.side_effect = err
        return resp

    with unittest.mock.patch("requests.get", side_effect=mock_get):
        result = fetch_s2_with_retry("abc123", max_retries=3)
        assert call_count == 1  # No retry for non-429
        assert result is None


def test_search_s2_retries_on_429():
    """search_s2_with_retry retries on 429 rate limit."""
    call_count = 0

    def mock_get(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            return _make_429_response()
        return _make_success_response({"data": [{"paperId": "xyz"}]})

    with unittest.mock.patch("requests.get", side_effect=mock_get):
        result = search_s2_with_retry("test paper", max_retries=3)
        assert call_count == 2
        assert len(result) == 1


def test_fetch_s2_uses_api_key_header():
    """fetch_s2_with_retry includes x-api-key header when provided."""
    captured_headers = {}

    def mock_get(*args, **kwargs):
        captured_headers["headers"] = kwargs.get("headers", {})
        return _make_success_response({"paperId": "abc"})

    with unittest.mock.patch("requests.get", side_effect=mock_get):
        fetch_s2_with_retry("abc123", api_key="test-key")
        assert captured_headers["headers"].get("x-api-key") == "test-key"
