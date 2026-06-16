"""Tests for webtools provider — mocked HTTP responses.

uspto_odp and uspto_ppubs are already tested in tests/test_uspto.py (51 tests).
This file covers webtools.py which had zero coverage.
"""

from __future__ import annotations

import json
from unittest import mock

from drbrain.providers.webtools import (
    _get_webextract_timeout,
    _get_webextract_url,
    _slugify_title,
    check_webextract_service,
    extract_web,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _mock_urlopen_response(data: dict, status: int = 200):
    """Build a mock HTTPResponse that returns *data* as JSON bytes."""
    body = json.dumps(data).encode("utf-8")
    resp = mock.MagicMock()
    resp.read.return_value = body
    resp.status = status
    resp.__enter__ = mock.MagicMock(return_value=resp)
    resp.__exit__ = mock.MagicMock(return_value=False)
    return resp


def _make_http_error(code: int, body: str = "Error") -> Exception:
    from urllib.error import HTTPError

    err = HTTPError("url", code, body, {}, None)
    err.read = mock.MagicMock(return_value=body.encode("utf-8"))
    return err


# ===================================================================
# _get_webextract_url
# ===================================================================


class TestGetWebextractURL:
    def test_default_url(self, monkeypatch):
        monkeypatch.delenv("WEBEXTRACT_URL", raising=False)
        monkeypatch.delenv("QT_WEB_EXTRACTOR_URL", raising=False)
        assert _get_webextract_url() == "http://127.0.0.1:8766"

    def test_webextract_url_env(self, monkeypatch):
        monkeypatch.setenv("WEBEXTRACT_URL", "http://custom:9999")
        monkeypatch.delenv("QT_WEB_EXTRACTOR_URL", raising=False)
        assert _get_webextract_url() == "http://custom:9999"

    def test_qt_web_extractor_url_env(self, monkeypatch):
        monkeypatch.delenv("WEBEXTRACT_URL", raising=False)
        monkeypatch.setenv("QT_WEB_EXTRACTOR_URL", "http://qt-host:5000")
        assert _get_webextract_url() == "http://qt-host:5000"

    def test_webextract_url_takes_priority(self, monkeypatch):
        monkeypatch.setenv("WEBEXTRACT_URL", "http://priority:1111")
        monkeypatch.setenv("QT_WEB_EXTRACTOR_URL", "http://fallback:2222")
        assert _get_webextract_url() == "http://priority:1111"

    def test_trailing_slash_stripped(self, monkeypatch):
        monkeypatch.setenv("WEBEXTRACT_URL", "http://host:8000/")
        assert _get_webextract_url() == "http://host:8000"

    def test_empty_env_uses_default(self, monkeypatch):
        monkeypatch.setenv("WEBEXTRACT_URL", "")
        monkeypatch.setenv("QT_WEB_EXTRACTOR_URL", "")
        assert _get_webextract_url() == "http://127.0.0.1:8766"


# ===================================================================
# _get_webextract_timeout
# ===================================================================


class TestGetWebextractTimeout:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("WEBEXTRACT_TIMEOUT", raising=False)
        assert _get_webextract_timeout() == 60.0

    def test_custom_env(self, monkeypatch):
        monkeypatch.setenv("WEBEXTRACT_TIMEOUT", "120")
        assert _get_webextract_timeout() == 120.0

    def test_float_value(self, monkeypatch):
        monkeypatch.setenv("WEBEXTRACT_TIMEOUT", "30.5")
        assert _get_webextract_timeout() == 30.5

    def test_invalid_value_fallback(self, monkeypatch):
        monkeypatch.setenv("WEBEXTRACT_TIMEOUT", "not-a-number")
        assert _get_webextract_timeout() == 60.0


# ===================================================================
# _slugify_title
# ===================================================================


class TestSlugifyTitle:
    def test_basic_title(self):
        assert _slugify_title("Hello World", "http://example.com") == "hello-world"

    def test_empty_title_uses_url_path(self):
        slug = _slugify_title("", "http://example.com/paper/2024/my-paper.html")
        assert slug == "my-paper"

    def test_empty_title_uses_host(self):
        slug = _slugify_title("", "http://example.com/")
        assert slug == "example-com"

    def test_special_chars(self):
        slug = _slugify_title("ML @ Scale: A 2024 Review!", "http://x.com")
        assert slug == "ml-scale-a-2024-review"

    def test_long_title_truncated(self):
        long_title = "A" * 200
        slug = _slugify_title(long_title, "http://x.com")
        assert len(slug) <= 120

    def test_empty_title_empty_url_fallback(self):
        slug = _slugify_title("", "")
        assert slug == "web-link"

    def test_dashes_and_underscores(self):
        slug = _slugify_title("foo_bar--baz", "http://x.com")
        assert slug == "foo-bar-baz"

    def test_strips_leading_trailing_dashes(self):
        slug = _slugify_title("  --hello--  ", "http://x.com")
        assert slug == "hello"


# ===================================================================
# extract_web — happy path
# ===================================================================


SAMPLE_SUCCESS_RESPONSE = {
    "url": "https://example.com/paper",
    "title": "Research Paper Title",
    "text": "# Abstract\nThis is the paper content.",
    "html": "<html><body>...</body></html>",
    "images": [{"src": "img1.png", "alt": "Figure 1"}],
    "extracted_at": "2025-06-01T12:00:00Z",
}


class TestExtractWeb:
    @mock.patch("drbrain.providers.webtools.urlopen")
    def test_successful_extraction(self, mock_open):
        mock_open.return_value = _mock_urlopen_response(SAMPLE_SUCCESS_RESPONSE)

        result = extract_web("https://example.com/paper")
        assert result["url"] == "https://example.com/paper"
        assert result["title"] == "Research Paper Title"
        assert result["text"] == "# Abstract\nThis is the paper content."
        assert result["html"] == "<html><body>...</body></html>"
        assert len(result["images"]) == 1
        assert result["extracted_at"] == "2025-06-01T12:00:00Z"
        assert result["error"] == ""

    @mock.patch("drbrain.providers.webtools.urlopen")
    def test_extraction_with_markdown_fallback(self, mock_open):
        """If 'text' key is missing, falls back to 'markdown' key."""
        resp_data = {
            "url": "https://example.com",
            "title": "Title",
            "markdown": "**Bold text**",
        }
        mock_open.return_value = _mock_urlopen_response(resp_data)

        result = extract_web("https://example.com")
        assert result["text"] == "**Bold text**"

    @mock.patch("drbrain.providers.webtools.urlopen")
    def test_extraction_uses_provided_timeout(self, mock_open):
        mock_open.return_value = _mock_urlopen_response(SAMPLE_SUCCESS_RESPONSE)

        extract_web("https://example.com", timeout=5.0)
        mock_open.assert_called_once()
        call_args = mock_open.call_args
        assert call_args[1]["timeout"] == 5.0

    @mock.patch("drbrain.providers.webtools.urlopen")
    def test_extraction_pdf_flag(self, mock_open):
        mock_open.return_value = _mock_urlopen_response(SAMPLE_SUCCESS_RESPONSE)

        extract_web("https://example.com/doc.pdf", pdf=True)
        # Verify the payload includes pdf=True
        call_args = mock_open.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        assert body["pdf"] is True

    @mock.patch("drbrain.providers.webtools.urlopen")
    def test_extraction_pdf_false(self, mock_open):
        mock_open.return_value = _mock_urlopen_response(SAMPLE_SUCCESS_RESPONSE)

        extract_web("https://example.com/doc.pdf", pdf=False)
        call_args = mock_open.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        assert body["pdf"] is False

    @mock.patch("drbrain.providers.webtools.urlopen")
    def test_extraction_custom_base_url(self, mock_open, monkeypatch):
        monkeypatch.setenv("WEBEXTRACT_URL", "http://custom:9000")
        mock_open.return_value = _mock_urlopen_response(SAMPLE_SUCCESS_RESPONSE)

        extract_web("https://example.com")
        call_args = mock_open.call_args
        req = call_args[0][0]
        assert "custom:9000" in req.full_url


# ===================================================================
# extract_web — error handling
# ===================================================================


class TestExtractWebErrors:
    @mock.patch("drbrain.providers.webtools.urlopen")
    def test_http_error_returns_error_dict(self, mock_open):
        mock_open.side_effect = _make_http_error(500, "Internal Server Error")

        result = extract_web("https://example.com")
        assert result["error"] != ""
        assert "500" in result["error"]
        assert result["url"] == "https://example.com"
        assert result["title"] == ""
        assert result["text"] == ""
        assert result["images"] == []

    @mock.patch("drbrain.providers.webtools.urlopen")
    def test_http_404_returns_error_dict(self, mock_open):
        mock_open.side_effect = _make_http_error(404, "Not Found")

        result = extract_web("https://example.com/missing")
        assert "404" in result["error"]
        assert result["title"] == ""

    @mock.patch("drbrain.providers.webtools.urlopen")
    def test_url_error_returns_error_dict(self, mock_open):
        from urllib.error import URLError

        mock_open.side_effect = URLError("Connection refused")

        result = extract_web("https://example.com")
        assert "Connection refused" in result["error"]
        assert result["url"] == "https://example.com"
        assert result["text"] == ""

    @mock.patch("drbrain.providers.webtools.urlopen")
    def test_oserror_returns_error_dict(self, mock_open):
        mock_open.side_effect = OSError("Network unreachable")

        result = extract_web("https://example.com")
        assert result["error"] != ""
        assert result["url"] == "https://example.com"

    @mock.patch("drbrain.providers.webtools.urlopen")
    def test_invalid_json_returns_error_dict(self, mock_open):
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = b"not json at all"
        mock_resp.__enter__ = mock.MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = mock.MagicMock(return_value=False)
        mock_open.return_value = mock_resp

        result = extract_web("https://example.com")
        assert result["error"] != ""

    @mock.patch("drbrain.providers.webtools.urlopen")
    def test_empty_response_body(self, mock_open):
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = b""
        mock_resp.__enter__ = mock.MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = mock.MagicMock(return_value=False)
        mock_open.return_value = mock_resp

        result = extract_web("https://example.com")
        # Empty body → json.loads("") → should be handled gracefully
        assert isinstance(result, dict)
        assert result["url"] == "https://example.com"

    @mock.patch("drbrain.providers.webtools.urlopen")
    def test_error_with_title_preserves_title(self, mock_open):
        """If error is returned but title is present, title is preserved."""
        resp_data = {
            "title": "Partial Page",
            "error": "Some extraction error",
        }
        mock_open.return_value = _mock_urlopen_response(resp_data)

        result = extract_web("https://example.com")
        assert result["title"] == "Partial Page"
        assert result["error"] == "Some extraction error"


# ===================================================================
# check_webextract_service
# ===================================================================


class TestCheckWebextractService:
    @mock.patch("drbrain.providers.webtools.urlopen")
    def test_service_reachable(self, mock_open):
        mock_open.return_value = _mock_urlopen_response({"status": "ok"}, status=200)

        assert check_webextract_service() is True

    @mock.patch("drbrain.providers.webtools.urlopen")
    def test_service_unreachable(self, mock_open):
        from urllib.error import URLError

        mock_open.side_effect = URLError("Connection refused")

        assert check_webextract_service() is False

    @mock.patch("drbrain.providers.webtools.urlopen")
    def test_service_non_200(self, mock_open):
        mock_open.return_value = _mock_urlopen_response({"status": "error"}, status=503)

        assert check_webextract_service() is False

    @mock.patch("drbrain.providers.webtools.urlopen")
    def test_service_timeout(self, mock_open):

        mock_open.side_effect = TimeoutError("timed out")

        assert check_webextract_service() is False

    @mock.patch("drbrain.providers.webtools.urlopen")
    def test_service_http_error(self, mock_open):
        mock_open.side_effect = _make_http_error(500, "Server Error")

        assert check_webextract_service() is False

    @mock.patch("drbrain.providers.webtools.urlopen")
    def test_custom_timeout(self, mock_open):
        from urllib.error import URLError

        mock_open.side_effect = URLError("nope")
        assert check_webextract_service(timeout=1.0) is False
        mock_open.assert_called_once()
        call_kwargs = mock_open.call_args[1]
        assert call_kwargs["timeout"] == 1.0

    @mock.patch("drbrain.providers.webtools.urlopen")
    def test_custom_base_url(self, mock_open, monkeypatch):
        monkeypatch.setenv("WEBEXTRACT_URL", "http://custom:9000")
        mock_open.return_value = _mock_urlopen_response({"status": "ok"}, status=200)

        assert check_webextract_service() is True
        call_args = mock_open.call_args
        req = call_args[0][0]
        assert "custom:9000" in req.full_url
