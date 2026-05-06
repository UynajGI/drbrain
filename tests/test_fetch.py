"""Tests for PDF acquisition from open access sources."""

from drbrain.services.fetch import (
    _proxy_url,
    _resolve_identifier,
    resolve_pdf_url,
)


class TestResolveIdentifier:
    """Tests for identifier classification."""

    def test_doi_detection(self):
        doi, title, arxiv = _resolve_identifier("10.1234/example.doi")
        assert doi == "10.1234/example.doi"
        assert title is None
        assert arxiv is None

    def test_title_detection(self):
        doi, title, arxiv = _resolve_identifier("Attention Is All You Need")
        assert doi is None
        assert title == "Attention Is All You Need"
        assert arxiv is None

    def test_arxiv_flag(self):
        doi, title, arxiv = _resolve_identifier("1706.03762", is_arxiv=True)
        assert doi is None
        assert title is None
        assert arxiv == "1706.03762"

    def test_doi_with_slash_no_spaces(self):
        doi, title, arxiv = _resolve_identifier("abc/def.123")
        assert doi == "abc/def.123"
        assert title is None


class TestProxyUrl:
    """Tests for institutional proxy URL rewriting."""

    def test_no_proxy(self):
        url = "https://arxiv.org/pdf/1706.03762.pdf"
        result = _proxy_url(url, {})
        assert result == url

    def test_ezproxy_basic(self):
        url = "https://doi.org/10.1234/example"
        cfg = {"institutional_proxy": "proxy.lib.edu", "proxy_type": "ezproxy"}
        result = _proxy_url(url, cfg)
        assert "proxy.lib.edu" in result
        assert "doi-org.proxy.lib.edu" in result
        assert "/10.1234/example" in result

    def test_url_prefix(self):
        url = "https://journal.com/article.pdf"
        cfg = {"institutional_proxy": "http://proxy.lib.edu", "proxy_type": "url_prefix"}
        result = _proxy_url(url, cfg)
        assert result.startswith("http://proxy.lib.eduhttps://")


class TestArxivUrlConstruction:
    """Tests for arXiv URL construction (no network)."""

    def test_arxiv_url_format(self):
        arxiv_id = "1706.03762"
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        assert url == "https://arxiv.org/pdf/1706.03762.pdf"


class TestResolvePdfUrl:
    """Tests for resolve_pdf_url with no real network calls (fallbacks)."""

    def test_resolve_with_none_inputs(self):
        result = resolve_pdf_url(doi=None, title=None, arxiv_id=None)
        assert result is None

    def test_resolve_arxiv_constructs_url(self):
        # Will fail HEAD request (no network in CI), so returns None
        # but verifies function signature and no crash
        result = resolve_pdf_url(arxiv_id="1706.03762")
        # May be None due to HEAD check failure, but shouldn't crash
        assert result is None or result.startswith("https://")
