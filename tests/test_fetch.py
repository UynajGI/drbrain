"""Tests for src/drbrain/services/fetch.py.

All HTTP / library calls are mocked: requests, openalex, mineru_parser.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from drbrain.services import fetch
from drbrain.services.fetch import (
    _proxy_url,
    _resolve_identifier,
    _resolve_metadata,
    _search_arxiv_by_title,
    _try_direct_doi,
    _try_openalex_oa,
    _try_unpaywall,
    download_pdf,
    fetch_paper,
    resolve_pdf_url,
)

# ── _resolve_identifier ────────────────────────────────────────────────────


class TestResolveIdentifier:
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

    def test_arxiv_flag_short_circuits(self):
        doi, title, arxiv = _resolve_identifier("anything", is_arxiv=True)
        assert doi is None
        assert title is None
        assert arxiv == "anything"

    def test_doi_with_slash_no_spaces(self):
        doi, title, arxiv = _resolve_identifier("abc/def.123")
        assert doi == "abc/def.123"
        assert title is None

    def test_arxiv_url_abs(self):
        doi, title, arxiv = _resolve_identifier("https://arxiv.org/abs/2301.00234")
        assert doi is None
        assert arxiv == "2301.00234"

    def test_arxiv_url_pdf_strips_extension(self):
        doi, title, arxiv = _resolve_identifier("https://arxiv.org/pdf/2301.00234.pdf")
        assert arxiv == "2301.00234"
        assert doi is None

    def test_arxiv_new_style_id(self):
        doi, title, arxiv = _resolve_identifier("1706.03762")
        assert arxiv == "1706.03762"
        assert doi is None

    def test_arxiv_old_style_id(self):
        doi, title, arxiv = _resolve_identifier("cs.AI/0703111")
        assert arxiv == "cs.AI/0703111"
        assert doi is None

    def test_title_with_spaces_not_doi(self):
        doi, title, arxiv = _resolve_identifier("some / path with spaces")
        assert doi is None
        assert title == "some / path with spaces"

    def test_strips_whitespace(self):
        doi, title, arxiv = _resolve_identifier("  10.1/xx  ")
        assert doi == "10.1/xx"


# ── _proxy_url ─────────────────────────────────────────────────────────────


class TestProxyUrl:
    def test_no_proxy(self):
        url = "https://arxiv.org/pdf/1706.03762.pdf"
        result = _proxy_url(url, {})
        assert result == url

    def test_no_proxy_host_empty(self):
        url = "https://example.com/a"
        result = _proxy_url(url, {"institutional_proxy": "", "proxy_type": "ezproxy"})
        assert result == url

    def test_unknown_proxy_type_passthrough(self):
        url = "https://example.com/a"
        result = _proxy_url(url, {"institutional_proxy": "proxy.lib", "proxy_type": "weird"})
        assert result == url

    def test_ezproxy_basic(self):
        url = "https://doi.org/10.1234/example"
        cfg = {"institutional_proxy": "proxy.lib.edu", "proxy_type": "ezproxy"}
        result = _proxy_url(url, cfg)
        assert "proxy.lib.edu" in result
        assert "doi-org.proxy.lib.edu" in result
        assert "/10.1234/example" in result

    def test_ezproxy_preserves_query(self):
        url = "https://doi.org/x?q=1"
        cfg = {"institutional_proxy": "p.lib", "proxy_type": "ezproxy"}
        result = _proxy_url(url, cfg)
        assert "?q=1" in result

    def test_url_prefix(self):
        url = "https://journal.com/article.pdf"
        cfg = {
            "institutional_proxy": "http://proxy.lib.edu",
            "proxy_type": "url_prefix",
        }
        result = _proxy_url(url, cfg)
        assert result.startswith("http://proxy.lib.eduhttps://")


# ── resolve_pdf_url (federated fallback) ───────────────────────────────────


class TestResolvePdfUrl:
    def test_all_none_returns_none(self):
        assert resolve_pdf_url() is None

    def test_arxiv_id_stage_succeeds(self):
        with patch.object(fetch, "_url_exists", return_value=True) as m:
            result = resolve_pdf_url(arxiv_id="2301.00234")
        assert result == "https://arxiv.org/pdf/2301.00234.pdf"
        m.assert_called_once_with("https://arxiv.org/pdf/2301.00234.pdf")

    def test_arxiv_id_stage_fails_falls_to_openalex(self):
        with (
            patch.object(fetch, "_url_exists", side_effect=[False, True]),
            patch.object(fetch, "_try_openalex_oa", return_value="https://oa/x.pdf") as oa,
        ):
            result = resolve_pdf_url(doi="10.1/x", arxiv_id="2301.00234")
        assert result == "https://oa/x.pdf"
        oa.assert_called_once_with("10.1/x")

    def test_openalex_stage(self):
        with patch.object(fetch, "_try_openalex_oa", return_value="https://oa/y.pdf"):
            result = resolve_pdf_url(doi="10.1/y")
        assert result == "https://oa/y.pdf"

    def test_unpaywall_stage_when_openalex_none(self):
        cfg = {"unpaywall_email": "u@e.com"}
        with (
            patch.object(fetch, "_try_openalex_oa", return_value=None),
            patch.object(fetch, "_try_unpaywall", return_value="https://uw/z.pdf") as uw,
        ):
            result = resolve_pdf_url(doi="10.1/z", fetch_config=cfg)
        assert result == "https://uw/z.pdf"
        uw.assert_called_once_with("10.1/z", cfg)

    def test_direct_doi_stage(self):
        with (
            patch.object(fetch, "_try_openalex_oa", return_value=None),
            patch.object(fetch, "_try_unpaywall", return_value=None),
            patch.object(fetch, "_try_direct_doi", return_value="https://dd/w.pdf") as dd,
        ):
            result = resolve_pdf_url(doi="10.1/w")
        assert result == "https://dd/w.pdf"
        dd.assert_called_once_with("10.1/w")

    def test_title_arxiv_search_stage(self):
        with (
            patch.object(fetch, "_try_openalex_oa", return_value=None),
            patch.object(fetch, "_try_unpaywall", return_value=None),
            patch.object(fetch, "_try_direct_doi", return_value=None),
            patch.object(fetch, "_search_arxiv_by_title", return_value="2301.999") as st,
        ):
            result = resolve_pdf_url(title="Some Paper Title")
        assert result == "https://arxiv.org/pdf/2301.999.pdf"
        st.assert_called_once_with("Some Paper Title")

    def test_title_stage_skipped_when_arxiv_id_present(self):
        with (
            patch.object(fetch, "_url_exists", return_value=True),
            patch.object(fetch, "_search_arxiv_by_title") as st,
        ):
            resolve_pdf_url(title="Some Title", arxiv_id="2301.00234")
        st.assert_not_called()

    def test_every_stage_fails_returns_none(self):
        with (
            patch.object(fetch, "_url_exists", return_value=False),
            patch.object(fetch, "_try_openalex_oa", return_value=None),
            patch.object(fetch, "_try_unpaywall", return_value=None),
            patch.object(fetch, "_try_direct_doi", return_value=None),
            patch.object(fetch, "_search_arxiv_by_title", return_value=None),
        ):
            result = resolve_pdf_url(doi="10.1/x", title="t")
        assert result is None


# ── _try_openalex_oa / _try_unpaywall / _try_direct_doi / _search_arxiv ────


class TestHelpers:
    def test_try_openalex_oa_from_oa_field(self):
        with patch("drbrain.extractor.openalex.get_work_enriched") as gw:
            gw.return_value = {"open_access": {"is_oa": True, "pdf_url": "https://oa/a.pdf"}}
            assert _try_openalex_oa("10.1/a") == "https://oa/a.pdf"

    def test_try_openalex_oa_from_primary_location(self):
        with patch("drbrain.extractor.openalex.get_work_enriched") as gw:
            gw.return_value = {
                "open_access": {},
                "primary_location": {"pdf_url": "https://oa/b.pdf"},
            }
            assert _try_openalex_oa("10.1/b") == "https://oa/b.pdf"

    def test_try_openalex_oa_returns_none_when_not_oa(self):
        with patch("drbrain.extractor.openalex.get_work_enriched") as gw:
            gw.return_value = {"open_access": {"is_oa": False}}
            assert _try_openalex_oa("10.1/c") is None

    def test_try_openalex_oa_handles_exception(self):
        with patch("drbrain.extractor.openalex.get_work_enriched", side_effect=Exception("boom")):
            assert _try_openalex_oa("10.1/d") is None

    def test_try_unpaywall_no_email_returns_none(self):
        assert _try_unpaywall("10.1/x", {}) is None

    def test_try_unpaywall_success(self):
        cfg = {"unpaywall_email": "u@e.com"}
        resp = MagicMock()
        resp.json.return_value = {"best_oa_location": {"url_for_pdf": "https://uw/x.pdf"}}
        with (
            patch("drbrain.services.fetch.requests.get", return_value=resp),
            patch.object(fetch, "_url_exists", return_value=True),
        ):
            assert _try_unpaywall("10.1/x", cfg) == "https://uw/x.pdf"

    def test_try_unpaywall_falls_back_to_url(self):
        cfg = {"email": "u@e.com"}
        resp = MagicMock()
        resp.json.return_value = {"best_oa_location": {"url": "https://uw/y"}}
        with (
            patch("drbrain.services.fetch.requests.get", return_value=resp),
            patch.object(fetch, "_url_exists", return_value=True),
        ):
            assert _try_unpaywall("10.1/y", cfg) == "https://uw/y"

    def test_try_unpaywall_returns_none_on_exception(self):
        cfg = {"unpaywall_email": "u@e.com"}
        with patch("drbrain.services.fetch.requests.get", side_effect=Exception("nope")):
            assert _try_unpaywall("10.1/z", cfg) is None

    def test_try_direct_doi_success(self):
        resp = MagicMock()
        resp.url = "https://publisher/x.pdf"
        with (
            patch("drbrain.services.fetch.requests.head", return_value=resp),
            patch.object(fetch, "_url_exists", return_value=True),
        ):
            assert _try_direct_doi("10.1/x") == "https://publisher/x.pdf"

    def test_try_direct_doi_handles_exception(self):
        with patch("drbrain.services.fetch.requests.head", side_effect=Exception("x")):
            assert _try_direct_doi("10.1/x") is None

    def test_search_arxiv_by_title_success(self):
        xml = (
            '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">'
            "<entry><id>http://arxiv.org/abs/2301.00999</id></entry></feed>"
        )
        resp = MagicMock(status_code=200, text=xml)
        with patch("drbrain.services.fetch.requests.get", return_value=resp):
            assert _search_arxiv_by_title("Some Title") == "2301.00999"

    def test_search_arxiv_by_title_non_200(self):
        resp = MagicMock(status_code=500, text="")
        with patch("drbrain.services.fetch.requests.get", return_value=resp):
            assert _search_arxiv_by_title("t") is None

    def test_search_arxiv_by_title_handles_exception(self):
        with patch("drbrain.services.fetch.requests.get", side_effect=Exception("x")):
            assert _search_arxiv_by_title("t") is None


# ── _url_exists ────────────────────────────────────────────────────────────


class TestUrlExists:
    def test_returns_true_on_200(self):
        from drbrain.services.fetch import _url_exists

        resp = MagicMock(status_code=200)
        with patch("drbrain.services.fetch.requests.head", return_value=resp):
            assert _url_exists("https://x") is True

    def test_returns_false_on_non_200(self):
        from drbrain.services.fetch import _url_exists

        resp = MagicMock(status_code=404)
        with patch("drbrain.services.fetch.requests.head", return_value=resp):
            assert _url_exists("https://x") is False

    def test_returns_false_on_exception(self):
        from drbrain.services.fetch import _url_exists

        with patch("drbrain.services.fetch.requests.head", side_effect=Exception("x")):
            assert _url_exists("https://x") is False


# ── download_pdf ───────────────────────────────────────────────────────────


def _pdf_resp(prefix: bytes = b"%PDF-1.4 body", content_type: str = "application/pdf"):
    """Build a fake streaming response whose raw.peek returns the PDF magic."""
    raw = MagicMock()
    raw.peek.return_value = b"%PDF-"
    resp = MagicMock()
    resp.headers = {"content-type": content_type}
    resp.raw = raw
    resp.raise_for_status = MagicMock()
    resp.iter_content = MagicMock(return_value=iter([prefix]))
    return resp


class TestDownloadPdf:
    def test_creates_dir_and_writes_pdf(self, tmp_path):
        dest_dir = tmp_path / "papers" / "p1"
        resp = _pdf_resp()
        with patch("drbrain.services.fetch.requests.get", return_value=resp):
            out = download_pdf("https://x/p.pdf", dest_dir)
        assert out is not None
        assert out.exists()
        assert out.name == "source.pdf"
        assert dest_dir.is_dir()

    def test_returns_none_when_not_pdf(self, tmp_path):
        # content-type not pdf, url has no .pdf suffix, peek magic not %PDF-
        raw = MagicMock()
        raw.peek.return_value = b"<html"
        resp = MagicMock(
            headers={"content-type": "text/html"}, raw=raw, raise_for_status=MagicMock()
        )
        resp.iter_content = MagicMock(return_value=iter([b"<html>"]))
        with patch("drbrain.services.fetch.requests.get", return_value=resp):
            out = download_pdf("https://x/page", tmp_path / "d")
        assert out is None

    def test_accepts_pdf_via_content_type_only(self, tmp_path):
        # peek returns no magic, but content-type says pdf and url ends .pdf.
        # Use a list so iter_content can be consumed twice (peek + write).
        chunks = [b"binary", b"-more"]

        def _iter(chunk_size=8192):
            yield from chunks

        raw = MagicMock()
        raw.peek.return_value = None
        resp = MagicMock(
            headers={"content-type": "application/pdf"}, raw=raw, raise_for_status=MagicMock()
        )
        resp.iter_content = MagicMock(side_effect=_iter)
        with patch("drbrain.services.fetch.requests.get", return_value=resp):
            out = download_pdf("https://x/a.pdf", tmp_path / "d")
        assert out is not None
        assert out.read_bytes() == b"binary-more"

    def test_uses_config_user_agent_and_timeout(self, tmp_path):
        cfg = {"user_agent": "Test/1.0", "timeout_per_fetch": 7}
        resp = _pdf_resp()
        with patch("drbrain.services.fetch.requests.get", return_value=resp) as g:
            download_pdf("https://x/p.pdf", tmp_path / "d", cfg)
        _, kwargs = g.call_args
        assert kwargs["headers"]["User-Agent"] == "Test/1.0"
        assert kwargs["timeout"] == 7

    def test_returns_none_on_exception(self, tmp_path):
        with patch("drbrain.services.fetch.requests.get", side_effect=Exception("boom")):
            out = download_pdf("https://x/p.pdf", tmp_path / "d")
        assert out is None


# ── _resolve_metadata ──────────────────────────────────────────────────────


class TestResolveMetadata:
    def test_from_doi_via_openalex(self):
        with patch("drbrain.extractor.openalex.get_work_by_doi") as gw:
            gw.return_value = {"title": "T", "publication_year": 2023}
            result = _resolve_metadata(doi="10.1/x")
        assert result["doi"] == "10.1/x"
        assert result["title"] == "T"
        assert result["year"] == 2023
        assert result["arxiv"] is None
        assert result["local_id"].startswith("p")

    def test_from_arxiv_id(self):
        with patch("drbrain.parser.mineru_parser._fetch_arxiv_metadata") as fm:
            fm.return_value = ("Arxiv Title", 2020)
            result = _resolve_metadata(arxiv_id="2301.001")
        assert result["arxiv"] == "2301.001"
        assert result["title"] == "Arxiv Title"
        assert result["doi"] is None

    def test_from_title_search(self):
        with (
            patch("drbrain.extractor.openalex.get_work_by_doi", side_effect=Exception("x")),
            patch("drbrain.extractor.openalex.search_work_by_title") as st,
        ):
            st.return_value = {"title": "FT", "publication_year": 2019, "doi": "10.1/z"}
            result = _resolve_metadata(title="anything")
        assert result["title"] == "FT"
        assert result["doi"] == "10.1/z"

    def test_returns_none_when_all_fail(self):
        with (
            patch("drbrain.extractor.openalex.get_work_by_doi", side_effect=Exception("x")),
            patch("drbrain.extractor.openalex.search_work_by_title", side_effect=Exception("x")),
        ):
            assert _resolve_metadata(doi="10.1/x", title="t") is None

    def test_doi_returns_none_when_openalex_empty(self):
        with patch("drbrain.extractor.openalex.get_work_by_doi", return_value=None):
            # No arxiv/title provided → falls through to return None
            assert _resolve_metadata(doi="10.1/x") is None


# ── fetch_paper ────────────────────────────────────────────────────────────


class TestFetchPaper:
    def test_full_pipeline_success(self, tmp_path):
        cfg = {"papers_root": str(tmp_path)}
        # Patch resolve_pdf_url, _proxy_url, _resolve_metadata, download_pdf
        with (
            patch.object(fetch, "resolve_pdf_url", return_value="https://x/p.pdf"),
            patch.object(fetch, "_proxy_url", side_effect=lambda u, c: u),
            patch.object(
                fetch,
                "_resolve_metadata",
                return_value={
                    "local_id": "pabc",
                    "title": "T",
                    "year": 2023,
                    "doi": "10.1/x",
                    "arxiv": None,
                },
            ),
            patch.object(
                fetch, "download_pdf", return_value=tmp_path / "pabc" / "source.pdf"
            ) as dl,
        ):
            result = fetch_paper(doi="10.1/x", fetch_config=cfg)
        assert result is not None
        assert result["title"] == "T"
        assert result["doi"] == "10.1/x"
        assert result["local_id"] == "pabc"
        dl.assert_called_once()

    def test_returns_none_when_no_pdf_url(self):
        with patch.object(fetch, "resolve_pdf_url", return_value=None):
            assert fetch_paper(doi="10.1/x") is None

    def test_returns_none_when_no_metadata(self):
        with (
            patch.object(fetch, "resolve_pdf_url", return_value="https://x/p.pdf"),
            patch.object(fetch, "_proxy_url", side_effect=lambda u, c: u),
            patch.object(fetch, "_resolve_metadata", return_value=None),
        ):
            assert fetch_paper(doi="10.1/x") is None

    def test_returns_none_when_download_fails(self):
        with (
            patch.object(fetch, "resolve_pdf_url", return_value="https://x/p.pdf"),
            patch.object(fetch, "_proxy_url", side_effect=lambda u, c: u),
            patch.object(
                fetch,
                "_resolve_metadata",
                return_value={"local_id": "p1", "title": "T", "year": 2020},
            ),
            patch.object(fetch, "download_pdf", return_value=None),
        ):
            assert fetch_paper(doi="10.1/x") is None

    def test_applies_proxy_to_pdf_url(self, tmp_path):
        cfg = {"institutional_proxy": "proxy.lib", "proxy_type": "url_prefix"}
        with (
            patch.object(fetch, "resolve_pdf_url", return_value="https://x/p.pdf"),
            patch.object(fetch, "_proxy_url", return_value="https://proxy+x") as pu,
            patch.object(
                fetch,
                "_resolve_metadata",
                return_value={"local_id": "p1", "title": "T", "year": 2020},
            ),
            patch.object(fetch, "download_pdf", return_value=tmp_path / "p1" / "source.pdf") as dl,
        ):
            fetch_paper(doi="10.1/x", fetch_config=cfg)
        pu.assert_called_once()
        # download_pdf gets the proxied URL
        assert dl.call_args[0][0] == "https://proxy+x"
