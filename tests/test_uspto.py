"""Tests for USPTO patent search providers — mocked HTTP responses."""

from __future__ import annotations

import json
from unittest import mock

import pytest

from drbrain.providers.uspto_odp import (
    PatentResult,
    USPTOAPIError,
    _extract_patent_result,
    get_patent_by_application_number,
    search_patents,
)
from drbrain.providers.uspto_odp import (
    clean_publication_number as _clean_publication_number,
)
from drbrain.providers.uspto_ppubs import (
    PpubsClient,
    PpubsError,
    PpubsPatent,
    _publication_search_query,
    _strip_markup,
)
from drbrain.providers.uspto_ppubs import (
    _extract_patent as ppubs_extract_patent,
)
from drbrain.providers.uspto_ppubs import (
    clean_publication_number as _normalize_publication_number,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SAMPLE_ODP_ITEM = {
    "applicationNumberText": "  17/123456  ",
    "applicationMetaData": {
        "inventionTitle": "Machine Learning for Patent Analysis",
        "inventorBag": [
            {"inventorNameText": "Alice Smith"},
            {"firstName": "Bob", "lastName": "Jones"},
        ],
        "applicantBag": [
            {"applicantName": "Acme Corp"},
            {"firstName": "Tech", "lastName": "Inc"},
        ],
        "firstApplicantName": "Acme Corp",
        "filingDate": "2023-01-15",
        "grantDate": "2025-06-10",
        "publicationDateBag": ["2024-03-20"],
        "earliestPublicationDate": "2024-03-20",
        "patentNumber": "1234567",
        "earliestPublicationNumber": "US2024001234A1",
        "applicationStatusDescriptionText": "Granted",
        "applicationTypeLabelName": "Utility",
    },
}

SAMPLE_ODP_RESPONSE = {
    "count": 42,
    "patentFileWrapperDataBag": [SAMPLE_ODP_ITEM],
}


def _mock_urlopen_response(data: dict, status: int = 200):
    """Build a mock HTTPResponse that returns *data* as JSON bytes."""
    body = json.dumps(data).encode("utf-8")

    mock_resp = mock.MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = mock.MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = mock.MagicMock(return_value=False)
    mock_resp.status = status
    return mock_resp


def _mock_urlopen_error(code: int, body: str = "Not Found"):
    """Build a mock that raises urllib.error.HTTPError on open."""
    from urllib.error import HTTPError

    err = HTTPError("url", code, body, {}, None)
    err.read = mock.MagicMock(return_value=body.encode("utf-8"))
    raise err


# ===================================================================
# uspto_odp — PatentResult dataclass
# ===================================================================


class TestPatentResult:
    def test_to_dict_keys(self):
        r = PatentResult(
            application_number="17/123456",
            title="Test",
            patent_number="9999",
        )
        d = r.to_dict()
        assert d["application_number"] == "17/123456"
        assert d["title"] == "Test"
        assert d["patent_number"] == "9999"  # to_dict includes patent_number

    def test_google_patents_url_with_publication_number(self):
        r = PatentResult(publication_number="US2024001234A1")
        assert r.google_patents_url() == "https://patents.google.com/patent/US2024001234A1/en"

    def test_google_patents_url_fallback(self):
        r = PatentResult(application_number="17/123456")
        assert "17%2F123456" in r.google_patents_url() or "17/123456" in r.google_patents_url()

    def test_best_year_from_grant(self):
        r = PatentResult(grant_date="2025-06-10")
        assert r._best_year() == "2025"

    def test_best_year_from_publication(self):
        r = PatentResult(publication_date="2024-03-20")
        assert r._best_year() == "2024"

    def test_best_year_none(self):
        r = PatentResult()
        assert r._best_year() is None

    def test_to_meta_dict_year_and_author(self):
        r = PatentResult(
            title="My Patent",
            inventors=["Alice Smith"],
            grant_date="2025-06-10",
            publication_number="US1234B2",
            application_number="17/123456",
        )
        m = r.to_meta_dict()
        assert m["year"] == 2025
        assert m["first_author"] == "Alice Smith"
        assert m["first_author_lastname"] == "Smith"
        assert m["paper_type"] == "patent"
        assert m["source_url"].startswith("https://patents.google.com")
        assert "uspto_odp" in m["api_sources"]

    def test_to_meta_dict_no_year(self):
        r = PatentResult(
            inventors=["Alice"],
            application_number="17/123456",
            publication_number="US1234B2",
        )
        m = r.to_meta_dict()
        assert m["year"] is None
        assert m["first_author_lastname"] == "Alice"


# ===================================================================
# uspto_odp — _clean_publication_number
# ===================================================================


class TestCleanPublicationNumber:
    def test_removes_spaces_and_dashes(self):
        assert _clean_publication_number("US 2024-001234 A1") == "US2024001234A1"

    def test_empty_string(self):
        assert _clean_publication_number("") == ""

    def test_none_returns_empty(self):
        assert _clean_publication_number(None) == ""


# ===================================================================
# uspto_odp — _extract_patent_result
# ===================================================================


class TestExtractPatentResult:
    def test_extracts_all_fields(self):
        r = _extract_patent_result(SAMPLE_ODP_ITEM)
        assert r.application_number == "17/123456"
        assert r.title == "Machine Learning for Patent Analysis"
        assert r.inventors == ["Alice Smith", "Bob Jones"]
        assert "Acme Corp" in r.applicants
        assert r.filing_date == "2023-01-15"
        assert r.grant_date == "2025-06-10"
        assert r.publication_date == "2024-03-20"
        assert r.patent_number == "1234567"
        assert r.publication_number == "US2024001234A1"
        assert r.application_status == "Granted"
        assert r.application_type == "Utility"

    def test_minimal_item(self):
        r = _extract_patent_result({"applicationNumberText": "  99/1  "})
        assert r.application_number == "99/1"
        assert r.title == ""
        assert r.inventors == []
        assert r.applicants == []

    def test_patent_number_fallback_publication(self):
        item = {
            "applicationNumberText": "17/999",
            "applicationMetaData": {"patentNumber": 5555},
        }
        r = _extract_patent_result(item)
        assert r.publication_number == "US5555"

    def test_earliest_publication_number_whitespace(self):
        item = {
            "applicationNumberText": "17/111",
            "applicationMetaData": {
                "earliestPublicationNumber": "  US 2024-001111 A1  ",
            },
        }
        r = _extract_patent_result(item)
        assert r.publication_number == "US2024001111A1"


# ===================================================================
# uspto_odp — search_patents
# ===================================================================


class TestUSPTOODPSearchPatents:
    @mock.patch("drbrain.providers.uspto_odp.urllib.request.urlopen")
    def test_search_returns_results(self, mock_open):
        mock_open.return_value = _mock_urlopen_response(SAMPLE_ODP_RESPONSE)

        results = search_patents("machine learning", api_key="test-key")
        assert len(results) == 1
        r = results[0]
        assert r.title == "Machine Learning for Patent Analysis"
        assert r.application_number == "17/123456"

    @mock.patch("drbrain.providers.uspto_odp.urllib.request.urlopen")
    def test_search_empty_results(self, mock_open):
        mock_open.return_value = _mock_urlopen_response(
            {"count": 0, "patentFileWrapperDataBag": []}
        )

        results = search_patents("nonexistent_xyz", api_key="test-key")
        assert results == []

    @mock.patch("drbrain.providers.uspto_odp.urllib.request.urlopen")
    def test_search_sends_correct_payload(self, mock_open):
        mock_open.return_value = _mock_urlopen_response(
            {"count": 0, "patentFileWrapperDataBag": []}
        )

        search_patents("neural network", api_key="my-key", limit=5, offset=10)

        # Verify urlopen was called once
        mock_open.assert_called_once()
        call_args = mock_open.call_args
        req = call_args[0][0]
        assert req.method == "POST"
        # urllib normalizes header names (e.g. X-API-Key -> X-api-key)
        assert any(k.lower() == "x-api-key" for k in req.headers)

    @mock.patch("drbrain.providers.uspto_odp.urllib.request.urlopen")
    def test_search_custom_base_url(self, mock_open):
        mock_open.return_value = _mock_urlopen_response(
            {"count": 0, "patentFileWrapperDataBag": []}
        )

        search_patents("test", api_key="k", base_url="https://custom.api")
        req = mock_open.call_args[0][0]
        assert "custom.api" in req.full_url

    @mock.patch("drbrain.providers.uspto_odp.urllib.request.urlopen")
    def test_search_api_error_raises(self, mock_open):
        # side_effect must be the exception instance so urlopen raises it,
        # not a callable returning the exception (which would be treated as a
        # normal return value).
        mock_open.side_effect = _make_http_error(500, "Server Error")

        with pytest.raises(USPTOAPIError, match="500"):
            search_patents("test", api_key="k")


def _make_http_error(code, msg):
    from urllib.error import HTTPError

    err = HTTPError("url", code, msg, {}, None)
    err.read = mock.MagicMock(return_value=b"")
    return err


# ===================================================================
# uspto_odp — get_patent_by_application_number
# ===================================================================


class TestUSPTOODPGetByApplicationNumber:
    @mock.patch("drbrain.providers.uspto_odp.urllib.request.urlopen")
    def test_returns_patent(self, mock_open):
        mock_open.return_value = _mock_urlopen_response(SAMPLE_ODP_ITEM)

        r = get_patent_by_application_number("17/123456", api_key="k")
        assert r is not None
        assert r.title == "Machine Learning for Patent Analysis"

    @mock.patch("drbrain.providers.uspto_odp.urllib.request.urlopen")
    def test_returns_none_on_404(self, mock_open):
        mock_open.side_effect = _make_http_error(404, "Not Found")

        r = get_patent_by_application_number("99/999999", api_key="k")
        assert r is None

    @mock.patch("drbrain.providers.uspto_odp.urllib.request.urlopen")
    def test_raises_on_non_404_error(self, mock_open):
        mock_open.side_effect = _make_http_error(500, "Internal Error")

        with pytest.raises(USPTOAPIError, match="500"):
            get_patent_by_application_number("17/123456", api_key="k")


# ===================================================================
# uspto_odp — _request_json error handling
# ===================================================================


class TestUSPTOODPRequestJSON:
    @mock.patch("drbrain.providers.uspto_odp.urllib.request.urlopen")
    def test_url_error_raises(self, mock_open):
        from urllib.error import URLError

        mock_open.side_effect = URLError("Connection refused")

        with pytest.raises(USPTOAPIError, match="Request failed"):
            from drbrain.providers.uspto_odp import _request_json

            _request_json("https://api.uspto.gov/test", api_key="k")


# ===================================================================
# uspto_ppubs — PpubsPatent dataclass
# ===================================================================


class TestPpubsPatent:
    def test_inventors_parsing_semicolon(self):
        p = PpubsPatent(inventors_short="Smith; Alice, Jones; Bob")
        inv = p.inventors
        assert "Alice Smith" in inv
        assert "Bob Jones" in inv

    def test_inventors_parsing_comma(self):
        p = PpubsPatent(inventors_short="Alice Smith, Bob Jones")
        inv = p.inventors
        assert len(inv) == 2

    def test_inventors_et_al_stripped(self):
        p = PpubsPatent(inventors_short="Alice Smith et al.")
        inv = p.inventors
        assert inv == ["Alice Smith"]

    def test_inventors_empty(self):
        p = PpubsPatent(inventors_short="")
        assert p.inventors == []

    def test_to_dict_keys(self):
        p = PpubsPatent(guid="abc", publication_number="US1234A1")
        d = p.to_dict()
        assert d["guid"] == "abc"
        assert d["publication_number"] == "US1234A1"
        assert "inventors" in d
        assert "cpc_codes" in d

    def test_google_patents_url(self):
        p = PpubsPatent(publication_number="US1234567B2")
        assert p.google_patents_url() == "https://patents.google.com/patent/US1234567B2/en"


# ===================================================================
# uspto_ppubs — helpers
# ===================================================================


class TestPpubsHelpers:
    def test_normalize_publication_number(self):
        assert _normalize_publication_number("US 2024-001234 A1") == "US2024001234A1"

    def test_normalize_empty(self):
        assert _normalize_publication_number("") == ""

    def test_normalize_none(self):
        assert _normalize_publication_number(None) == ""

    def test_publication_search_query_match(self):
        assert _publication_search_query("US1234567B2") == "1234567"

    def test_publication_search_query_no_match(self):
        assert _publication_search_query("WO2024ABC") == "WO2024ABC"

    def test_strip_markup(self):
        assert _strip_markup("<b>Hello</b>  <i>World</i>") == "Hello World"

    def test_strip_markup_empty(self):
        assert _strip_markup("") == ""


# ===================================================================
# uspto_ppubs — _extract_patent
# ===================================================================


SAMPLE_PPUBS_ITEM = {
    "guid": "guid-123",
    "publicationReferenceDocumentNumber": "1234567",
    "type": "USPAT",
    "kindCode": "B2",
    "datePublished": "2025-01-15T00:00:00Z",
    "applicationFilingDate": ["2023-06-10T00:00:00Z"],
    "inventionTitle": "Neural Network Accelerator",
    "inventorsShort": "Smith; Alice, Jones; Bob",
    "applicantName": ["Acme Corp"],
    "assigneeName": ["Big Tech Inc"],
    "applicationNumber": "17/999888",
    "ipcCodeFlattened": "G06N; H04L",
    "cpcInventiveFlattened": "G06N3/08",
    "pageCount": 25,
    "imageLocation": "/path/to/image",
    "primaryExaminer": "John Doe",
}


class TestPpubsExtractPatent:
    def test_extracts_all_fields(self):
        p = ppubs_extract_patent(SAMPLE_PPUBS_ITEM)
        assert p.guid == "guid-123"
        assert p.publication_number == "US1234567B2"
        assert p.title == "Neural Network Accelerator"
        assert p.patent_type == "USPAT"
        assert p.filing_date == "2023-06-10"
        assert p.publication_date == "2025-01-15"
        assert p.page_count == 25
        assert "G06N" in p.ipc_codes
        assert "G06N3/08" in p.cpc_codes
        assert p.applicants == ["Acme Corp"]
        assert p.assignees == ["Big Tech Inc"]
        assert p.primary_examiner == "John Doe"

    def test_pgpub_type(self):
        item = {
            "publicationReferenceDocumentNumber": "0001234",
            "type": "US-PGPUB",
            "kindCode": "A1",
        }
        p = ppubs_extract_patent(item)
        assert p.publication_number == "US0001234A1"

    def test_default_kind_code_uspat(self):
        item = {
            "publicationReferenceDocumentNumber": "0001234",
            "type": "USPAT",
        }
        p = ppubs_extract_patent(item)
        assert p.publication_number == "US0001234B2"

    def test_default_kind_code_pgpub(self):
        item = {
            "publicationReferenceDocumentNumber": "0001234",
            "type": "US-PGPUB",
        }
        p = ppubs_extract_patent(item)
        assert p.publication_number == "US0001234A1"

    def test_minimal_item(self):
        p = ppubs_extract_patent({})
        assert p.guid == ""
        assert p.publication_number == ""
        assert p.page_count == 0

    def test_filing_date_scalar(self):
        item = {"applicationFilingDate": "2024-01-01T00:00:00Z"}
        p = ppubs_extract_patent(item)
        assert p.filing_date == "2024-01-01"

    def test_applicant_name_scalar(self):
        item = {"applicantName": "Solo Applicant"}
        p = ppubs_extract_patent(item)
        assert p.applicants == ["Solo Applicant"]


# ===================================================================
# uspto_ppubs — PpubsClient (mocked at urllib level)
# ===================================================================


def _mock_opener_response(data: dict, headers: dict | None = None):
    """Build a mock response for opener.open() context manager."""
    body = json.dumps(data).encode("utf-8")
    resp = mock.MagicMock()
    resp.read.return_value = body
    resp.headers = mock.MagicMock()
    if headers:
        for k, v in headers.items():
            setattr(resp.headers, k, v) if not isinstance(
                resp.headers.get, mock.MagicMock
            ) else resp.headers.get.return_value
            resp.headers.get.return_value = headers.get("X-Access-Token", v)
    resp.__enter__ = mock.MagicMock(return_value=resp)
    resp.__exit__ = mock.MagicMock(return_value=False)
    return resp


SAMPLE_SESSION_RESPONSE = {
    "userCase": {"caseId": 42},
}

SAMPLE_PPUBS_SEARCH_RESPONSE = {
    "numFound": 1,
    "patents": [SAMPLE_PPUBS_ITEM],
}


class TestPpubsClient:
    @mock.patch("drbrain.providers.uspto_ppubs.urllib.request.build_opener")
    def test_search_returns_patents(self, mock_build_opener):
        """PpubsClient.search returns parsed patents from mocked session."""
        mock_opener = mock.MagicMock()
        mock_build_opener.return_value = mock_opener

        # First call: GET /pubwebapp/ (session init)
        session_resp = _mock_opener_response(SAMPLE_SESSION_RESPONSE, {"X-Access-Token": "tok123"})
        # Second call: POST /api/users/me/session
        session_resp2 = _mock_opener_response(SAMPLE_SESSION_RESPONSE, {"X-Access-Token": "tok123"})
        # Third call: POST /api/searches/searchWithBeFamily
        search_resp = _mock_opener_response(SAMPLE_PPUBS_SEARCH_RESPONSE)

        mock_opener.open.side_effect = [session_resp, session_resp2, search_resp]

        client = PpubsClient()
        total, results = client.search("neural network")
        assert total == 1
        assert len(results) == 1
        assert results[0].title == "Neural Network Accelerator"

    @mock.patch("drbrain.providers.uspto_ppubs.urllib.request.build_opener")
    def test_search_empty_results(self, mock_build_opener):
        mock_opener = mock.MagicMock()
        mock_build_opener.return_value = mock_opener

        session_resp = _mock_opener_response(SAMPLE_SESSION_RESPONSE, {"X-Access-Token": "tok"})
        search_resp = _mock_opener_response({"numFound": 0, "patents": []})

        mock_opener.open.side_effect = [session_resp, session_resp, search_resp]

        client = PpubsClient()
        total, results = client.search("nonexistent_xyz")
        assert total == 0
        assert results == []

    @mock.patch("drbrain.providers.uspto_ppubs.urllib.request.build_opener")
    def test_session_failure_raises(self, mock_build_opener):
        from urllib.error import HTTPError

        mock_opener = mock.MagicMock()
        mock_build_opener.return_value = mock_opener

        mock_opener.open.side_effect = HTTPError("url", 500, "Error", {}, None)

        with pytest.raises(PpubsError, match="Failed to establish"):
            PpubsClient()._ensure_session()

    @mock.patch("drbrain.providers.uspto_ppubs.urllib.request.build_opener")
    def test_find_by_publication_number_match(self, mock_build_opener):
        mock_opener = mock.MagicMock()
        mock_build_opener.return_value = mock_opener

        session_resp = _mock_opener_response(SAMPLE_SESSION_RESPONSE, {"X-Access-Token": "tok"})
        search_resp = _mock_opener_response(SAMPLE_PPUBS_SEARCH_RESPONSE)

        mock_opener.open.side_effect = [session_resp, session_resp, search_resp]

        client = PpubsClient()
        result = client.find_by_publication_number("US1234567B2")
        assert result is not None
        assert result.title == "Neural Network Accelerator"

    @mock.patch("drbrain.providers.uspto_ppubs.urllib.request.build_opener")
    def test_find_by_publication_number_no_match(self, mock_build_opener):
        mock_opener = mock.MagicMock()
        mock_build_opener.return_value = mock_opener

        session_resp = _mock_opener_response(SAMPLE_SESSION_RESPONSE, {"X-Access-Token": "tok"})
        search_resp = _mock_opener_response({"numFound": 1, "patents": [SAMPLE_PPUBS_ITEM]})

        mock_opener.open.side_effect = [session_resp, session_resp, search_resp]

        client = PpubsClient()
        result = client.find_by_publication_number("US9999999B2")
        assert result is None

    def test_find_by_publication_number_empty_returns_none(self):
        client = PpubsClient()
        assert client.find_by_publication_number("") is None


# ===================================================================
# uspto_ppubs — search_patents module-level function
# ===================================================================


class TestPpubsSearchPatents:
    @mock.patch("drbrain.providers.uspto_ppubs.urllib.request.build_opener")
    def test_search_patents_module_function(self, mock_build_opener):
        mock_opener = mock.MagicMock()
        mock_build_opener.return_value = mock_opener

        session_resp = _mock_opener_response(SAMPLE_SESSION_RESPONSE, {"X-Access-Token": "tok"})
        search_resp = _mock_opener_response(SAMPLE_PPUBS_SEARCH_RESPONSE)

        mock_opener.open.side_effect = [session_resp, session_resp, search_resp]

        from drbrain.providers.uspto_ppubs import search_patents as ppubs_search

        results = ppubs_search("neural network")
        assert len(results) == 1
        assert results[0].title == "Neural Network Accelerator"
