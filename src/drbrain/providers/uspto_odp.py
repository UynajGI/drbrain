"""USPTO Open Data Portal (ODP) patent search.

API docs: https://data.uspto.gov/apis/getting-started
Requires API key — register at https://data.uspto.gov/apis/getting-started
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from loguru import logger

from drbrain.providers.base import PatentBase, clean_publication_number

USPTO_ODP_BASE_URL = "https://api.uspto.gov"


class USPTOAPIError(Exception):
    """USPTO ODP API error."""


@dataclass
class PatentResult(PatentBase):
    """USPTO patent search result.

    Inherits ``google_patents_url()`` from PatentBase.
    """

    inventors: list[str] = field(default_factory=list)
    applicants: list[str] = field(default_factory=list)
    filing_date: str = ""
    grant_date: str = ""
    publication_date: str = ""
    patent_number: str = ""
    application_status: str = ""
    application_type: str = ""
    earliest_publication_number: str = ""
    raw: dict = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        return {
            **self._common_dict(),
            "inventors": self.inventors,
            "applicants": self.applicants,
            "filing_date": self.filing_date,
            "grant_date": self.grant_date,
            "publication_date": self.publication_date,
            "patent_number": self.patent_number,
            "application_status": self.application_status,
            "application_type": self.application_type,
        }

    def _best_year(self) -> str | None:
        for d in (self.grant_date, self.publication_date, self.filing_date):
            if d:
                return d[:4]
        return None

    def to_meta_dict(self) -> dict:
        """Convert to paper metadata dict for DB ingestion."""
        year_str = self._best_year()
        year = int(year_str) if year_str and year_str.isdigit() else None
        first_author = self.inventors[0] if self.inventors else "Unknown"
        first_author_lastname = first_author.split()[-1] if first_author else "Unknown"

        return {
            "id": "",
            "title": self.title or f"US Patent Application {self.application_number}",
            "authors": self.inventors[:],
            "first_author": first_author,
            "first_author_lastname": first_author_lastname,
            "year": year,
            "publication_number": self.publication_number,
            "journal": "",
            "abstract": self._build_abstract(),
            "paper_type": "patent",
            "source_url": self.google_patents_url(),
            "source_file": "",
            "citation_count": {},
            "ids": {
                "patent_publication_number": self.publication_number,
                "uspto_application_number": self.application_number,
            },
            "api_sources": ["uspto_odp"],
            "references": [],
            "extraction_method": "uspto_odp_lookup",
            "extracted_at": "",
        }

    def _build_abstract(self) -> str:
        lines: list[str] = []
        if self.title:
            lines.append(f"**Title:** {self.title}")
        lines.append(f"**Application Number:** {self.application_number}")
        if self.patent_number:
            lines.append(f"**Patent Number:** US{self.patent_number}")
        if self.publication_number and self.publication_number != f"US{self.patent_number}":
            lines.append(f"**Publication Number:** {self.publication_number}")
        if self.inventors:
            lines.append(f"**Inventors:** {', '.join(self.inventors)}")
        if self.applicants:
            lines.append(f"**Applicants:** {', '.join(self.applicants)}")
        if self.filing_date:
            lines.append(f"**Filing Date:** {self.filing_date}")
        if self.grant_date:
            lines.append(f"**Grant Date:** {self.grant_date}")
        if self.application_type:
            lines.append(f"**Type:** {self.application_type}")
        if self.application_status:
            lines.append(f"**Status:** {self.application_status}")
        return "\n\n".join(lines)


def _clean_publication_number(raw: str) -> str:
    if not raw:
        return ""
    return raw.replace(" ", "").replace("-", "")


def _extract_patent_result(item: dict) -> PatentResult:
    app_num = item.get("applicationNumberText", "").strip()
    meta = item.get("applicationMetaData") or {}

    title = meta.get("inventionTitle", "")

    inventors: list[str] = []
    for inv in meta.get("inventorBag") or []:
        name = inv.get("inventorNameText", "").strip()
        if not name:
            name = f"{inv.get('firstName', '').strip()} {inv.get('lastName', '').strip()}".strip()
        if name:
            inventors.append(name)

    applicants: list[str] = []
    for app in meta.get("applicantBag") or []:
        name = app.get("applicantName", "").strip()
        if not name:
            name = f"{app.get('firstName', '').strip()} {app.get('lastName', '').strip()}".strip()
        if name:
            applicants.append(name)
    if not applicants and meta.get("firstApplicantName"):
        applicants.append(meta["firstApplicantName"].strip())

    filing_date = meta.get("filingDate", "")
    grant_date = meta.get("grantDate", "")
    pub_date = ""
    pub_dates = meta.get("publicationDateBag") or []
    if pub_dates:
        pub_date = pub_dates[0]
    if not pub_date and meta.get("earliestPublicationDate"):
        pub_date = meta["earliestPublicationDate"]

    patent_number = str(meta.get("patentNumber", "")).strip()
    earliest_pub = str(meta.get("earliestPublicationNumber", "")).strip()

    publication_number = clean_publication_number(earliest_pub)
    if not publication_number and patent_number:
        publication_number = f"US{patent_number}"

    return PatentResult(
        application_number=app_num,
        title=title,
        inventors=inventors,
        applicants=applicants,
        filing_date=filing_date,
        grant_date=grant_date,
        publication_date=pub_date,
        patent_number=patent_number,
        publication_number=publication_number,
        application_status=meta.get("applicationStatusDescriptionText", ""),
        application_type=meta.get("applicationTypeLabelName", ""),
        earliest_publication_number=earliest_pub,
        raw=item,
    )


def _request_json(
    url: str,
    *,
    api_key: str | None = None,
    method: str = "GET",
    data: dict | None = None,
    timeout: float = 30.0,
) -> dict:
    headers = {
        "User-Agent": "DrBrain/0.1 (https://github.com/UynajGI/DrBrain)",
        "Accept": "application/json",
    }
    if api_key:
        headers["X-API-Key"] = api_key

    body = None
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8")
        except Exception:
            detail = ""
        raise USPTOAPIError(f"HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise USPTOAPIError(f"Request failed: {e.reason}") from e
    except json.JSONDecodeError as e:
        raise USPTOAPIError(f"Invalid JSON response: {e}") from e


def search_patents(
    query: str,
    *,
    api_key: str,
    limit: int = 10,
    offset: int = 0,
    base_url: str = USPTO_ODP_BASE_URL,
    timeout: float = 30.0,
) -> list[PatentResult]:
    """Search USPTO patents via ODP API.

    Args:
        query: Search query (OpenSearch syntax).
        api_key: USPTO ODP API key.
        limit: Max results (default 10).
        offset: Pagination offset.
        base_url: API base URL.
        timeout: Request timeout in seconds.

    Returns:
        List of PatentResult.
    """
    url = f"{base_url.rstrip('/')}/api/v1/patent/applications/search"
    payload = {
        "q": query,
        "pagination": {"offset": offset, "limit": min(max(limit, 1), 100)},
    }

    logger.info("[patent] ODP search — query=%r limit=%d", query, limit)
    data = _request_json(url, api_key=api_key, method="POST", data=payload, timeout=timeout)

    count = data.get("count", 0)
    results = data.get("patentFileWrapperDataBag") or []
    logger.info("[patent] ODP returned %d / %d results", len(results), count)

    return [_extract_patent_result(item) for item in results]


def get_patent_by_application_number(
    app_number: str,
    *,
    api_key: str,
    base_url: str = USPTO_ODP_BASE_URL,
    timeout: float = 30.0,
) -> PatentResult | None:
    """Look up a patent by application number.

    Args:
        app_number: Patent application number, e.g. "17123456".
        api_key: USPTO ODP API key.
        base_url: API base URL.
        timeout: Request timeout in seconds.

    Returns:
        PatentResult or None if not found.
    """
    app_number = app_number.strip()
    url = f"{base_url.rstrip('/')}/api/v1/patent/applications/{app_number}"

    logger.info("[patent] ODP lookup — app_number=%r", app_number)
    try:
        data = _request_json(url, api_key=api_key, method="GET", timeout=timeout)
    except USPTOAPIError as e:
        if "404" in str(e):
            return None
        raise

    return _extract_patent_result(data)
