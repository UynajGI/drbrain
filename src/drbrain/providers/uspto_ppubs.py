"""USPTO Patent Public Search (PPUBS) client — no API key required.

Session-based web client for ppubs.uspto.gov. Auto-manages cookies,
session creation, and token refresh.
"""

from __future__ import annotations

import http.cookiejar
import json
import re as _re
import time as _time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from html import unescape

from loguru import logger

from drbrain.providers.base import PatentBase, clean_publication_number

PPUBS_BASE_URL = "https://ppubs.uspto.gov"
US_PUBLICATION_NUMBER_PATTERN = _re.compile(
    r"^US(?P<number>\d{6,})(?P<kind>[A-Z]\d?)?$", _re.IGNORECASE
)


class PpubsError(Exception):
    """PPUBS request error."""


@dataclass
class PpubsPatent(PatentBase):
    """PPUBS patent search result.

    Inherits ``google_patents_url()`` and ``_common_dict()`` from PatentBase.
    """

    guid: str = ""
    inventors_short: str = ""
    applicants: list[str] = field(default_factory=list)
    assignees: list[str] = field(default_factory=list)
    filing_date: str = ""
    publication_date: str = ""
    patent_type: str = ""
    page_count: int = 0
    image_location: str = ""
    ipc_codes: list[str] = field(default_factory=list)
    cpc_codes: list[str] = field(default_factory=list)
    primary_examiner: str = ""
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def inventors(self) -> list[str]:
        if not self.inventors_short:
            return []
        text = self.inventors_short.replace(" et al.", "").strip()
        parts = [p.strip() for p in text.split(",") if p.strip()]
        result = []
        for part in parts:
            if ";" in part:
                last, first = part.split(";", 1)
                result.append(f"{first.strip()} {last.strip()}")
            else:
                result.append(part)
        return result

    def to_dict(self) -> dict:
        return {
            **self._common_dict(),
            "guid": self.guid,
            "inventors": self.inventors,
            "applicants": self.applicants,
            "assignees": self.assignees,
            "filing_date": self.filing_date,
            "publication_date": self.publication_date,
            "patent_type": self.patent_type,
            "page_count": self.page_count,
            "ipc_codes": self.ipc_codes,
            "cpc_codes": self.cpc_codes,
        }


class PpubsClient:
    """USPTO PPUBS session client with auto-refresh."""

    def __init__(self, base_url: str = PPUBS_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self._cookie_jar = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self._cookie_jar)
        )
        self._token: str | None = None
        self._case_id: int | None = None

    def _ensure_session(self) -> None:
        if self._token and self._case_id:
            return

        logger.debug("[patent] establishing new PPUBS session")

        req1 = urllib.request.Request(f"{self.base_url}/pubwebapp/", method="GET")
        req1.add_header(
            "User-Agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        )
        try:
            self._opener.open(req1)
        except urllib.error.HTTPError as e:
            raise PpubsError(f"Failed to establish PPUBS session: {e}") from e

        req2 = urllib.request.Request(
            f"{self.base_url}/api/users/me/session",
            data=b"-1",
            method="POST",
        )
        req2.add_header("X-Access-Token", "null")
        req2.add_header("referer", f"{self.base_url}/pubwebapp/")
        req2.add_header("Content-Type", "application/json")

        try:
            with self._opener.open(req2) as resp:
                session = json.loads(resp.read().decode("utf-8"))
                self._case_id = session["userCase"]["caseId"]
                self._token = resp.headers.get("X-Access-Token")
        except (urllib.error.HTTPError, KeyError) as e:
            raise PpubsError(f"Failed to establish PPUBS session: {e}") from e

        if not self._token or not self._case_id:
            raise PpubsError("PPUBS session returned empty token or caseId")

        logger.debug("[patent] PPUBS session established — caseId=%s", self._case_id)

    def _request_json(self, method: str, url: str, data: dict | None = None) -> dict:
        self._ensure_session()

        body = json.dumps(data).encode("utf-8") if data else None
        for attempt in range(3):
            req = urllib.request.Request(url, data=body, method=method)
            req.add_header("X-Access-Token", self._token or "")
            req.add_header(
                "User-Agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            )
            req.add_header("referer", f"{self.base_url}/pubwebapp/")
            req.add_header("Origin", self.base_url)
            req.add_header("X-Requested-With", "XMLHttpRequest")
            if data:
                req.add_header("Content-Type", "application/json")

            try:
                with self._opener.open(req) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code == 403 and attempt < 2:
                    logger.debug("[patent] PPUBS session expired, refreshing")
                    self._token = None
                    self._case_id = None
                    self._ensure_session()
                    continue
                try:
                    detail = e.read().decode("utf-8", "replace")
                except Exception:
                    detail = ""
                raise PpubsError(f"HTTP {e.code}: {detail}") from e
            except urllib.error.URLError as e:
                if attempt == 0:
                    logger.debug("[patent] transient PPUBS failure, retrying — %s", e.reason)
                    _time.sleep(0.2)
                    continue
                raise PpubsError(f"Request failed: {e.reason}") from e
            except json.JSONDecodeError as e:
                raise PpubsError(f"Invalid JSON response: {e}") from e

        raise PpubsError("PPUBS request failed after session refresh")

    def search(
        self,
        query: str,
        *,
        start: int = 0,
        limit: int = 10,
        sort: str = "date_publ desc",
    ) -> tuple[int, list[PpubsPatent]]:
        """Search USPTO patents via PPUBS."""
        sources = ["US-PGPUB", "USPAT", "USOCR"]
        self._ensure_session()

        query_data: dict = {
            "caseId": self._case_id,
            "hl_snippets": "2",
            "op": "OR",
            "q": query,
            "queryName": query,
            "highlights": "1",
            "qt": "brs",
            "spellCheck": False,
            "viewName": "tile",
            "plurals": True,
            "britishEquivalents": True,
            "databaseFilters": [{"databaseName": s, "countryCodes": []} for s in sources],
            "searchType": 1,
            "ignorePersist": True,
            "userEnteredQuery": query,
        }

        search_payload: dict = {
            "start": start,
            "pageCount": min(max(limit, 1), 100),
            "sort": sort,
            "docFamilyFiltering": "familyIdFiltering",
            "searchType": 1,
            "familyIdEnglishOnly": True,
            "familyIdFirstPreferred": "US-PGPUB",
            "familyIdSecondPreferred": "USPAT",
            "familyIdThirdPreferred": "FPRS",
            "showDocPerFamilyPref": "showEnglish",
            "queryId": 0,
            "tagDocSearch": False,
            "query": query_data,
        }

        logger.info("[patent] PPUBS search — query=%r limit=%d", query, limit)
        result = self._request_json(
            "POST",
            f"{self.base_url}/api/searches/searchWithBeFamily",
            search_payload,
        )

        patents = result.get("patents") or []
        total = result.get("numFound", 0)
        logger.info("[patent] PPUBS returned %d / %d results", len(patents), total)
        return total, [_extract_patent(p) for p in patents]

    def find_by_publication_number(
        self, publication_number: str, *, limit: int = 10
    ) -> PpubsPatent | None:
        """Find a patent by publication number."""
        normalized = clean_publication_number(publication_number)
        if not normalized:
            return None

        query = _publication_search_query(normalized)
        _, results = self.search(query, limit=limit)
        for patent in results:
            if clean_publication_number(patent.publication_number) == normalized:
                return patent
        return None


def _publication_search_query(pn: str) -> str:
    normalized = clean_publication_number(pn)
    match = US_PUBLICATION_NUMBER_PATTERN.match(normalized)
    return match.group("number") if match else normalized


def _strip_markup(text: str) -> str:
    if not text:
        return ""
    plain = _re.sub(r"<[^>]+>", "", text)
    return _re.sub(r"\s+", " ", unescape(plain)).strip()


def _extract_patent(item: dict) -> PpubsPatent:
    pub_num_raw = str(item.get("publicationReferenceDocumentNumber", "")).strip()
    patent_type = item.get("type", "")

    publication_number = ""
    if pub_num_raw and patent_type:
        kind = ""
        kc = item.get("kindCode")
        if kc:
            kind = kc[0] if isinstance(kc, list) else kc
        if patent_type == "US-PGPUB":
            publication_number = f"US{pub_num_raw}{kind or 'A1'}"
        elif patent_type == "USPAT":
            publication_number = f"US{pub_num_raw}{kind or 'B2'}"
        else:
            publication_number = f"US{pub_num_raw}{kind}"

    pub_date = item.get("datePublished", "")
    if pub_date:
        pub_date = pub_date[:10]

    filing_date = ""
    afd = item.get("applicationFilingDate")
    if afd:
        filing_date = (afd[0][:10] if isinstance(afd, list) else str(afd))[:10]

    ipc = []
    if item.get("ipcCodeFlattened"):
        ipc = [c.strip() for c in str(item["ipcCodeFlattened"]).split(";") if c.strip()]
    cpc = []
    if item.get("cpcInventiveFlattened"):
        cpc = [c.strip() for c in str(item["cpcInventiveFlattened"]).split(";") if c.strip()]

    applicants = []
    an = item.get("applicantName")
    if an:
        applicants = an if isinstance(an, list) else [an]

    assignees = []
    an2 = item.get("assigneeName")
    if an2:
        assignees = an2 if isinstance(an2, list) else [an2]

    return PpubsPatent(
        guid=item.get("guid", ""),
        publication_number=publication_number,
        title=_strip_markup(str(item.get("inventionTitle", ""))),
        inventors_short=item.get("inventorsShort", ""),
        applicants=applicants,
        assignees=assignees,
        application_number=str(item.get("applicationNumber", "")),
        filing_date=filing_date,
        publication_date=pub_date,
        patent_type=patent_type,
        page_count=int(item.get("pageCount", 0) or 0),
        image_location=str(item.get("imageLocation", "")),
        ipc_codes=ipc,
        cpc_codes=cpc,
        primary_examiner=str(item.get("primaryExaminer", "")),
        raw=item,
    )


def search_patents(
    query: str,
    *,
    limit: int = 10,
    offset: int = 0,
) -> list[PpubsPatent]:
    """Search USPTO patents via PPUBS — no API key needed.

    Args:
        query: Search query string.
        limit: Max results (default 10).
        offset: Pagination offset.

    Returns:
        List of PpubsPatent.
    """
    client = PpubsClient()
    _, results = client.search(query, start=offset, limit=limit)
    return results
