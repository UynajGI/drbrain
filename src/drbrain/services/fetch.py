"""PDF acquisition from open access sources with multi-stage fallback."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import requests
from loguru import logger

from drbrain.utils.http_retry import http_retry


@http_retry(max_retries=3, base_delay=1.0)
def resolve_pdf_url(
    doi: str | None = None,
    title: str | None = None,
    arxiv_id: str | None = None,
    fetch_config: dict | None = None,
) -> str | None:
    """Try to find a PDF URL through multiple fallback stages.

    Returns None if no source can provide a PDF.
    """
    cfg = fetch_config or {}

    # Stage 1: arXiv (most reliable for papers with arXiv ID)
    if arxiv_id:
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        if _url_exists(url):
            return url

    # Stage 2: OpenAlex OA location
    if doi and (oa_url := _try_openalex_oa(doi)):
        return oa_url

    # Stage 3: Unpaywall
    if doi and (uw_url := _try_unpaywall(doi, cfg)):
        return uw_url

    # Stage 4: Direct DOI with Accept header
    if doi and (direct_url := _try_direct_doi(doi)):
        return direct_url

    # Stage 5: Title-based arXiv search
    if title and not arxiv_id:
        found_id = _search_arxiv_by_title(title)
        if found_id:
            return f"https://arxiv.org/pdf/{found_id}.pdf"

    return None


def _url_exists(url: str) -> bool:
    """HEAD request to check if URL is accessible."""
    try:
        resp = requests.head(url, timeout=10, allow_redirects=True)
        return resp.status_code == 200
    except Exception:
        return False


def _try_openalex_oa(doi: str) -> str | None:
    """Try OpenAlex for OA PDF URL."""
    try:
        from drbrain.extractor.openalex import get_work_enriched

        data = get_work_enriched(doi)
        if not data:
            return None
        oa = data.get("open_access", {})
        if oa.get("is_oa") and oa.get("pdf_url"):
            return oa["pdf_url"]
        # Also check primary_location
        primary = data.get("primary_location", {})
        if primary.get("pdf_url"):
            return primary["pdf_url"]
    except Exception:
        logger.exception("OpenAlex OA lookup failed")
    return None


def _try_unpaywall(doi: str, cfg: dict) -> str | None:
    """Try Unpaywall API for legal OA PDF URL."""
    email = cfg.get("unpaywall_email") or cfg.get("email", "")
    if not email:
        return None
    try:
        url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        best = data.get("best_oa_location") or {}
        pdf_url = best.get("url_for_pdf") or best.get("url")
        if pdf_url and _url_exists(pdf_url):
            return pdf_url
    except Exception:
        logger.exception("Unpaywall lookup failed")
    return None


def _try_direct_doi(doi: str) -> str | None:
    """Try direct DOI resolution with PDF accept header."""
    try:
        url = f"https://doi.org/{doi}"
        resp = requests.head(
            url,
            timeout=15,
            allow_redirects=True,
            headers={"Accept": "application/pdf"},
        )
        final_url = resp.url
        if _url_exists(final_url):
            return final_url
    except Exception:
        pass
    return None


def _search_arxiv_by_title(title: str) -> str | None:
    """Search arXiv API by title, return arxiv_id if found."""
    try:
        import urllib.parse
        from xml.etree import ElementTree

        query = urllib.parse.quote(title[:200])
        url = f"https://export.arxiv.org/api/query?search_query=ti:{query}&max_results=1"
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        # Parse arXiv ID from response
        root = ElementTree.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns):
            id_url = entry.find("atom:id", ns)
            if id_url is not None and id_url.text:
                # Extract ID from http://arxiv.org/abs/XXXX.XXXXX
                match = re.search(r"arxiv\.org/abs/([\w.\-]+)", id_url.text)
                if match:
                    return match.group(1)
    except Exception:
        logger.exception("arXiv title search failed")
    return None


def _proxy_url(url: str, cfg: dict) -> str:
    """Apply institutional proxy to a URL."""
    proxy_type = cfg.get("proxy_type", "")
    proxy_host = cfg.get("institutional_proxy", "")
    if not proxy_host:
        return url

    from urllib.parse import urlparse

    if proxy_type == "ezproxy":
        # Replace domain dots with dashes, append proxy host
        parsed = urlparse(url)
        proxy_domain = parsed.netloc.replace(".", "-")
        proxy_url_str = f"{parsed.scheme}://{proxy_domain}.{proxy_host}{parsed.path}"
        if parsed.query:
            proxy_url_str += f"?{parsed.query}"
        return proxy_url_str
    elif proxy_type == "url_prefix":
        return f"{proxy_host}{url}"

    return url


def download_pdf(url: str, paper_dir: Path, fetch_config: dict | None = None) -> Path | None:
    """Download PDF to paper_dir/source.pdf. Returns path or None."""
    cfg = fetch_config or {}
    paper_dir.mkdir(parents=True, exist_ok=True)
    dest = paper_dir / "source.pdf"

    user_agent = cfg.get("user_agent", "DrBrain/0.1")
    timeout = cfg.get("timeout_per_fetch", 60)
    headers = {"User-Agent": user_agent}
    try:
        resp = requests.get(url, stream=True, timeout=timeout, headers=headers)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")

        # Detect PDF by reading first bytes (some servers serve incorrect content-type)
        peek = resp.raw.peek(5) if hasattr(resp.raw, "peek") else None
        if peek is None:
            # Read first 5 bytes from iter_content
            chunks = []
            for chunk in resp.iter_content(chunk_size=5):
                chunks.append(chunk)
                break
            peek = b"".join(chunks) if chunks else b""

        is_pdf = False
        if peek and peek[:5] == b"%PDF-":
            is_pdf = True
        elif "pdf" in content_type or url.endswith(".pdf"):
            is_pdf = True

        if not is_pdf:
            logger.warning(f"Response is not a PDF: {url} (content-type: {content_type})")
            return None

        # Write to disk
        with open(dest, "wb") as f:
            if peek and peek[:5] == b"%PDF-":
                f.write(peek)
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        # Verify file written
        if dest.stat().st_size == 0:
            dest.unlink(missing_ok=True)
            return None

        return dest
    except Exception:
        logger.exception(f"Download failed: {url}")
        return None


def fetch_paper(
    doi: str | None = None,
    title: str | None = None,
    arxiv_id: str | None = None,
    fetch_config: dict | None = None,
) -> dict | None:
    """Fetch a paper: find PDF -> download -> return metadata for ingest.

    Returns dict with: title, year, doi, arxiv, local_id, pdf_path
    Returns None if PDF cannot be acquired.
    """
    cfg = fetch_config or {}

    # Resolve PDF URL through fallback stages
    pdf_url = resolve_pdf_url(doi=doi, title=title, arxiv_id=arxiv_id, fetch_config=cfg)
    if not pdf_url:
        logger.warning(f"No PDF URL found for doi={doi} title={title} arxiv={arxiv_id}")
        return None

    # Apply proxy if configured
    pdf_url = _proxy_url(pdf_url, cfg)

    # Resolve metadata
    meta = _resolve_metadata(doi=doi, title=title, arxiv_id=arxiv_id)
    if not meta:
        logger.warning(f"Could not resolve metadata for doi={doi} title={title}")
        return None

    # Download PDF to paper directory
    from drbrain.storage.paths import paper_dir

    papers_root = Path(cfg.get("papers_root", "data/papers"))
    pdir = paper_dir(papers_root, meta["local_id"])

    pdf_path = download_pdf(pdf_url, pdir, cfg)
    if not pdf_path:
        return None

    return {
        "local_id": meta["local_id"],
        "title": meta["title"],
        "year": meta["year"],
        "doi": meta.get("doi"),
        "arxiv": meta.get("arxiv"),
        "pdf_path": str(pdf_path),
    }


def _resolve_metadata(
    doi: str | None = None,
    title: str | None = None,
    arxiv_id: str | None = None,
) -> dict | None:
    """Quick metadata resolution from OpenAlex or arXiv.

    Returns a dict with local_id, title, year, doi, arxiv, or None.
    The local_id assigned here is preliminary and will be replaced by
    the dedup engine during proper ingestion.
    """
    local_id = f"p{uuid.uuid4().hex[:6]}"

    if doi:
        try:
            from drbrain.extractor.openalex import get_work_by_doi

            data = get_work_by_doi(doi)
            if data:
                return {
                    "local_id": local_id,
                    "title": data.get("title", ""),
                    "year": data.get("publication_year"),
                    "doi": doi,
                    "arxiv": None,
                }
        except Exception:
            pass

    if arxiv_id:
        try:
            from drbrain.parser.mineru_parser import _fetch_arxiv_metadata

            title_result, year = _fetch_arxiv_metadata(arxiv_id)
            if title_result:
                return {
                    "local_id": local_id,
                    "title": title_result,
                    "year": year,
                    "doi": None,
                    "arxiv": arxiv_id,
                }
        except Exception:
            pass

    if title:
        try:
            from drbrain.extractor.openalex import search_work_by_title

            data = search_work_by_title(title)
            if data:
                return {
                    "local_id": local_id,
                    "title": data.get("title", title),
                    "year": data.get("publication_year"),
                    "doi": data.get("doi"),
                    "arxiv": None,
                }
        except Exception:
            pass

    return None


def _resolve_identifier(
    identifier: str, is_arxiv: bool = False
) -> tuple[str | None, str | None, str | None]:
    """Classify an identifier as DOI, arXiv ID, arXiv URL, or title.

    Returns (doi, title, arxiv_id).
    """
    if is_arxiv:
        return (None, None, identifier)

    ident = identifier.strip()

    # arXiv URL: https://arxiv.org/abs/2301.00234 or /pdf/2301.00234
    arxiv_url_match = re.search(r"arxiv\.org/(?:abs|pdf)/([\w.\-]+)", ident)
    if arxiv_url_match:
        arxiv_id = arxiv_url_match.group(1).replace(".pdf", "")
        return (None, None, arxiv_id)

    # arXiv ID: new-style 4 digits + dot + 4-5 digits (e.g. 1706.03762)
    # Old-style: subject-class/digits (e.g. cs.AI/0703111)
    if re.match(r"^\d{4}\.\d{4,5}$", ident) or re.match(r"^[A-Za-z\-\.]+/\d{7}$", ident):
        return (None, None, ident)

    # DOI detection: starts with "10." or contains "/" and no spaces
    if ident.startswith("10.") or ("/" in ident and " " not in ident):
        return (ident, None, None)

    # Otherwise treat as title
    return (None, ident, None)
