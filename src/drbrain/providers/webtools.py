"""HTTP connector for external web extraction service (qt-web-extractor).

ScholarAIO-compatible interface. The external service renders web pages
and returns extracted text, HTML, and metadata — DrBrain never renders pages itself.
"""

from __future__ import annotations

import json
import os as _os
import re as _re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from loguru import logger

_DEFAULT_WEBEXTRACT_URL = "http://127.0.0.1:8766"


def _get_webextract_url() -> str:
    url = _os.environ.get("WEBEXTRACT_URL") or _os.environ.get("QT_WEB_EXTRACTOR_URL") or ""
    if url:
        return url.rstrip("/")
    return _DEFAULT_WEBEXTRACT_URL


def _get_webextract_timeout() -> float:
    try:
        return float(_os.environ.get("WEBEXTRACT_TIMEOUT", "60"))
    except ValueError:
        return 60.0


def _http_json_post(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        logger.warning("[webtools] HTTP %s from %s — %s", e.code, url, detail)
        return {"error": f"HTTP {e.code}: {detail}"}
    except URLError as e:
        logger.warning("[webtools] connection error %s — %s", url, e.reason)
        return {"error": str(e.reason)}
    except (OSError, ValueError) as e:
        logger.warning("[webtools] request failed %s — %s", url, e)
        return {"error": str(e)}


def _slugify_title(title: str, url: str) -> str:
    """Generate a filesystem-safe slug from title or URL."""
    raw = (title or "").strip() or url
    # Extract host+path from URL if used as fallback
    if raw.startswith(("http://", "https://")):
        from urllib.parse import urlparse as _urlparse

        parsed = _urlparse(raw)
        raw = Path(parsed.path).stem or parsed.netloc
    slug = _re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")[:120].strip("-")
    return slug or "web-link"


def extract_web(
    url: str,
    *,
    pdf: bool | None = None,
    timeout: float | None = None,
) -> dict:
    """Extract rendered content from a web URL via external qt-web-extractor service.

    Args:
        url: Target URL to extract.
        pdf: If True, force PDF extraction mode. If None, auto-detect from URL.
        timeout: Request timeout in seconds (default: WEBEXTRACT_TIMEOUT env or 60).

    Returns:
        Dict with keys: ``url``, ``title``, ``text`` (markdown), ``html``,
        ``images`` (list of dicts with ``data``/``alt``/``src``),
        ``extracted_at`` (ISO timestamp), ``error`` (str if failed).
    """
    base_url = _get_webextract_url()
    endpoint = f"{base_url}/extract"
    timeout_val = timeout if timeout is not None else _get_webextract_timeout()

    payload: dict = {"url": url}
    if pdf is not None:
        payload["pdf"] = pdf

    logger.info("[webtools] extracting %s — timeout=%.0fs", url, timeout_val)
    result = _http_json_post(endpoint, payload, timeout_val)

    if result.get("error") and not result.get("title"):
        logger.warning("[webtools] extraction failed for %s — %s", url, result["error"])
        return {
            "url": url,
            "title": "",
            "text": "",
            "html": "",
            "images": [],
            "extracted_at": "",
            "error": result["error"],
        }

    return {
        "url": result.get("url", url),
        "title": result.get("title", ""),
        "text": result.get("text", result.get("markdown", "")),
        "html": result.get("html", ""),
        "images": result.get("images", []),
        "extracted_at": result.get("extracted_at", ""),
        "error": result.get("error", ""),
    }


def check_webextract_service(timeout: float = 3.0) -> bool:
    """Check if the web extraction service is reachable.

    Args:
        timeout: Health check timeout in seconds.

    Returns:
        True if the service responds, False otherwise.
    """
    try:
        url = f"{_get_webextract_url()}/health"
        req = Request(url)
        with urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False
