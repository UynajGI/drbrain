"""HTTP transport for Sciverse: Bearer auth, rate limiting and retry.

Sciverse enforces a per-account rate limit (default 30 requests/minute). This
module provides a small token-bucket limiter plus a thin client that reuses the
project-wide :func:`drbrain.utils.http_retry.http_retry` decorator for transient
failures (429 / 502 / 503 / 504) with exponential backoff.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import requests
from loguru import logger

from drbrain.concept_graph.sources.base import is_success_envelope
from drbrain.utils.http_retry import http_retry


class SciverseAPIError(RuntimeError):
    """Raised when a Sciverse call fails (HTTP error or non-success envelope)."""

    def __init__(self, message: str, *, status: int | None = None, code: int | str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code


class TokenBucketRateLimiter:
    """A simple sliding-window rate limiter (``rate_limit`` calls per 60s).

    Thread-safe. ``acquire`` blocks until a slot is available within the window.

    Args:
        rate_limit: Maximum number of calls allowed per 60-second window.
            Values <= 0 disable limiting.
        clock: Monotonic clock callable (injectable for tests).
        sleep: Sleep callable (injectable for tests).
    """

    window: float = 60.0

    def __init__(
        self,
        rate_limit: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.rate_limit = rate_limit
        self._clock = clock
        self._sleep = sleep
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a request slot is available within the sliding window."""
        if self.rate_limit <= 0:
            return
        while True:
            with self._lock:
                now = self._clock()
                while self._calls and now - self._calls[0] >= self.window:
                    self._calls.popleft()
                if len(self._calls) < self.rate_limit:
                    self._calls.append(now)
                    return
                wait = self.window - (now - self._calls[0])
            self._sleep(max(wait, 0.01))


class SciverseClient:
    """Minimal JSON client for the Sciverse REST API.

    Args:
        token: Bearer API token.
        base_url: API base URL (no trailing slash).
        rate_limit: Requests per minute (token-bucket capacity).
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        token: str,
        base_url: str = "https://api.sciverse.space",
        *,
        rate_limit: int = 30,
        timeout: int = 60,
    ):
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.limiter = TokenBucketRateLimiter(rate_limit)
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
        )

    def post(self, path: str, json: dict[str, Any] | None = None) -> dict[str, Any]:
        """POST JSON to ``path`` and return the decoded body."""
        return self._request("POST", path, json=json or {})

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET ``path`` with query params and return the decoded body."""
        return self._request("GET", path, params=params or {})

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        retryable = {429, 502, 503, 504}

        @http_retry(max_retries=3, base_delay=1.0)
        def _do_request() -> requests.Response:
            self.limiter.acquire()
            resp = self._session.request(method, url, timeout=self.timeout, **kwargs)
            # Raise retryable statuses INSIDE the wrapper so http_retry actually
            # backs off and retries (a returned response is treated as success).
            if resp.status_code in retryable:
                raise SciverseAPIError(
                    f"Sciverse {method} {path}: HTTP {resp.status_code} (retryable)",
                    status=resp.status_code,
                    code=resp.status_code,
                )
            return resp

        resp = _do_request()
        if resp.status_code >= 400:
            raise SciverseAPIError(
                f"Sciverse {method} {path} failed: HTTP {resp.status_code} — {resp.text[:300]}",
                status=resp.status_code,
            )
        try:
            payload = resp.json()
        except ValueError as exc:  # pragma: no cover - defensive
            raise SciverseAPIError(f"Sciverse {method} {path}: invalid JSON response") from exc
        if not is_success_envelope(payload):
            raise SciverseAPIError(
                f"Sciverse {method} {path}: {payload.get('message', 'non-success envelope')}",
                code=str(payload.get("code")),
            )
        logger.debug("[sciverse] {} {} ok", method, path)
        return payload
