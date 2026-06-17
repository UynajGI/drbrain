"""HTTP retry decorator for external API calls.

Provides exponential backoff with jitter for network operations.
Works with both ``urllib.request.urlopen`` and ``requests.get/post``.

Usage::

    from drbrain.utils.http_retry import http_retry

    @http_retry(max_retries=3, base_delay=1.0)
    def fetch_patent(url: str) -> dict:
        ...
"""

from __future__ import annotations

import functools
import random
import time
from collections.abc import Callable
from typing import Any

from loguru import logger

# Exceptions that indicate a transient/network error worth retrying
_RETRYABLE_NETWORK_ERRORS = (
    ConnectionError,
    TimeoutError,
    OSError,  # covers socket errors, DNS failures
)

# HTTP status codes worth retrying
_RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})


def _is_retryable_exception(exc: Exception) -> bool:
    """Check if an exception represents a transient failure."""
    if isinstance(exc, _RETRYABLE_NETWORK_ERRORS):
        return True
    # requests.RequestException — check by class name to avoid hard import
    cls_name = type(exc).__name__
    if cls_name in (
        "RequestException",
        "ConnectionError",
        "Timeout",
        "ConnectTimeout",
        "ReadTimeout",
    ):
        return True
    # urllib HTTPError with retryable status
    status = getattr(exc, "code", None)
    if status and status in _RETRYABLE_STATUS_CODES:
        return True
    return False


def http_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> Callable[..., Any]:
    """Decorator: retry on transient network errors with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (0 = no retry).
        base_delay: Initial delay in seconds. Doubles each retry.
        max_delay: Maximum delay cap in seconds.

    Returns:
        Decorated function that retries on transient failures.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if not _is_retryable_exception(e) or attempt >= max_retries:
                        raise
                    delay = min(base_delay * (2**attempt), max_delay)
                    delay += random.uniform(0, delay * 0.1)  # jitter
                    logger.warning(
                        "[http_retry] %s failed (attempt %d/%d): %s — retrying in %.1fs",
                        func.__name__,
                        attempt + 1,
                        max_retries + 1,
                        e,
                        delay,
                    )
                    time.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator
