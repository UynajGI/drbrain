"""Token counting for chunking math — tiktoken with graceful fallback.

Single seam for "how many tokens is this text": ``count_tokens(text) -> int``.
Uses the ``o200k_base`` encoding (OpenAI's current default; close enough for
chunk-budget arithmetic across providers — this is heuristic budgeting, not
billing). The encoding object is loaded lazily once per process; if tiktoken
is unavailable or the vocab cannot be loaded (e.g. offline first run), the
count falls back to ``max(1, len(text) // 4)`` so ingestion never breaks. A
failed vocab load retries after a short cooldown instead of hammering the
download on every call.
"""

from __future__ import annotations

import threading
import time

_ENCODING_RETRY_S = 60.0

_encoding_lock = threading.Lock()
_encoding: object | None = None
_encoding_failed_at: float | None = None


def _get_encoding() -> object | None:
    """Load the ``o200k_base`` encoding once; ``None`` while unavailable.

    A failed load is remembered for ``_ENCODING_RETRY_S`` (so the chunking hot
    path cannot hammer a failing download per paragraph) and retried after the
    cooldown, so token counting recovers automatically.
    """
    global _encoding, _encoding_failed_at
    if _encoding is not None:
        return _encoding
    with _encoding_lock:
        if _encoding is not None:
            return _encoding
        if (
            _encoding_failed_at is not None
            and time.monotonic() - _encoding_failed_at < _ENCODING_RETRY_S
        ):
            return None
        try:
            import tiktoken

            _encoding = tiktoken.get_encoding("o200k_base")
        except Exception:  # noqa: BLE001 — offline/vocab-download failure must not break callers
            _encoding_failed_at = time.monotonic()
            _encoding = None
    return _encoding


def count_tokens(text: str) -> int:
    """Token count of *text* (tiktoken ``o200k_base``; heuristic fallback).

    Empty input counts as 0. When the encoding cannot be loaded or encoding
    fails, falls back to ``max(1, len(text) // 4)`` for non-empty text.
    """
    if not text:
        return 0
    enc = _get_encoding()
    if enc is None:
        return max(1, len(text) // 4)
    try:
        return len(enc.encode(text))  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 — never let tokenizer errors break chunking
        return max(1, len(text) // 4)
