"""Token counting for chunking math — tiktoken with graceful fallback.

Single seam for "how many tokens is this text": ``count_tokens(text) -> int``.
Uses the ``o200k_base`` encoding (OpenAI's current default; close enough for
chunk-budget arithmetic across providers — this is heuristic budgeting, not
billing). The encoding object is loaded lazily once per process; if tiktoken
is unavailable or the vocab cannot be loaded (e.g. offline first run), the
count falls back to ``max(1, len(text) // 4)`` so ingestion never breaks.
"""

from __future__ import annotations

import threading

_encoding_lock = threading.Lock()
_encoding: object | None = None


def _get_encoding() -> object | None:
    """Load the ``o200k_base`` encoding once; ``None`` while unavailable.

    A failed load (offline first run, transient vocab-download error) is NOT
    latched: the next call retries, so token counting recovers automatically
    once tiktoken/the vocab becomes available.
    """
    global _encoding
    if _encoding is not None:
        return _encoding
    with _encoding_lock:
        if _encoding is None:
            try:
                import tiktoken

                _encoding = tiktoken.get_encoding("o200k_base")
            except Exception:  # noqa: BLE001 — offline/vocab-download failure must not break callers
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
