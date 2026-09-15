"""Exact PDF page alignment for canonical text (plan T22).

The parsers return one markdown body without page offsets, so a page number
is only recorded when it can be *verified* against the canonical text: each
real PDF page's leading text is located in order, and alignment succeeds only
when every page matches monotonically.  Anything else yields no page marks —
blocks then carry no page fields rather than a guessed range.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from loguru import logger

_WS_RE = re.compile(r"\s+")
#: How many normalized characters identify one page start.
_SIGNATURE_CHARS = 48


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs; keep a normalized-index -> source-index map."""
    out: list[str] = []
    index_map: list[int] = []
    position = 0
    length = len(text)
    while position < length:
        char = text[position]
        if char.isspace():
            start = position
            while position < length and text[position].isspace():
                position += 1
            out.append(" ")
            index_map.append(start)
        else:
            out.append(char)
            index_map.append(position)
            position += 1
    index_map.append(length)
    return "".join(out), index_map


def pdf_page_texts(pdf_path: str | Path) -> list[str]:
    """Real per-page text via PyMuPDF (empty list when unavailable)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:  # pragma: no cover - PyMuPDF is an ingest dependency
        logger.warning("[align] PyMuPDF unavailable; page marks will be omitted")
        return []
    pages: list[str] = []
    try:
        with fitz.open(str(pdf_path)) as document:
            for page in document:
                pages.append(page.get_text("text"))
    except Exception as exc:  # noqa: BLE001 - an unreadable PDF gets no marks
        logger.warning("[align] cannot read pages of {}: {}", pdf_path, exc)
        return []
    return pages


def align_page_marks(
    pdf_path: str | Path,
    canonical_text: str,
    *,
    pages: Sequence[str] | None = None,
) -> list[tuple[int, int]] | None:
    """Return ``[(page_number, start_char), ...]`` or ``None`` when unverifiable.

    Every page must match, in order, using its normalized leading text; the
    first page always starts at char 0.
    """
    page_texts = list(pages) if pages is not None else pdf_page_texts(pdf_path)
    if not page_texts:
        return None
    normalized, index_map = normalize_with_map(canonical_text)
    marks: list[tuple[int, int]] = []
    cursor = 0
    for page_number, page_text in enumerate(page_texts, start=1):
        signature = _WS_RE.sub(" ", page_text or "").strip()
        if not signature:
            # A blank page has no anchor text; it cannot be located reliably.
            return None
        if page_number == 1:
            head = signature[:_SIGNATURE_CHARS]
            if not normalized.startswith(head):
                # Some parsers drop cover-page furniture; still require the
                # signature to appear early rather than pretending precision.
                found = normalized.find(head, 0, max(1, len(head) * 4))
                if found < 0:
                    return None
                cursor = found
            marks.append((1, 0))
            cursor = max(cursor, normalized.find(signature[:_SIGNATURE_CHARS], cursor))
            continue
        probe = signature[:_SIGNATURE_CHARS]
        found = normalized.find(probe, cursor)
        if found < 0:
            return None
        marks.append((page_number, index_map[found]))
        cursor = found
    if marks and marks[0] != (1, 0):
        return None
    starts = [offset for _page, offset in marks]
    if starts != sorted(starts) or len(set(starts)) != len(starts):
        return None
    if marks[-1][1] >= len(canonical_text):
        return None
    return marks


def page_span_for_offset(marks: Sequence[tuple[int, int]], offset: int) -> int | None:
    """Which 1-based page contains ``offset`` (marks must start at page 1)."""
    if not marks:
        return None
    page = None
    for page_number, start in marks:
        if offset >= start:
            page = page_number
        else:
            break
    return page
