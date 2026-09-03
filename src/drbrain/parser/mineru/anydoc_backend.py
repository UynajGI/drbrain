"""anydoc (firecrawl-anydoc) + OCRmyPDF fallback backends for PDF parsing.

Fallback chain when the mineru-open-api CLI is unavailable:

    anydoc (text-based PDFs)
        -> OCRmyPDF text layer + anydoc again (scanned PDFs, opt-in)
        -> caller drops to pymupdf4llm / plain text

All heavy dependencies (anydoc, ocrmypdf) are imported lazily so the core
pipeline keeps working when they are not installed.
"""

from __future__ import annotations

import importlib
from enum import Enum
from pathlib import Path

from loguru import logger as _log


class AnydocStatus(Enum):
    """Result classification for anydoc_to_markdown."""

    OK = "ok"
    UNSUPPORTED = "unsupported"  # scanned / image-only PDF
    ERROR = "error"  # Malformed / Encrypted / ResourceLimit / Io / ...
    NOT_INSTALLED = "not_installed"


def anydoc_to_markdown(pdf_path: str | Path) -> tuple[AnydocStatus, str]:
    """Convert a PDF to Markdown via anydoc (pip install firecrawl-anydoc).

    Failures are classified by exception class name / error code rather than
    exception hierarchy, so an anydoc upgrade cannot easily break the chain.
    (Real anydoc raises ``UnsupportedError`` etc. — subclasses of ConvertError.)
    """
    try:
        anydoc = importlib.import_module("anydoc")
    except ImportError as e:
        _log.debug("anydoc not installed: {}", e)
        return AnydocStatus.NOT_INSTALLED, ""

    try:
        md = anydoc.to_markdown(str(pdf_path))
    except Exception as e:
        name = type(e).__name__
        code = str(getattr(e, "code", "") or "").lower()
        _log.debug("anydoc failed for {} ({}): {}", Path(pdf_path).name, name, e)
        if "unsupported" in name.lower() or "unsupported" in code:
            return AnydocStatus.UNSUPPORTED, ""
        return AnydocStatus.ERROR, ""

    if md and md.strip():
        return AnydocStatus.OK, md
    return AnydocStatus.ERROR, ""


def ocr_pdf(src: str | Path, dst: str | Path, language: str = "eng", force: bool = False) -> bool:
    """Add a text layer to a PDF via OCRmyPDF (requires system tesseract).

    First attempt uses skip_text (only OCR pages without a text layer); if it
    raises (e.g. PriorOcrFoundError or missing tesseract binaries are caught
    per attempt) a single retry with force_ocr follows. Returns False when
    ocrmypdf is not installed or OCR ultimately fails.
    """
    try:
        ocrmypdf = importlib.import_module("ocrmypdf")
    except ImportError as e:
        _log.debug("ocrmypdf not installed: {}", e)
        return False

    modes: list[dict] = (
        [{"force_ocr": True}] if force else [{"skip_text": True}, {"force_ocr": True}]
    )
    for attempt, kwargs in enumerate(modes):
        try:
            ocrmypdf.ocr(str(src), str(dst), language=language, **kwargs)
            if Path(dst).exists():
                return True
        except Exception as e:
            _log.debug("ocrmypdf attempt {} failed for {}: {}", attempt + 1, Path(src).name, e)
    return False
