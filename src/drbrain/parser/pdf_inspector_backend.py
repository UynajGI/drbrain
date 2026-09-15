"""Optional CPU-first PDF to Markdown backend using Firecrawl pdf-inspector."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def extract_pdf_inspector(path: str | Path) -> dict[str, Any] | None:
    """Extract a text PDF and return normalized Markdown/provenance metadata."""
    try:
        import pdf_inspector

        result = pdf_inspector.process_pdf(str(path))
        markdown = getattr(result, "markdown", None)
        if not markdown or not str(markdown).strip():
            return None
        return {
            "markdown": str(markdown),
            "backend": "pdf_inspector",
            "pdf_type": str(getattr(result, "pdf_type", "unknown")),
            "confidence": float(getattr(result, "confidence", 0.0) or 0.0),
            "pages_needing_ocr": list(getattr(result, "pages_needing_ocr", []) or []),
            "warnings": list(getattr(result, "warnings", []) or []),
            "provenance": {"path": str(Path(path).resolve()), "backend": "pdf_inspector"},
        }
    except Exception:
        return None
