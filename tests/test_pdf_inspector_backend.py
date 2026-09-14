"""A/B contract checks for the optional CPU pdf-inspector backend.

The real-corpus test is opt-in because the physics corpus is local and large:
``DRBRAIN_PHYSICS_PDF_DIR=/path/to/physics/data/arxiv-pdf pytest -m integration``.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from drbrain.parser.pdf_inspector_backend import extract_pdf_inspector

PDF_ROOT = Path(
    os.environ.get(
        "DRBRAIN_PHYSICS_PDF_DIR",
        "/home/jiangyuan/drbrain-phy/physics/data/arxiv-pdf",
    )
)


def _sample_pdfs() -> list[Path]:
    if not PDF_ROOT.is_dir():
        return []
    return sorted(PDF_ROOT.glob("*.pdf"))[:24]


@pytest.mark.integration
def test_pdf_inspector_real_text_pdf_contract():
    """At least one local arXiv PDF yields structured Markdown and provenance."""
    samples = _sample_pdfs()
    if not samples:
        pytest.skip("physics PDF corpus is not available")
    result = next((extract_pdf_inspector(p) for p in samples), None)
    if result is None:
        pytest.skip("sample set contains no text-based PDF")
    assert result["backend"] == "pdf_inspector"
    assert result["markdown"].strip()
    assert result["provenance"]["path"]
    assert isinstance(result["pages_needing_ocr"], list)


@pytest.mark.integration
def test_pdf_inspector_ab_preserves_content_against_pymupdf():
    """The CPU backend must not regress to an empty extraction on real papers."""
    fitz = pytest.importorskip("fitz")
    samples = _sample_pdfs()
    if not samples:
        pytest.skip("physics PDF corpus is not available")
    for path in samples[:3]:
        inspected = extract_pdf_inspector(path)
        if inspected is None:
            continue
        with fitz.open(path) as doc:
            baseline = "\n".join(page.get_text("text") for page in doc)
        assert len(inspected["markdown"]) >= max(32, int(len(baseline) * 0.05))
        return
    pytest.skip("sample set contains no text-based PDF")


@pytest.mark.integration
@pytest.mark.parametrize("pdf_kind", ["two_column_formula", "scanned_or_mixed"])
def test_pdf_inspector_layout_classes_are_explicit(pdf_kind: str):
    """Layout-sensitive classes are recorded, or skipped when absent locally."""
    samples = _sample_pdfs()
    if not samples:
        pytest.skip("physics PDF corpus is not available")
    for path in samples:
        result = extract_pdf_inspector(path)
        if result is None:
            continue
        kind = result["pdf_type"].lower()
        if pdf_kind == "scanned_or_mixed" and kind in {"scanned", "mixed", "imagebased"}:
            assert isinstance(result["pages_needing_ocr"], list)
            return
        if pdf_kind == "two_column_formula" and kind == "textbased":
            assert result["markdown"].strip()
            return
    pytest.skip(f"no {pdf_kind} sample found in {PDF_ROOT}")
