"""Tests for the anydoc + OCRmyPDF fallback chain in MinerUParser."""

import unittest.mock
from pathlib import Path

from drbrain.parser.mineru.anydoc_backend import (
    AnydocStatus,
    anydoc_to_markdown,
    ocr_pdf,
)
from drbrain.parser.mineru_parser import MinerUParser, extract_pdf


def _parser(**kwargs) -> MinerUParser:
    defaults = {"max_retries": 1, "retry_delay": 0.01}
    defaults.update(kwargs)
    return MinerUParser(**defaults)


def test_fallback_chain_uses_anydoc_when_available():
    """anydoc success short-circuits the chain; OCR and pymupdf never run."""
    parser = _parser()
    with (
        unittest.mock.patch(
            "drbrain.parser.mineru.parser.anydoc_to_markdown",
            return_value=(AnydocStatus.OK, "# Title\n\nBody"),
        ) as mock_anydoc,
        unittest.mock.patch("drbrain.parser.mineru.parser.ocr_pdf") as mock_ocr,
        unittest.mock.patch.object(MinerUParser, "_fallback_pymupdf") as mock_pymupdf,
    ):
        md = parser._fallback_chain(Path("/tmp/x.pdf"))
    assert md == "# Title\n\nBody"
    mock_anydoc.assert_called_once()
    mock_ocr.assert_not_called()
    mock_pymupdf.assert_not_called()


def test_fallback_chain_ocrs_scanned_pdf():
    """Unsupported + ocr_enabled triggers OCRmyPDF, then anydoc on the OCR'd PDF."""
    parser = _parser(ocr_enabled=True)
    results = [
        (AnydocStatus.UNSUPPORTED, ""),
        (AnydocStatus.OK, "# Scanned\n\nOCR text"),
    ]
    with (
        unittest.mock.patch(
            "drbrain.parser.mineru.parser.anydoc_to_markdown", side_effect=results
        ) as mock_anydoc,
        unittest.mock.patch("drbrain.parser.mineru.parser.ocr_pdf", return_value=True) as mock_ocr,
        unittest.mock.patch.object(MinerUParser, "_fallback_pymupdf") as mock_pymupdf,
    ):
        md = parser._fallback_chain(Path("/tmp/scan.pdf"))
    assert md == "# Scanned\n\nOCR text"
    assert mock_anydoc.call_count == 2
    mock_ocr.assert_called_once()
    mock_pymupdf.assert_not_called()


def test_fallback_chain_skips_ocr_when_disabled():
    """Unsupported + ocr_enabled=False drops straight to pymupdf."""
    parser = _parser(ocr_enabled=False)
    with (
        unittest.mock.patch(
            "drbrain.parser.mineru.parser.anydoc_to_markdown",
            return_value=(AnydocStatus.UNSUPPORTED, ""),
        ),
        unittest.mock.patch("drbrain.parser.mineru.parser.ocr_pdf") as mock_ocr,
        unittest.mock.patch.object(
            MinerUParser, "_fallback_pymupdf", return_value="plain text"
        ) as mock_pymupdf,
    ):
        md = parser._fallback_chain(Path("/tmp/scan.pdf"))
    assert md == "plain text"
    mock_ocr.assert_not_called()
    mock_pymupdf.assert_called_once()


def test_fallback_chain_without_anydoc_installed():
    """Missing anydoc package degrades to pymupdf without raising."""
    parser = _parser()
    with (
        unittest.mock.patch(
            "drbrain.parser.mineru.parser.anydoc_to_markdown",
            return_value=(AnydocStatus.NOT_INSTALLED, ""),
        ),
        unittest.mock.patch.object(
            MinerUParser, "_fallback_pymupdf", return_value="plain text"
        ) as mock_pymupdf,
    ):
        md = parser._fallback_chain(Path("/tmp/x.pdf"))
    assert md == "plain text"
    mock_pymupdf.assert_called_once()


def test_fallback_chain_ocr_failure_falls_to_pymupdf():
    """OCR failure (e.g. missing tesseract) degrades to pymupdf."""
    parser = _parser(ocr_enabled=True)
    with (
        unittest.mock.patch(
            "drbrain.parser.mineru.parser.anydoc_to_markdown",
            return_value=(AnydocStatus.UNSUPPORTED, ""),
        ),
        unittest.mock.patch("drbrain.parser.mineru.parser.ocr_pdf", return_value=False),
        unittest.mock.patch.object(
            MinerUParser, "_fallback_pymupdf", return_value="plain text"
        ) as mock_pymupdf,
    ):
        md = parser._fallback_chain(Path("/tmp/scan.pdf"))
    assert md == "plain text"
    mock_pymupdf.assert_called_once()


def test_fallback_chain_disabled_anydoc_uses_pymupdf():
    """use_anydoc=False bypasses anydoc entirely."""
    parser = _parser(use_anydoc=False)
    with (
        unittest.mock.patch("drbrain.parser.mineru.parser.anydoc_to_markdown") as mock_anydoc,
        unittest.mock.patch.object(
            MinerUParser, "_fallback_pymupdf", return_value="plain text"
        ) as mock_pymupdf,
    ):
        md = parser._fallback_chain(Path("/tmp/x.pdf"))
    assert md == "plain text"
    mock_anydoc.assert_not_called()
    mock_pymupdf.assert_called_once()


def test_anydoc_to_markdown_classifies_unsupported():
    """Exceptions named UnsupportedError map to AnydocStatus.UNSUPPORTED."""
    fake = unittest.mock.MagicMock()

    class UnsupportedError(Exception):
        pass

    fake.to_markdown.side_effect = UnsupportedError("image-only pdf")
    with unittest.mock.patch(
        "drbrain.parser.mineru.anydoc_backend.importlib.import_module",
        return_value=fake,
    ):
        status, md = anydoc_to_markdown(Path("/tmp/scan.pdf"))
    assert status is AnydocStatus.UNSUPPORTED
    assert md == ""


def test_anydoc_to_markdown_classifies_other_errors():
    """Other ConvertError subclasses map to AnydocStatus.ERROR."""
    fake = unittest.mock.MagicMock()

    class EncryptedError(Exception):
        pass

    fake.to_markdown.side_effect = EncryptedError("password protected")
    with unittest.mock.patch(
        "drbrain.parser.mineru.anydoc_backend.importlib.import_module",
        return_value=fake,
    ):
        status, md = anydoc_to_markdown(Path("/tmp/enc.pdf"))
    assert status is AnydocStatus.ERROR
    assert md == ""


def test_anydoc_to_markdown_not_installed():
    """ImportError maps to AnydocStatus.NOT_INSTALLED."""
    with unittest.mock.patch(
        "drbrain.parser.mineru.anydoc_backend.importlib.import_module",
        side_effect=ImportError("no anydoc"),
    ):
        status, md = anydoc_to_markdown(Path("/tmp/x.pdf"))
    assert status is AnydocStatus.NOT_INSTALLED
    assert md == ""


def test_ocr_pdf_retries_with_force_ocr(tmp_path):
    """First skip_text attempt fails, force_ocr retry succeeds."""
    fake = unittest.mock.MagicMock()
    calls = []

    def fake_ocr(src, dst, language="eng", **kwargs):
        calls.append(kwargs)
        if "skip_text" in kwargs:
            raise RuntimeError("prior ocr found")
        Path(dst).write_bytes(b"%PDF-1.4 fake")

    fake.ocr.side_effect = fake_ocr
    src = tmp_path / "in.pdf"
    src.write_bytes(b"fake")
    dst = tmp_path / "out.pdf"
    with unittest.mock.patch(
        "drbrain.parser.mineru.anydoc_backend.importlib.import_module",
        return_value=fake,
    ):
        ok = ocr_pdf(src, dst)
    assert ok is True
    assert calls == [{"skip_text": True}, {"force_ocr": True}]


def test_extract_pdf_passes_anydoc_config():
    """extract_pdf forwards use_anydoc / ocr_enabled / ocr_language from config."""
    cfg = {
        "mineru": {
            "use_anydoc": False,
            "ocr_enabled": True,
            "ocr_language": "eng+chi_sim",
        }
    }
    with unittest.mock.patch("drbrain.parser.mineru.parser.MinerUParser") as mock_cls:
        mock_cls.return_value.extract.return_value = unittest.mock.Mock()
        extract_pdf("/tmp/x.pdf", cfg)
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["use_anydoc"] is False
        assert kwargs["ocr_enabled"] is True
        assert kwargs["ocr_language"] == "eng+chi_sim"
