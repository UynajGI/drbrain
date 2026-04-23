"""Tests for mineru-open-api CLI retry and skip logic in parser."""
import unittest.mock
import subprocess
import tempfile
from pathlib import Path

from brbrain.parser.mineru_parser import (
    MinerUParser, filter_sections, _extract_arxiv_from_filename,
    normalize_arxiv, normalize_doi, _find_cli,
)


def _mock_mineru_run(captured_cmd=None, fail_count=0, md_content="# Title\n\nIntroduction.—Content here.\n\nConclusion.—Done."):
    """Helper to mock mineru-open-api -o behavior."""
    call_count = 0

    def mock_run(cmd, **kwargs):
        nonlocal call_count
        call_count += 1
        if captured_cmd is not None:
            captured_cmd.append(list(cmd))
        if call_count <= fail_count:
            result = unittest.mock.Mock()
            result.returncode = 1
            result.stdout = ""
            result.stderr = "error"
            return result
        # Simulate -o output: create temp dir with images
        out_dir = Path(cmd[cmd.index("-o") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "paper.md").write_text(md_content)
        img_dir = out_dir / "images"
        img_dir.mkdir(exist_ok=True)
        (img_dir / "test.jpg").write_bytes(b"fake")
        result = unittest.mock.Mock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    return mock_run, lambda: call_count


def test_parser_retries_mineru_on_failure():
    """MinerUParser retries mineru-open-api up to max_retries times."""
    mock_run, get_count = _mock_mineru_run(fail_count=2)

    with unittest.mock.patch("subprocess.run", side_effect=mock_run), \
         unittest.mock.patch("brbrain.parser.mineru_parser._find_cli", return_value="mineru-open-api"):
        parser = MinerUParser(token="test", max_retries=3, retry_delay=0.01)
        result = parser.extract("/tmp/test.pdf")
        assert get_count() == 3


def test_parser_skips_to_fallback_after_max_retries():
    """MinerUParser falls back to pypdfium2 after max_retries failures."""
    def mock_run(*args, **kwargs):
        result = unittest.mock.Mock()
        result.returncode = 1
        result.stdout = ""
        result.stderr = "error"
        return result

    def mock_pdfium(path):
        return "No headings, just text."

    with unittest.mock.patch("subprocess.run", side_effect=mock_run), \
         unittest.mock.patch("brbrain.parser.mineru_parser._find_cli", return_value="mineru-open-api"), \
         unittest.mock.patch.object(MinerUParser, "_fallback_pypdfium2", side_effect=mock_pdfium):
        parser = MinerUParser(token="test", max_retries=2, retry_delay=0.01)
        result = parser.extract("/tmp/test.pdf")
        assert result.raw_md == "No headings, just text."


def test_parser_succeeds_on_first_try():
    """MinerUParser succeeds without retry when mineru-open-api works."""
    mock_run, _ = _mock_mineru_run()

    with unittest.mock.patch("subprocess.run", side_effect=mock_run), \
         unittest.mock.patch("brbrain.parser.mineru_parser._find_cli", return_value="mineru-open-api"):
        parser = MinerUParser(token="test", max_retries=3, retry_delay=0.01)
        result = parser.extract("/tmp/test.pdf")
        assert result.title == "Title"


def test_mineru_open_api_uses_extract_with_token():
    """When token is provided, mineru-open-api uses 'extract' subcommand."""
    captured = []
    mock_run, _ = _mock_mineru_run(captured_cmd=captured)

    with unittest.mock.patch("subprocess.run", side_effect=mock_run), \
         unittest.mock.patch("brbrain.parser.mineru_parser._find_cli", return_value="mineru-open-api"):
        parser = MinerUParser(token="abc123", model="vlm", is_ocr=True)
        parser.extract("/tmp/test.pdf")
        cmd = captured[0]
        assert cmd[1] == "extract"
        assert "--token" in cmd
        assert "abc123" in cmd
        assert "--ocr" in cmd


def test_mineru_open_api_uses_extract_without_token():
    """When no token, mineru-open-api uses 'extract' with -o (not flash-extract)."""
    captured = []
    mock_run, _ = _mock_mineru_run(captured_cmd=captured)

    with unittest.mock.patch("subprocess.run", side_effect=mock_run), \
         unittest.mock.patch("brbrain.parser.mineru_parser._find_cli", return_value="mineru-open-api"):
        parser = MinerUParser()
        parser.extract("/tmp/test.pdf")
        cmd = captured[0]
        assert cmd[1] == "extract"
        assert "-o" in cmd


def test_parser_strips_thinking_header():
    """MinerUParser output from -o doesn't have 'Thinking...' since it's file-based."""
    mock_run, _ = _mock_mineru_run(md_content="# Title\n\nContent.")

    with unittest.mock.patch("subprocess.run", side_effect=mock_run), \
         unittest.mock.patch("brbrain.parser.mineru_parser._find_cli", return_value="mineru-open-api"):
        parser = MinerUParser()
        result = parser.extract("/tmp/test.pdf")
        assert not result.raw_md.startswith("Thinking...")
        assert "# Title" in result.raw_md
