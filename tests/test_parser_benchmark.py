"""Tests for parser benchmark harness.

TDD: tests written before implementation.
"""

from __future__ import annotations


class TestParserBenchmark:
    """Test parser benchmark utilities."""

    def test_module_imports(self):
        from drbrain.services import parser_benchmark

        assert parser_benchmark is not None

    def test_benchmark_result_dataclass(self):
        from drbrain.services.parser_benchmark import BenchmarkResult

        r = BenchmarkResult(
            parser="mineru",
            pdf_path="/tmp/test.pdf",
            elapsed_sec=2.5,
            output_size_bytes=5000,
            success=True,
        )
        d = r.to_dict()
        assert d["parser"] == "mineru"
        assert d["elapsed_sec"] == 2.5
        assert d["success"] is True

    def test_benchmark_result_failure(self):
        from drbrain.services.parser_benchmark import BenchmarkResult

        r = BenchmarkResult(
            parser="pymupdf",
            pdf_path="/tmp/test.pdf",
            elapsed_sec=0.0,
            success=False,
            error="PDF not found",
        )
        d = r.to_dict()
        assert d["success"] is False
        assert d["error"] == "PDF not found"

    def test_run_benchmark_with_missing_parser(self, tmp_path):
        from drbrain.services.parser_benchmark import (
            BenchmarkResult,
            run_single_benchmark,
        )

        pdf = tmp_path / "test.pdf"
        pdf.write_text("fake pdf content")

        result = run_single_benchmark("nonexistent-parser", pdf)
        assert isinstance(result, BenchmarkResult)
        assert result.parser == "nonexistent-parser"
        # Unknown parser should report failure
        assert result.success is False

    def test_print_results_table(self):
        from drbrain.services.parser_benchmark import (
            BenchmarkResult,
            format_results_table,
        )

        results = [
            BenchmarkResult("mineru", "/a.pdf", 2.5, 5000, True),
            BenchmarkResult("pymupdf", "/a.pdf", 0.3, 3000, True),
            BenchmarkResult("docling", "/a.pdf", 4.1, 8000, True),
        ]
        table = format_results_table(results)
        assert "mineru" in table
        assert "pymupdf" in table
        assert "docling" in table
        assert "2.50s" in table
