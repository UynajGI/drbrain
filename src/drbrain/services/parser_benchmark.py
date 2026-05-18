"""Parser benchmark harness — compare PDF parser outputs (MinerU, PyMuPDF, Docling)."""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Single parser benchmark result."""

    parser: str
    pdf_path: str
    elapsed_sec: float
    output_size_bytes: int = 0
    success: bool = True
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "parser": self.parser,
            "pdf_path": self.pdf_path,
            "elapsed_sec": round(self.elapsed_sec, 3),
            "output_size_bytes": self.output_size_bytes,
            "success": self.success,
            "error": self.error,
        }


PARSERS = {
    "pymupdf": "_run_pymupdf",
    "mineru": "_run_mineru",
    "pymupdf4llm": "_run_pymupdf4llm",
}


def run_single_benchmark(parser_name: str, pdf_path: Path) -> BenchmarkResult:
    """Benchmark a single parser on one PDF.

    Args:
        parser_name: Parser key (``"pymupdf"``, ``"mineru"``, ``"pymupdf4llm"``).
        pdf_path: Path to the PDF file.

    Returns:
        BenchmarkResult with timing and output size.
    """
    if parser_name not in PARSERS:
        return BenchmarkResult(
            parser=parser_name,
            pdf_path=str(pdf_path),
            elapsed_sec=0,
            success=False,
            error=f"Unknown parser: {parser_name}. Available: {', '.join(PARSERS)}",
        )

    try:
        t0 = _time.monotonic()

        if parser_name == "pymupdf":
            import fitz

            doc = fitz.open(str(pdf_path))
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            output_size = len(text)

        elif parser_name == "pymupdf4llm":
            import pymupdf4llm

            text = pymupdf4llm.to_markdown(str(pdf_path))
            output_size = len(text)

        elif parser_name == "mineru":
            from drbrain.parser.mineru_parser import extract_pdf

            text = extract_pdf(pdf_path)
            output_size = len(text) if text else 0

        else:
            raise ValueError(f"Unknown parser: {parser_name}")

        elapsed = _time.monotonic() - t0
        return BenchmarkResult(
            parser=parser_name,
            pdf_path=str(pdf_path),
            elapsed_sec=elapsed,
            output_size_bytes=output_size,
            success=True,
        )
    except Exception as exc:
        return BenchmarkResult(
            parser=parser_name,
            pdf_path=str(pdf_path),
            elapsed_sec=_time.monotonic() - t0 if "t0" in dir() else 0,
            success=False,
            error=str(exc),
        )


def run_benchmark(pdf_paths: list[Path], parsers: list[str] | None = None) -> list[BenchmarkResult]:
    """Run benchmark across multiple PDFs and parsers.

    Args:
        pdf_paths: List of PDF files to benchmark.
        parsers: Parser names to test (default: all registered).

    Returns:
        List of BenchmarkResult, one per (parser, pdf) combination.
    """
    names = parsers if parsers else list(PARSERS)
    results: list[BenchmarkResult] = []
    for pdf in pdf_paths:
        for name in names:
            results.append(run_single_benchmark(name, pdf))
    return results


def format_results_table(results: list[BenchmarkResult]) -> str:
    """Format benchmark results as a text table.

    Args:
        results: List of BenchmarkResult from ``run_benchmark``.

    Returns:
        Formatted table string.
    """
    lines = [
        f"{'Parser':<15} {'PDF':<40} {'Time':>8} {'Size':>10} {'Status':>8}",
        "-" * 85,
    ]
    for r in results:
        pdf_name = Path(r.pdf_path).name[:38]
        status = "OK" if r.success else f"FAIL: {r.error[:20]}"
        size_str = f"{r.output_size_bytes:,}" if r.output_size_bytes else "-"
        lines.append(
            f"{r.parser:<15} {pdf_name:<40} {r.elapsed_sec:>7.2f}s {size_str:>10} {status:>8}"
        )
    # Summary
    parsers_tested = sorted(set(r.parser for r in results))
    by_parser: dict[str, list[BenchmarkResult]] = {p: [] for p in parsers_tested}
    for r in results:
        by_parser[r.parser].append(r)
    lines.append("-" * 85)
    for p in parsers_tested:
        prs = by_parser[p]
        ok = sum(1 for r in prs if r.success)
        avg_time = sum(r.elapsed_sec for r in prs) / len(prs) if prs else 0
        lines.append(f"  {p}: {ok}/{len(prs)} success, avg {avg_time:.2f}s")
    return "\n".join(lines)
