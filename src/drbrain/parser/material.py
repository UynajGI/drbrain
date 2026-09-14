"""Lightweight adapters for non-PDF source material.

PDFs keep the MinerU fallback chain.  Plain text, Markdown and LaTeX are
already machine-readable, so sending them through a PDF parser only adds
failure points and loses source fidelity.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from drbrain.parser.mineru.parser import ParsedPaper, extract_pdf

TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt", ".tex", ".text"})


def extract_material(path: str | Path, config: dict) -> ParsedPaper:
    """Extract a supported material, falling back to the PDF parser."""
    source = Path(path)
    if source.suffix.lower() not in TEXT_SUFFIXES:
        return extract_pdf(source, config)
    raw_md = source.read_text(encoding="utf-8")
    title = ""
    for line in raw_md.splitlines()[:20]:
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            break
    if not title:
        title = next((line.strip() for line in raw_md.splitlines() if line.strip()), source.stem)
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", raw_md[:4000])
    doi_match = re.search(r"10\.\d{4,9}/[^\s<>]+", raw_md, re.IGNORECASE)
    arxiv_match = re.search(
        r"\b(?:arXiv:\s*|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})(?:v\d+)?\b",
        raw_md,
        re.IGNORECASE,
    )
    blocks = [block.strip() for block in re.split(r"\n(?=#+\s)", raw_md) if block.strip()]
    doi = doi_match.group(0).rstrip(".,;)]*") if doi_match else None
    arxiv = arxiv_match.group(1) if arxiv_match else None
    file_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    isbn_match = re.search(r"\b(?:ISBN(?:-1[03])?:?\s*)?((?:97[89][ -]?)?\d[\d -]{9,16}\d)\b", raw_md, re.IGNORECASE)
    isbn = re.sub(r"[^0-9Xx]", "", isbn_match.group(1)) if isbn_match else None
    url_match = re.search(r"https?://[^\s<>]+", raw_md)
    url = url_match.group(0).rstrip(".,;)]") if url_match else None
    candidates = {"doi": doi, "arxiv": arxiv, "isbn": isbn, "url": url, "file_sha256": file_hash}
    return ParsedPaper(
        title=title,
        year=int(year_match.group(1)) if year_match else None,
        doi=doi,
        arxiv=arxiv,
        text_blocks=blocks or ([raw_md] if raw_md.strip() else []),
        raw_md=raw_md,
        backend="text-native",
        provenance={"path": str(source.resolve()), "backend": "text-native"},
        identifiers={k: v for k, v in candidates.items() if v},
        identifier_candidates=[
            {"kind": k, "value": v, "source": "content"}
            for k, v in candidates.items()
            if v
        ],
        source_path=str(source.resolve()),
    )


__all__ = ["TEXT_SUFFIXES", "extract_material"]
