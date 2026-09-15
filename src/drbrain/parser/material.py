"""Lightweight adapters for non-PDF source material.

PDFs keep the MinerU fallback chain.  Plain text, Markdown and LaTeX are
already machine-readable, so sending them through a PDF parser only adds
failure points and loses source fidelity.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from loguru import logger as _material_log

from drbrain.parser.mineru.parser import ParsedPaper, extract_pdf

TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt", ".tex", ".text"})

_DRBRAIN_HEADER_RE = re.compile(
    r"^%\s*(?:DrBrain\s+)?(?P<name>source_id|Title|DOI|Categories)\s*:\s*(?P<value>.+?)\s*$",
    re.MULTILINE,
)
_BANNER_RE = re.compile(r"^={10,}\s*$")
_OLD_STYLE_ARXIV_RE = re.compile(
    r"^[a-z][a-z-]*(?:\.[A-Za-z]+)?/(\d{2})(0[1-9]|1[0-2])(\d{3})(?:v\d+)?$"
)
_NEW_STYLE_ARXIV_RE = re.compile(r"^(\d{2})(0[1-9]|1[0-2])\.\d{4,5}(?:v\d+)?$")


def _parse_drbrain_header(raw_md: str) -> dict[str, str]:
    """Parse the ``% DrBrain ...`` wrapper header written by corpus extractors."""
    fields: dict[str, str] = {}
    for match in _DRBRAIN_HEADER_RE.finditer(raw_md[:4000]):
        fields.setdefault(match.group("name").lower(), match.group("value").strip())
    return fields


def _strip_drbrain_wrapper(raw_md: str) -> str:
    """Return the LaTeX body below the ``====`` banner wrapper."""
    lines = raw_md.splitlines()
    banners = [i for i, line in enumerate(lines[:40]) if _BANNER_RE.match(line)]
    if len(banners) >= 2:
        return "\n".join(lines[banners[1] + 1 :])
    # No wrapper: drop the leading comment/blank preamble of a bare .tex file.
    start = 0
    while start < len(lines) and (
        not lines[start].strip() or lines[start].lstrip().startswith("%")
    ):
        start += 1
    return "\n".join(lines[start:])


def _year_from_arxiv_like(arxiv_id: str) -> int | None:
    """Derive the submission year encoded in an arXiv id (old- or new-style)."""
    match = _OLD_STYLE_ARXIV_RE.match(arxiv_id) or _NEW_STYLE_ARXIV_RE.match(arxiv_id)
    if not match:
        return None
    yy = int(match.group(1))
    return 1900 + yy if yy >= 91 else 2000 + yy


def _extract_latex_material(source: Path, raw_md: str) -> ParsedPaper:
    """Convert wrapped arXiv LaTeX source into pipeline-ready material.

    A raw LaTeX body has no markdown headings, so without the conversion pass
    the PageIndex tree collapses to a single node and the ``% DrBrain`` wrapper
    comment becomes the title.  Metadata comes from the wrapper header fields
    (``% Title`` / ``% DOI`` / ``% DrBrain source_id``); the body is converted
    with :mod:`drbrain.parser.latex_md` so section headings, math atoms and
    ``[CITE:...]`` markers survive into ``raw.md``.
    """
    from drbrain.dedup.resolver import normalize_arxiv, normalize_doi
    from drbrain.parser.latex_md import latex_to_document

    fields = _parse_drbrain_header(raw_md)
    body = _strip_drbrain_wrapper(raw_md)
    try:
        markdown = latex_to_document(body).markdown
    except Exception as exc:  # noqa: BLE001 — one bad source must not stop a corpus run
        _material_log.warning("latex conversion failed for {}: {}", source.name, exc)
        markdown = body
    title = fields.get("title", "")
    if not title:
        heading = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
        title = heading.group(1).strip() if heading else source.stem
    doi = normalize_doi(fields["doi"]) or None if fields.get("doi") else None
    arxiv = normalize_arxiv(fields["source_id"]) if fields.get("source_id") else None
    year = _year_from_arxiv_like(arxiv) if arxiv else None
    blocks = [block.strip() for block in re.split(r"\n(?=#+\s)", markdown) if block.strip()]
    file_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    candidates = {"doi": doi, "arxiv": arxiv, "file_sha256": file_hash}
    return ParsedPaper(
        title=title,
        year=year,
        doi=doi,
        arxiv=arxiv,
        text_blocks=blocks or ([markdown] if markdown.strip() else []),
        raw_md=markdown,
        backend="latex-native",
        provenance={"path": str(source.resolve()), "backend": "latex-native"},
        identifiers={k: v for k, v in candidates.items() if v},
        identifier_candidates=[
            {"kind": kind, "value": value, "source": "header"}
            for kind, value in candidates.items()
            if value
        ],
        source_path=str(source.resolve()),
    )


def extract_material(path: str | Path, config: dict) -> ParsedPaper:
    """Extract a supported material, falling back to the PDF parser."""
    source = Path(path)
    if source.suffix.lower() not in TEXT_SUFFIXES:
        return extract_pdf(source, config)
    raw_md = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".tex":
        return _extract_latex_material(source, raw_md)
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
    isbn_match = re.search(
        r"\b(?:ISBN(?:-1[03])?:?\s*)?((?:97[89][ -]?)?\d[\d -]{9,16}\d)\b", raw_md, re.IGNORECASE
    )
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
            {"kind": k, "value": v, "source": "content"} for k, v in candidates.items() if v
        ],
        source_path=str(source.resolve()),
    )


__all__ = ["TEXT_SUFFIXES", "extract_material"]
