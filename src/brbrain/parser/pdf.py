"""PDF parser module."""

import re
from dataclasses import dataclass
from pathlib import Path

# High-signal sections to extract
TARGET_SECTIONS = re.compile(
    r"^(abstract|introduction|related\s*work|conclusion|limitations?|future\s*work|discussion)",
    re.IGNORECASE,
)

MAX_CHARS = 12_000


@dataclass
class ParsedPaper:
    """Result of parsing a PDF."""

    title: str
    year: int | None
    doi: str | None
    arxiv: str | None
    text_blocks: list[str]
    raw_md: str


def extract_pdf(pdf_path: str | Path) -> ParsedPaper:
    """Parse PDF with MinerU, filter to high-signal sections.

    TODO: integrate mineru CLI when environment is ready.
    For now, returns placeholder from plain text/markdown files.
    """
    path = Path(pdf_path)
    if path.suffix == ".md":
        raw = path.read_text(encoding="utf-8")
        return ParsedPaper(
            title=path.stem,
            year=None,
            doi=None,
            arxiv=None,
            text_blocks=[raw[:MAX_CHARS]],
            raw_md=raw,
        )
    raise NotImplementedError(f"PDF parsing for {path.suffix} requires MinerU")


def filter_sections(raw_md: str) -> list[str]:
    """Split markdown by headings, keep only target sections."""
    blocks: list[str] = []
    current_section = ""
    current_body: list[str] = []

    for line in raw_md.splitlines():
        if line.startswith("#"):
            if current_body and TARGET_SECTIONS.match(current_section):
                blocks.append("\n".join(current_body))
            current_section = line.lstrip("# ").strip()
            current_body = []
        else:
            current_body.append(line)

    if current_body and TARGET_SECTIONS.match(current_section):
        blocks.append("\n".join(current_body))

    return [b[:MAX_CHARS] for b in blocks if b.strip()]
