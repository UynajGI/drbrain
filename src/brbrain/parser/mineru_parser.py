"""MinerU PDF parser with flash/token mode and chapter filtering."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

MAX_CHARS = 12_000

TARGET_SECTIONS = re.compile(
    r"^(abstract|introduction|related\s*work|method(ology)?|"
    r"conclusion|limitations?|future\s*work|discussion|results)",
    re.IGNORECASE,
)

ID_PATTERN = re.compile(r"(10\.\d{4,}/[\S]+|arxiv[:\s]+(\d{4}\.\d{4,5}))", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


@dataclass
class ParsedPaper:
    """Result of parsing a PDF."""

    title: str = ""
    year: int | None = None
    doi: str | None = None
    arxiv: str | None = None
    s2_id: str | None = None
    openalex_id: str | None = None
    text_blocks: list[str] = field(default_factory=list)
    raw_md: str = ""


class MinerUParser:
    """MinerU-based PDF parser with configurable mode."""

    def __init__(self, token: str = "", model: str = "vlm",
                 is_ocr: bool = False, enable_formula: bool = True,
                 enable_table: bool = True):
        self.token = token
        self.mode = "token" if token else "flash"
        self.model = model
        self.is_ocr = is_ocr
        self.enable_formula = enable_formula
        self.enable_table = enable_table

    def extract(self, pdf_path: str | Path) -> ParsedPaper:
        """Extract structured content from PDF via MinerU."""
        from mineru import MinerU

        if self.mode == "token":
            client = MinerU(token=self.token)
        else:
            client = MinerU()

        result = client.extract(
            pdf_path=str(pdf_path),
            model=self.model,
            is_ocr=self.is_ocr,
            enable_formula=self.enable_formula,
            enable_table=self.enable_table,
        )

        raw_md = self._extract_markdown(result)
        title = self._extract_title(raw_md, str(pdf_path))
        year = self._extract_year(raw_md)
        doi, arxiv = self._extract_ids(raw_md)
        blocks = filter_sections(raw_md)

        return ParsedPaper(
            title=title, year=year, doi=doi, arxiv=arxiv,
            text_blocks=blocks, raw_md=raw_md,
        )

    def _extract_markdown(self, result) -> str:
        """Extract markdown text from MinerU result."""
        if isinstance(result, dict):
            return result.get("markdown", result.get("content", ""))
        if isinstance(result, str):
            return result
        if hasattr(result, "markdown"):
            return result.markdown
        if hasattr(result, "content"):
            return result.content
        return str(result)

    def _extract_title(self, raw_md: str, fallback_path: str) -> str:
        """Extract title from first heading or filename."""
        for line in raw_md.splitlines()[:10]:
            if line.startswith("# "):
                return line[2:].strip()
        return Path(fallback_path).stem

    def _extract_year(self, raw_md: str) -> int | None:
        """Extract year from metadata or first paragraph."""
        for line in raw_md.splitlines()[:20]:
            m = YEAR_PATTERN.search(line)
            if m:
                year = int(m.group())
                if 1900 <= year <= 2030:
                    return year
        return None

    def _extract_ids(self, raw_md: str) -> tuple[str | None, str | None]:
        """Extract DOI and arXiv ID from raw text."""
        doi = None
        arxiv = None
        for line in raw_md.splitlines()[:30]:
            m = ID_PATTERN.search(line)
            if m:
                raw_id = m.group(1)
                if raw_id.startswith("10."):
                    doi = normalize_doi(raw_id)
                elif "arxiv" in raw_id.lower() or raw_id[:4].isdigit():
                    arxiv = normalize_arxiv(raw_id)
        return doi, arxiv


def normalize_doi(raw: str) -> str:
    """Strip URL prefix, lowercase."""
    doi = raw.strip().lower()
    doi = re.sub(r"^https?://doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi.strip()


def normalize_arxiv(raw: str) -> str:
    """Strip version suffix, standardize format."""
    raw = raw.strip()
    raw = re.sub(r"v\d+$", "", raw)
    m = re.search(r"(\d{4}\.\d{4,5})", raw)
    return m.group(1) if m else raw


def filter_sections(raw_md: str) -> list[str]:
    """Split markdown by headings, keep only target sections."""
    blocks: list[str] = []
    current_section = ""
    current_body: list[str] = []

    for line in raw_md.splitlines():
        if line.startswith("#"):
            if current_body and TARGET_SECTIONS.match(current_section):
                joined = "\n".join(current_body)
                if joined.strip():
                    blocks.append(joined[:MAX_CHARS])
            current_section = line.lstrip("# ").strip()
            current_body = []
        else:
            current_body.append(line)

    if current_body and TARGET_SECTIONS.match(current_section):
        joined = "\n".join(current_body)
        if joined.strip():
            blocks.append(joined[:MAX_CHARS])

    return blocks


def extract_pdf(pdf_path: str | Path, config: dict) -> ParsedPaper:
    """Convenience function: create parser from config and extract."""
    mineru_cfg = config.get("mineru", {})
    parser = MinerUParser(
        token=mineru_cfg.get("token", ""),
        model=mineru_cfg.get("model", "vlm"),
        is_ocr=mineru_cfg.get("is_ocr", False),
        enable_formula=mineru_cfg.get("enable_formula", True),
        enable_table=mineru_cfg.get("enable_table", True),
    )
    return parser.extract(pdf_path)
