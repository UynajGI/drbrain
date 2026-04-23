"""MinerU PDF parser via mineru-open-api CLI + pypdfium2 fallback."""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

MAX_CHARS = 12_000

# Markdown heading sections
HEADING_SECTIONS = re.compile(
    r"^(abstract|introduction|related\s*work|method(ology)?|"
    r"conclusion|limitations?|future\s*work|discussion|results|"
    r"supplementary\s*material)$",
    re.IGNORECASE,
)

# Inline section markers: "Introduction.—", "Conclusion. —", etc.
INLINE_SECTION = re.compile(
    r"^(Abstract|Introduction|Related\s*(Work|Work).{0,5}|Methods?|"
    r"Methodology|Conclusion|Limitations?|Future\s*Work|Discussion|"
    r"Results|GME and reflected entropy|A new measure)"
    r"[.\s:—–-]",
    re.IGNORECASE,
)

ID_PATTERN = re.compile(r"(10\.\d{4,}/[\S]+|arxiv[:\s]+(\d{4}\.\d{4,5}))", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
ARXIV_FILENAME = re.compile(r"(\d{4}\.\d{4,5})v\d*\.pdf$", re.IGNORECASE)

log = logging.getLogger(__name__)


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
    images_dir: Path | None = None  # extracted images directory


class MinerUParser:
    """PDF parser: mineru-open-api CLI (flash-extract/extract) -> pypdfium2 fallback."""

    def __init__(self, token: str = "", model: str = "vlm",
                 is_ocr: bool = False, enable_formula: bool = True,
                 enable_table: bool = True, max_retries: int = 3,
                 retry_delay: float = 2.0):
        self.token = token
        self.model = model
        self.is_ocr = is_ocr
        self.enable_formula = enable_formula
        self.enable_table = enable_table
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def extract(self, pdf_path: str | Path) -> ParsedPaper:
        """Extract structured content from PDF."""
        pdf_path = Path(pdf_path)

        # Extract arXiv ID from filename first
        arxiv_from_name = _extract_arxiv_from_filename(pdf_path)

        # Try mineru-open-api CLI with -o to get images
        out_dir = self._try_mineru_open_api(pdf_path)
        if out_dir is not None:
            raw_md = self._read_output_md(out_dir)
        else:
            raw_md = self._fallback_pypdfium2(pdf_path)
            out_dir = None

        title = self._extract_title(raw_md, str(pdf_path))
        year = self._extract_year(raw_md)
        doi, arxiv = self._extract_ids(raw_md)

        if not arxiv and arxiv_from_name:
            arxiv = arxiv_from_name

        # Enrich metadata from arXiv API
        if arxiv:
            api_title, api_year = _fetch_arxiv_metadata(arxiv)
            if api_title:
                title = api_title
            if api_year and not year:
                year = api_year

        blocks = filter_sections(raw_md)

        images_dir = out_dir / "images" if out_dir and (out_dir / "images").exists() else None

        return ParsedPaper(
            title=title, year=year, doi=doi, arxiv=arxiv,
            text_blocks=blocks, raw_md=raw_md,
            images_dir=images_dir,
        )

    def _try_mineru_open_api(self, pdf_path: Path) -> Path | None:
        """Invoke mineru-open-api CLI with retry. Returns output dir or None on failure."""
        cli = _find_cli()
        if cli is None:
            return None

        # Use extract if token available, else flash-extract
        if self.token:
            out_dir = Path(tempfile.mkdtemp(prefix="mineru_")) / "out"
            cmd = [cli, "extract", str(pdf_path), "--model", self.model, "-o", str(out_dir)]
            if self.is_ocr:
                cmd.append("--ocr")
            if not self.enable_formula:
                cmd.append("--formula=false")
            if not self.enable_table:
                cmd.append("--table=false")
            cmd.extend(["--token", self.token])
        else:
            out_dir = Path(tempfile.mkdtemp(prefix="mineru_")) / "out"
            cmd = [cli, "extract", str(pdf_path), "-o", str(out_dir)]

        for attempt in range(self.max_retries):
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=600,
                )
                if result.returncode != 0:
                    log.warning("mineru-open-api failed (attempt %d/%d): %s",
                                attempt + 1, self.max_retries, result.stderr[:500])
                    time.sleep(self.retry_delay)
                    continue
                if not (out_dir / "images").exists():
                    time.sleep(self.retry_delay)
                    continue
                return out_dir
            except subprocess.TimeoutExpired:
                log.warning("mineru-open-api timeout (attempt %d/%d)",
                            attempt + 1, self.max_retries)
                time.sleep(self.retry_delay)
            except (FileNotFoundError, OSError) as e:
                log.warning("mineru-open-api error: %s", e)
                return None
        return None

    def _read_output_md(self, out_dir: Path) -> str:
        """Read the generated markdown from the output directory."""
        md_files = list(out_dir.glob("*.md"))
        if not md_files:
            return ""
        return md_files[0].read_text(encoding="utf-8")

    def _fallback_pypdfium2(self, pdf_path: Path) -> str:
        """Extract text via pypdfium2 as fallback."""
        import pypdfium2 as pdfium

        try:
            doc = pdfium.PdfDocument(str(pdf_path))
            lines = []
            for page in doc:
                text_page = page.get_textpage()
                text = text_page.get_text_bounded()
                if text.strip():
                    lines.append(text)
            return "\n\n".join(lines)
        finally:
            doc.close()

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
                if 1900 <= year <= 2999:
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


def _find_cli() -> str | None:
    """Find mineru-open-api CLI binary."""
    import shutil
    return shutil.which("mineru-open-api")


def _extract_arxiv_from_filename(pdf_path: Path) -> str | None:
    """Extract arXiv ID from filename like '2602.00617v1.pdf'."""
    m = ARXIV_FILENAME.search(pdf_path.name)
    if m:
        return m.group(1)
    return None


def _fetch_arxiv_metadata(arxiv_id: str) -> tuple[str | None, int | None]:
    """Fetch title and year from arXiv API."""
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "DrBrain/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            xml = resp.read().decode("utf-8")

        title_match = re.search(r"<title>(.+?)</title>", xml, re.DOTALL)
        published_match = re.search(r"<published>(\d{4})-", xml)

        title = None
        year = None

        if title_match:
            all_titles = re.findall(r"<title>(.+?)</title>", xml, re.DOTALL)
            if len(all_titles) > 1:
                title = all_titles[1].strip()
            else:
                title = title_match.group(1).strip()

        if published_match:
            year = int(published_match.group(1))

        return title, year
    except Exception:
        return None, None


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
    """Extract target academic sections from mineru output.

    Handles both markdown headings (# Introduction) and inline markers (Introduction.—).
    Falls back to returning all text if no sections detected.
    """
    lines = raw_md.splitlines()
    blocks: list[str] = []
    current_section = ""
    current_body: list[str] = []
    found_any_target = False

    for line in lines:
        # Check for markdown heading sections
        if line.startswith("#"):
            heading_text = line.lstrip("# ").strip()
            if current_body and HEADING_SECTIONS.match(current_section):
                joined = "\n".join(current_body)
                if joined.strip():
                    blocks.append(joined[:MAX_CHARS])
                    found_any_target = True
            current_section = heading_text
            current_body = []
        else:
            # Check for inline section marker at start of a line/paragraph
            if INLINE_SECTION.match(line.strip()):
                if current_body and HEADING_SECTIONS.match(current_section):
                    joined = "\n".join(current_body)
                    if joined.strip():
                        blocks.append(joined[:MAX_CHARS])
                        found_any_target = True
                # Extract section name from inline marker
                m = INLINE_SECTION.match(line.strip())
                if m:
                    current_section = m.group(1)
                    # The rest of the line after the marker is content
                    rest = line.strip()[m.end():]
                    current_body = [rest] if rest.strip() else []
                else:
                    current_body.append(line)
            else:
                current_body.append(line)

    # Handle last section
    if current_body and HEADING_SECTIONS.match(current_section):
        joined = "\n".join(current_body)
        if joined.strip():
            blocks.append(joined[:MAX_CHARS])
            found_any_target = True

    # If no target sections found, return all text (excluding title line and thinking line)
    if not found_any_target:
        filtered_lines = [l for l in lines if not l.startswith("Thinking...")]
        text = "\n".join(filtered_lines).strip()
        if text:
            # Split into chunks if too large
            if len(text) > MAX_CHARS:
                return [text[:MAX_CHARS]]
            return [text]
        return []

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
