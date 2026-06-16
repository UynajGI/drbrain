"""MinerU PDF parser via mineru-open-api CLI + PyMuPDF fallback."""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from loguru import logger as _parse_log

from drbrain.parser.mineru.fallback import (
    ARXIV_FILENAME,
    ID_PATTERN,
    YEAR_PATTERN,
    filter_sections,
)
from drbrain.parser.mineru.metadata import _resolve_metadata


@dataclass
class PDFValidation:
    """Result of pre-validating a PDF before extraction."""

    ok: bool
    error: str = ""
    encrypted: bool = False
    page_count: int = 0


def _validate_pdf(pdf_path: Path) -> PDFValidation:
    """Pre-validate a PDF file before passing to extraction pipeline.

    Checks: file existence, page count, encryption status.
    """
    import fitz

    if not pdf_path.exists():
        return PDFValidation(ok=False, error=f"File not found: {pdf_path}")

    try:
        doc = fitz.open(str(pdf_path))
        try:
            pc = doc.page_count
            enc = doc.is_encrypted
            if pc == 0:
                return PDFValidation(ok=False, error="0 pages")
            if enc:
                return PDFValidation(ok=False, error="encrypted", encrypted=True, page_count=pc)
            return PDFValidation(ok=True, page_count=pc)
        finally:
            doc.close()
    except Exception as e:
        return PDFValidation(ok=False, error=f"PDF open error: {e}")


@dataclass
class ParsedPaper:
    """Result of parsing a PDF."""

    title: str = ""
    year: int | None = None
    doi: str | None = None
    arxiv: str | None = None
    s2_id: str | None = None
    openalex_id: str | None = None
    journal: str = ""
    publisher: str = ""
    citation_count: int = 0
    authors: list[dict] = field(default_factory=list)
    text_blocks: list[str] = field(default_factory=list)
    raw_md: str = ""
    images_dir: Path | None = None  # extracted images directory


class MinerUParser:
    """PDF parser: mineru-open-api CLI -> PyMuPDF fallback."""

    def __init__(
        self,
        token: str = "",
        model: str = "vlm",
        is_ocr: bool = False,
        enable_formula: bool = True,
        enable_table: bool = True,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        deepxiv_token: str = "",
        s2_api_key: str = "",
    ):
        self.token = token
        self.model = model
        self.is_ocr = is_ocr
        self.enable_formula = enable_formula
        self.enable_table = enable_table
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.deepxiv_token = deepxiv_token
        self.s2_api_key = s2_api_key

    def extract(self, pdf_path: str | Path, max_pages: int = 150) -> ParsedPaper:
        """Extract structured content from PDF. Splits into chunks if > max_pages."""
        pdf_path = Path(pdf_path)
        page_count = self._count_pages(pdf_path)
        _parse_log.info("[parse] %s — %d pages (split at %d)", pdf_path.name, page_count, max_pages)

        if page_count <= max_pages:
            return self._extract_single(pdf_path)

        # Split and process chunks under a single temp directory
        _parse_log.info(
            "[parse] %s split into %d chunks (max %d pages each)",
            pdf_path.name,
            (page_count + max_pages - 1) // max_pages,
            max_pages,
        )
        tmp = TemporaryDirectory(prefix="mineru_split_")
        tmp_path = Path(tmp.name)
        managed_tmps: list[TemporaryDirectory] = []
        try:
            chunk_paths = self._split_pdf(pdf_path, max_pages, tmp_path)
            raw_mds: list[str] = []
            image_dirs: list[Path | None] = []
            for chunk in chunk_paths:
                chunk_tmp = TemporaryDirectory(prefix="mineru_chunk_", dir=str(tmp_path))
                out_dir = Path(chunk_tmp.name) / "out"
                md, img_dir, chunk_managed_tmp = self._extract_mineru_only(chunk, out_dir=out_dir)
                if chunk_managed_tmp:
                    managed_tmps.append(chunk_managed_tmp)
                raw_mds.append(md)
                image_dirs.append(img_dir)

            merged_tmp = TemporaryDirectory(prefix="mineru_merged_", dir=str(tmp_path))
            merged_md = self._merge_markdown(raw_mds)
            merged_images = self._merge_images(image_dirs, Path(merged_tmp.name) / "images")

            title = self._extract_title(merged_md, str(pdf_path))
            year = self._extract_year(merged_md)
            doi, arxiv = self._extract_ids(merged_md)
            arxiv_from_name = _extract_arxiv_from_filename(pdf_path)
            if not arxiv and arxiv_from_name:
                arxiv = arxiv_from_name

            meta = _resolve_metadata(
                arxiv=arxiv,
                raw_title=title,
                raw_year=year,
                raw_doi=doi,
                deepxiv_token=self.deepxiv_token,
                s2_api_key=self.s2_api_key,
            )
            title = meta["title"] or title
            year = meta["year"] or year
            doi = meta["doi"] or doi
            s2_id = meta["s2_id"]
            oa_id = meta["openalex_id"]
            journal = meta["journal"]
            publisher = meta["publisher"]
            citation_count = meta["citation_count"]

            blocks = filter_sections(merged_md)

            return ParsedPaper(
                title=title,
                year=year,
                doi=doi,
                arxiv=arxiv,
                s2_id=s2_id,
                openalex_id=oa_id,
                journal=journal,
                publisher=publisher,
                citation_count=citation_count,
                text_blocks=blocks,
                raw_md=merged_md,
                images_dir=merged_images,
            )
        finally:
            for t in managed_tmps:
                t.cleanup()
            tmp.cleanup()

    def _count_pages(self, pdf_path: Path) -> int:
        """Count pages in PDF using PyMuPDF."""
        import fitz

        doc = fitz.open(str(pdf_path))
        try:
            return doc.page_count
        finally:
            doc.close()

    def _split_pdf(self, pdf_path: Path, max_pages: int, base_dir: Path) -> list[Path]:
        """Split PDF into chunks of max_pages each. Returns list of temp PDF paths."""
        import fitz

        doc = fitz.open(str(pdf_path))
        total = doc.page_count
        chunks: list[Path] = []
        for start in range(0, total, max_pages):
            end = min(start + max_pages, total)
            out = base_dir / f"chunk_{start}-{end}.pdf"
            out_doc = fitz.open()
            out_doc.insert_pdf(doc, from_page=start, to_page=end - 1)
            out_doc.save(str(out))
            out_doc.close()
            chunks.append(out)
        doc.close()
        return chunks

    def _extract_mineru_only(
        self, pdf_path: Path, out_dir: Path | None = None
    ) -> tuple[str, Path | None, TemporaryDirectory | None]:
        """Run MinerU CLI on a PDF chunk, return (raw_md, images_dir, managed_tmp)."""
        out_dir, managed_tmp = self._try_mineru_open_api(pdf_path, out_dir=out_dir)
        if out_dir is not None:
            raw_md = self._read_output_md(out_dir)
            img_dir = out_dir / "images" if (out_dir / "images").exists() else None
        else:
            raw_md = self._fallback_pymupdf(pdf_path)
            img_dir = None
        return raw_md, img_dir, managed_tmp

    def _extract_single(self, pdf_path: Path) -> ParsedPaper:
        """Extract a single PDF without splitting (existing logic)."""
        arxiv_from_name = _extract_arxiv_from_filename(pdf_path)
        out_dir, managed_tmp = self._try_mineru_open_api(pdf_path)
        try:
            if out_dir is not None:
                raw_md = self._read_output_md(out_dir)
                _parse_log.info("[parse] MinerU succeeded for %s", pdf_path.name)
            else:
                _parse_log.warning(
                    "[parse] MinerU unavailable, falling back to PyMuPDF for %s", pdf_path.name
                )
                raw_md = self._fallback_pymupdf(pdf_path)
                out_dir = None

            title = self._extract_title(raw_md, str(pdf_path))
            year = self._extract_year(raw_md)
            doi, arxiv = self._extract_ids(raw_md)

            if not arxiv and arxiv_from_name:
                arxiv = arxiv_from_name

            # Cross-validate metadata from all available sources
            meta = _resolve_metadata(
                arxiv=arxiv,
                raw_title=title,
                raw_year=year,
                raw_doi=doi,
                deepxiv_token=self.deepxiv_token,
                s2_api_key=self.s2_api_key,
            )
            title = meta["title"] or title
            year = meta["year"] or year
            doi = meta["doi"] or doi
            s2_id = meta["s2_id"]
            oa_id = meta["openalex_id"]
            journal = meta["journal"]
            publisher = meta["publisher"]
            citation_count = meta["citation_count"]

            # Fetch authorships from OpenAlex
            from drbrain.extractor.openalex import search_authors_by_work

            authors = search_authors_by_work(doi=doi, title=title)

            blocks = filter_sections(raw_md)
            _parse_log.info(
                "[parse] %s — title=%s year=%s doi=%s blocks=%d md_size=%d",
                pdf_path.name,
                title[:60] if title else "N/A",
                year,
                doi,
                len(blocks),
                len(raw_md),
            )

            images_dir = out_dir / "images" if out_dir and (out_dir / "images").exists() else None

            return ParsedPaper(
                title=title,
                year=year,
                doi=doi,
                arxiv=arxiv,
                s2_id=s2_id,
                openalex_id=oa_id,
                journal=journal,
                publisher=publisher,
                citation_count=citation_count,
                authors=authors or [],
                text_blocks=blocks,
                raw_md=raw_md,
                images_dir=images_dir,
            )
        finally:
            if managed_tmp:
                managed_tmp.cleanup()

    def _merge_markdown(self, raw_mds: list[str]) -> str:
        """Merge multiple markdown outputs: keep first chunk fully, append rest with separator."""
        result = raw_mds[0]
        for md in raw_mds[1:]:
            # Strip duplicate title from subsequent chunks
            lines = md.splitlines()
            content_start = 0
            for i, line in enumerate(lines[:10]):
                if line.startswith("# ") and i < 3:
                    content_start = i + 1
                    break
            stripped = "\n".join(lines[content_start:]).strip()
            if stripped:
                result += f"\n\n---\n\n{stripped}"
        return result

    def _merge_images(self, image_dirs: list[Path | None], dest: Path) -> Path | None:
        """Copy all images from chunk dirs into dest/images."""
        dest.mkdir(parents=True, exist_ok=True)
        any_images = False
        for img_dir in image_dirs:
            if img_dir and img_dir.exists():
                for f in img_dir.iterdir():
                    if f.is_file():
                        shutil.copy2(f, dest / f.name)
                        any_images = True
        return dest if any_images else None

    def _try_mineru_open_api(
        self, pdf_path: Path, out_dir: Path | None = None
    ) -> tuple[Path | None, TemporaryDirectory | None]:
        """Invoke mineru-open-api CLI with retry. Returns (output_dir, managed_tmp).

        If out_dir was provided, managed_tmp is None (caller owns the dir).
        If a temp dir was created here, managed_tmp must be cleaned up by the caller.
        """
        cli = _find_cli()
        if cli is None:
            return None, None

        # Pre-validate PDF before invoking MinerU (only when file exists)
        if pdf_path.exists():
            validation = _validate_pdf(pdf_path)
            if not validation.ok:
                _parse_log.warning(
                    "PDF validation failed for %s: %s (encrypted=%s, pages=%d)",
                    pdf_path,
                    validation.error,
                    validation.encrypted,
                    validation.page_count,
                )
                return None, None

        # Create temp dir if not provided
        managed_tmp: TemporaryDirectory | None = None
        if out_dir is None:
            managed_tmp = TemporaryDirectory(prefix="mineru_")
            out_dir = Path(managed_tmp.name) / "out"

        if self.token:
            cmd = [cli, "extract", str(pdf_path), "--model", self.model, "-o", str(out_dir)]
            if self.is_ocr:
                cmd.append("--ocr")
            if not self.enable_formula:
                cmd.append("--formula=false")
            if not self.enable_table:
                cmd.append("--table=false")
            cmd.extend(["--token", self.token])
        else:
            cmd = [cli, "extract", str(pdf_path), "-o", str(out_dir)]

        for attempt in range(self.max_retries):
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if result.returncode != 0:
                    _parse_log.warning(
                        "mineru-open-api failed (attempt %d/%d): %s",
                        attempt + 1,
                        self.max_retries,
                        result.stderr[:500],
                    )
                    time.sleep(2.0**attempt)
                    continue
                if not (out_dir / "images").exists():
                    time.sleep(2.0**attempt)
                    continue
                return out_dir, managed_tmp
            except subprocess.TimeoutExpired:
                _parse_log.warning(
                    "mineru-open-api timeout (attempt %d/%d)", attempt + 1, self.max_retries
                )
                time.sleep(2.0**attempt)
            except (FileNotFoundError, OSError) as e:
                _parse_log.warning("mineru-open-api error: %s", e)
                if managed_tmp:
                    managed_tmp.cleanup()
                return None, None
        if managed_tmp:
            managed_tmp.cleanup()
        return None, None

    def _read_output_md(self, out_dir: Path) -> str:
        """Read the generated markdown from the output directory."""
        md_files = list(out_dir.glob("*.md"))
        if not md_files:
            return ""
        return md_files[0].read_text(encoding="utf-8")

    def _fallback_pymupdf(self, pdf_path: Path) -> str:
        """Extract markdown via pymupdf4llm. Falls back to plain text."""
        try:
            import pymupdf4llm

            md = pymupdf4llm.to_markdown(str(pdf_path))
            if md.strip():
                return md
        except Exception as e:
            _parse_log.debug("pymupdf4llm failed, falling back to plain text: {}", e)

        # Last resort: plain text
        import fitz

        doc = fitz.open(str(pdf_path))
        try:
            lines = []
            for page in doc:
                t = page.get_text("text")
                if t.strip():
                    lines.append(t)
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
    return shutil.which("mineru-open-api")


def _extract_arxiv_from_filename(pdf_path: Path) -> str | None:
    """Extract arXiv ID from filename like '2602.00617v1.pdf'."""
    m = ARXIV_FILENAME.search(pdf_path.name)
    if m:
        return m.group(1)
    return None


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


def extract_pdf(pdf_path: str | Path, config: dict) -> ParsedPaper:
    """Convenience function: create parser from config and extract."""
    mineru_cfg = config.get("mineru", {})
    api_cfg = config.get("api", {})
    max_pages = mineru_cfg.get("max_pages", 150)
    parser = MinerUParser(
        token=mineru_cfg.get("token", ""),
        model=mineru_cfg.get("model", "vlm"),
        is_ocr=mineru_cfg.get("is_ocr", False),
        enable_formula=mineru_cfg.get("enable_formula", True),
        enable_table=mineru_cfg.get("enable_table", True),
        deepxiv_token=api_cfg.get("deepxiv_token", ""),
        s2_api_key=api_cfg.get("s2_api_key", ""),
    )
    return parser.extract(pdf_path, max_pages=max_pages)
