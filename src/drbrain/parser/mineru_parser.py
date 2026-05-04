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

        if page_count <= max_pages:
            return self._extract_single(pdf_path)

        # Split and process chunks under a single temp directory
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
            else:
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
                    time.sleep(self.retry_delay)
                    continue
                if not (out_dir / "images").exists():
                    time.sleep(self.retry_delay)
                    continue
                return out_dir, managed_tmp
            except subprocess.TimeoutExpired:
                _parse_log.warning(
                    "mineru-open-api timeout (attempt %d/%d)", attempt + 1, self.max_retries
                )
                time.sleep(self.retry_delay)
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
        except Exception:
            pass

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


def _titles_match(a: str, b: str) -> bool:
    """Check if two titles likely refer to the same paper. Uses word overlap ratio."""
    a_words = set(a.strip().lower().rstrip(".").split())
    b_words = set(b.strip().lower().rstrip(".").split())
    if not a_words or not b_words:
        return False
    smaller = a_words if len(a_words) <= len(b_words) else b_words
    larger = b_words if smaller == a_words else a_words
    overlap = len(smaller & larger)
    return overlap >= len(smaller) * 0.6  # 60% word overlap


def _resolve_metadata(
    arxiv: str | None = None,
    raw_title: str | None = None,
    raw_year: int | None = None,
    raw_doi: str | None = None,
    deepxiv_token: str = "",
    s2_api_key: str = "",
) -> dict:
    """Cross-validate metadata from arXiv, CrossRef, S2, and OpenAlex.

    Strategy:
    1. If arXiv ID: fetch from arXiv API (authoritative for arXiv papers)
    2. If title: search CrossRef, OpenAlex, S2
    3. DOI from CrossRef with title+year consistency check
    4. Multiple source consensus → high confidence
    5. Returns {title, year, doi, s2_id, openalex_id}
    """
    sources: dict[str, dict] = {}

    # ── arXiv ──
    if arxiv:
        arxiv_title, arxiv_year = _fetch_arxiv_metadata(arxiv)
        if arxiv_title or arxiv_year:
            sources["arxiv"] = {
                "title": arxiv_title,
                "year": arxiv_year,
                "doi": None,
                "s2_id": None,
                "openalex_id": None,
            }

    # ── CrossRef (keyed by title from arXiv, or raw title) ──
    search_title = sources.get("arxiv", {}).get("title") or raw_title or ""
    if search_title:
        cr_title, cr_year, cr_doi, cr_journal, cr_publisher = _fetch_crossref_metadata(search_title)
        if cr_doi or cr_year:
            sources["crossref"] = {
                "title": cr_title,
                "year": cr_year,
                "doi": cr_doi,
                "s2_id": None,
                "openalex_id": None,
                "journal": cr_journal,
                "publisher": cr_publisher,
                "citation_count": 0,
            }

    # ── OpenAlex ──
    if search_title:
        oa_title, oa_year, oa_id, oa_journal, oa_cited = _fetch_openalex_metadata(search_title)
        if oa_title or oa_year:
            sources["openalex"] = {
                "title": oa_title,
                "year": oa_year,
                "doi": None,
                "s2_id": None,
                "openalex_id": oa_id,
                "journal": oa_journal,
                "publisher": "",
                "citation_count": oa_cited,
            }

    # ── Semantic Scholar ──
    if search_title:
        s2_title, s2_year, s2_id, s2_journal, s2_cited = _fetch_s2_metadata(search_title)
        if s2_title or s2_year:
            sources["s2"] = {
                "title": s2_title,
                "year": s2_year,
                "doi": None,
                "s2_id": s2_id,
                "openalex_id": None,
                "journal": s2_journal,
                "publisher": "",
                "citation_count": s2_cited,
            }

    # ── DeepXiv (arXiv papers, adds TLDR + keywords + citation count) ──
    if arxiv:
        dx = _fetch_deepxiv_metadata(arxiv, token=deepxiv_token)
        if dx and dx.get("title"):
            sources["deepxiv"] = {
                "title": dx["title"],
                "year": dx["year"],
                "doi": None,
                "s2_id": None,
                "openalex_id": None,
                "journal": "",
                "publisher": "",
                "citation_count": dx.get("citations") or 0,
            }

    # ── Resolution ──
    final_doi = raw_doi
    final_title = raw_title
    final_year = raw_year

    # Use text-extracted year as anchor to filter API results
    _text_year = raw_year  # from PDF text parsing

    def _year_consistent(api_year, anchor):
        if not api_year or not anchor:
            return True
        return abs(api_year - anchor) <= 5

    # Only trust CrossRef's DOI if title matches AND year is consistent
    cr_data = sources.get("crossref", {})
    cr_doi = cr_data.get("doi")
    cr_year = cr_data.get("year")
    if cr_doi and not final_doi:
        cr_title = cr_data.get("title") or ""
        ref_title = final_title or ""
        if _titles_match(ref_title, cr_title):
            anchor = _text_year or sources.get("arxiv", {}).get("year")
            if _year_consistent(cr_year, anchor):
                final_doi = cr_doi

    if final_doi:
        if cr_data.get("year"):
            final_year = cr_data["year"]
        if cr_data.get("title"):
            final_title = cr_data["title"]

    # Filter API years by text-year consistency
    filtered_sources = {
        k: v
        for k, v in sources.items()
        if not _text_year or not v.get("year") or _year_consistent(v["year"], _text_year)
    }

    if not final_year:
        # Use filtered sources (year-consistent with text anchor)
        years = [(k, v["year"]) for k, v in filtered_sources.items() if v.get("year")]
        if len(years) >= 2 and len(set(y for _, y in years)) == 1:
            final_year = years[0][1]
        elif sources.get("arxiv", {}).get("year"):
            final_year = sources["arxiv"]["year"]
        elif years:
            final_year = years[0][1]

    if not final_title or final_title == raw_title:
        for src in ["arxiv", "crossref", "s2", "openalex"]:
            if sources.get(src, {}).get("title"):
                final_title = sources[src]["title"]
                break

    # Collect external IDs from sources
    final_s2_id = None
    final_openalex_id = None
    for src_name in ["crossref", "s2", "openalex"]:
        s = sources.get(src_name, {})
        if s.get("s2_id") and not final_s2_id:
            final_s2_id = s["s2_id"]
        if s.get("openalex_id") and not final_openalex_id:
            final_openalex_id = s["openalex_id"]

    # Collect venue metadata: prefer CrossRef for journal/publisher, then OpenAlex, then S2
    final_journal = ""
    final_publisher = ""
    final_citation_count = 0
    for src_name in ["crossref", "openalex", "s2", "deepxiv"]:
        s = sources.get(src_name, {})
        if s.get("journal") and not final_journal:
            final_journal = s["journal"]
        if s.get("publisher") and not final_publisher:
            final_publisher = s["publisher"]
        if s.get("citation_count") and not final_citation_count:
            final_citation_count = s["citation_count"]

    return {
        "title": final_title,
        "year": final_year,
        "doi": final_doi,
        "s2_id": final_s2_id,
        "openalex_id": final_openalex_id,
        "journal": final_journal,
        "publisher": final_publisher,
        "citation_count": final_citation_count,
    }


def _fetch_arxiv_metadata(arxiv_id: str) -> tuple[str | None, int | None]:
    """Fetch title and year from arXiv API via arxiv library."""
    try:
        import arxiv

        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(client.results(search))
        year = paper.published.year if paper.published else None
        return paper.title, year
    except Exception:
        # Fallback: infer year from arXiv ID (1706.03762 → 2017)
        m = re.match(r"(\d{2})(\d{2})\.\d{4,5}", arxiv_id)
        if m:
            yy = int(m.group(1))
            year = 2000 + yy if yy <= 50 else 1900 + yy
            return None, year
        return None, None


def _fetch_openalex_metadata(
    title: str,
) -> tuple[str | None, int | None, str | None, str | None, int]:
    """Fetch title, year, OpenAlex ID, journal, and citation count via pyalex."""
    if not title:
        return None, None, None, None, 0
    try:
        from pyalex import Works as _Works

        works = _Works()
        results = list(works.search(title).get(per_page=1))
        if results:
            w = results[0]
            oa_id = w.get("id", "").replace("https://openalex.org/", "")
            journal = ""
            loc = w.get("primary_location") or {}
            source = loc.get("source") or {}
            if source:
                journal = source.get("display_name") or ""
            cited = w.get("cited_by_count") or 0
            return w.get("title"), w.get("publication_year"), oa_id or None, journal, cited
    except Exception:
        pass
    return None, None, None, None, 0


def _fetch_s2_metadata(
    title: str, api_key: str = ""
) -> tuple[str | None, int | None, str | None, str | None, int]:
    """Fetch title, year, paperId, journal, and citationCount from Semantic Scholar API."""
    if not title:
        return None, None, None, None, 0
    try:
        import json as _json
        import urllib.parse as _uparse
        import urllib.request as _ureq

        fields = "title,year,paperId,journal,citationCount"
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={_uparse.quote(title)}&limit=1&fields={fields}"
        headers = {"Accept": "application/json"}
        if api_key:
            headers["x-api-key"] = api_key
        req = _ureq.Request(url, headers=headers)
        resp = _ureq.urlopen(req, timeout=10)
        data = _json.loads(resp.read())
        papers = data.get("data", [])
        if papers:
            p = papers[0]
            journal = ""
            j = p.get("journal") or {}
            if j:
                journal = j.get("name", "")
            return (
                p.get("title"),
                p.get("year"),
                p.get("paperId"),
                journal,
                p.get("citationCount") or 0,
            )
    except Exception:
        pass
    return None, None, None, None, 0


def _fetch_deepxiv_metadata(arxiv_id: str, token: str = "") -> dict | None:
    """Fetch metadata from DeepXiv API (title, year, TLDR, keywords, citations)."""
    if not arxiv_id:
        return None
    try:
        import os as _os

        from deepxiv_sdk import Reader as _Reader

        _token = token or _os.environ.get("DEEPXIV_TOKEN", "")
        r = _Reader(token=_token) if _token else _Reader()
        data = r.brief(arxiv_id)
        year = None
        if data.get("publish_at"):
            year = int(data["publish_at"][:4])
        return {
            "title": data.get("title"),
            "year": year,
            "tldr": data.get("tldr"),
            "keywords": data.get("keywords", []),
            "citations": data.get("citations"),
        }
    except Exception:
        return None


def _fetch_crossref_metadata(
    title: str,
) -> tuple[str | None, int | None, str | None, str | None, str | None]:
    """Fetch title, year, DOI, journal, and publisher from CrossRef by title search."""
    if not title:
        return None, None, None, None, None
    try:
        import json as _json
        import urllib.parse as _uparse
        import urllib.request as _ureq

        clean = title.strip()[:200]
        url = f"https://api.crossref.org/works?query.bibliographic={_uparse.quote(clean)}&rows=1"
        req = _ureq.Request(url, headers={"Accept": "application/json"})
        resp = _ureq.urlopen(req, timeout=10)
        data = _json.loads(resp.read())
        items = data.get("message", {}).get("items", [])
        if items:
            item = items[0]
            doi = item.get("DOI")
            cr_title = item.get("title", [None])[0]
            year = item.get("published-print", {}).get("date-parts", [[None]])[0][0]
            if not year:
                year = item.get("created", {}).get("date-parts", [[None]])[0][0]
            journal = item.get("container-title", [None])[0] or ""
            publisher = item.get("publisher", "")
            return cr_title, year, doi, journal, publisher
    except Exception:
        pass
    return None, None, None, None, None


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
                    rest = line.strip()[m.end() :]
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
        filtered_lines = [line for line in lines if not line.startswith("Thinking...")]
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
