"""Transient structure hints for candidate grouping (plan T20/T21).

This module produces *hints only*: ordered :class:`StructureHint` records that
say "this document appears to have these sections, over these real locators".
Hints exist to propose grouping candidates (T31); they are never a second
tree, never a document library, and nothing here persists anything.  The
canonical text stays in ``content_blocks`` (design ``§3.1``), and the hints are
discarded once a grouping pass has consumed them.

What is reused, and what is not
-------------------------------
* Markdown/TeX text: the vendored ``page_index_md.md_to_tree`` is the primary
  source.  Upstream's entry point takes a *path*, so the in-memory text is
  written to a temporary file with an explicit lifetime (``NamedTemporaryFile``
  in the system temp dir, deleted in ``finally``).  That file is an adapter for
  the upstream signature, never fact storage.  Only the LLM-free shape is
  requested (``if_add_node_summary="no"``, ``if_thinning=False``), so no model
  is ever called on this path.  Upstream fences code on triple backticks only,
  so every node is re-checked against the canonical block segmentation
  (``blocks.segment_text``): a ``###`` inside a ``~~~`` fence or a ``$$`` block
  is code, not a heading, and is dropped.
* TeX / heading-less text: upstream only recognizes Markdown headings, so when
  the upstream tree contains no node we fall back to the canonical heading
  scanner in :mod:`drbrain.tree.blocks` (``segment_text``) plus a deterministic
  scan for LaTeX sectioning commands (``\\section``, ``\\subsection``, ...).
  This is the one documented deviation: the fallback exists because upstream's
  Markdown path cannot see TeX structure.  It never runs on a document where
  upstream already found headings, and it never invents a section for text
  that has no heading at all (empty input and heading-less text yield ``[]``).
* PDF: the vendored ``page_index_classic`` pipeline is reused for its **page
  extraction** (``utils.get_page_tokens``, PyPDF2) and, when a caller explicitly
  supplies ``index_model``, for its section tree
  (``page_index_classic.page_index_main(doc, opt, logger, page_list=...)``).
  The classic pipeline is *not* LLM-free end to end: ``tree_parser`` calls
  ``toc_detector_single_page``, ``toc_extractor``, ``toc_transformer``,
  ``generate_toc_*``, ``verify_toc`` and ``fix_incorrect_toc_*``, all of which
  go through ``utils.llm_completion``.  ``opt`` can only switch off the
  *summary* work (``if_add_node_summary="no"``, ``if_add_doc_description="no"``,
  ``if_add_node_text="no"``), so ``page_index_main`` is invoked solely when an
  index model was explicitly given; otherwise the PDF path derives a flat,
  page-level cover from the classic module's own real page extraction.  No page
  number is ever fabricated, and no temporary pseudo-PDF is created.
* Flash is deliberately unused: it consumes PDF/BytesIO geometry only, its
  private layout modules are not on the loader's audited allowlist, and the
  design forbids turning in-memory text into a fake PDF.

Model access
------------
Nothing in this module calls a model unless the caller passes ``index_model``
(optionally with a ``backend`` mapping that is installed on upstream's audited
``utils._llm_backend`` context variable, e.g. to point the pipeline at a local
OpenAI-compatible endpoint).  Upstream's ``JsonLogger`` is never used: it would
create a ``./logs`` directory and write JSON files, so the classic call gets a
loguru-backed adapter instead.

Locator contract
----------------
A hint carries exactly one locator family: PDF hints carry 1-based inclusive
page ranges, text hints carry 1-based inclusive line ranges.  Mixing the two,
or carrying none, raises ``ValueError`` at construction.  ``level`` is the
outline depth and always equals ``len(heading_path)`` (Markdown heading ranks
that skip a level are normalized to their depth, matching the nesting that is
the only hierarchy the upstream Markdown tree exposes).
"""

from __future__ import annotations

import asyncio
import bisect
import io
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from drbrain.tree.blocks import segment_text
from drbrain.tree.upstream import load_pageindex_module

__all__ = [
    "DEFAULT_MAX_NODES",
    "HINT_KINDS",
    "STRUCTURE_HINT_ORIGINS",
    "TEXT_HINT_LIMIT",
    "TITLE_LIMIT",
    "StructureHint",
    "extract_md_outline",
    "extract_pdf_outline",
    "structure_coverage",
]

#: Where a hint came from: the vendored Markdown path, the vendored classic
#: PDF path, or the deterministic heading fallback (TeX / heading-less text).
STRUCTURE_HINT_ORIGINS: tuple[str, ...] = ("pageindex-md", "pageindex-classic", "headings")

#: A hint describes either a titled section or a bare page container.
HINT_KINDS: tuple[str, ...] = ("section", "page")

DEFAULT_MAX_NODES = 200

#: ``text_hint`` is a budget probe, not evidence: it is whitespace-collapsed and
#: capped here so a hint can never smuggle a document's text around.
TEXT_HINT_LIMIT = 200
TITLE_LIMIT = 160

#: Markdown/TeX block kinds whose spans must not yield headings.
_OPAQUE_KINDS = frozenset({"code", "formula"})

#: LaTeX sectioning commands, mapped to a scan rank (ordering only; the emitted
#: ``level`` is the resulting depth, so a paper using only ``\section`` starts
#: at depth 1).
_TEX_SECTION_RE = re.compile(
    r"^[ \t]*\\(?P<command>part|chapter|section|subsection|subsubsection|paragraph"
    r"|subparagraph)\*?[ \t]*(?:\[[^\]]*\])?[ \t]*\{(?P<title>[^}]*)\}"
)
_TEX_SECTION_RANK: dict[str, int] = {
    "part": 1,
    "chapter": 2,
    "section": 3,
    "subsection": 4,
    "subsubsection": 5,
    "paragraph": 6,
    "subparagraph": 7,
}

#: LaTeX environments whose body is literal text: a ``\section`` line inside one
#: is not structure.  Only literal-code environments are listed — skipping
#: ``document`` or ``equation`` bodies would hide real sections.
_TEX_VERBATIM_ENVIRONMENTS: tuple[str, ...] = (
    "verbatim",
    "Verbatim",
    "lstlisting",
    "minted",
    "comment",
)
_TEX_VERBATIM_BEGIN_RE = re.compile(
    r"^[ \t]*\\begin\{(?P<environment>" + "|".join(_TEX_VERBATIM_ENVIRONMENTS) + r")\}"
)
_TEX_VERBATIM_END_RE = re.compile(r"^[ \t]*\\end\{(?P<environment>[A-Za-z*]+)\}")

_WHITESPACE_RE = re.compile(r"\s+")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_max_nodes(max_nodes: int) -> None:
    _require(isinstance(max_nodes, int) and max_nodes >= 1, "max_nodes must be a positive int")


def _short(text: str, limit: int = TEXT_HINT_LIMIT) -> str:
    """Whitespace-collapsed, capped probe text (never document text storage)."""
    return _WHITESPACE_RE.sub(" ", text).strip()[:limit]


def _short_join(chunks: Iterable[str], limit: int = TEXT_HINT_LIMIT) -> str:
    """Collapse-and-cap across chunks without materializing them all."""
    parts: list[str] = []
    used = 0
    for chunk in chunks:
        collapsed = _WHITESPACE_RE.sub(" ", chunk).strip()
        if not collapsed:
            continue
        parts.append(collapsed)
        used += len(collapsed) + 1
        if used >= limit:
            break
    return " ".join(parts)[:limit]


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for index, char in enumerate(text):
        if char == "\n":
            offsets.append(index + 1)
    return offsets


def _line_count(text: str) -> int:
    return text.count("\n") + 1


def _line_of(offsets: Sequence[int], offset: int) -> int:
    """1-based line number containing ``offset`` (same rule as ``blocks``)."""
    return bisect.bisect_right(offsets, offset)


@dataclass(frozen=True)
class StructureHint:
    """One transient structure hint over real locators.

    ``kind="section"`` carries a titled heading (``heading_path`` non-empty,
    ``anchor == title``, ``level == len(heading_path)``); ``kind="page"`` is a
    bare page container (no heading path, ``level == 0``).  Exactly one locator
    family is populated: PDF hints carry pages, text hints carry lines.
    """

    kind: str
    title: str
    heading_path: tuple[str, ...]
    anchor: str
    level: int
    page_start: int | None = None
    page_end: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    text_hint: str = ""
    origin: str = "headings"

    def __post_init__(self) -> None:
        _require(self.kind in HINT_KINDS, f"kind must be one of {HINT_KINDS}, got {self.kind!r}")
        _require(
            self.origin in STRUCTURE_HINT_ORIGINS,
            f"origin must be one of {STRUCTURE_HINT_ORIGINS}, got {self.origin!r}",
        )
        _require(isinstance(self.heading_path, tuple), "heading_path must be a tuple of str")
        _require(
            all(isinstance(part, str) and part for part in self.heading_path),
            "heading_path entries must be non-empty strings",
        )
        _require(len(self.title) <= TITLE_LIMIT, f"title exceeds {TITLE_LIMIT} characters")
        _require(len(self.text_hint) <= TEXT_HINT_LIMIT, f"text_hint exceeds {TEXT_HINT_LIMIT}")

        has_pages = self.page_start is not None or self.page_end is not None
        has_lines = self.line_start is not None or self.line_end is not None
        _require(
            not (has_pages and has_lines),
            "a hint carries one locator family: pages for PDF material, lines for text",
        )
        _require(has_pages or has_lines, "a hint needs at least one real locator family")
        if has_pages:
            _require(
                isinstance(self.page_start, int) and isinstance(self.page_end, int),
                "page locators are 1-based inclusive pairs; both ends are required",
            )
            _require(
                self.page_start >= 1 and self.page_end >= self.page_start,
                f"invalid page range {self.page_start}..{self.page_end}",
            )
        else:
            _require(
                isinstance(self.line_start, int) and isinstance(self.line_end, int),
                "line locators are 1-based inclusive pairs; both ends are required",
            )
            _require(
                self.line_start >= 1 and self.line_end >= self.line_start,
                f"invalid line range {self.line_start}..{self.line_end}",
            )

        if self.kind == "page":
            _require(not self.heading_path, "page hints carry no heading path")
            _require(self.level == 0, "page hints have level 0")
        else:
            _require(bool(self.heading_path), "section hints require a heading path")
            _require(
                self.level == len(self.heading_path),
                "level is the outline depth and equals len(heading_path)",
            )
            _require(
                self.heading_path[-1] == self.title,
                "a section hint's heading path ends with its own title",
            )
            _require(bool(self.anchor), "section hints require an anchor")
            _require(self.anchor == self.title, "anchor is the heading text (see blocks.py)")


def extract_md_outline(text: str, *, max_nodes: int = DEFAULT_MAX_NODES) -> list[StructureHint]:
    """Structure hints for in-memory Markdown/TeX text.

    Uses the vendored Markdown tree when it finds headings; otherwise falls back
    to the canonical heading scanner (TeX sectioning included).  Text without a
    single heading yields an empty list: sections are never invented.
    """
    _require_max_nodes(max_nodes)
    if not text.strip():
        return []
    hints = _md_hints_from_upstream(text)
    if not hints:
        hints = _heading_hints_from_text(text)
    return hints[:max_nodes]


def _md_hints_from_upstream(text: str) -> list[StructureHint]:
    """Run the vendored ``page_index_md.md_to_tree`` over a transient temp file."""
    module = load_pageindex_module("page_index_md")
    utils_module = load_pageindex_module("utils")
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".md", delete=False, encoding="utf-8", dir=tempfile.gettempdir()
    )
    path = handle.name
    try:
        with handle:
            handle.write(text)
        # ``md_to_tree`` is async; run_off_loop keeps this safe inside a loop.
        with redirect_stdout(io.StringIO()):
            tree = utils_module.run_off_loop(
                asyncio.run,
                module.md_to_tree(
                    path,
                    if_thinning=False,
                    if_add_node_summary="no",
                    if_add_doc_description="no",
                    if_add_node_text="yes",
                    if_add_node_id="yes",
                ),
            )
    finally:
        # The temp file is an adapter for upstream's path-based signature only.
        os.unlink(path)
        if os.path.exists(path):  # pragma: no cover - defensive on odd filesystems
            logger.warning("temporary Markdown adapter file survived cleanup: {}", path)

    structure = tree.get("structure") if isinstance(tree, Mapping) else None
    if not structure:
        return []
    line_total = int(tree.get("line_count") or _line_count(text))
    flat = _flatten_md_nodes(structure)
    flat = _drop_opaquely_fenced(text, flat)
    return _section_hints_from_flat(flat, line_total)


def _flatten_md_nodes(
    structure: Sequence[Mapping[str, Any]],
) -> list[tuple[tuple[str, ...], Mapping[str, Any], int]]:
    """Depth-first flatten of the vendored tree: (heading_path, node, line)."""
    flat: list[tuple[tuple[str, ...], Mapping[str, Any], int]] = []

    def walk(nodes: Sequence[Mapping[str, Any]], path: tuple[str, ...]) -> None:
        for node in nodes:
            title = str(node.get("title") or "").strip()
            child_path = path + (title,) if title else path
            start = node.get("line_num")
            if not title:
                logger.debug("vendored md tree node without a title skipped")
            elif not isinstance(start, int) or start < 1:
                logger.warning("vendored md tree node {!r} has no 1-based line_num; skipped", path)
            else:
                flat.append((child_path, node, start))
            children = node.get("nodes") or []
            if children:
                walk(children, child_path)

    walk(structure, ())
    return flat


def _opaque_spans(text: str) -> list[tuple[int, int]]:
    """Spans the canonical blocks call code or formula (never heading text)."""
    return [
        (segment.start, segment.end)
        for segment in segment_text(text)
        if segment.kind in _OPAQUE_KINDS
    ]


def _drop_opaquely_fenced(
    text: str, flat: Sequence[tuple[tuple[str, ...], Mapping[str, Any], int]]
) -> list[tuple[tuple[str, ...], Mapping[str, Any], int]]:
    """Drop headings that canonical segmentation calls code/formula body.

    Upstream's Markdown scanner only fences on triple backticks, so a ``###``
    inside a ``~~~`` fence (or display math) reaches us as a node.  Sections are
    cut afterwards, so a dropped span belongs to the section that encloses it
    instead of leaving a fake gap in the line cover.
    """
    opaque = _opaque_spans(text)
    if not opaque:
        return list(flat)
    offsets = _line_offsets(text)
    kept = []
    for entry in flat:
        index = entry[2] - 1
        offset = offsets[index] if index < len(offsets) else len(text)
        if any(start <= offset < end for start, end in opaque):
            logger.debug("dropped a vendored md heading inside code/formula: {}", entry[1])
            continue
        kept.append(entry)
    return kept


def _section_hints_from_flat(
    flat: Sequence[tuple[tuple[str, ...], Mapping[str, Any], int]], line_total: int
) -> list[StructureHint]:
    """Ordered section hints; a section ends where the next heading starts."""
    hints: list[StructureHint] = []
    for index, (path, node, start) in enumerate(flat):
        end = flat[index + 1][2] - 1 if index + 1 < len(flat) else line_total
        end = max(start, min(end, max(line_total, start)))
        hints.append(
            StructureHint(
                kind="section",
                title=path[-1],
                heading_path=path,
                anchor=path[-1],
                level=len(path),
                line_start=start,
                line_end=end,
                text_hint=_short(str(node.get("text") or "")),
                origin="pageindex-md",
            )
        )
    return hints


def _heading_hints_from_text(text: str) -> list[StructureHint]:
    """Deterministic fallback: canonical heading segments plus TeX sectioning.

    Runs only when the vendored Markdown path found no heading, so the two
    sources never double-count the same document.
    """
    segments = segment_text(text)
    offsets = _line_offsets(text)
    line_total = _line_count(text)

    opaque = [(segment.start, segment.end) for segment in segments if segment.kind in _OPAQUE_KINDS]

    def blocked(offset: int) -> bool:
        return any(start <= offset < end for start, end in opaque)

    found: dict[int, tuple[int, str]] = {}
    for segment in segments:
        if segment.kind != "title":
            continue
        title = (segment.anchor or segment.heading_path[-1]).strip()
        if title:
            found[segment.start] = (segment.level, title)

    cursor = 0
    environment: str | None = None
    for line in text.split("\n"):
        offset = cursor
        cursor += len(line) + 1
        begin = _TEX_VERBATIM_BEGIN_RE.match(line)
        if environment is None and begin is not None:
            environment = begin.group("environment")
            continue
        if environment is not None:
            end = _TEX_VERBATIM_END_RE.match(line)
            if end is not None and end.group("environment") == environment:
                environment = None
            continue
        if offset in found or blocked(offset):
            continue
        match = _TEX_SECTION_RE.match(line)
        if match is None:
            continue
        title = match.group("title").strip()
        if title:
            found[offset] = (_TEX_SECTION_RANK[match.group("command")], title)

    if not found:
        return []

    ordered = sorted(found.items())
    stack: list[tuple[int, str]] = []
    entries: list[tuple[int, tuple[str, ...], int]] = []
    for offset, (rank, title) in ordered:
        while stack and stack[-1][0] >= rank:
            stack.pop()
        stack.append((rank, title))
        entries.append((offset, tuple(name for _, name in stack), rank))

    hints: list[StructureHint] = []
    for index, (offset, path, _rank) in enumerate(entries):
        next_offset = entries[index + 1][0] if index + 1 < len(entries) else len(text)
        start_line = _line_of(offsets, offset)
        end_line = _line_of(offsets, next_offset) - 1 if index + 1 < len(entries) else line_total
        hints.append(
            StructureHint(
                kind="section",
                title=path[-1],
                heading_path=path,
                anchor=path[-1],
                level=len(path),
                line_start=start_line,
                line_end=max(start_line, min(end_line, max(line_total, start_line))),
                text_hint=_short(text[offset:next_offset]),
                origin="headings",
            )
        )
    return hints


class _UpstreamLogAdapter:
    """loguru-backed stand-in for upstream's ``JsonLogger`` (which writes files)."""

    def info(self, message: Any, **_kwargs: Any) -> None:
        logger.debug("pageindex-classic: {}", message)

    def debug(self, message: Any, **_kwargs: Any) -> None:
        logger.debug("pageindex-classic: {}", message)

    def error(self, message: Any, **_kwargs: Any) -> None:
        logger.warning("pageindex-classic error: {}", message)

    def exception(self, message: Any, **_kwargs: Any) -> None:
        logger.warning("pageindex-classic exception: {}", message)


def extract_pdf_outline(
    pdf_path: str | Path,
    *,
    max_nodes: int = DEFAULT_MAX_NODES,
    page_list: Sequence[Sequence[Any]] | None = None,
    index_model: str | None = None,
    backend: Mapping[str, Any] | None = None,
) -> list[StructureHint]:
    """Structure hints for a real PDF, over real 1-based inclusive page ranges.

    ``page_list`` may inject an already-parsed ``[(page_text, token_count), ...]``
    map (T20); it must describe the same PDF, which is checked when the file is
    readable.  Without it the vendored classic module's own page extraction
    (``utils.get_page_tokens``, PyPDF2) is used — the same routine upstream
    would run — so page numbers are never fabricated.

    ``index_model`` is the explicit opt-in to the vendored classic section tree:
    pass a model name (plus an optional ``backend`` mapping of client kwargs) to
    run ``page_index_classic.page_index_main``.  Without it no model is called
    and the result is a flat, honest page-level cover.
    """
    _require_max_nodes(max_nodes)
    utils_module = load_pageindex_module("utils")
    path = Path(pdf_path)
    pages = _resolve_page_list(path, page_list, utils_module)
    if not pages:
        return []
    if index_model:
        hints = _classic_section_hints(path, pages, index_model, backend)
    else:
        hints = _page_hints(pages)
    return hints[:max_nodes]


def _validated_page(entry: Any, position: int) -> tuple[str, int]:
    _require(
        isinstance(entry, Sequence) and not isinstance(entry, str | bytes) and len(entry) == 2,
        f"page_list[{position}] must be a (page_text, token_count) pair",
    )
    text, tokens = entry[0], entry[1]
    _require(isinstance(text, str), f"page_list[{position}][0] must be the page text as str")
    _require(
        isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0,
        f"page_list[{position}][1] must be a nonnegative token count",
    )
    return text, tokens


def _resolve_page_list(
    path: Path, page_list: Sequence[Sequence[Any]] | None, utils_module: Any
) -> tuple[tuple[str, int], ...]:
    if page_list is None:
        _require(path.is_file(), f"PDF not found: {path}")
        extracted = utils_module.get_page_tokens(str(path))
        return tuple(_validated_page(entry, index) for index, entry in enumerate(extracted))

    pages = tuple(_validated_page(entry, index) for index, entry in enumerate(page_list))
    _require(bool(pages), "page_list must not be empty")
    if path.is_file():
        declared = int(utils_module.get_number_of_pages(str(path)))
        _require(
            declared == len(pages),
            f"injected page_list has {len(pages)} pages but {path.name} has {declared}; "
            "hint page ranges must describe the real PDF",
        )
    return pages


def _page_hints(pages: Sequence[tuple[str, int]]) -> list[StructureHint]:
    """Flat cover: one page container per real page (never an invented section)."""
    hints: list[StructureHint] = []
    for number, (text, _tokens) in enumerate(pages, start=1):
        leading = next((line for line in text.split("\n") if line.strip()), "")
        hints.append(
            StructureHint(
                kind="page",
                title=_short(leading, TITLE_LIMIT),
                heading_path=(),
                anchor="",
                level=0,
                page_start=number,
                page_end=number,
                text_hint=_short(text),
                origin="pageindex-classic",
            )
        )
    return hints


def _classic_opt(utils_module: Any, index_model: str) -> Any:
    """Vendored defaults with every model-backed output switched off.

    The model still runs the table-of-contents stages (detection, synthesis,
    verification) — those are inherent to the classic pipeline — but no
    summary, node text or document description is requested.
    """
    return utils_module.ConfigLoader().load(
        {
            "model": index_model,
            "index_model": index_model,
            "if_add_node_id": "yes",
            "if_add_node_summary": "no",
            "if_add_doc_description": "no",
            "if_add_node_text": "no",
        }
    )


def _classic_section_hints(
    path: Path,
    pages: Sequence[tuple[str, int]],
    index_model: str,
    backend: Mapping[str, Any] | None,
) -> list[StructureHint]:
    classic = load_pageindex_module("page_index_classic")
    utils_module = load_pageindex_module("utils")
    opt = _classic_opt(utils_module, index_model)
    token = None
    if backend:
        token = utils_module._llm_backend.set(dict(backend))
    try:
        with redirect_stdout(io.StringIO()):
            tree = classic.page_index_main(
                str(path), opt, _UpstreamLogAdapter(), page_list=list(pages)
            )
    finally:
        if token is not None:
            utils_module._llm_backend.reset(token)
    structure = tree.get("structure") if isinstance(tree, Mapping) else None
    if not structure:
        return []
    return _section_hints_from_classic_nodes(structure, pages)


def _section_hints_from_classic_nodes(
    structure: Sequence[Mapping[str, Any]], pages: Sequence[tuple[str, int]]
) -> list[StructureHint]:
    """Flatten the classic TOC tree; parent/child page ranges may overlap."""
    hints: list[StructureHint] = []

    def walk(nodes: Sequence[Mapping[str, Any]], path: tuple[str, ...]) -> None:
        for node in nodes:
            title = str(node.get("title") or "").strip()
            child_path = path + (title,) if title else path
            start = node.get("start_index")
            end = node.get("end_index")
            if not title:
                logger.debug("classic tree node without a title skipped")
            elif not isinstance(start, int) or not isinstance(end, int):
                logger.warning("classic tree node {!r} has no page range; skipped", child_path)
            else:
                first = max(1, start)
                last = min(len(pages), end)
                if first > last:
                    logger.warning(
                        "classic tree node {!r} has out-of-document pages {}-{}; skipped",
                        child_path,
                        start,
                        end,
                    )
                else:
                    hints.append(
                        StructureHint(
                            kind="section",
                            title=title,
                            heading_path=child_path,
                            anchor=title,
                            level=len(child_path),
                            page_start=first,
                            page_end=last,
                            text_hint=_short_join(page[0] for page in pages[first - 1 : last]),
                            origin="pageindex-classic",
                        )
                    )
            children = node.get("nodes") or []
            if children:
                walk(children, child_path)

    walk(structure, ())
    return hints


def structure_coverage(
    hints: Sequence[StructureHint],
    *,
    total_pages: int | None = None,
    total_lines: int | None = None,
) -> dict[str, Any]:
    """Audit which units the hints cover and which are left over (design §3.1).

    Returns ``{"family", "hint_count", "pages": {...}, "lines": {...}}`` where
    each family block reports ``total`` (when the caller knows it), ``covered``,
    ``uncovered``, ``ratio``, ``residual_ranges`` (contiguous gaps with no hint)
    and ``overlap_units`` (units claimed more than once — PageIndex parent/child
    ranges may overlap by design) plus ``out_of_range_units`` (claims beyond the
    declared total, which must stay 0 for honest page numbers).
    """
    page_ranges = [
        (hint.page_start, hint.page_end)
        for hint in hints
        if hint.page_start is not None and hint.page_end is not None
    ]
    line_ranges = [
        (hint.line_start, hint.line_end)
        for hint in hints
        if hint.line_start is not None and hint.line_end is not None
    ]
    if page_ranges and line_ranges:
        family = "mixed"
    elif page_ranges:
        family = "page"
    elif line_ranges:
        family = "line"
    else:
        family = "none"
    return {
        "family": family,
        "hint_count": len(hints),
        "pages": _family_coverage(page_ranges, total_pages),
        "lines": _family_coverage(line_ranges, total_lines),
    }


def _family_coverage(ranges: Sequence[tuple[int, int]], total: int | None) -> dict[str, Any]:
    if total is not None:
        _require(isinstance(total, int) and total >= 0, "total must be a nonnegative int")
    merged, overlap = _merge_ranges(ranges)
    covered = sum(end - start + 1 for start, end in merged)
    out_of_range = 0
    if total is not None:
        out_of_range = sum(
            max(0, end - max(start, total + 1) + 1) for start, end in merged if end > total
        )
    residual = _residual_ranges(merged, total) if total is not None else []
    return {
        "total": total,
        "covered": covered,
        "uncovered": None if total is None else max(0, total - min(covered, total)),
        "ratio": None if total is None else (round(covered / total, 6) if total else 0.0),
        "residual_ranges": residual,
        "overlap_units": overlap,
        "out_of_range_units": out_of_range,
    }


def _merge_ranges(ranges: Sequence[tuple[int, int]]) -> tuple[list[tuple[int, int]], int]:
    """Merge inclusive ranges; return (merged, units claimed more than once)."""
    merged: list[tuple[int, int]] = []
    overlap = 0
    for start, end in sorted(ranges):
        if not merged:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        if start <= previous_end:  # overlapping or touching
            overlap += min(end, previous_end) - start + 1
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged, overlap


def _residual_ranges(merged: Sequence[tuple[int, int]], total: int) -> list[list[int]]:
    """Contiguous uncovered runs inside ``[1, total]``; no silent gaps."""
    residual: list[list[int]] = []
    cursor = 1
    for start, end in merged:
        if start > cursor:
            residual.append([cursor, start - 1])
        cursor = max(cursor, end + 1)
    if cursor <= total:
        residual.append([cursor, total])
    return residual
