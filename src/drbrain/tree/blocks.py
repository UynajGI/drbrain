"""Boundary-faithful canonical block generation (plan T09).

Input is the parser's canonical text (``raw_md`` from PDF/MD/TeX parses) plus
optional locator maps; output is the ordered, contiguous ``ContentBlock``
sequence whose concatenation reproduces the canonical text *verbatim* —
separators, punctuation, formulas, tables and residual material included.
Nothing is silently truncated: a span over the token budget is split at
structural boundaries (headings, blank lines, sentence/whitespace edges) and
only as a last resort at an exact character offset, which keeps every source
range exact even when the text has no usable boundary.

Locator rules: PDF blocks carry 1-based inclusive page spans when the caller
supplies page marks; MD/TeX blocks carry 1-based inclusive line spans.  A
material exposes one family; missing marks mean ``None``, never a fabricated
page number.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from drbrain.tree.contracts import ContentBlock, content_block_id

#: Structural boundary candidates inside an oversized span, in priority order.
_BOUNDARY_RE = re.compile(r"(?:\n[ \t]*\n)|(?:\n)|(?<=[.!?;:])\s+|(?<=\s)")

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SETEXT_RE = re.compile(r"^(=+|-{3,})\s*$")

DEFAULT_MAX_BLOCK_TOKENS = 512


def _default_count_tokens(text: str) -> int:
    from drbrain.services.tokens import count_tokens

    return int(count_tokens(text))


@dataclass(frozen=True)
class Segment:
    """A structural slice of the canonical text."""

    start: int
    end: int
    kind: str
    heading_path: tuple[str, ...]
    anchor: str
    level: int = 0


@dataclass
class BlockPolicy:
    max_tokens: int = DEFAULT_MAX_BLOCK_TOKENS
    count_tokens: Callable[[str], int] = field(default=_default_count_tokens)
    #: kinds that are never split internally (formula/table stay whole when
    #: they fit; oversized ones still split on their own line boundaries).
    atomic_kinds: tuple[str, ...] = ("formula",)
    #: Merge adjacent spans within one section up to this many characters so a
    #: leaf carries a paragraph-sized chunk instead of a line/sentence
    #: fragment (the 10k flow test measured a 118-char median leaf, 23% under
    #: 40 chars, against RAPTOR's 100-token ≈ 400-char chunking).  ``0`` keeps
    #: the raw structural segmentation.
    min_chars: int = 400

    def __post_init__(self) -> None:
        if self.max_tokens < 16:
            raise ValueError("max_tokens must be >= 16")
        if self.min_chars < 0:
            raise ValueError("min_chars must be >= 0")


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for index, char in enumerate(text):
        if char == "\n":
            starts.append(index + 1)
    return starts


def _line_of(starts: Sequence[int], offset: int) -> int:
    """1-based line number containing ``offset``."""
    return bisect.bisect_right(starts, offset)


def segment_text(text: str) -> list[Segment]:
    """Split canonical text into structural segments with exact ranges.

    Every character of ``text`` belongs to exactly one segment (segments
    absorb their trailing separators), so the pieces reconstruct the input.
    """
    if not text:
        return []
    starts = _line_starts(text)
    segments: list[Segment] = []
    heading_stack: list[tuple[int, str]] = []
    index = 0
    total_lines = len(starts)

    def line_bounds(line_index: int) -> tuple[int, int]:
        start = starts[line_index]
        end = starts[line_index + 1] if line_index + 1 < total_lines else len(text)
        return start, end

    while index < total_lines:
        start_line, end_line = line_bounds(index)
        line = text[start_line:end_line]
        stripped = line.strip()

        # Fenced code block: consume until the closing fence (or EOF).
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence = stripped[:3]
            block_start = start_line
            cursor = index + 1
            while cursor < total_lines:
                candidate_start, _ = line_bounds(cursor)
                candidate = text[candidate_start : line_bounds(cursor)[1]]
                if candidate.strip().startswith(fence):
                    cursor += 1
                    break
                cursor += 1
            block_end = line_bounds(cursor - 1)[1] if cursor > index else end_line
            segments.append(
                Segment(
                    start=block_start,
                    end=block_end,
                    kind="code",
                    heading_path=tuple(title for _, title in heading_stack),
                    anchor=heading_stack[-1][1] if heading_stack else "",
                )
            )
            index = cursor
            continue

        # Display math block ($$ ... $$), with or without delimiters on their
        # own lines.
        if stripped.startswith("$$"):
            block_start = start_line
            cursor = index
            if stripped.count("$$") >= 2 and len(stripped) > 2:
                cursor = index + 1  # single-line $$...$$
            else:
                cursor = index + 1
                while cursor < total_lines:
                    candidate_start, candidate_end = line_bounds(cursor)
                    if "$$" in text[candidate_start:candidate_end]:
                        cursor += 1
                        break
                    cursor += 1
            block_end = line_bounds(cursor - 1)[1]
            segments.append(
                Segment(
                    start=block_start,
                    end=block_end,
                    kind="formula",
                    heading_path=tuple(title for _, title in heading_stack),
                    anchor=heading_stack[-1][1] if heading_stack else "",
                )
            )
            index = cursor
            continue

        heading_match = _HEADING_RE.match(stripped)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            segments.append(
                Segment(
                    start=start_line,
                    end=end_line,
                    kind="title",
                    heading_path=tuple(name for _, name in heading_stack),
                    anchor=title,
                    level=level,
                )
            )
            index += 1
            continue

        if stripped and index + 1 < total_lines:
            next_start, next_end = line_bounds(index + 1)
            if _SETEXT_RE.match(text[next_start:next_end].strip()) and len(stripped) < 120:
                title = stripped
                level = 1 if text[next_start:next_end].strip().startswith("=") else 2
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))
                segments.append(
                    Segment(
                        start=start_line,
                        end=next_end,
                        kind="title",
                        heading_path=tuple(name for _, name in heading_stack),
                        anchor=title,
                        level=level,
                    )
                )
                index += 2
                continue

        # Blank line: absorb into the previous segment when possible.
        if not stripped:
            if segments and segments[-1].end == start_line:
                previous = segments[-1]
                segments[-1] = Segment(
                    start=previous.start,
                    end=end_line,
                    kind=previous.kind,
                    heading_path=previous.heading_path,
                    anchor=previous.anchor,
                    level=previous.level,
                )
            else:
                segments.append(
                    Segment(
                        start=start_line,
                        end=end_line,
                        kind="residual",
                        heading_path=tuple(title for _, title in heading_stack),
                        anchor=heading_stack[-1][1] if heading_stack else "",
                    )
                )
            index += 1
            continue

        # Paragraph (or table): consume until a blank line, heading, fence or
        # display-math start.
        block_start = start_line
        cursor = index
        kind = "table" if stripped.startswith("|") else "paragraph"
        while cursor < total_lines:
            current_start, current_end = line_bounds(cursor)
            current = text[current_start:current_end]
            current_stripped = current.strip()
            if cursor > index:
                if not current_stripped:
                    break
                if _HEADING_RE.match(current_stripped):
                    break
                if current_stripped.startswith("```") or current_stripped.startswith("~~~"):
                    break
                if current_stripped.startswith("$$"):
                    break
                if kind == "table" and not current_stripped.startswith("|"):
                    break
            cursor += 1
        block_end = line_bounds(cursor - 1)[1]
        segments.append(
            Segment(
                start=block_start,
                end=block_end,
                kind=kind,
                heading_path=tuple(title for _, title in heading_stack),
                anchor=heading_stack[-1][1] if heading_stack else "",
            )
        )
        index = cursor

    # Absorb trailing separators into their preceding segment and guarantee
    # exact coverage of the whole text.
    fixed: list[Segment] = []
    for segment in segments:
        if fixed and fixed[-1].end < segment.start:
            gap_start = fixed[-1].end
            previous = fixed[-1]
            fixed[-1] = Segment(
                start=previous.start,
                end=segment.start,
                kind=previous.kind,
                heading_path=previous.heading_path,
                anchor=previous.anchor,
                level=previous.level,
            )
            del gap_start
        fixed.append(segment)
    if fixed and fixed[-1].end < len(text):
        previous = fixed[-1]
        fixed[-1] = Segment(
            start=previous.start,
            end=len(text),
            kind=previous.kind,
            heading_path=previous.heading_path,
            anchor=previous.anchor,
            level=previous.level,
        )
    return fixed


def _merge_spans(
    spans: Sequence[tuple[int, int, Segment]],
    policy: BlockPolicy,
) -> list[tuple[int, int, Segment]]:
    """Group adjacent spans into paragraph-sized blocks for the leaf layer.

    The canonical partition is preserved exactly — merging only re-groups
    adjacent ranges — while granularity becomes a section chunk instead of a
    line fragment.  A heading always opens a new group (its title stays with
    the section it introduces), groups stop growing at ``min_chars``, and a
    rough token ceiling keeps an embedding's 512-token window covering the
    whole block.
    """
    if policy.min_chars <= 0 or len(spans) < 2:
        return list(spans)
    ceiling = 4 * int(policy.max_tokens)

    merged: list[tuple[int, int, Segment]] = []
    begin, end, segment = spans[0]
    for next_begin, next_end, next_segment in spans[1:]:
        length = end - begin
        opens_section = next_segment.kind == "title"
        same_section = next_segment.heading_path == segment.heading_path
        grows = length < policy.min_chars and length + (next_end - next_begin) <= ceiling
        if not opens_section and same_section and grows:
            end = next_end
            if segment.kind == "title" and next_segment.kind != "title":
                segment = next_segment
            continue
        merged.append((begin, end, segment))
        begin, end, segment = next_begin, next_end, next_segment
    merged.append((begin, end, segment))
    return merged


def _split_oversized(
    text: str,
    start: int,
    end: int,
    policy: BlockPolicy,
) -> list[tuple[int, int]]:
    """Split an oversized span at structural boundaries, keeping exact ranges."""
    span = text[start:end]
    total = policy.count_tokens(span)
    if total <= policy.max_tokens:
        return [(start, end)]
    # Character budget estimated from the span's own token density; avoids
    # re-counting growing candidates (quadratic on huge paragraphs).
    char_budget = max(16, int(len(span) * (policy.max_tokens / max(1, total))))
    pieces: list[tuple[int, int]] = []
    piece_start = 0
    while piece_start < len(span):
        target = min(len(span), piece_start + char_budget)
        if target >= len(span):
            pieces.append((piece_start, len(span)))
            break
        window = span[piece_start:target]
        cut = None
        for match in _BOUNDARY_RE.finditer(window):
            cut = match.end()
        if cut is None or cut < max(1, len(window) // 4):
            cut = len(window)  # no usable boundary: exact character split
        pieces.append((piece_start, piece_start + cut))
        piece_start += cut
    return [(start + begin, start + finish) for begin, finish in pieces]


def build_content_blocks(
    text: str,
    *,
    local_id: str,
    revision: int,
    media_type: str,
    parser: str = "",
    policy: BlockPolicy | None = None,
    page_marks: Sequence[tuple[int, int]] | None = None,
    provenance: dict | None = None,
) -> list[ContentBlock]:
    """Build the canonical contiguous block sequence for one revision.

    ``page_marks`` is a sequence of ``(page_number, start_char)`` boundaries in
    the canonical text; the end of each page is the next mark (or the text
    end).  Only PDF materials should pass marks.
    """
    if not text:
        raise ValueError("canonical text must be non-empty")
    if media_type not in ("pdf", "tex", "md"):
        raise ValueError(f"unsupported media_type {media_type!r}")
    # PDF blocks carry page spans only when the caller verified real page
    # offsets; a PDF without verified marks gets no page fields rather than a
    # guessed range (never fabricate pages, and never invent absence as data).
    policy = policy or BlockPolicy()
    starts = _line_starts(text)

    page_ranges: list[tuple[int, int, int]] = []
    if page_marks:
        ordered = sorted((int(page), int(offset)) for page, offset in page_marks)
        if ordered[0][1] != 0:
            raise ValueError("first page mark must start at char 0")
        for index, (page, offset) in enumerate(ordered):
            end = ordered[index + 1][1] if index + 1 < len(ordered) else len(text)
            page_ranges.append((offset, end, page))

    def page_of(offset: int) -> int | None:
        for begin, finish, page in page_ranges:
            if begin <= offset < finish:
                return page
        if page_ranges:
            return page_ranges[-1][2]
        return None

    spans: list[tuple[int, int, Segment]] = []
    page_bounds = [offset for offset, _finish, _page in page_ranges[1:]] if page_ranges else []
    for segment in segment_text(text):
        if segment.end <= segment.start:
            continue
        pieces: list[tuple[int, int]] = [(segment.start, segment.end)]
        if page_bounds:
            # Split at real page boundaries so every block's page span is exact.
            split: list[tuple[int, int]] = []
            for begin, finish in pieces:
                cut_points = [b for b in page_bounds if begin < b < finish]
                cursor = begin
                for cut in cut_points:
                    split.append((cursor, cut))
                    cursor = cut
                split.append((cursor, finish))
            pieces = split
        for begin, finish in pieces:
            if (
                segment.kind in policy.atomic_kinds
                or policy.count_tokens(text[begin:finish]) <= policy.max_tokens
            ):
                spans.append((begin, finish, segment))
                continue
            for sub_begin, sub_finish in _split_oversized(text, begin, finish, policy):
                spans.append((sub_begin, sub_finish, segment))

    spans = _merge_spans(spans, policy)

    blocks: list[ContentBlock] = []
    for ordinal, (begin, finish, segment) in enumerate(spans):
        piece = text[begin:finish]
        if not piece:
            continue
        import hashlib

        page_start = page_of(begin) if media_type == "pdf" else None
        page_end = page_of(max(begin, finish - 1)) if media_type == "pdf" else None
        line_start = _line_of(starts, begin) if media_type != "pdf" else None
        line_end = _line_of(starts, max(begin, finish - 1)) if media_type != "pdf" else None
        blocks.append(
            ContentBlock(
                block_id=content_block_id(local_id, revision, ordinal),
                local_id=local_id,
                revision=revision,
                ordinal=ordinal,
                text=piece,
                text_hash=hashlib.sha256(piece.encode("utf-8")).hexdigest(),
                char_start=begin,
                char_end=finish,
                page_start=page_start,
                page_end=page_end,
                line_start=line_start,
                line_end=line_end,
                heading_path=segment.heading_path,
                anchor=segment.anchor,
                kind=segment.kind,
                parser=parser,
                provenance=dict(provenance or {}),
            )
        )
    if not blocks:
        raise ValueError("block generation produced no blocks")
    for previous, current in zip(blocks, blocks[1:]):
        if previous.char_end != current.char_start:
            raise AssertionError("block generation violated contiguity")
    if blocks[0].char_start != 0 or blocks[-1].char_end != len(text):
        raise AssertionError("block generation did not cover the canonical text")
    return blocks


def reconstruct(blocks: Iterable[ContentBlock]) -> str:
    """Rebuild canonical text from blocks (must equal the parser output)."""
    return "".join(block.text for block in sorted(blocks, key=lambda item: item.ordinal))
