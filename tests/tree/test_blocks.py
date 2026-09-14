"""T09: boundary-faithful block generation and exact reconstruction."""

from __future__ import annotations

import hashlib

import pytest

from drbrain.tree.blocks import (
    BlockPolicy,
    build_content_blocks,
    reconstruct,
    segment_text,
)

FIXTURE_MD = (
    "# Title\n"
    "\n"
    "Pre-heading abstract paragraph with an equation $x=1$.\n"
    "\n"
    "## Methods\n"
    "\n"
    "First paragraph.\n"
    "Second line of the same paragraph.\n"
    "\n"
    "$$\n"
    "E = mc^2\n"
    "$$\n"
    "\n"
    "| a | b |\n"
    "| - | - |\n"
    "| 1 | 2 |\n"
    "\n"
    "```python\n"
    "print('# not a heading')\n"
    "```\n"
    "\n"
    "### Appendix\n"
    "\n"
    "Extra material without trailing newline."
)


def _blocks(text: str = FIXTURE_MD, **kw):
    return build_content_blocks(
        text,
        local_id="p1",
        revision=1,
        media_type=kw.pop("media_type", "md"),
        parser="test",
        **kw,
    )


class TestExactReconstruction:
    def test_segments_cover_text_exactly(self):
        segments = segment_text(FIXTURE_MD)
        assert segments[0].start == 0
        assert segments[-1].end == len(FIXTURE_MD)
        for previous, current in zip(segments, segments[1:]):
            assert previous.end == current.start
        assert "".join(FIXTURE_MD[s.start : s.end] for s in segments) == FIXTURE_MD

    def test_blocks_reconstruct_verbatim(self):
        assert reconstruct(_blocks()) == FIXTURE_MD

    def test_formula_table_code_and_titles_are_classified(self):
        blocks = _blocks()
        kinds = {block.kind for block in blocks}
        assert {"title", "paragraph", "formula", "table", "code"} <= kinds
        formula = next(block for block in blocks if block.kind == "formula")
        assert "E = mc^2" in formula.text
        table = next(block for block in blocks if block.kind == "table")
        assert "| a | b |" in table.text
        code = next(block for block in blocks if block.kind == "code")
        assert "print('# not a heading')" in code.text
        # The fenced '#' must not open a heading.
        assert all("not a heading" not in " > ".join(block.heading_path) for block in blocks)

    def test_heading_paths_and_anchors(self):
        blocks = _blocks()
        methods = next(block for block in blocks if block.anchor == "Methods")
        assert methods.heading_path == ("Title", "Methods")
        appendix = next(block for block in blocks if block.anchor == "Appendix")
        assert appendix.heading_path == ("Title", "Methods", "Appendix")
        # Appendix body inherits the appendix path.
        tail = blocks[-1]
        assert tail.heading_path[-1] == "Appendix"

    def test_pre_heading_text_is_preserved(self):
        blocks = _blocks()
        assert "Pre-heading abstract" in blocks[1].text
        assert blocks[1].heading_path == ("Title",)

    def test_blank_separators_are_absorbed_not_dropped(self):
        blocks = _blocks()
        assert any(block.text.endswith("\n\n") for block in blocks)
        assert reconstruct(blocks) == FIXTURE_MD


class TestLocators:
    def test_line_spans_for_text_materials(self):
        blocks = _blocks()
        assert blocks[0].line_start == 1
        assert all(block.line_start is None or block.line_start >= 1 for block in blocks)
        assert all(block.page_start is None for block in blocks)

    def test_pdf_pages_require_marks_and_map_ranges(self):
        text = "Page one text.\nPage two text.\n"
        with pytest.raises(ValueError, match="page marks"):
            build_content_blocks(text, local_id="p", revision=1, media_type="pdf")
        split = text.index("Page two")
        blocks = build_content_blocks(
            text,
            local_id="p",
            revision=1,
            media_type="pdf",
            page_marks=[(1, 0), (2, split)],
        )
        assert reconstruct(blocks) == text
        assert blocks[0].page_start == 1 and blocks[0].page_end == 1
        assert blocks[-1].page_end == 2
        assert all(block.line_start is None for block in blocks)

    def test_tex_lines_are_recorded(self):
        tex = "\\section{Intro}\n\nBody line.\n"
        blocks = _blocks(tex, media_type="tex")
        assert reconstruct(blocks) == tex
        assert blocks[0].line_start == 1
        assert any(block.line_start and block.line_start >= 2 for block in blocks)
        assert all(block.page_start is None for block in blocks)


class TestOversizedSpans:
    def test_long_paragraph_is_split_without_losing_text(self):
        paragraph = "Sentence number one stays intact. " * 400
        text = "# H\n\n" + paragraph
        policy = BlockPolicy(max_tokens=50, count_tokens=lambda value: max(1, len(value) // 4))
        blocks = build_content_blocks(
            text, local_id="p", revision=1, media_type="md", policy=policy
        )
        assert len(blocks) > 3
        assert reconstruct(blocks) == text
        assert all(policy.count_tokens(block.text) <= policy.max_tokens * 1.5 for block in blocks)

    def test_boundaryless_blob_still_splits_exactly(self):
        blob = "x" * 5000
        policy = BlockPolicy(max_tokens=100, count_tokens=lambda value: max(1, len(value) // 4))
        blocks = build_content_blocks(
            blob, local_id="p", revision=1, media_type="md", policy=policy
        )
        assert reconstruct(blocks) == blob
        assert len(blocks) > 1

    def test_single_formula_over_budget_is_not_dropped(self):
        formula = "$$\n" + "a + b = c\n" * 50 + "$$\n"
        policy = BlockPolicy(max_tokens=16, count_tokens=lambda value: max(1, len(value) // 4))
        blocks = build_content_blocks(
            formula, local_id="p", revision=1, media_type="md", policy=policy
        )
        assert reconstruct(blocks) == formula


class TestBlockIdentity:
    def test_block_hashes_and_ids_match_contracts(self):
        blocks = _blocks()
        for block in blocks:
            assert block.text_hash == hashlib.sha256(block.text.encode()).hexdigest()
            assert block.block_id.startswith("cb-")
        assert [block.ordinal for block in blocks] == list(range(len(blocks)))

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_content_blocks("", local_id="p", revision=1, media_type="md")

    def test_unknown_media_type_rejected(self):
        with pytest.raises(ValueError, match="media_type"):
            build_content_blocks("x", local_id="p", revision=1, media_type="docx")
