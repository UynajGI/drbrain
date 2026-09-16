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
    # Segmentation-contract tests pin the raw structural split; paragraph
    # merging (the production default) is exercised in TestParagraphMerge.
    kw.setdefault("policy", BlockPolicy(min_chars=0))
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

    def test_pdf_pages_map_only_from_verified_marks(self):
        text = "Page one text.\nPage two text.\n"
        # A PDF without verified marks gets no page fields (never fabricated).
        unmarked = build_content_blocks(
            text, local_id="p", revision=1, media_type="pdf", policy=BlockPolicy(min_chars=0)
        )
        assert all(block.page_start is None and block.page_end is None for block in unmarked)
        split = text.index("Page two")
        blocks = build_content_blocks(
            text,
            local_id="p",
            revision=1,
            media_type="pdf",
            page_marks=[(1, 0), (2, split)],
            policy=BlockPolicy(min_chars=0),
        )
        assert reconstruct(blocks) == text
        assert blocks[0].page_start == 1 and blocks[0].page_end == 1
        assert blocks[-1].page_end == 2
        assert all(block.line_start is None for block in blocks)

    def test_merged_pdf_blocks_span_the_pages_they_cover(self):
        text = "Page one text.\nPage two text.\n"
        split = text.index("Page two")
        blocks = build_content_blocks(
            text,
            local_id="p",
            revision=1,
            media_type="pdf",
            page_marks=[(1, 0), (2, split)],
        )
        assert len(blocks) == 1
        assert blocks[0].page_start == 1 and blocks[0].page_end == 2

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


MERGE_MD = (
    "# Section One\n"
    "\n"
    "Short intro.\n"
    "\n"
    "Another short line.\n"
    "\n"
    "### Sub\n"
    "\n"
    "Third short line under the subheading.\n"
    "\n"
    "# Section Two\n"
    "\n"
    "Different section body.\n"
)


class TestParagraphMerge:
    """Leaves carry a paragraph-sized chunk, not a structural line fragment."""

    def test_short_lines_merge_into_one_block_per_section(self):
        blocks = build_content_blocks(
            MERGE_MD, local_id="p1", revision=1, media_type="md", parser="test"
        )
        assert reconstruct(blocks) == MERGE_MD
        assert len(blocks) == 3
        # A heading stays glued to the content it introduces, and the merged
        # block keeps the section's kind rather than the heading's.
        assert blocks[0].kind == "paragraph"
        assert blocks[0].heading_path == ("Section One",)
        assert blocks[0].text.startswith("# Section One")
        assert "Another short line." in blocks[0].text
        assert blocks[1].heading_path == ("Section One", "Sub")
        assert blocks[2].heading_path == ("Section Two",)

    def test_a_new_heading_always_opens_a_new_block(self):
        blocks = build_content_blocks(
            MERGE_MD, local_id="p1", revision=1, media_type="md", parser="test"
        )
        assert blocks[1].text.startswith("### Sub")
        assert blocks[2].text.startswith("# Section Two")

    def test_min_chars_zero_keeps_the_raw_segmentation(self):
        merged = build_content_blocks(
            MERGE_MD, local_id="p1", revision=1, media_type="md", parser="test"
        )
        raw = build_content_blocks(
            MERGE_MD,
            local_id="p1",
            revision=1,
            media_type="md",
            parser="test",
            policy=BlockPolicy(min_chars=0),
        )
        assert len(raw) > len(merged)
        assert reconstruct(raw) == reconstruct(merged) == MERGE_MD

    def test_merge_never_crosses_a_section_boundary(self):
        blocks = build_content_blocks(
            MERGE_MD, local_id="p1", revision=1, media_type="md", parser="test"
        )
        paths = [block.heading_path for block in blocks]
        assert paths == [("Section One",), ("Section One", "Sub"), ("Section Two",)]

    def test_merge_is_contiguous_and_hash_consistent(self):
        blocks = build_content_blocks(
            MERGE_MD, local_id="p1", revision=1, media_type="md", parser="test"
        )
        for previous, current in zip(blocks, blocks[1:]):
            assert previous.char_end == current.char_start
        assert blocks[0].char_start == 0
        assert blocks[-1].char_end == len(MERGE_MD)
        for block in blocks:
            assert block.text_hash == hashlib.sha256(block.text.encode()).hexdigest()

    def test_merge_respects_the_token_ceiling_when_characters_underestimate(self):
        # One token per character (CJK-like density): a 4-chars-per-token
        # ceiling let two 300-character paragraphs merge past ``max_tokens``.
        paragraph = "字" * 300
        text = "# 标题\n\n" + paragraph + "\n\n" + paragraph
        policy = BlockPolicy(max_tokens=512, count_tokens=lambda value: max(1, len(value)))
        blocks = build_content_blocks(
            text, local_id="p", revision=1, media_type="md", parser="test", policy=policy
        )
        assert reconstruct(blocks) == text
        # the second 300-token paragraph no longer fits the 512-token window
        assert len(blocks) == 2
        assert all(policy.count_tokens(block.text) <= policy.max_tokens for block in blocks)
