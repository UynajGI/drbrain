"""Tests for paper translation service."""

from unittest import mock

from drbrain.services.translate import _chunk_text, translate_paper, translate_text

# The LLM client function is imported locally inside translate_text(), so we
# patch at its source module.
ACALL_PATH = "drbrain.extractor.llm_client.acall_text_with_fallback"


# ---------------------------------------------------------------------------
# _chunk_text tests
# ---------------------------------------------------------------------------


def test_chunk_text_short_returns_single_element():
    """Short text (< CHUNK_SIZE) returns a single-element list."""
    text = "A short paragraph."
    chunks = _chunk_text(text, max_chars=3000)
    assert isinstance(chunks, list)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_splits_at_markdown_sections():
    """Text with ## headings splits into multiple chunks when content is long."""
    text = "Preamble.\n\n## Introduction\n" + "A" * 200 + "\n\n## Methods\n" + "B" * 200
    chunks = _chunk_text(text, max_chars=100)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.strip()


def test_chunk_text_long_text_no_sections_splits_at_paragraphs():
    """Long text without markdown sections splits at paragraph boundaries."""
    para = "X" * 800  # each paragraph 800 chars
    text = "\n\n".join([para] * 10)  # 8000 chars total
    chunks = _chunk_text(text, max_chars=3000)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 3000 + len("\n\n")  # allow small margin for joining


def test_chunk_text_empty_text():
    """Empty text returns empty list."""
    chunks = _chunk_text("", max_chars=3000)
    assert chunks == []


def test_chunk_text_no_headings_single_paragraph():
    """Single paragraph without headings, shorter than max_chars."""
    text = "Single paragraph with no headings."
    chunks = _chunk_text(text, max_chars=3000)
    assert len(chunks) == 1
    assert chunks[0] == "Single paragraph with no headings."


def test_chunk_text_multiple_sections_stay_within_limit():
    """Each resulting chunk from sectioned text should be reasonably sized."""
    heading = "## H\n"
    content = "X" * 800
    sections = [f"{heading}{content}" for _ in range(10)]
    text = "\n\n".join(sections)
    chunks = _chunk_text(text, max_chars=1200)
    # With large sections, we get multiple chunks
    assert len(chunks) >= 2
    for c in chunks:
        # Each chunk should be non-empty and not excessively large
        assert c.strip()
        # Allow generous margin since section joining can create bigger chunks
        assert len(c) <= 5000


def test_chunk_text_section_boundary_with_h3():
    """### (h3) headings are also recognized as section boundaries."""
    text = "Intro.\n\n### Results\n" + "R" * 200 + "\n\n### Discussion\n" + "D" * 200
    chunks = _chunk_text(text, max_chars=100)
    assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# translate_text tests (async)
# ---------------------------------------------------------------------------


async def test_translate_text_returns_translated_content():
    """translate_text calls LLM and returns translated content."""
    text = "First paragraph.\n\n## Section\nSection text."

    async def fake_acall(*, prompt, models, system_prompt, max_tokens):
        return f"[TRANS] {prompt[:20]}..."

    with mock.patch(ACALL_PATH, side_effect=fake_acall):
        result = await translate_text(text, models=[{"provider": "openai", "model": "gpt-4"}])
        assert result is not None
        assert "[TRANS]" in result


async def test_translate_text_llm_fails_returns_none():
    """If LLM returns None for any chunk, translate_text returns None."""
    # Need >3000 chars to produce 2+ chunks with default CHUNK_SIZE
    text = "Preamble.\n\n## Section A\n" + "X" * 2500 + "\n\n## Section B\n" + "Y" * 2500

    call_count = 0

    async def fake_acall(*, prompt, models, system_prompt, max_tokens):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            return None  # fail on second chunk
        return f"[TRANS] {prompt[:20]}"

    with mock.patch(ACALL_PATH, side_effect=fake_acall):
        result = await translate_text(text, models=[{"provider": "openai", "model": "gpt-4"}])
        assert result is None


async def test_translate_text_passes_correct_params():
    """translate_text passes the correct system_prompt, max_tokens to the LLM."""
    text = "Just some text to translate."

    async def fake_acall(*, prompt, models, system_prompt, max_tokens):
        return f"SYS:{system_prompt[:30]} | MAX:{max_tokens}"

    with mock.patch(ACALL_PATH, side_effect=fake_acall):
        result = await translate_text(
            text,
            models=[{"provider": "openai", "model": "gpt-4"}],
            target_lang="Japanese",
            source_lang="English",
        )
        assert result is not None
        assert "SYS:" in result
        assert "MAX:4096" in result


# ---------------------------------------------------------------------------
# translate_paper tests
# ---------------------------------------------------------------------------


def test_translate_paper_file_not_found(tmp_path):
    """translate_paper returns None when the input file does not exist."""
    nonexistent = tmp_path / "nonexistent.md"
    result = translate_paper(
        md_path=nonexistent,
        models=[{"provider": "openai", "model": "gpt-4"}],
    )
    assert result is None


def test_translate_paper_creates_output_file(tmp_path):
    """translate_paper reads source, translates, and writes output file."""
    md_path = tmp_path / "raw.md"
    md_path.write_text("Hello world content.", encoding="utf-8")

    async def fake_acall(*, prompt, models, system_prompt, max_tokens):
        return f"[TRANSLATED] {prompt}"

    with mock.patch(ACALL_PATH, side_effect=fake_acall):
        result = translate_paper(
            md_path=md_path,
            models=[{"provider": "openai", "model": "gpt-4"}],
            target_lang="Chinese",
        )

    assert result is not None
    assert result.exists()
    content = result.read_text(encoding="utf-8")
    assert "[TRANSLATED]" in content


def test_translate_paper_custom_output_path(tmp_path):
    """translate_paper writes to a custom output_path when provided."""
    md_path = tmp_path / "raw.md"
    md_path.write_text("Translate me.", encoding="utf-8")
    custom_out = tmp_path / "custom_output.md"

    async def fake_acall(*, prompt, models, system_prompt, max_tokens):
        return f"[DONE] {prompt}"

    with mock.patch(ACALL_PATH, side_effect=fake_acall):
        result = translate_paper(
            md_path=md_path,
            models=[{"provider": "openai", "model": "gpt-4"}],
            output_path=custom_out,
        )

    assert result == custom_out
    assert custom_out.exists()
    assert "[DONE]" in custom_out.read_text(encoding="utf-8")


def test_translate_paper_llm_fails_returns_none(tmp_path):
    """If translation fails on any chunk, translate_paper returns None."""
    md_path = tmp_path / "raw.md"
    # Need >3000 chars to produce 2+ chunks with default CHUNK_SIZE
    md_path.write_text(
        "Preamble.\n\n## First\n" + "A" * 2500 + "\n\n## Second\n" + "B" * 2500,
        encoding="utf-8",
    )

    call_count = 0

    async def fake_acall(*, prompt, models, system_prompt, max_tokens):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            return None
        return f"[OK] {prompt}"

    with mock.patch(ACALL_PATH, side_effect=fake_acall):
        result = translate_paper(
            md_path=md_path,
            models=[{"provider": "openai", "model": "gpt-4"}],
        )

    assert result is None
