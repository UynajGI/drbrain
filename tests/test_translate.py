"""Tests for paper translation service."""

from unittest import mock

from drbrain.services.translate import (
    _split_into_chunks,
    translate_paper,
    translate_text,
)

# The LLM client function is imported locally inside translate_text(), so we
# patch at its source module.
ACALL_PATH = "drbrain.extractor.llm_client.acall_text_with_fallback"


# ---------------------------------------------------------------------------
# _split_into_chunks — basic behaviour
# ---------------------------------------------------------------------------


def test_split_into_chunks_short_returns_single_element():
    """Short text (< chunk_size) returns a single-element list."""
    text = "A short paragraph."
    chunks = _split_into_chunks(text, chunk_size=3000)
    assert isinstance(chunks, list)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_into_chunks_respects_chunk_size():
    """Text with multiple paragraphs exceeding chunk_size produces multiple chunks."""
    para = "X" * 800  # each paragraph 800 chars
    text = "\n\n".join([para] * 10)  # 8000 chars total
    chunks = _split_into_chunks(text, chunk_size=3000)
    assert len(chunks) > 1


def test_split_into_chunks_empty_text():
    """Empty text returns empty list."""
    chunks = _split_into_chunks("", chunk_size=3000)
    assert chunks == []


def test_split_into_chunks_single_paragraph():
    """Single paragraph, shorter than chunk_size."""
    text = "Single paragraph with no special blocks."
    chunks = _split_into_chunks(text, chunk_size=3000)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_into_chunks_whitespace_only_paragraphs_filtered():
    """Paragraphs containing only whitespace are filtered out."""
    text = "First.\n\n   \n\nSecond."
    chunks = _split_into_chunks(text, chunk_size=3000)
    assert len(chunks) == 1
    assert "First." in chunks[0]
    assert "Second." in chunks[0]


# ---------------------------------------------------------------------------
# Protected block tests — code / math / images must never be bisected
# ---------------------------------------------------------------------------


def test_code_block_preserved_intact():
    """Code blocks (```...```) are never split across chunks."""
    code = "```python\n" + "print('hello world')\n" * 10 + "```"
    text = f"Before the code.\n\n{code}\n\nAfter the code."
    chunks = _split_into_chunks(text, chunk_size=50)
    # The code block must appear complete in exactly one chunk
    found_in = [i for i, c in enumerate(chunks) if "```python" in c]
    assert len(found_in) == 1, "code block must be in exactly one chunk"
    chunk_idx = found_in[0]
    assert "print('hello world')" in chunks[chunk_idx]
    assert chunks[chunk_idx].count("```") >= 2  # opening + closing


def test_display_math_preserved_intact():
    """Display math ($$...$$) is never split across chunks."""
    math = r"$$\sum_{i=1}^{n} x_i = \frac{n(n+1)}{2}$$"
    text = f"Before.\n\n{math}\n\nAfter."
    chunks = _split_into_chunks(text, chunk_size=50)
    found_in = [i for i, c in enumerate(chunks) if r"\sum" in c]
    assert len(found_in) == 1, "display math must be in exactly one chunk"
    chunk = chunks[found_in[0]]
    assert chunk.count("$$") >= 2  # opening + closing


def test_inline_math_preserved():
    """Inline math ($...$) appears complete in exactly one chunk."""
    text = "Prefix text here. The formula $E=mc^2$ is famous.\n\nSuffix trailing text."
    chunks = _split_into_chunks(text, chunk_size=50)
    found_in = [i for i, c in enumerate(chunks) if "$E=mc^2$" in c]
    assert len(found_in) == 1, "inline math must be in exactly one chunk"


def test_inline_math_with_escaped_dollar():
    """Inline math with escaped dollar signs like $\\$5$ stays intact."""
    text = "Cost is $\\$5$ per unit.\n\nMore text after break."
    chunks = _split_into_chunks(text, chunk_size=50)
    # The escaped-dollar math should appear whole in one chunk
    found_in = [i for i, c in enumerate(chunks) if "$\\$5$" in c]
    assert len(found_in) == 1


def test_image_preserved_intact():
    """Image markup (![...](...)) is never split across chunks."""
    img = "![architecture diagram](images/arch.png)"
    text = f"Before.\n\n{img}\n\nAfter."
    chunks = _split_into_chunks(text, chunk_size=50)
    found_in = [i for i, c in enumerate(chunks) if "images/arch.png" in c]
    assert len(found_in) == 1, "image must be in exactly one chunk"
    assert "![" in chunks[found_in[0]]


def test_multiple_protected_blocks_in_same_paragraph():
    """Multiple protected blocks in a single paragraph all survive intact."""
    text = (
        "Here is `inline code` and $x^2$ and ![img](a.png) all together. "
        "Plus $$\\alpha$$ at the end."
    )
    chunks = _split_into_chunks(text, chunk_size=3000)
    combined = "".join(chunks)
    assert "`inline code`" in combined
    assert "$x^2$" in combined
    assert "![img](a.png)" in combined
    assert "$$\\alpha$$" in combined


# ---------------------------------------------------------------------------
# Hard-split tests
# ---------------------------------------------------------------------------


def test_oversized_paragraph_hard_split():
    """A paragraph > chunk_size gets hard-split at sentence boundaries."""
    sentence = "This is sentence number {} with some extra words to make it longer. "
    para = "".join(sentence.format(i) for i in range(50))
    assert len(para) > 500  # ensure it is oversized for small chunk_size
    chunks = _split_into_chunks(para, chunk_size=200)
    assert len(chunks) > 1, "oversized paragraph must be split"
    for c in chunks:
        assert c.strip()


def test_oversized_paragraph_with_protected_blocks():
    """Hard-split does not cut through a protected block in an oversized paragraph."""
    # A long paragraph with an inline math block in the middle
    prefix = "A. " * 50  # ~150 chars
    suffix = " B." * 50  # ~150 chars
    para = f"{prefix}The value is $E=mc^2$ in this context.{suffix}"
    assert len(para) > 200
    chunks = _split_into_chunks(para, chunk_size=100)
    assert len(chunks) > 1
    # The inline math must be complete in exactly one chunk
    found = [i for i, c in enumerate(chunks) if "$E=mc^2$" in c]
    assert len(found) == 1, "inline math must not be bisected during hard-split"


def test_oversized_paragraph_with_code_block():
    """Code block inside a paragraph is preserved intact even if chunk is large."""
    code = "```\n" + "x = 1\n" * 20 + "```"
    para = f"Preamble sentence. {code} After the code block."
    # The code block is placeholder-protected so the masked paragraph is small
    # enough to stay in one chunk. After restoration the chunk may be large.
    chunks = _split_into_chunks(para, chunk_size=80)
    # Code block must appear complete in exactly one chunk
    found = [i for i, c in enumerate(chunks) if "```" in c]
    assert len(found) == 1, "code fence must not be bisected"
    # The chunk containing the code block should have both fences
    chunk = chunks[found[0]]
    assert chunk.count("```") >= 2


# ---------------------------------------------------------------------------
# Order preservation
# ---------------------------------------------------------------------------


def test_order_preserved_across_chunks():
    """Content order is preserved: first chunk has beginning, last has ending."""
    text = "AAAA\n\nBBBB\n\nCCCC\n\nDDDD"
    chunks = _split_into_chunks(text, chunk_size=8)
    assert len(chunks) >= 2
    assert "AAAA" in chunks[0]
    assert "DDDD" in chunks[-1]


# ---------------------------------------------------------------------------
# Placeholder integrity
# ---------------------------------------------------------------------------


def test_no_placeholder_leaks():
    """After restoration, no raw \\x00PROTECTED_ placeholders remain in output."""
    text = "Text with `code` and $math$.\n\nMore text with ```\nfence\n```."
    chunks = _split_into_chunks(text, chunk_size=3000)
    for c in chunks:
        assert "\x00PROTECTED_" not in c


def test_all_protected_blocks_restored():
    """Every protected block that goes in comes back out somewhere."""
    text = "A $x$ B $$y$$ C ```\ncode\n``` D ![img](a.png) E $z$ F"
    chunks = _split_into_chunks(text, chunk_size=3000)
    combined = "".join(chunks)
    assert "$x$" in combined
    assert "$$y$$" in combined
    assert "```" in combined
    assert "![img](a.png)" in combined
    assert "$z$" in combined


def test_placeholder_display_math_before_inline_math():
    """Display math $$...$$ is not incorrectly parsed as two inline $ tokens."""
    text = "Before. $$E=mc^2$$ After."
    chunks = _split_into_chunks(text, chunk_size=3000)
    combined = "".join(chunks)
    # The display math must appear intact, not as two separate $ blocks
    assert "$$E=mc^2$$" in combined


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
    # Need >CHUNK_SIZE chars to produce 2+ chunks
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
    # Need >CHUNK_SIZE chars to produce 2+ chunks
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
