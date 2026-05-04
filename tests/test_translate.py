"""Tests for paper translation service."""

from unittest import mock

import pytest

from drbrain.services.translate import (
    _split_into_chunks,
    _subdivide_chunk_for_retry,
    _translate_chunk,
    _translate_chunk_resilient,
    _translate_chunk_with_retry,
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


# ---------------------------------------------------------------------------
# detect_language tests
# ---------------------------------------------------------------------------


class TestDetectLanguage:
    """Heuristic language detection without external dependencies."""

    def test_english_text(self):
        from drbrain.services.translate import detect_language

        assert (
            detect_language(
                "The quick brown fox jumps over the lazy dog. "
                "This is a test of the emergency broadcast system."
            )
            == "en"
        )

    def test_chinese_text(self):
        from drbrain.services.translate import detect_language

        assert (
            detect_language("本文提出了一种新型湍流模型，用于预测高雷诺数流动中的湍流结构。")
            == "zh"
        )

    def test_japanese_kana(self):
        from drbrain.services.translate import detect_language

        assert detect_language("これはテストです") == "ja"

    def test_korean_text(self):
        from drbrain.services.translate import detect_language

        assert detect_language("이 논문은 새로운 기계 학습 방법을 제안합니다") == "ko"

    def test_german_text(self):
        from drbrain.services.translate import detect_language

        # Contains >=2 German stopwords uniquely
        assert (
            detect_language("Der Prozess der Wissensextraktion ist ein wichtiger Teil des Systems.")
            == "de"
        )

    def test_french_text(self):
        from drbrain.services.translate import detect_language

        assert (
            detect_language("Le modele de langage est une approche pour le traitement des donnees.")
            == "fr"
        )

    def test_spanish_text(self):
        from drbrain.services.translate import detect_language

        assert (
            detect_language("El modelo de lenguaje es una herramienta para el analisis de datos.")
            == "es"
        )

    def test_empty_defaults_to_english(self):
        from drbrain.services.translate import detect_language

        assert detect_language("") == "en"

    def test_math_only_defaults_to_english(self):
        from drbrain.services.translate import detect_language

        assert detect_language("$$E=mc^2$$") == "en"

    def test_code_blocks_stripped(self):
        from drbrain.services.translate import detect_language

        text = (
            "This is an English document about machine learning.\n\n"
            "```python\n"
            "# これは日本語のコメントです\n"
            "x = 1 + 2\n"
            "```\n\n"
            "The results show significant improvement over the baseline."
        )
        assert detect_language(text) == "en"


# ---------------------------------------------------------------------------
# validate_lang tests
# ---------------------------------------------------------------------------


class TestValidateLang:
    """Language code validation and normalization."""

    def test_normalizes_case_and_whitespace(self):
        from drbrain.services.translate import validate_lang

        assert validate_lang(" ZH ") == "zh"
        assert validate_lang("En") == "en"
        assert validate_lang("  ja  ") == "ja"

    def test_rejects_non_string(self):
        from drbrain.services.translate import validate_lang

        with pytest.raises(ValueError, match="str"):
            validate_lang(None)
        with pytest.raises(ValueError, match="str"):
            validate_lang(123)

    def test_rejects_invalid_pattern(self):
        from drbrain.services.translate import validate_lang

        with pytest.raises(ValueError):
            validate_lang("zh-cn")
        with pytest.raises(ValueError):
            validate_lang("")
        with pytest.raises(ValueError):
            validate_lang("e" * 10)


# ---------------------------------------------------------------------------
# _translate_chunk tests
# ---------------------------------------------------------------------------


class TestTranslateChunk:
    """Synchronous single-chunk translation via LLM."""

    def test_returns_translated_text(self, monkeypatch):
        """_translate_chunk returns the LLM result stripped."""

        async def fake_acall(prompt, models, *, system_prompt="", max_tokens=1024):
            return "  translated text here  "

        monkeypatch.setattr(
            "drbrain.extractor.llm_client.acall_text_with_fallback",
            fake_acall,
        )
        result = _translate_chunk("hello world", "zh", [{"provider": "test", "model": "x"}])
        assert result == "translated text here"

    def test_llm_returns_none_raises(self, monkeypatch):
        """When the LLM returns None, _translate_chunk raises RuntimeError."""

        async def fake_acall(prompt, models, *, system_prompt="", max_tokens=1024):
            return None

        monkeypatch.setattr(
            "drbrain.extractor.llm_client.acall_text_with_fallback",
            fake_acall,
        )
        with pytest.raises(RuntimeError, match="Translation failed"):
            _translate_chunk("hello", "zh", [{"provider": "test", "model": "x"}])

    def test_passes_lang_name_in_prompt(self, monkeypatch):
        """The prompt contains the human-readable language name."""
        prompts: list[str] = []

        async def fake_acall(prompt, models, *, system_prompt="", max_tokens=1024):
            prompts.append(prompt)
            return "ok"

        monkeypatch.setattr(
            "drbrain.extractor.llm_client.acall_text_with_fallback",
            fake_acall,
        )
        _translate_chunk("test text", "ja", [{"provider": "test", "model": "x"}])
        assert len(prompts) == 1
        assert "Japanese" in prompts[0]


# ---------------------------------------------------------------------------
# _translate_chunk_with_retry tests
# ---------------------------------------------------------------------------


class TestTranslateChunkWithRetry:
    """Exponential-backoff retry wrapper around _translate_chunk."""

    def test_retry_succeeds_on_first_attempt(self, monkeypatch):
        """Returns (result, 1) when the first call succeeds."""

        def mock_translate(text, target_lang, models):
            return f"T:{text}"

        monkeypatch.setattr("drbrain.services.translate._translate_chunk", mock_translate)
        result, attempts = _translate_chunk_with_retry(
            "hello", "zh", [], max_attempts=5, backoff_base=1.0
        )
        assert result == "T:hello"
        assert attempts == 1

    def test_retry_after_timeout(self, monkeypatch):
        """Succeeds on the third attempt after two failures."""
        call_count = 0

        def mock_translate(text, target_lang, models):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("timeout")
            return f"T:{text}"

        monkeypatch.setattr("drbrain.services.translate._translate_chunk", mock_translate)
        # Neutralise sleep so the test runs instantly
        monkeypatch.setattr("time.sleep", lambda _s: None)

        result, attempts = _translate_chunk_with_retry(
            "hello", "zh", [], max_attempts=5, backoff_base=1.0
        )
        assert result == "T:hello"
        assert attempts == 3

    def test_exponential_backoff_delays(self, monkeypatch):
        """Verifies the sleep calls are [1.0, 2.0, 4.0, 8.0] with base=1.0."""
        call_count = 0

        def mock_translate(text, target_lang, models):
            nonlocal call_count
            call_count += 1
            if call_count < 5:
                raise RuntimeError("timeout")
            return f"T:{text}"

        sleeps: list[float] = []
        monkeypatch.setattr("drbrain.services.translate._translate_chunk", mock_translate)
        monkeypatch.setattr("time.sleep", sleeps.append)

        _translate_chunk_with_retry("hello", "zh", [], max_attempts=5, backoff_base=1.0)
        assert sleeps == [1.0, 2.0, 4.0, 8.0]

    def test_max_attempts_exhausted_raises(self, monkeypatch):
        """Raises the last exception when all retries are exhausted."""

        def mock_translate(text, target_lang, models):
            raise RuntimeError("always fails")

        monkeypatch.setattr("drbrain.services.translate._translate_chunk", mock_translate)
        monkeypatch.setattr("time.sleep", lambda _s: None)

        with pytest.raises(RuntimeError, match="always fails"):
            _translate_chunk_with_retry("hello", "zh", [], max_attempts=3, backoff_base=1.0)


# ---------------------------------------------------------------------------
# _subdivide_chunk_for_retry tests
# ---------------------------------------------------------------------------


class TestSubdivideChunkForRetry:
    """Chunk subdivision helper for resilient retry."""

    def test_small_text_returns_single_element(self):
        """Returns [text] when target >= len(text)."""
        # Single-char text: len=1, target=max(1, min(chunk_size, 0))=1, 1>=1 → [text]
        result = _subdivide_chunk_for_retry("a", chunk_size=100)
        assert result == ["a"]

    def test_target_size_is_half_of_text_length(self):
        """The target chunk size is capped at len(text)//2."""
        para = "X" * 800
        text = "\n\n".join([para] * 6)  # ~4800 chars
        # chunk_size=3000 but target = min(3000, 4800//2) = 2400
        result = _subdivide_chunk_for_retry(text, chunk_size=3000)
        assert len(result) >= 2
        # No chunk should exceed target substantially (allow large-protected-block overflow)
        # Target = max(1, min(3000, ~4800 // 2)) = 2400
        target = max(1, min(3000, len(text) // 2))
        assert all(len(c) <= target * 2 for c in result)

    def test_subdivides_long_text(self):
        """Splits long text into multiple chunks."""
        para = "Y" * 500
        text = "\n\n".join([para] * 10)  # ~5000 chars
        result = _subdivide_chunk_for_retry(text, chunk_size=2000)
        assert len(result) > 1

    def test_target_clamped_to_min_one(self):
        """Target size is at least 1 even for tiny chunk_size."""
        result = _subdivide_chunk_for_retry("abcdef", chunk_size=0)
        # len("abcdef") = 6, target = max(1, min(0, 3)) = max(1, 0) = 1
        # 1 < 6, so it subdivides — hard_split into 6 chars
        assert len(result) >= 1
        assert "".join(result) == "abcdef"


# ---------------------------------------------------------------------------
# _translate_chunk_resilient tests
# ---------------------------------------------------------------------------


class TestTranslateChunkResilient:
    """Timeout-resilient translation with chunk subdivision."""

    def test_returns_on_first_retry_success(self, monkeypatch):
        """Returns immediately when _translate_chunk_with_retry succeeds."""

        def mock_retry(text, target_lang, models, *, max_attempts=5, backoff_base=1.0):
            return f"[OK:{len(text)}]", 1

        monkeypatch.setattr("drbrain.services.translate._translate_chunk_with_retry", mock_retry)
        result, attempts = _translate_chunk_resilient(
            "short text", "zh", [], chunk_size=3000, max_attempts=5
        )
        assert result == "[OK:10]"
        assert attempts == 1

    def test_subdivides_on_timeout(self, monkeypatch):
        """When the full chunk times out it is subdivided and each subchunk
        is translated independently."""
        retry_calls: list[tuple[int, int]] = []

        def mock_retry(text, target_lang, models, *, max_attempts=5, backoff_base=1.0):
            retry_calls.append((len(text), max_attempts))
            # Fail on the very first call (full chunk), succeed on subchunks
            if len(retry_calls) == 1:
                raise RuntimeError("timeout")
            return f"[SUB:{len(text)}]", 1

        monkeypatch.setattr("drbrain.services.translate._translate_chunk_with_retry", mock_retry)

        # Text long enough to subdivide with small chunk_size
        text = "AAA\n\nBBB\n\nCCC\n\nDDD"
        result, attempts = _translate_chunk_resilient(
            text, "zh", [], chunk_size=5, max_attempts=2, backoff_base=0.01
        )
        # The first call (full text) timed out → subdivided → subchunks succeeded
        assert len(retry_calls) > 1, "should have subdivided and retried subchunks"
        assert "[SUB:" in result
        assert attempts >= 1

    def test_no_subdivision_when_cannot_subdivide(self, monkeypatch):
        """When text cannot be subdivided (too short), raises the exception."""

        def mock_retry(text, target_lang, models, *, max_attempts=5, backoff_base=1.0):
            raise RuntimeError("persistent failure")

        monkeypatch.setattr("drbrain.services.translate._translate_chunk_with_retry", mock_retry)
        with pytest.raises(RuntimeError, match="persistent failure"):
            _translate_chunk_resilient("abc", "zh", [], chunk_size=3000, max_attempts=2)

    def test_recursive_subdivision(self, monkeypatch):
        """When even subchunks timeout they are subdivided again."""
        retry_calls: list[tuple[int, int]] = []

        def mock_retry(text, target_lang, models, *, max_attempts=5, backoff_base=1.0):
            retry_calls.append((len(text), max_attempts))
            # Fail first call (full chunk) and fail any chunk > 5 chars
            if len(retry_calls) == 1 or len(text) > 5:
                raise RuntimeError(f"timeout on len={len(text)}")
            return f"[OK:{len(text)}]", 1

        monkeypatch.setattr("drbrain.services.translate._translate_chunk_with_retry", mock_retry)
        # Text that will subdivide, and subchunks may also subdivide
        para = "Z" * 300
        text = "\n\n".join([para] * 8)  # ~2400 chars
        result, attempts = _translate_chunk_resilient(
            text, "zh", [], chunk_size=200, max_attempts=2, backoff_base=0.01
        )
        assert "[OK:" in result
        assert attempts >= 1
