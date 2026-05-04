"""Paper translation via LLM — chunks sections and translates with fallback chain."""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

CHUNK_SIZE = 3000  # chars per translation chunk

# Combined pattern for protected blocks (code fences, display/inline math, images).
# Order matters: display math ($$...$$) must be matched before inline math ($...$)
# to avoid consuming the opening $$ as two inline $ tokens.
_PROTECTED_RE = re.compile(
    r"(```[\s\S]*?```|\$\$[\s\S]*?\$\$|(?<!\$)\$(?!\$)(?:[^$\\]|\\.)+\$(?!\$)|!\[.*?\]\(.*?\))",
    re.MULTILINE,
)

_PLACEHOLDER_FMT = "\x00PROTECTED_{}\x00"
_PLACEHOLDER_RE = re.compile(r"\x00PROTECTED_(\d+)\x00")

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

# Script coverage regexes
_CJK_RE = re.compile(r"[一-鿿㐀-䶿]")
_HANGUL_RE = re.compile(r"[가-힯]")
_KANA_RE = re.compile(r"[぀-ゟ゠-ヿ]")
_ALPHA_RE = re.compile(r"[^\W\d_]")

_LANG_PATTERN_RE = re.compile(r"^[a-z]{2,5}$")

_LATIN_STOPWORDS: dict[str, set[str]] = {
    "en": {"the", "and", "of", "to", "in", "for", "with", "is", "that", "this"},
    "de": {"der", "die", "das", "und", "ist", "mit", "eine", "ein", "den", "von"},
    "fr": {"le", "la", "les", "de", "des", "et", "une", "un", "dans", "pour"},
    "es": {"el", "la", "los", "las", "de", "del", "y", "una", "un", "para"},
}

_LANG_NAMES: dict[str, str] = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
}

_TERMINOLOGY_RULES = {
    "zh": "- 对于专业术语，在首次出现时用「英文 (中文翻译)」格式",
    "ja": "- 専門用語は初出時に「英語 (日本語訳)」の形式で記載すること",
    "ko": "- 전문 용어는 처음 등장할 때 「영어 (한국어 번역)」 형식을 사용",
}


def detect_language(text: str) -> str:
    """Detect language of *text* using heuristics (no external dependencies).

    Strips code blocks, LaTeX math, and images before analysis to avoid
    false positives from non-linguistic content.
    """
    if not text:
        return "en"

    # Sample first 2000 chars and strip protected blocks (code, math, images)
    sample = text[:2000]
    cleaned = _PROTECTED_RE.sub("", sample)

    alpha_chars = len(_ALPHA_RE.findall(cleaned))
    if alpha_chars == 0:
        return "en"

    cjk_count = len(_CJK_RE.findall(cleaned))
    hangul_count = len(_HANGUL_RE.findall(cleaned))
    kana_count = len(_KANA_RE.findall(cleaned))

    # CJK-dominant text: check for kana to disambiguate zh vs ja
    if cjk_count / alpha_chars > 0.15:
        if kana_count / alpha_chars > 0.10:
            return "ja"
        return "zh"

    # Hangul-dominant text
    if hangul_count / alpha_chars > 0.15:
        return "ko"

    # Pure-kana text (no CJK)
    if kana_count / alpha_chars > 0.10:
        return "ja"

    # Latin-script stopword scoring
    words = re.findall(r"[a-zA-Z]+", cleaned.lower())
    if words:
        scores: dict[str, int] = {}
        for lang, stopwords in _LATIN_STOPWORDS.items():
            scores[lang] = sum(1 for w in words if w in stopwords)
        max_score = max(scores.values())
        if max_score >= 2:
            top_langs = [lang for lang, s in scores.items() if s == max_score]
            if len(top_langs) == 1:
                return top_langs[0]

    return "en"


def validate_lang(lang: str) -> str:
    """Validate and normalize a language code.

    Args:
        lang: Raw language code string.

    Returns:
        Normalized lowercase code (e.g. ``"zh"``).

    Raises:
        ValueError: If *lang* is not a string or does not match ``^[a-z]{2,5}$``.
    """
    if not isinstance(lang, str):
        raise ValueError(f"expected str, got {type(lang).__name__}")
    lang = lang.strip().lower()
    if not _LANG_PATTERN_RE.match(lang):
        raise ValueError(f"invalid language code: {lang!r}")
    return lang


def _build_translate_prompt(text: str, target_lang: str, lang_name: str) -> str:
    """Build a translation prompt for an academic paper chunk."""
    header = (
        f"翻译以下学术论文段落至{lang_name}。\n\n"
        "重要事项：\n"
        "- 保留所有 markdown 格式\n"
        "- 保留 LaTeX 公式不翻译\n"
        "- 保留代码块不翻译\n"
        "- 保留图片引用不翻译\n"
        "- 保留作者姓名和引用格式"
    )
    parts = [header]
    rule = _TERMINOLOGY_RULES.get(target_lang)
    if rule:
        parts.append(rule)
    parts.append(f"- 只返回翻译文本，不要任何解释\n\n原文：\n{text}")
    return "\n".join(parts)


def _adjust_for_placeholder(text: str, cut: int) -> int:
    """Move cut point outside any placeholder span it would bisect."""
    last_start = text.rfind("\x00PROTECTED_", 0, cut)
    if last_start == -1:
        return cut
    # Find the closing NUL of that placeholder
    end = text.find("\x00", last_start + 1)
    if end == -1:
        return cut
    end += 1  # include the closing NUL
    if cut < end:
        # cut falls inside the placeholder — move past it
        return end
    return cut


def _hard_split(text: str, chunk_size: int) -> list[str]:
    """Split an oversized text block into pieces targeting chunk_size.

    Tries sentence boundaries first (``". "``), falls back to hard cut.
    Avoids cutting through ``\\x00PROTECTED_N\\x00`` placeholder tokens.
    A piece may exceed chunk_size only if a single placeholder token is
    longer than chunk_size (unavoidable).
    """
    if len(text) <= chunk_size:
        return [text]
    parts: list[str] = []
    while len(text) > chunk_size:
        cut = text.rfind(". ", 0, chunk_size)
        if cut == -1 or cut < chunk_size // 4:
            cut = chunk_size  # hard cut
        else:
            cut += 2  # include ". "
        # Ensure we don't split inside a placeholder token
        orig_cut = cut
        cut = _adjust_for_placeholder(text, cut)
        # If placeholder adjustment pushed cut beyond chunk_size, split
        # *before* the placeholder instead (unless that gives us nothing).
        if cut > orig_cut and cut > chunk_size:
            before_placeholder = text.rfind("\x00PROTECTED_", 0, orig_cut)
            if before_placeholder > 0:
                cut = before_placeholder
        parts.append(text[:cut])
        text = text[cut:]
    if text:
        parts.append(text)
    return parts


def _split_into_chunks(text: str, chunk_size: int) -> list[str]:
    """Split markdown text into translatable chunks respecting structure.

    Protected blocks (code fences, display math ``$$...$$``, inline math
    ``$...$``, and images) are replaced with placeholders before splitting
    to prevent them from being broken across chunks. After splitting,
    placeholders are restored.

    Args:
        text: Full markdown text.
        chunk_size: Target maximum chunk size in characters.

    Returns:
        List of text chunks.
    """
    # Mask protected blocks with placeholders
    protected: list[str] = []

    def _mask(m: re.Match) -> str:
        idx = len(protected)
        protected.append(m.group(0))
        return _PLACEHOLDER_FMT.format(idx)

    masked = _PROTECTED_RE.sub(_mask, text)

    # Split on paragraph boundaries (filter empty/whitespace-only paragraphs)
    paragraphs = [p for p in re.split(r"\n{2,}", masked) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        # Oversized paragraph: flush current, then hard-split the paragraph
        if para_len > chunk_size:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            for frag in _hard_split(para, chunk_size):
                chunks.append(frag)
            continue
        if current_len + para_len > chunk_size and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += para_len + 2  # +2 for \n\n

    if current:
        chunks.append("\n\n".join(current))

    # Restore protected blocks in each chunk; warn if restoration inflates beyond limit
    def _restore(chunk: str) -> str:
        return _PLACEHOLDER_RE.sub(lambda m: protected[int(m.group(1))], chunk)

    restored = [_restore(c) for c in chunks]
    for i, c in enumerate(restored):
        if len(c) > chunk_size * 2:
            logger.warning(
                "chunk {}/{} restored to {} chars (limit {}) due to large protected blocks",
                i + 1,
                len(restored),
                len(c),
                chunk_size,
            )
    return restored


def _translate_chunk(text: str, target_lang: str, models: list[dict]) -> str:
    """Translate a single chunk via LLM. Raises on failure."""
    import asyncio

    from drbrain.extractor.llm_client import acall_text_with_fallback

    lang_name = _LANG_NAMES.get(target_lang, target_lang)
    prompt = _build_translate_prompt(text, target_lang, lang_name)
    result = asyncio.run(acall_text_with_fallback(prompt, models, max_tokens=4096))
    if result is None:
        raise RuntimeError(f"Translation failed for chunk ({len(text)} chars)")
    return result.strip()


def _translate_chunk_with_retry(
    text: str,
    target_lang: str,
    models: list[dict],
    *,
    max_attempts: int = 5,
    backoff_base: float = 1.0,
) -> tuple[str, int]:
    """Translate with exponential backoff retry. Returns (translated_text, attempts)."""
    import time

    for attempt in range(1, max_attempts + 1):
        try:
            return _translate_chunk(text, target_lang, models), attempt
        except Exception:
            if attempt >= max_attempts:
                raise
            time.sleep(backoff_base * (2 ** (attempt - 1)))
    raise RuntimeError("unreachable")


def _subdivide_chunk_for_retry(text: str, chunk_size: int) -> list[str]:
    """Produce sub-chunks for retry at a smaller target size.

    Target size is ``max(1, min(chunk_size, len(text) // 2))``.
    Returns ``[text]`` if the target is not smaller than *text*.
    """
    target = max(1, min(chunk_size, len(text) // 2))
    if target >= len(text):
        return [text]
    try:
        return _split_into_chunks(text, target)
    except Exception:
        return _hard_split(text, target)


def _translate_chunk_resilient(
    text: str,
    target_lang: str,
    models: list[dict],
    *,
    chunk_size: int = 3000,
    max_attempts: int = 5,
    backoff_base: float = 1.0,
) -> tuple[str, int]:
    """Translate with retry; on repeated timeout, subdivide and retry parts."""
    subchunks = _subdivide_chunk_for_retry(text, chunk_size)
    split_budget = min(max_attempts, 2) if len(subchunks) > 1 else max_attempts
    try:
        return _translate_chunk_with_retry(
            text,
            target_lang,
            models,
            max_attempts=split_budget,
            backoff_base=backoff_base,
        )
    except Exception:
        if len(subchunks) <= 1:
            raise
        logger.warning(f"chunk timed out, retrying as {len(subchunks)} subchunks")
        translated_parts = []
        total_attempts = split_budget
        for sub in subchunks:
            t, used = _translate_chunk_resilient(
                sub,
                target_lang,
                models,
                chunk_size=max(1, min(chunk_size, len(sub))),
                max_attempts=max_attempts,
                backoff_base=backoff_base,
            )
            translated_parts.append(t)
            total_attempts += used
        return "\n\n".join(translated_parts), total_attempts


async def translate_text(
    text: str,
    models: list[dict],
    target_lang: str = "Chinese",
    source_lang: str = "English",
) -> str | None:
    """Translate text chunk by chunk via LLM fallback chain."""
    from drbrain.extractor.llm_client import acall_text_with_fallback

    chunks = _split_into_chunks(text, CHUNK_SIZE)
    translated = []

    system_prompt = (
        f"You are a professional academic translator. Translate the following text "
        f"from {source_lang} to {target_lang}. Preserve all formatting, LaTeX math "
        f"expressions, citations, and technical terms. Keep the academic tone. "
        f"Return ONLY the translated text, no explanations."
    )

    for i, chunk in enumerate(chunks):
        logger.debug(f"Translating chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)")
        result = await acall_text_with_fallback(
            prompt=chunk,
            models=models,
            system_prompt=system_prompt,
            max_tokens=4096,
        )
        if result is None:
            logger.error(f"Translation failed on chunk {i + 1}")
            return None
        translated.append(result)

    return "\n\n".join(translated)


def translate_paper(
    md_path: Path,
    models: list[dict],
    target_lang: str = "Chinese",
    source_lang: str = "English",
    output_path: Path | None = None,
) -> Path | None:
    """Translate a paper's raw.md and save result. Returns output path."""
    import asyncio

    if not md_path.exists():
        logger.error(f"Markdown file not found: {md_path}")
        return None

    text = md_path.read_text(encoding="utf-8")
    logger.info(f"Translating {md_path} ({len(text)} chars) to {target_lang}")

    result = asyncio.run(
        translate_text(text, models, target_lang=target_lang, source_lang=source_lang)
    )

    if result is None:
        return None

    output_path = output_path or md_path.parent / f"paper_{target_lang.lower()}.md"
    output_path.write_text(result, encoding="utf-8")
    logger.info(f"Translation saved to {output_path}")
    return output_path
