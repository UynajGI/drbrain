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
