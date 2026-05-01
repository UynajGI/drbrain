"""Paper translation via LLM — chunks sections and translates with fallback chain."""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

CHUNK_SIZE = 3000  # chars per translation chunk
SECTION_BREAK = re.compile(r"^#{1,3}\s", re.MULTILINE)


def _chunk_text(text: str, max_chars: int = CHUNK_SIZE) -> list[str]:
    """Split text at section boundaries, keeping chunks under max_chars."""
    parts = SECTION_BREAK.split(text)
    if not parts or len(parts) < 2:
        # No sections found, chunk by paragraphs
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""
        for p in paragraphs:
            if len(current) + len(p) > max_chars and current:
                chunks.append(current.strip())
                current = p
            else:
                current += "\n\n" + p if current else p
        if current.strip():
            chunks.append(current.strip())
        return chunks

    # Reconstruct: parts[0] is before first heading, then alternating (heading, content, heading, ...)
    chunks = []
    current = parts[0] if parts[0].strip() else ""
    for i in range(1, len(parts) - 1, 2):
        heading = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        section = f"{heading} {content}"
        if len(current) + len(section) > max_chars and current:
            chunks.append(current.strip())
            current = section
        else:
            current += "\n\n" + section
    if current.strip():
        chunks.append(current.strip())
    return chunks


async def translate_text(
    text: str,
    models: list[dict],
    target_lang: str = "Chinese",
    source_lang: str = "English",
) -> str | None:
    """Translate text chunk by chunk via LLM fallback chain."""
    from drbrain.extractor.llm_client import acall_text_with_fallback

    chunks = _chunk_text(text)
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
