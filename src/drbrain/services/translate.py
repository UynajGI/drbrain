"""Paper translation via LLM — chunks sections and translates with fallback chain."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from drbrain.storage.paths import raw_md_path

CHUNK_SIZE = 3000  # chars per translation chunk

# ---------------------------------------------------------------------------
# Skip reason constants
# ---------------------------------------------------------------------------

SKIP_NO_MD = "no_paper_md"
SKIP_ALREADY_EXISTS = "already_exists"
SKIP_EMPTY = "empty_source"
SKIP_SAME_LANG = "same_language"
SKIP_ALL_CHUNKS_FAILED = "all_chunks_failed"

# ---------------------------------------------------------------------------
# TranslateResult
# ---------------------------------------------------------------------------


@dataclass
class TranslateResult:
    path: Path | None = None
    skip_reason: str = ""
    partial: bool = False
    completed_chunks: int = 0
    total_chunks: int = 0

    @property
    def ok(self) -> bool:
        return self.path is not None and not self.partial


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


# ---------------------------------------------------------------------------
# Workdir helpers
# ---------------------------------------------------------------------------


def _translation_workdir(paper_dir: Path, lang: str) -> Path:
    return paper_dir / f".translate_{lang}"


def _translation_state_path(workdir: Path) -> Path:
    return workdir / "state.json"


def _translation_parts_dir(workdir: Path) -> Path:
    return workdir / "parts"


def _translation_part_path(workdir: Path, index: int) -> Path:
    return _translation_parts_dir(workdir) / f"{index + 1:06d}.md"


def _source_digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Translation state management
# ---------------------------------------------------------------------------


def _load_translation_state(state_path: Path) -> dict | None:
    """Read state.json if it exists and is valid JSON dict."""
    if not state_path.exists():
        return None
    try:
        data = json.loads(state_path.read_text("utf-8"))
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _build_translation_state(
    workdir: Path,
    lang: str,
    source_digest_val: str,
    chunk_size: int,
    chunks: list[str],
) -> dict:
    total = len(chunks)
    return {
        "target_lang": lang,
        "source_digest": source_digest_val,
        "chunk_size": chunk_size,
        "total_chunks": total,
        "chunks": [{"index": i, "status": "pending", "attempts": 0} for i in range(total)],
        "updated_at": datetime.now(UTC).isoformat(),
    }


def _write_translation_workspace_files(
    workdir: Path,
    state: dict,
    chunks: list[str],
) -> None:
    """Write state.json and chunks.json atomically (tmp->rename)."""
    workdir.mkdir(parents=True, exist_ok=True)

    # chunks.json — per-chunk digests for resume validation
    chunks_data = [{"index": i, "digest": _source_digest(c)} for i, c in enumerate(chunks)]
    chunks_tmp = workdir / "chunks.json.tmp"
    chunks_tmp.write_text(json.dumps(chunks_data, indent=2, ensure_ascii=False), encoding="utf-8")
    chunks_tmp.rename(workdir / "chunks.json")

    # state.json
    state_tmp = workdir / "state.json.tmp"
    state_tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    state_tmp.rename(workdir / "state.json")


def _write_translation_state(workdir: Path, state: dict) -> None:
    """Write state.json atomically (incremental update)."""
    tmp = workdir / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(workdir / "state.json")


def _load_or_init_translation_workspace(
    paper_dir: Path,
    lang: str,
    force: bool,
    out_path: Path,
    source_digest_val: str,
    chunk_size: int,
    chunks: list[str],
) -> dict:
    """Load existing translation workspace if valid, otherwise initialise fresh.

    If *force* is True the old workdir and output are deleted.
    Otherwise the existing state is validated (same lang, source digest,
    chunk size, chunk count, and per-chunk digests).  A valid state is
    returned for resumption; an invalid or missing state causes the old
    workdir to be removed and a fresh one created.
    """
    workdir = _translation_workdir(paper_dir, lang)
    state_path = _translation_state_path(workdir)
    chunks_path = workdir / "chunks.json"

    if force:
        if workdir.exists():
            shutil.rmtree(workdir)
        if out_path.exists():
            out_path.unlink()
    else:
        existing = _load_translation_state(state_path)
        if existing is not None:
            # Validate key fields
            if (
                existing.get("target_lang") == lang
                and existing.get("source_digest") == source_digest_val
                and existing.get("chunk_size") == chunk_size
                and existing.get("total_chunks") == len(chunks)
            ):
                # Validate per-chunk digests
                if chunks_path.exists():
                    try:
                        stored_chunks = json.loads(chunks_path.read_text("utf-8"))
                        if len(stored_chunks) == len(chunks):
                            all_match = all(
                                sc.get("digest") == _source_digest(chunks[i])
                                for i, sc in enumerate(stored_chunks)
                            )
                            if all_match:
                                return existing  # valid — resume
                    except (json.JSONDecodeError, OSError):
                        pass

    # Invalid, missing, or force — start fresh
    if workdir.exists():
        shutil.rmtree(workdir)
    state = _build_translation_state(workdir, lang, source_digest_val, chunk_size, chunks)
    _write_translation_workspace_files(workdir, state, chunks)
    return state


def _load_success_prefix(workdir: Path, state: dict) -> list[str]:
    """Walk chunks from index 0; collect translations for all SUCCESS chunks
    until the first non-SUCCESS or missing part file."""
    translated: list[str] = []
    for ci in state["chunks"]:
        if ci["status"] != "success":
            break
        part_path = _translation_part_path(workdir, ci["index"])
        if not part_path.exists():
            break
        translated.append(part_path.read_text("utf-8"))
    return translated


def _persist_prefix_output(out_path: Path, translated_chunks: list[str]) -> None:
    """Write translated prefix to *out_path*, or remove it if empty."""
    if translated_chunks:
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text("\n\n".join(translated_chunks), encoding="utf-8")
        tmp.replace(out_path)
    elif out_path.exists():
        out_path.unlink()


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


def translate_paper(
    paper_dir: Path,
    models: list[dict],
    *,
    target_lang: str = "zh",
    force: bool = False,
    chunk_workers: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> TranslateResult:
    """Translate a paper's raw.md with resume, concurrency, and skip logic.

    Args:
        paper_dir: Directory containing ``raw.md`` (e.g. ``data/papers/<id>/``).
        models: List of litellm-compatible provider configs.
        target_lang: Language code (``"zh"``, ``"ja"``, etc.).  Default ``"zh"``.
        force: If True, delete any existing output and workdir before translating.
        chunk_workers: Number of concurrent translation threads.  Default 3.
        progress_callback: Called with a progress string after each chunk completes.

    Returns:
        TranslateResult with outcome details.
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    md_path = raw_md_path(paper_dir)
    if not md_path.exists():
        return TranslateResult(skip_reason=SKIP_NO_MD)

    text = md_path.read_text(encoding="utf-8")
    if not text.strip():
        return TranslateResult(skip_reason=SKIP_EMPTY)

    src_lang = detect_language(text)
    if src_lang == target_lang:
        return TranslateResult(skip_reason=SKIP_SAME_LANG)

    out_path = paper_dir / f"paper_{target_lang}.md"
    workdir = _translation_workdir(paper_dir, target_lang)

    # Skip if output already exists and there is no partial workdir to resume
    if not force and not workdir.exists() and out_path.exists():
        return TranslateResult(skip_reason=SKIP_ALREADY_EXISTS)

    chunks = _split_into_chunks(text, CHUNK_SIZE)
    if not chunks:
        return TranslateResult(skip_reason=SKIP_EMPTY)

    source_digest_val = _source_digest(text)

    state = _load_or_init_translation_workspace(
        paper_dir,
        target_lang,
        force,
        out_path,
        source_digest_val,
        CHUNK_SIZE,
        chunks,
    )

    total = state["total_chunks"]

    # Load already-completed prefix and persist to output
    prefix = _load_success_prefix(workdir, state)
    _persist_prefix_output(out_path, prefix)

    completed = len(prefix)
    if completed == total:
        shutil.rmtree(workdir)
        return TranslateResult(path=out_path, completed_chunks=total, total_chunks=total)

    # Determine chunks that still need translation
    pending = [i for i, ci in enumerate(state["chunks"]) if ci["status"] != "success"]

    workers = max(1, chunk_workers if chunk_workers is not None else 3)

    logger.info(
        "Translating paper {} ({} chars, {} chunks, {} workers, {} already done)",
        paper_dir.name,
        len(text),
        total,
        workers,
        completed,
    )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx: dict = {}
        for idx in pending:
            chunk = chunks[idx]
            future = executor.submit(
                _translate_chunk_resilient,
                chunk,
                target_lang,
                models,
                chunk_size=CHUNK_SIZE,
                max_attempts=5,
                backoff_base=1.0,
            )
            future_to_idx[future] = idx

        while future_to_idx:
            done, _ = wait(future_to_idx.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                idx = future_to_idx.pop(future)
                try:
                    translated_text, attempts = future.result()
                    # Write part file
                    part_path = _translation_part_path(workdir, idx)
                    part_path.parent.mkdir(parents=True, exist_ok=True)
                    part_tmp = part_path.with_suffix(part_path.suffix + ".tmp")
                    part_tmp.write_text(translated_text, encoding="utf-8")
                    part_tmp.replace(part_path)
                    state["chunks"][idx]["status"] = "success"
                    state["chunks"][idx]["attempts"] = attempts
                except Exception as exc:
                    logger.warning("Translation failed for chunk {}/{}: {}", idx + 1, total, exc)
                    state["chunks"][idx]["status"] = "failed"

                state["updated_at"] = datetime.now(UTC).isoformat()
                _write_translation_state(workdir, state)

                # Persist current prefix to output
                prefix = _load_success_prefix(workdir, state)
                _persist_prefix_output(out_path, prefix)

                if progress_callback:
                    progress_callback(f"Translation progress: {len(prefix)}/{total}")

    # Final status
    final_prefix = _load_success_prefix(workdir, state)

    if len(final_prefix) == total:
        shutil.rmtree(workdir)
        return TranslateResult(
            path=out_path,
            completed_chunks=total,
            total_chunks=total,
        )
    elif len(final_prefix) > 0:
        return TranslateResult(
            path=out_path,
            partial=True,
            completed_chunks=len(final_prefix),
            total_chunks=total,
        )
    else:
        return TranslateResult(skip_reason=SKIP_ALL_CHUNKS_FAILED)
