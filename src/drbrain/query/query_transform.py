"""HyDE — Hypothetical Document Embeddings query transform.

HyDE (Gao et al., 2022, "Precise Zero-Shot Dense Retrieval without Relevance
Labels") reformulates a retrieval query by first asking an LLM to *answer* the
question with a short hypothetical paragraph, then using that paragraph as the
retrieval query. The hypothesis lives in the same semantic space as the target
documents (both are "answers"), so it matches better than the question itself
— especially for embedding search, where a question and its answer are
textually dissimilar.

This is a **query-side** transform: it sits before retrieval, not after. It is
optional and must degrade gracefully:

    LLM unavailable / fails  →  return the original query unchanged

The retriever never sees a failure; HyDE is an enhancement, not a dependency.

Design:
    - ``hyde_transform``      sync, takes a caller for testability
    - ``ahyde_transform``     async, wraps ``acall_text_with_fallback``
    - Both return ``HydeResult`` carrying the (possibly augmented) query plus
      provenance, so callers can inspect whether HyDE actually fired.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from drbrain.extractor.cache import ApiCache

log = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 300  # a hypothesis paragraph, not a full essay
DEFAULT_N_DOCS = 1  # how many hypothetical docs to generate then merge

_SYSTEM_PROMPT = (
    "You are a scientific retrieval assistant. Given a research question, "
    "write a short, concrete paragraph that could plausibly appear in a "
    "relevant paper's methods or results section — as if the question were "
    "already answered. Use domain terminology. Do not preface with "
    "'Here is'. Just the paragraph."
)


@dataclass
class HydeResult:
    """Outcome of a HyDE transform.

    Attributes:
        query: The query to feed to retrieval. Equal to the original on
            failure; the hypothetical document(s) on success.
        original: The untouched input query, for logging/debug.
        transformed: True iff HyDE replaced the query with a hypothesis.
        hypothesis: The generated hypothetical text (None if not transformed).
    """

    query: str
    original: str
    transformed: bool = False
    hypothesis: str | None = None


# A sync caller has the same shape as call_text_with_fallback; an async caller
# matches acall_text_with_fallback. Typing both is verbose, so we accept a
# Callable and document the contract.
SyncCaller = Callable[..., str | None]
AsyncCaller = Callable[..., Awaitable[str | None]]


def _build_prompt(question: str) -> str:
    return (
        f"Research question: {question}\n\n"
        f"Write one concrete paragraph (3-5 sentences) that a relevant paper "
        f"might contain addressing this question."
    )


def _merge(question: str, hypotheses: list[str]) -> str:
    """Combine the original question with one or more hypotheses.

    A single hypothesis replaces the query (canonical HyDE). Multiple are
    concatenated so the retriever sees several semantic angles at once.
    """
    clean = [h.strip() for h in hypotheses if h and h.strip()]
    if not clean:
        return question
    if len(clean) == 1:
        return clean[0]
    return "\n\n".join(clean)


def hyde_transform(
    question: str,
    models: list[dict],
    *,
    n_docs: int = DEFAULT_N_DOCS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    caller: SyncCaller | None = None,
) -> HydeResult:
    """Sync HyDE transform.

    Args:
        question: Original retrieval query.
        models: LLM model configs (same shape used across the codebase).
        n_docs: Number of hypothetical documents to generate, then merge.
        max_tokens: Token cap per generated document.
        caller: Injectable LLM caller matching ``call_text_with_fallback``
            signature. Defaults to the real one. Inject a stub in tests.

    Returns:
        ``HydeResult`` — ``query`` is the (possibly transformed) query to
        pass to retrieval. On any failure, returns the original unmodified.
    """
    if not question.strip() or not models:
        return HydeResult(query=question, original=question)

    if caller is None:
        from drbrain.extractor.llm_client import call_text_with_fallback

        caller = call_text_with_fallback

    prompt = _build_prompt(question)
    hypotheses: list[str] = []
    for _ in range(n_docs):
        try:
            text = caller(
                prompt=prompt,
                models=models,
                system_prompt=_SYSTEM_PROMPT,
                max_tokens=max_tokens,
            )
        except Exception as e:  # caller must not crash retrieval
            log.warning("[hyde] LLM call failed (%s); using original query", e)
            break
        if text:
            hypotheses.append(text)

    if not hypotheses:
        return HydeResult(query=question, original=question)

    merged = _merge(question, hypotheses)
    return HydeResult(
        query=merged,
        original=question,
        transformed=True,
        hypothesis=merged,
    )


async def ahyde_transform(
    question: str,
    models: list[dict],
    *,
    n_docs: int = DEFAULT_N_DOCS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    caller: AsyncCaller | None = None,
    _cache: ApiCache | None = None,
) -> HydeResult:
    """Async HyDE transform.

    Wraps ``acall_text_with_fallback``. Same contract as ``hyde_transform``;
    use this from async retrieval paths (e.g. ``query_by_structure_hybrid``).
    """
    if not question.strip() or not models:
        return HydeResult(query=question, original=question)

    if caller is None:
        from drbrain.extractor.llm_client import acall_text_with_fallback

        caller = acall_text_with_fallback

    prompt = _build_prompt(question)
    hypotheses: list[str] = []
    for _ in range(n_docs):
        try:
            text = await caller(
                prompt=prompt,
                models=models,
                system_prompt=_SYSTEM_PROMPT,
                max_tokens=max_tokens,
                _cache=_cache,
            )
        except Exception as e:
            log.warning("[hyde] async LLM call failed (%s); using original query", e)
            break
        if text:
            hypotheses.append(text)

    if not hypotheses:
        return HydeResult(query=question, original=question)

    merged = _merge(question, hypotheses)
    return HydeResult(
        query=merged,
        original=question,
        transformed=True,
        hypothesis=merged,
    )


__all__ = ["HydeResult", "hyde_transform", "ahyde_transform"]
