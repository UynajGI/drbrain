"""Model-judge prompts and response parsing; scores are not calibrated probabilities."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any

log = logging.getLogger(__name__)
_CONTEXT_CHUNK_MAX_CHARS = 1500


def _prompt_faithfulness(question: str, answer: str, context: str) -> str:
    return (
        "You are an evaluation judge for a retrieval-augmented question answering system.\n"
        "Score the FAITHFULNESS of the Answer with respect to the Context.\n"
        "Faithfulness measures whether every factual claim in the answer is supported by "
        "the provided context (0.0 = fully unsupported/hallucinated, 1.0 = fully supported).\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        f"Context:\n{context}\n\n"
        "Reply with exactly one line: SCORE: <number between 0 and 1>"
    )


def _prompt_answer_relevancy(question: str, answer: str) -> str:
    return (
        "You are an evaluation judge for a question answering system.\n"
        "Score the ANSWER RELEVANCY: how well the answer directly addresses the question, "
        "without being evasive or off-topic (0.0 = completely irrelevant, 1.0 = perfectly on-topic).\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        "Reply with exactly one line: SCORE: <number between 0 and 1>"
    )


def _prompt_context_precision(question: str, context: str) -> str:
    return (
        "You are an evaluation judge for a retrieval-augmented question answering system.\n"
        "Score the CONTEXT PRECISION: what fraction of the retrieved context is relevant to "
        "answering the question (0.0 = none of it is relevant, 1.0 = all of it is relevant).\n\n"
        f"Question: {question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Reply with exactly one line: SCORE: <number between 0 and 1>"
    )


def _prompt_answer_correctness(question: str, answer: str, reference: str) -> str:
    return (
        "You are an evaluation judge for a question answering system.\n"
        "Score the ANSWER CORRECTNESS: how factually consistent the answer is with the "
        "reference answer from the source paper (0.0 = contradicts/ignores the reference, "
        "1.0 = fully consistent).\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        f"Reference answer (source paper abstract):\n{reference}\n\n"
        "Reply with exactly one line: SCORE: <number between 0 and 1>"
    )


_SCORE_RE = re.compile(r"\bSCORE\s*[:=]\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def _parse_score(text: str | None) -> float | None:
    """Parse ``SCORE: 0.75`` (or a bare number) out of an LLM verdict."""
    if not text:
        return None
    m = _SCORE_RE.search(text)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except ValueError:  # pragma: no cover - defensive
            return None
    # Tolerate a bare numeric reply (some models skip the prefix).
    for token in text.split():
        try:
            value = float(token.strip(".,:[]()"))
            if 0.0 <= value <= 1.0:
                return value
        except ValueError:
            continue
    return None


def _score_metric(llm: Any, prompt: str) -> float | None:
    """Run one scoring prompt through the DrbrainLLM bridge."""
    try:
        response = llm.complete(prompt, max_tokens=128)
        return _parse_score(getattr(response, "text", None))
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("[rag] metric scoring call failed: %s", exc)
        return None


def _context_for(nodes: Sequence[Any], limit: int = 3) -> str:
    """Join top retrieved node texts for the context-based metrics."""
    chunks: list[str] = []
    for nws in (nodes or [])[:limit]:
        node = getattr(nws, "node", None)
        text = (getattr(node, "text", None) or "").strip()
        if not text:
            continue
        if len(text) > _CONTEXT_CHUNK_MAX_CHARS:
            text = text[:_CONTEXT_CHUNK_MAX_CHARS].rstrip() + "…"
        chunks.append(text)
    return "\n\n".join(chunks)
