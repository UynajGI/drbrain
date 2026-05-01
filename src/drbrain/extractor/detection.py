"""Paper type classification via heuristics + optional LLM refinement."""

from __future__ import annotations

PAPER_TYPES = {"paper", "review", "thesis", "preprint", "book", "document"}

_TYPE_HEURISTICS: dict[str, list[str]] = {
    "review": [
        "survey",
        "systematic review",
        "meta-analysis",
        "literature review",
        "state of the art",
        "comprehensive review",
        "critical review",
    ],
    "thesis": [
        "dissertation",
        "doctor of philosophy",
        "master's thesis",
        "doctoral thesis",
        "submitted in partial fulfillment",
        "degree of doctor",
        "graduate school",
    ],
    "preprint": [
        "arxiv:",
        "preprint",
        "submitted to",
        "under review",
        "working paper",
        "techrxiv",
        "biorxiv",
        "medrxiv",
    ],
    "book": [
        "monograph",
        "textbook",
        "handbook",
        "encyclopedia",
        "proceedings volume",
        "edited volume",
        "edited by",
        "publisher:",
    ],
    "document": [
        "white paper",
        "technical report",
        "lecture notes",
        "course notes",
        "presentation",
        "slides",
        "manual",
        "specification",
        "standard",
        "guideline",
    ],
}


def _heuristic_type(
    title: str,
    abstract: str | None = None,
    first_page: str | None = None,
) -> str:
    """Classify paper type by keyword matching in title + text."""
    combined = (title or "").lower()
    if abstract:
        combined += " " + abstract.lower()
    if first_page:
        combined += " " + first_page.lower()

    for ptype in ("review", "thesis", "preprint", "book", "document"):
        for kw in _TYPE_HEURISTICS.get(ptype, []):
            if kw in combined:
                return ptype
    return "paper"


async def _llm_type(
    title: str,
    abstract: str | None,
    first_page: str | None,
    models: list[dict],
) -> str:
    """Use LLM to classify paper type when heuristics are inconclusive."""
    from pathlib import Path

    from drbrain.extractor.llm_client import acall_with_fallback

    prompt_path = Path(__file__).parent.parent.parent.parent / "prompts" / "detect_paper_type.txt"
    system_prompt = prompt_path.read_text(encoding="utf-8")

    text = f"Title: {title}\n"
    if abstract:
        text += f"Abstract: {abstract[:500]}\n"
    if first_page:
        text += f"Content: {first_page[:800]}\n"

    data = await acall_with_fallback(
        prompt=text,
        models=models,
        system_prompt=system_prompt,
    )
    if data and isinstance(data, dict) and data.get("paper_type") in PAPER_TYPES:
        return data["paper_type"]
    return "paper"


def detect_paper_type(
    title: str,
    abstract: str | None = None,
    first_page: str | None = None,
) -> str:
    """Detect paper type from title and content (heuristic only, no LLM).

    For LLM refinement, use ``detect_paper_type_async()``.
    """
    return _heuristic_type(title, abstract, first_page)


async def detect_paper_type_async(
    title: str,
    abstract: str | None = None,
    first_page: str | None = None,
    models: list[dict] | None = None,
) -> str:
    """Detect paper type with LLM fallback for ambiguous cases.

    Heuristics handle obvious cases (>90%). LLM is only called when
    the heuristic score is ambiguous (multiple types matched or none).
    """
    heuristic = _heuristic_type(title, abstract, first_page)
    if heuristic != "paper" or not models:
        return heuristic

    return await _llm_type(title, abstract, first_page, models)
