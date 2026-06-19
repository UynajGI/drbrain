"""Concept types and flat extraction (non-tree)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from drbrain.extractor.argument import ExtractedArgument, parse_arguments
from drbrain.extractor.llm_client import acall_with_fallback

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    Path(__file__).parent.parent.parent.parent.parent / "prompts" / "extract_concepts.txt"
)
ONTOLOGY_PROMPT = Path(__file__).parent.parent.parent.parent.parent / "prompts" / "ontology.txt"
ENTITIES_PROMPT = Path(__file__).parent.parent.parent.parent.parent / "prompts" / "entities.txt"
RELATIONS_PROMPT = Path(__file__).parent.parent.parent.parent.parent / "prompts" / "relations.txt"
COREFERENCE_PROMPT = (
    Path(__file__).parent.parent.parent.parent.parent / "prompts" / "coreference.txt"
)
REFINE_PROMPT = Path(__file__).parent.parent.parent.parent.parent / "prompts" / "refine.txt"


class ExtractedConcepts:
    """Structured extraction result from a paper."""

    def __init__(self, data: dict):
        self.problems: list[dict] = data.get("problems", [])
        self.methods: list[dict] = data.get("methods", [])
        self.conclusions: list[dict] = data.get("conclusions", [])
        self.debates: list[dict] = data.get("debates", [])
        self.gaps: list[dict] = data.get("gaps", [])
        self.actors: list[dict] = data.get("actors", [])
        self.relations: list[dict] = data.get("relations", [])
        self.arguments: list[ExtractedArgument] = parse_arguments(data.get("arguments", []))

    def to_dict(self) -> dict:
        return {
            "problems": self.problems,
            "methods": self.methods,
            "conclusions": self.conclusions,
            "debates": self.debates,
            "gaps": self.gaps,
            "actors": self.actors,
            "relations": self.relations,
            "arguments": [a.to_dict() for a in self.arguments],
        }


async def extract_concepts(
    text: str,
    models: list[dict],
) -> ExtractedConcepts | None:
    """Extract academic concepts + arguments from paper text using LLM fallback chain."""
    system_prompt = PROMPT_TEMPLATE.read_text(encoding="utf-8")
    data = await acall_with_fallback(
        prompt=text[:8000],
        models=models,
        system_prompt=system_prompt,
    )
    if data is None or not isinstance(data, dict):
        return None
    return ExtractedConcepts(data)


def validate_extraction(concepts: ExtractedConcepts) -> list[str]:
    """Validate extracted concepts against TBox rules before DB insertion.

    Returns a list of error strings. Empty list means valid.
    """
    from drbrain.validator.schema import TBOX

    errors = []

    # Build label → type lookup from concept categories
    label_type: dict[str, str] = {}
    for cat_name in ("problems", "methods", "conclusions", "debates", "gaps", "actors"):
        cat_type = cat_name.rstrip("s").capitalize()  # "problems" → "Problem"
        if cat_type == "Conclusion":
            cat_type = "Conclusion"
        for item in getattr(concepts, cat_name, []):
            label = item.get("label", "").strip()
            if label:
                label_type[label.lower()] = cat_type

    # Check each relation against TBox
    for rel in concepts.relations:
        head = rel.get("head", "").strip()
        rel_name = rel.get("rel", "").strip()
        if not head or not rel_name:
            continue
        head_type = label_type.get(head.lower())
        if not head_type:
            continue
        allowed = TBOX.get(head_type, set())
        if allowed and rel_name not in allowed:
            errors.append(
                f"TBox violation: {head_type} '{head}' cannot use relation '{rel_name}'. "
                f"Allowed: {sorted(allowed)}"
            )

    return errors
