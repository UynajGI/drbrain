"""Academic concept extraction via LLM."""

from __future__ import annotations

import json
from pathlib import Path

import litellm

PROMPT_TEMPLATE = Path(__file__).parent.parent.parent.parent / "prompts" / "extract_concepts.txt"


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

    def to_dict(self) -> dict:
        return {
            "problems": self.problems,
            "methods": self.methods,
            "conclusions": self.conclusions,
            "debates": self.debates,
            "gaps": self.gaps,
            "actors": self.actors,
            "relations": self.relations,
        }


async def extract_concepts(
    text: str,
    model: str = "openai/gpt-4o",
    api_base: str | None = None,
) -> ExtractedConcepts:
    """Extract academic concepts from paper text using LLM.

    Uses litellm for provider-agnostic calls.
    """
    prompt = PROMPT_TEMPLATE.read_text(encoding="utf-8")

    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text[:12000]},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
    }
    if api_base:
        kwargs["api_base"] = api_base

    response = await litellm.acompletion(**kwargs)
    content = response.choices[0].message.content
    data = json.loads(content)
    return ExtractedConcepts(data)
