"""Pipeline step definitions and presets for chained processing.

Presets:
    full    = ingest → build → embed → closure
    quick   = build → embed → closure
    embed   = embed → closure
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StepDef:
    """Definition of a pipeline step."""

    name: str
    scope: str  # "inbox" | "papers" | "global"
    desc: str


STEPS = {
    "ingest": StepDef(
        name="ingest",
        scope="inbox",
        desc="Parse PDFs from inbox, identify, tree-structure, register",
    ),
    "build": StepDef(
        name="build",
        scope="papers",
        desc="5-stage LLM extraction: ontology → entities → relations → coref → refine",
    ),
    "embed": StepDef(
        name="embed",
        scope="global",
        desc="Train TransE graph embeddings + tree text embeddings",
    ),
    "closure": StepDef(
        name="closure",
        scope="global",
        desc="Rule-based inference (8 symbolic + 4 embedding rules)",
    ),
}

PRESETS = {
    "full": ["ingest", "build", "embed", "closure"],
    "quick": ["build", "embed", "closure"],
    "embed": ["embed", "closure"],
}


def resolve_steps(
    preset: str | None = None,
    steps_str: str | None = None,
) -> list[str]:
    """Resolve a list of step names from preset name or comma-separated string.

    Args:
        preset: Preset name (``"full"``, ``"quick"``, ``"embed"``).
        steps_str: Comma-separated step names, e.g. ``"build,embed"``.

    Returns:
        Ordered list of validated step names.

    Raises:
        ValueError: If preset is unknown, a step name is unknown, or
            neither preset nor steps_str is provided.
    """
    if preset:
        if preset not in PRESETS:
            available = ", ".join(PRESETS)
            raise ValueError(f"Unknown preset '{preset}'. Available presets: {available}")
        return list(PRESETS[preset])

    if steps_str:
        names = [s.strip() for s in steps_str.split(",") if s.strip()]
        seen: set[str] = set()
        result: list[str] = []
        for name in names:
            if name not in STEPS:
                available = ", ".join(STEPS)
                raise ValueError(f"Unknown step '{name}'. Available steps: {available}")
            if name not in seen:
                seen.add(name)
                result.append(name)
        return result

    raise ValueError("Specify --preset or --steps")


def list_steps_info() -> tuple[list[dict], list[dict]]:
    """Return structured step and preset info for display.

    Returns:
        Tuple of (steps_list, presets_list) where each is a list of dicts
        with display-ready keys.
    """
    steps = [
        {
            "name": name,
            "scope": sdef.scope,
            "description": sdef.desc,
        }
        for name, sdef in STEPS.items()
    ]
    presets = [{"name": name, "steps": slist} for name, slist in PRESETS.items()]
    return steps, presets
