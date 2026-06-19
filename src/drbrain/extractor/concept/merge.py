"""Section-level extraction and multi-section merge."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from drbrain.extractor.concept.tree_helpers import (
    _collect_leaf_nodes,
    _is_quality_content,
)
from drbrain.extractor.concept.types import ExtractedConcepts
from drbrain.extractor.llm_client import acall_with_fallback
from drbrain.parser.pageindex_parser import get_document_structure_json, get_node_content

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


def _merge_concepts(
    results: list[ExtractedConcepts],
    sections: list[str] | None = None,
) -> ExtractedConcepts:
    """Merge multiple ExtractedConcepts, deduplicating by label (highest confidence wins).

    Args:
        results: List of ExtractedConcepts from each section.
        sections: Optional list of section titles parallel to results.
    """
    merged: dict = {
        "problems": [],
        "methods": [],
        "conclusions": [],
        "debates": [],
        "gaps": [],
        "actors": [],
        "relations": [],
        "arguments": [],
    }

    for category in ("problems", "methods", "conclusions", "debates", "gaps", "actors"):
        seen: dict[str, float] = {}
        items: list[dict] = []
        for idx, result in enumerate(results):
            section = sections[idx] if sections and idx < len(sections) else ""
            for item in getattr(result, category, []):
                label = item.get("label", "").strip().lower()
                conf = item.get("confidence", 0.0)
                if label and (label not in seen or conf > seen[label]):
                    seen[label] = conf
                    # Remove previous entry with lower confidence
                    items = [i for i in items if i.get("label", "").strip().lower() != label]
                    if section:
                        item = {**item, "section": section}
                    items.append(item)
        merged[category] = items

    # Relations: deduplicate by (head, rel, tail)
    seen_rels: set[tuple[str, str, str]] = set()
    for result in results:
        for rel in result.relations:
            key = (
                rel.get("head", "").strip().lower(),
                rel.get("rel", "").strip().lower(),
                rel.get("tail", "").strip().lower(),
            )
            if key not in seen_rels:
                seen_rels.add(key)
                merged["relations"].append(rel)

    # Arguments: deduplicate by (claim, target) pair, keep highest confidence
    seen_args: dict[tuple[str, str], int] = {}  # (claim, target) -> index
    raw_args: list[dict] = []
    for result in results:
        for arg in result.arguments:
            arg_key = (arg.claim.strip().lower(), arg.target.strip().lower())
            if arg_key in seen_args:
                # Keep higher confidence
                idx = seen_args[arg_key]
                if arg.confidence > raw_args[idx].get("confidence", 0):
                    raw_args[idx] = arg.to_dict()
            else:
                seen_args[arg_key] = len(raw_args)
                raw_args.append(arg.to_dict())
    merged["arguments"] = raw_args

    return ExtractedConcepts(merged)


async def extract_section_concepts(
    section_title: str,
    section_text: str,
    structure_json: str,
    models: list[dict],
) -> ExtractedConcepts | None:
    """Extract concepts from a single document section with tree context."""
    system_prompt = PROMPT_TEMPLATE.read_text(encoding="utf-8")
    user_prompt = (
        f"Document Structure:\n{structure_json}\n\n"
        f"Section: {section_title}\n\n"
        f"Section Content:\n{section_text}"
    )
    data = await acall_with_fallback(
        prompt=user_prompt,
        models=models,
        system_prompt=system_prompt,
    )
    if data is None or not isinstance(data, dict):
        return None
    return ExtractedConcepts(data)


async def extract_concepts_from_tree(
    md_path: str | Path,
    structure: list[dict],
    models: list[dict],
    max_concurrent: int = 10,
) -> ExtractedConcepts | None:
    """Extract concepts using PageIndex tree structure (structure-first, content-on-demand).

    Instead of sending the full paper text (truncated to 8000 chars), this:
    1. Sends the tree skeleton (summaries without text) as LLM context
    2. Extracts content per section via get_node_content()
    3. Merges results with deduplication
    """
    if not models:
        return None

    # Get tree skeleton for LLM context
    structure_json = get_document_structure_json(structure)

    # Collect leaf nodes (actual content sections)
    leaves = _collect_leaf_nodes(structure)
    if not leaves:
        log.warning("No leaf nodes found in tree structure")
        return None

    md_path = Path(md_path)

    # Extract from each leaf node with concurrency limit
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _extract_one(title: str, content: str):
        async with semaphore:
            return await extract_section_concepts(title, content, structure_json, models)

    tasks = []
    section_names = []
    for leaf in leaves:
        content = get_node_content(md_path, structure, leaf["node_id"])
        if not content or not _is_quality_content(content):
            continue
        tasks.append(_extract_one(leaf["title"], content))
        section_names.append(leaf["title"])

    if not tasks:
        log.warning("No content found in any tree section")
        return None

    results = await asyncio.gather(*tasks)
    valid_with_sections = [(r, s) for r, s in zip(results, section_names) if r is not None]

    if not valid_with_sections:
        return None

    valid = [r for r, _ in valid_with_sections]
    sections = [s for _, s in valid_with_sections]
    merged = _merge_concepts(valid, sections=sections)
    return merged
