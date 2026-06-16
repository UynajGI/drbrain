"""5-stage tree-based extraction pipeline (ontology→entities→relations→coref→refine)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from drbrain.extractor.concept.tree_helpers import (
    _apply_tree_weights,
    _build_tree_edges,
    _build_tree_hierarchy_text,
    _collect_leaf_nodes,
    _is_quality_content,
    _section_type_hints,
)
from drbrain.extractor.llm_client import acall_with_fallback
from drbrain.parser.pageindex_parser import get_node_content

if TYPE_CHECKING:
    from drbrain.extractor.cache import ApiCache

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


async def build_graph_from_tree(
    md_path: str | Path,
    structure: list[dict],
    models: list[dict],
    skip_refine: bool = False,
    *,
    cache: ApiCache | None = None,
) -> dict:
    """5-stage graph extraction from a document tree.

    .. deprecated::
        This function duplicates the logic in
        ``extractor.agent.OntologyAgent`` / ``EntityAgent`` /
        ``RelationAgent`` / ``CorefAgent`` / ``RefineAgent``.
        Prefer the Agent-based path (``agent.get_agent(name)``) for new
        code.  This function is kept for backward compatibility with
        ``cli/build_commands.py``.

    Stages: ontology -> entities -> relations -> coreference -> refine.
    When ``cache`` is provided, every LLM call in the pipeline is de-duplicated
    via the ApiCache (re-runs / re-ingestion of unchanged sections hit cache).
    Returns {"concepts": [...], "relations": [...], "merges": [...], "corrections": [...]}
    """
    md_path = Path(md_path)
    leaves = _collect_leaf_nodes(structure)
    if not leaves:
        return {"concepts": [], "relations": [], "merges": [], "corrections": []}

    import time as _ctime

    _ct0 = _ctime.monotonic()
    log.info("[build] Stage 1/5 ontology — %d leaf nodes", len(leaves))

    # Stage 1: Ontology Extension
    ontology = await _build_ontology(structure, models, cache=cache)
    _ct1 = _ctime.monotonic()
    log.info("[build] ontology done in %.1fs — %d types", _ct1 - _ct0, len(ontology))

    # Stage 2: Entity Extraction (tree-guided with section hints)
    concepts = await _extract_entities(md_path, structure, leaves, ontology, models, cache=cache)

    if not concepts:
        return {"concepts": [], "relations": [], "merges": [], "corrections": []}

    _ct2 = _ctime.monotonic()
    log.info("[build] Stage 2/5 entities done in %.1fs — %d concepts", _ct2 - _ct1, len(concepts))

    # Stage 2.5: Apply tree position → confidence weight
    _apply_tree_weights(concepts, leaves, structure)

    # Stage 3: Relation Extraction
    relations = await _extract_relations(concepts, models, cache=cache)

    _ct3 = _ctime.monotonic()
    log.info("[build] Stage 3/5 relations done in %.1fs — %d edges", _ct3 - _ct2, len(relations))

    # Stage 3.5: Add tree hierarchy edges (section contains subsection)
    tree_edges = _build_tree_edges(structure)
    relations.extend(tree_edges)

    # Stage 4: Coreference Resolution
    concepts, merges = await _resolve_coreferences(concepts, models, cache=cache)

    _ct4 = _ctime.monotonic()
    log.info("[build] Stage 4/5 coref done in %.1fs — %d merges", _ct4 - _ct3, len(merges))

    # Stage 5: Iterative Refinement (optional)
    corrections = []
    if not skip_refine:
        corrections = await _refine_extraction(concepts, relations, models, cache=cache)
        _ct5 = _ctime.monotonic()
        log.info(
            "[build] Stage 5/5 refine done in %.1fs — %d corrections", _ct5 - _ct4, len(corrections)
        )

    log.info(
        "[build] total %.1fs — %d concepts, %d relations, %d merges",
        _ctime.monotonic() - _ct0,
        len(concepts),
        len(relations),
        len(merges),
    )

    return {
        "concepts": concepts,
        "relations": relations,
        "merges": merges,
        "corrections": corrections,
    }


async def _build_ontology(
    structure: list[dict],
    models: list[dict],
    *,
    cache: ApiCache | None = None,
) -> dict[str, list[str]]:
    """Stage 1: Tree-to-ontology with iterative extension + plateau detection.

    Upgrades the document TOC hierarchy to ontology classes under TBox 6 types.
    Section headings (e.g., "3.1.1 Datasets") become ontology subcategories,
    preserving the author-crafted structure. Iterative sampling extends the
    ontology with content-level discoveries.

    Synthesis: PageIndex tree-structure (author intent) × 2511.11017 iterative
    ontology expansion (LLM-guided).
    """
    import json as _json
    import random

    from loguru import logger as _onto_log

    valid_types = {"Problem", "Method", "Conclusion", "Gap", "Debate", "Actor"}
    prompt = ONTOLOGY_PROMPT.read_text(encoding="utf-8")

    # Build hierarchical structure summary: show parent-child relationships
    tree_hierarchy_text = _build_tree_hierarchy_text(structure)

    # Initial ontology from hierarchical TOC (preserves author structure)
    data = await acall_with_fallback(
        prompt=(
            f"Document Section Hierarchy (TOC with parent-child relationships):\n"
            f"{tree_hierarchy_text}\n\n"
            f"Map section headings to ontology subcategories under the 6 TBox types: "
            f"{', '.join(sorted(valid_types))}. Preserve the TOC hierarchy: "
            f"child sections should map to subcategories of their parent section's type "
            f"when semantically appropriate."
        ),
        models=models,
        system_prompt=prompt,
        _cache=cache,
    )
    if not data:
        return {}

    ontology = {k: v for k, v in data.items() if k in valid_types and isinstance(v, list)}
    prev_total = sum(len(v) for v in ontology.values())
    _onto_log.info(
        f"Ontology round 1 (from TOC): {prev_total} subcategories across {len(ontology)} types"
    )

    # Collect leaf nodes for content-level sampling
    leaves = _collect_leaf_nodes(structure)
    if not leaves:
        return ontology

    # Iterative extension: sample leaf content for additions beyond TOC.
    # Optimization: send a compact existing-subcategory summary instead of the
    # full indented ontology JSON. The LLM still sees what exists (to avoid
    # duplicates) but the prompt is far smaller, reducing token cost per round.
    sample_size = min(5, len(leaves))
    for round_num in range(2, 7):
        sampled = random.sample(leaves, min(sample_size, len(leaves)))
        context = "\n---\n".join(
            f"{leaf.get('title', '')}:\n{_json.dumps(leaf, indent=2, default=str)}"
            for leaf in sampled
        )
        # Flatten existing subcategories into one compact line per type
        existing_summary = "\n".join(f"{k}: {', '.join(v)}" for k, v in ontology.items() if v)

        data = await acall_with_fallback(
            prompt=(
                f"Existing ontology subcategories (do NOT repeat these):\n"
                f"{existing_summary}\n\nNew Sections:\n{context}"
            ),
            models=models,
            system_prompt=f"{prompt}\n\nExtend the ontology with new subcategories "
            f"found in the new sections. Add only genuinely NEW categories "
            f"not already present. Do not remove existing entries.",
            _cache=cache,
        )
        if not data:
            break

        new_count = 0
        for k, v in data.items():
            if k in valid_types and isinstance(v, list):
                existing = set(ontology.get(k, []))
                for item in v:
                    if item not in existing:
                        ontology.setdefault(k, []).append(item)
                        new_count += 1

        total = sum(len(v) for v in ontology.values())
        _onto_log.info(f"Ontology round {round_num}: +{new_count} new, {total} total")

        if _is_plateau_reached(new_count, total):
            _onto_log.info(f"Ontology plateau reached at round {round_num}")
            break
        prev_total = total

    return ontology


def _is_plateau_reached(new_count: int, total: int) -> bool:
    """Adaptive plateau detection for iterative ontology extension.

    Returns True when growth has diminished sufficiently:
    - Zero growth: no new elements at all
    - Relative threshold: new elements < 5% of total ontology size

    Inspired by 2511.11017's iterative ontology expansion plateau detection.
    """
    if new_count == 0:
        return True
    if total > 0 and new_count / total < 0.05:
        return True
    return False


async def _extract_entities(
    md_path: Path,
    structure: list[dict],
    leaves: list[dict],
    ontology: dict[str, list[str]],
    models: list[dict],
    *,
    cache: ApiCache | None = None,
) -> list[dict]:
    """Stage 2: Per leaf node, extract concepts with subcategories.

    Uses section-type hints to bias extraction: e.g., Method sections
    are prompted to focus on Method concepts, Results on Conclusions.
    Leaves are ordered by priority: Methods/Results first, then others.
    """
    import json as _json

    prompt_tpl = ENTITIES_PROMPT.read_text(encoding="utf-8")
    semaphore = asyncio.Semaphore(10)
    all_concepts: list[dict] = []
    seen: set[tuple[str, str]] = set()

    # Tree-guided ordering: prioritize high-signal sections first
    priority_keywords = ["method", "approach", "experiment", "results", "evaluation"]

    def _leaf_priority(leaf: dict) -> int:
        t = leaf.get("title", "").lower()
        for i, kw in enumerate(priority_keywords):
            if kw in t:
                return i
        return len(priority_keywords)

    ordered_leaves = sorted(leaves, key=_leaf_priority)

    async def _extract_one(leaf: dict) -> list[dict]:
        content = get_node_content(md_path, structure, leaf["node_id"])
        if not content or not _is_quality_content(content):
            return []
        hints = _section_type_hints(leaf.get("title", ""))
        hints_str = ", ".join(f"{k}" for k in hints)
        user = prompt_tpl.format(
            ontology=_json.dumps(ontology, indent=2),
            section_title=leaf.get("title", ""),
            section_text=content,
        )
        # Append section hint to guide extraction focus
        if hints_str:
            user += f"\n\n[Section hint: this section is likely about {hints_str}. Focus extraction on these concept types.]"
        async with semaphore:
            data = await acall_with_fallback(
                prompt=user, models=models, system_prompt=prompt_tpl, _cache=cache
            )
        if not data:
            return []
        concepts = data.get("concepts", [])
        # Tag each concept with its source section + node_id for tree provenance
        section_title = leaf.get("title", "")
        leaf_node_id = leaf.get("node_id", "")
        for c in concepts:
            c["section"] = section_title
            c["node_id"] = leaf_node_id
        return concepts

    tasks = [_extract_one(leaf) for leaf in ordered_leaves]
    results = await asyncio.gather(*tasks)
    for concepts in results:
        for c in concepts:
            key = (c.get("label", "").strip().lower(), c.get("type", ""))
            if key not in seen:
                seen.add(key)
                all_concepts.append(c)
    return all_concepts


async def _extract_relations(
    concepts: list[dict], models: list[dict], *, cache: ApiCache | None = None
) -> list[dict]:
    """Stage 3: LLM connects entities with TBox relations.

    Each relation inherits node_id and section from its head (source) concept,
    maintaining the provenance chain: edge → concept → tree node → paper.
    """
    # Build label→concept lookup for provenance transfer
    concept_map: dict[str, dict] = {}
    for c in concepts:
        key = c.get("label", "").strip().lower()
        if key:
            concept_map[key] = c

    concept_list = [f"{c['label']}: {c['type']}" for c in concepts]
    prompt = RELATIONS_PROMPT.read_text(encoding="utf-8")
    user = prompt.format(concepts="\n".join(concept_list))
    data = await acall_with_fallback(prompt=user, models=models, system_prompt=prompt, _cache=cache)
    if not data:
        return []
    relations = data.get("relations", [])
    # Attach provenance from source concept
    for r in relations:
        head_label = r.get("head", "").strip().lower()
        src = concept_map.get(head_label)
        if src:
            r["node_id"] = src.get("node_id", "")
            r["section"] = src.get("section", "")
    return relations


async def _resolve_coreferences(
    concepts: list[dict],
    models: list[dict],
    *,
    cache: ApiCache | None = None,
) -> tuple[list[dict], list[dict]]:
    """Stage 4: LLM merges duplicate entity labels."""
    concept_list = [f"{c['label']}: {c['type']}" for c in concepts]
    prompt = COREFERENCE_PROMPT.read_text(encoding="utf-8")
    user = prompt.format(concepts="\n".join(concept_list))
    data = await acall_with_fallback(prompt=user, models=models, system_prompt=prompt, _cache=cache)
    if not data:
        return concepts, []

    merges = data.get("merges", [])
    # Build canonical map from merges
    canonical_map: dict[str, str] = {}
    for m in merges:
        canonical = m.get("canonical", "")
        for variant in m.get("variants", []):
            canonical_map[variant.strip().lower()] = canonical

    # Apply merges
    merged = []
    for c in concepts:
        label = c.get("label", "")
        if label.strip().lower() in canonical_map:
            c = dict(c)
            c["_merged_from"] = label
            c["label"] = canonical_map[label.strip().lower()]
        merged.append(c)
    return merged, merges


async def _refine_extraction(
    concepts: list[dict],
    relations: list[dict],
    models: list[dict],
    *,
    cache: ApiCache | None = None,
) -> list[dict]:
    """Stage 5: LLM self-reviews and corrects the extraction."""
    prompt = REFINE_PROMPT.read_text(encoding="utf-8")
    user = prompt.format(concept_count=len(concepts), relation_count=len(relations))
    data = await acall_with_fallback(prompt=user, models=models, system_prompt=prompt, _cache=cache)
    if not data:
        return []
    return data.get("corrections", [])
