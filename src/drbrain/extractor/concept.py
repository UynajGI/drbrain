"""Academic concept + argument extraction via LLM with fallback chain.

Tree-based extraction adapted from PageIndex (https://github.com/vectify-ai/pageindex).
Original code Copyright (c) 2025 Vectify AI, MIT License.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from drbrain.extractor.argument import ExtractedArgument, parse_arguments
from drbrain.extractor.llm_client import acall_with_fallback
from drbrain.parser.pageindex_parser import get_document_structure_json, get_node_content

log = logging.getLogger(__name__)

PROMPT_TEMPLATE = Path(__file__).parent.parent.parent.parent / "prompts" / "extract_concepts.txt"
ONTOLOGY_PROMPT = Path(__file__).parent.parent.parent.parent / "prompts" / "ontology.txt"
ENTITIES_PROMPT = Path(__file__).parent.parent.parent.parent / "prompts" / "entities.txt"
RELATIONS_PROMPT = Path(__file__).parent.parent.parent.parent / "prompts" / "relations.txt"
COREFERENCE_PROMPT = Path(__file__).parent.parent.parent.parent / "prompts" / "coreference.txt"
REFINE_PROMPT = Path(__file__).parent.parent.parent.parent / "prompts" / "refine.txt"


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
    if data is None:
        return None
    return ExtractedConcepts(data)


# -- Tree-based extraction (PageIndex approach) --


def _collect_leaf_nodes(nodes: list[dict]) -> list[dict]:
    """Collect leaf nodes from tree structure — nodes with no children or empty nodes."""
    leaves = []
    for node in nodes:
        children = node.get("nodes", [])
        if not children:
            leaves.append(
                {
                    "node_id": node.get("node_id", ""),
                    "title": node.get("title", ""),
                    "line_num": node.get("line_num", 0),
                    "summary": node.get("summary", ""),
                }
            )
        else:
            leaves.extend(_collect_leaf_nodes(children))
    return leaves


def _is_quality_content(text: str, min_chars: int = 100) -> bool:
    """Check if content is worth sending to LLM.

    Rejects short text, reference lists, and low-alpha-ratio content.
    """
    if len(text.strip()) < min_chars:
        return False
    # Filter reference lists (lines starting with [数字])
    lines = text.strip().split("\n")
    ref_lines = sum(1 for line in lines if re.match(r"^\[\d+\]", line.strip()))
    if ref_lines > len(lines) * 0.6:
        return False
    # Filter pages that are mostly numbers/captions
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    return alpha_ratio > 0.3


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


def _link_cross_section_arguments(concepts: ExtractedConcepts) -> ExtractedConcepts:
    """Link arguments across sections that share targets.

    For arguments with the same target but different sections, adds
    synthetic relations (cross_section_support / cross_section_challenge).
    """
    from collections import defaultdict

    # Group arguments by target
    by_target: dict[str, list[ExtractedArgument]] = defaultdict(list)
    for arg in concepts.arguments:
        target = arg.target.strip()
        if target:
            by_target[target].append(arg)

    new_relations = []
    for target, args in by_target.items():
        # Collect unique sections
        sections_seen: dict[str, str] = {}  # section -> claim_type
        for arg in args:
            section = arg.section.strip()
            claim_type = arg.claim_type.strip().lower()
            if section and section not in sections_seen:
                sections_seen[section] = claim_type

        # Only link if args come from 2+ different sections
        if len(sections_seen) < 2:
            continue

        # Determine relation type: challenge if opposing claim_types exist
        claim_types = set(sections_seen.values())
        has_opposition = ("limitation" in claim_types and "advantage" in claim_types) or (
            "challenges" in claim_types and "supports" in claim_types
        )
        rel_type = "cross_section_challenge" if has_opposition else "cross_section_support"

        # Cross-section argument links are logged, not added as graph edges.
        # The section provenance is already captured in each argument's section field.
        log.debug(
            "Cross-section %s: target=%r from %d sections (%s)",
            rel_type,
            target,
            len(sections_seen),
            ", ".join(sections_seen.keys()),
        )

    if new_relations:
        concepts.relations.extend(new_relations)
    return concepts


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
            key = (arg.claim.strip().lower(), arg.target.strip().lower())
            if key in seen_args:
                # Keep higher confidence
                idx = seen_args[key]
                if arg.confidence > raw_args[idx].get("confidence", 0):
                    raw_args[idx] = arg.to_dict()
            else:
                seen_args[key] = len(raw_args)
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
    if data is None:
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
    return _link_cross_section_arguments(merged)


async def build_graph_from_tree(
    md_path: str | Path,
    structure: list[dict],
    models: list[dict],
    skip_refine: bool = False,
) -> dict:
    """5-stage graph extraction from a document tree.

    Stages: ontology -> entities -> relations -> coreference -> refine.
    Returns {"concepts": [...], "relations": [...], "merges": [...], "corrections": [...]}
    """
    md_path = Path(md_path)
    leaves = _collect_leaf_nodes(structure)
    if not leaves:
        return {"concepts": [], "relations": [], "merges": [], "corrections": []}

    # Stage 1: Ontology Extension
    ontology = await _build_ontology(structure, models)

    # Stage 2: Entity Extraction
    concepts = await _extract_entities(md_path, structure, leaves, ontology, models)

    if not concepts:
        return {"concepts": [], "relations": [], "merges": [], "corrections": []}

    # Stage 3: Relation Extraction
    relations = await _extract_relations(concepts, models)

    # Stage 4: Coreference Resolution
    concepts, merges = await _resolve_coreferences(concepts, models)

    # Stage 5: Iterative Refinement (optional)
    corrections = []
    if not skip_refine:
        corrections = await _refine_extraction(concepts, relations, models)

    return {
        "concepts": concepts,
        "relations": relations,
        "merges": merges,
        "corrections": corrections,
    }


async def _build_ontology(structure: list[dict], models: list[dict]) -> dict[str, list[str]]:
    """Stage 1: LLM suggests domain-specific subcategories under 6 TBox types."""
    structure_json = get_document_structure_json(structure)
    prompt = ONTOLOGY_PROMPT.read_text(encoding="utf-8")
    user = f"Document Structure:\n{structure_json}"
    data = await acall_with_fallback(prompt=user, models=models, system_prompt=prompt)
    if not data:
        return {}
    # Filter to only valid TBox types with list values
    valid_types = {"Problem", "Method", "Conclusion", "Gap", "Debate", "Actor"}
    return {k: v for k, v in data.items() if k in valid_types and isinstance(v, list)}


async def _extract_entities(
    md_path: Path,
    structure: list[dict],
    leaves: list[dict],
    ontology: dict[str, list[str]],
    models: list[dict],
) -> list[dict]:
    """Stage 2: Per leaf node, extract concepts with subcategories."""
    import json as _json

    prompt_tpl = ENTITIES_PROMPT.read_text(encoding="utf-8")
    semaphore = asyncio.Semaphore(10)
    all_concepts: list[dict] = []
    seen: set[tuple[str, str]] = set()

    async def _extract_one(leaf: dict) -> list[dict]:
        content = get_node_content(md_path, structure, leaf["node_id"])
        if not content or not _is_quality_content(content):
            return []
        user = prompt_tpl.format(
            ontology=_json.dumps(ontology, indent=2),
            section_title=leaf.get("title", ""),
            section_text=content,
        )
        async with semaphore:
            data = await acall_with_fallback(prompt=user, models=models, system_prompt=prompt_tpl)
        if not data:
            return []
        return data.get("concepts", [])

    tasks = [_extract_one(leaf) for leaf in leaves]
    results = await asyncio.gather(*tasks)
    for concepts in results:
        for c in concepts:
            key = (c.get("label", "").strip().lower(), c.get("type", ""))
            if key not in seen:
                seen.add(key)
                all_concepts.append(c)
    return all_concepts


async def _extract_relations(concepts: list[dict], models: list[dict]) -> list[dict]:
    """Stage 3: LLM connects entities with TBox relations."""
    concept_list = [f"{c['label']}: {c['type']}" for c in concepts]
    prompt = RELATIONS_PROMPT.read_text(encoding="utf-8")
    user = prompt.format(concepts="\n".join(concept_list))
    data = await acall_with_fallback(prompt=user, models=models, system_prompt=prompt)
    if not data:
        return []
    return data.get("relations", [])


async def _resolve_coreferences(
    concepts: list[dict],
    models: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Stage 4: LLM merges duplicate entity labels."""
    concept_list = [f"{c['label']}: {c['type']}" for c in concepts]
    prompt = COREFERENCE_PROMPT.read_text(encoding="utf-8")
    user = prompt.format(concepts="\n".join(concept_list))
    data = await acall_with_fallback(prompt=user, models=models, system_prompt=prompt)
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
) -> list[dict]:
    """Stage 5: LLM self-reviews and corrects the extraction."""
    prompt = REFINE_PROMPT.read_text(encoding="utf-8")
    user = prompt.format(concept_count=len(concepts), relation_count=len(relations))
    data = await acall_with_fallback(prompt=user, models=models, system_prompt=prompt)
    if not data:
        return []
    return data.get("corrections", [])


def dedup_concepts_by_label(db) -> int:
    """Merge concepts with identical labels (case-insensitive) across papers.
    Keeps the highest confidence entry, updates edges to point to it.
    Returns number of merged pairs.
    """
    # Find exact label matches
    rows = db.conn.execute("""
        SELECT LOWER(label) as norm_label, type, COUNT(*) as cnt
        FROM concepts
        GROUP BY LOWER(label), type
        HAVING cnt > 1
    """).fetchall()

    merged = 0
    for norm_label, ctype, count in rows:
        # Get all entries for this label
        entries = db.conn.execute(
            "SELECT concept_id, label, confidence, local_id "
            "FROM concepts WHERE LOWER(label) = ? AND type = ? "
            "ORDER BY confidence DESC",
            (norm_label, ctype),
        ).fetchall()

        if len(entries) < 2:
            continue

        canonical = entries[0]  # highest confidence
        canonical_label = canonical[1]

        for dup in entries[1:]:
            dup_id = dup[0]
            dup_label = dup[1]
            # Update edges pointing to duplicate
            db.conn.execute(
                "UPDATE edges SET src_id = ? WHERE src_id = ?",
                (canonical_label, dup_label),
            )
            db.conn.execute(
                "UPDATE edges SET dst_id = ? WHERE dst_id = ?",
                (canonical_label, dup_label),
            )
            # Delete duplicate concept
            db.conn.execute("DELETE FROM concepts WHERE concept_id = ?", (dup_id,))
            merged += 1

    db.commit()
    return merged


def find_similar_labels(db, threshold: float = 0.6) -> list[tuple[str, str, float]]:
    """Find pairs of concept labels that are similar but not identical.
    Uses word overlap ratio. Returns list of (label_a, label_b, similarity).
    """
    rows = db.conn.execute(
        "SELECT DISTINCT label, type FROM concepts ORDER BY type, label"
    ).fetchall()

    pairs = []
    # Group by type
    by_type: dict[str, list[str]] = {}
    for label, ctype in rows:
        by_type.setdefault(ctype, []).append(label)

    for ctype, labels in by_type.items():
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                sim = _label_similarity(labels[i], labels[j])
                if sim >= threshold and sim < 1.0:
                    pairs.append((labels[i], labels[j], round(sim, 3)))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def _label_similarity(a: str, b: str) -> float:
    """Jaccard similarity between two label word sets."""
    import re

    a_words = set(re.split(r"[\s\-_]+", a.strip().lower()))
    b_words = set(re.split(r"[\s\-_]+", b.strip().lower()))
    if not a_words or not b_words:
        return 0.0
    union = a_words | b_words
    overlap = len(a_words & b_words)
    return overlap / len(union)
