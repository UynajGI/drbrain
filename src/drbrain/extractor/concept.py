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
    """Log cross-section argument patterns for debugging.

    Detects arguments referencing the same target from different sections
    and logs the pattern. No graph edges are created — section provenance
    is already captured in each argument's section field.
    """
    from collections import defaultdict

    by_target: dict[str, list[ExtractedArgument]] = defaultdict(list)
    for arg in concepts.arguments:
        target = arg.target.strip()
        if target:
            by_target[target].append(arg)

    for target, args in by_target.items():
        sections_seen: dict[str, str] = {}
        for arg in args:
            section = arg.section.strip()
            claim_type = arg.claim_type.strip().lower()
            if section and section not in sections_seen:
                sections_seen[section] = claim_type

        if len(sections_seen) < 2:
            continue

        claim_types = set(sections_seen.values())
        has_opposition = ("limitation" in claim_types and "advantage" in claim_types) or (
            "challenges" in claim_types and "supports" in claim_types
        )
        rel_type = "cross_section_challenge" if has_opposition else "cross_section_support"

        log.debug(
            "Cross-section %s: target=%r from %d sections (%s)",
            rel_type,
            target,
            len(sections_seen),
            ", ".join(sections_seen.keys()),
        )

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


def _section_type_hints(title: str) -> dict[str, float]:
    """Map section title to likely concept type probabilities.

    Uses keyword matching against common academic section names.
    Returns dict of {type: weight} for use in extraction prompts.
    """
    t = title.lower().strip()
    hints: dict[str, dict[str, float]] = {
        "abstract": {"Problem": 0.9, "Gap": 0.5},
        "introduction": {"Problem": 0.8, "Gap": 0.6, "Method": 0.2},
        "related work": {"Method": 0.3, "Gap": 0.5},
        "background": {"Problem": 0.7, "Method": 0.3},
        "method": {"Method": 0.9},
        "methodology": {"Method": 0.9},
        "approach": {"Method": 0.8},
        "experiment": {"Method": 0.5, "Conclusion": 0.3},
        "results": {"Conclusion": 0.7, "Method": 0.2},
        "evaluation": {"Conclusion": 0.6, "Method": 0.3},
        "discussion": {"Conclusion": 0.5, "Debate": 0.4, "Gap": 0.3},
        "conclusion": {"Conclusion": 0.9},
        "future work": {"Gap": 0.8},
        "limitation": {"Gap": 0.7, "Debate": 0.3},
    }
    # Find best matching section
    for key, weights in hints.items():
        if key in t:
            return weights
    # Default: slight Problem bias for unknown sections
    return {"Problem": 0.3, "Method": 0.3, "Conclusion": 0.2}


def _tree_position_weight(node: dict, depth: int = 0, max_depth: int = 5) -> float:
    """Compute confidence weight for concepts extracted from a tree node.

    Concepts from deep in the tree (specialized subsections) get higher weight.
    Concepts from shallow sections (e.g. Abstract, Introduction) get lower weight.
    Returns weight in [0.5, 1.0].
    """
    title = node.get("title", "").lower().strip()
    # Shallow sections: lower confidence
    shallow_keywords = {"abstract", "introduction", "related work", "background"}
    for kw in shallow_keywords:
        if kw in title and depth <= 2:
            return 0.6
    # Deep specialized sections: higher confidence
    if depth >= 4:
        return 0.95
    # Scale by depth
    return min(1.0, 0.5 + depth / max_depth * 0.5)


def _build_tree_edges(structure: list[dict], parent_id: str = "root") -> list[dict]:
    """Create 'contains' edges from tree parent-child relationships.

    Returns list of {head, rel, tail} matching the LLM extraction format
    so build_cmd's edge insertion loop handles them correctly.
    Uses section titles as node identifiers.
    """
    edges = []
    for node in structure:
        title = node.get("title", "")
        if title:
            edges.append(
                {
                    "head": parent_id if parent_id != "root" else "document",
                    "rel": "contains",
                    "tail": title,
                    "weight": 0.9,
                }
            )
        children = node.get("nodes", [])
        if children:
            edges.extend(_build_tree_edges(children, title))
    return edges


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

    # Stage 2: Entity Extraction (tree-guided with section hints)
    concepts = await _extract_entities(md_path, structure, leaves, ontology, models)

    if not concepts:
        return {"concepts": [], "relations": [], "merges": [], "corrections": []}

    # Stage 2.5: Apply tree position → confidence weight
    _apply_tree_weights(concepts, leaves, structure)

    # Stage 3: Relation Extraction
    relations = await _extract_relations(concepts, models)

    # Stage 3.5: Add tree hierarchy edges (section contains subsection)
    tree_edges = _build_tree_edges(structure)
    relations.extend(tree_edges)

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

    # Iterative extension: sample leaf content for additions beyond TOC
    sample_size = min(5, len(leaves))
    for round_num in range(2, 7):
        sampled = random.sample(leaves, min(sample_size, len(leaves)))
        context = "\n---\n".join(
            f"{leaf.get('title', '')}:\n{_json.dumps(leaf, indent=2, default=str)}"
            for leaf in sampled
        )

        data = await acall_with_fallback(
            prompt=(
                f"Current Ontology:\n{_json.dumps(ontology, indent=2)}\n\nNew Sections:\n{context}"
            ),
            models=models,
            system_prompt=f"{prompt}\n\nExtend the ontology with new subcategories "
            f"found in the new sections. Add only genuinely NEW categories "
            f"not already present. Do not remove existing entries.",
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


def _build_tree_hierarchy_text(structure: list[dict], indent: int = 0) -> str:
    """Render TOC hierarchy with parent-child relationships for ontology mapping.

    Output format:
        ├── 3. Methodology [depth=1]
        │   ├── 3.1 Dataset Construction [depth=2]
        │   │   └── 3.1.1 Data Sources [depth=3]
        │   └── 3.2 Evaluation Metrics [depth=2]

    This preserves the author's organizational intent so the LLM can map
    section hierarchy to ontology class hierarchy.
    """
    lines: list[str] = []

    def _walk(nodes: list[dict], depth: int, prefix: str = ""):
        for i, node in enumerate(nodes):
            title = node.get("title", "(untitled)")
            is_last = i == len(nodes) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{title} [depth={depth}]")
            children = node.get("nodes", []) or node.get("children", [])
            if children:
                child_prefix = prefix + ("    " if is_last else "│   ")
                _walk(children, depth + 1, child_prefix)

    _walk(structure, 0)
    return "\n".join(lines)


def _apply_tree_weights(concepts: list[dict], leaves: list[dict], structure: list[dict]) -> None:
    """Apply tree-position-based confidence weighting to concepts.

    Builds a leaf-node lookup indexed by node_id, then walks depth for
    each concept's source leaf. Concepts from deeper, specialized sections
    get higher confidence than shallow/general sections.
    """
    # Build node_id → depth map by walking tree
    node_depths: dict[str, int] = {}

    def _walk(nodes: list[dict], depth: int):
        for node in nodes:
            nid = node.get("node_id", "")
            if nid:
                node_depths[nid] = depth
            _walk(node.get("nodes", []), depth + 1)

    _walk(structure, 0)

    # Leaf lookup: {leaf_title: leaf_node}
    leaf_by_title: dict[str, dict] = {}
    for leaf in leaves:
        t = leaf.get("title", "")
        if t:
            leaf_by_title[t.lower().strip()] = leaf

    for c in concepts:
        section = (c.get("section", "") or "").lower().strip()
        leaf = leaf_by_title.get(section)
        if leaf:
            depth = node_depths.get(leaf.get("node_id", ""), 2)
            weight = _tree_position_weight(leaf, depth)
            # Blend with existing confidence
            existing = c.get("confidence", 1.0)
            c["confidence"] = round(existing * weight, 3)


async def _extract_entities(
    md_path: Path,
    structure: list[dict],
    leaves: list[dict],
    ontology: dict[str, list[str]],
    models: list[dict],
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
            data = await acall_with_fallback(prompt=user, models=models, system_prompt=prompt_tpl)
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


async def _extract_relations(concepts: list[dict], models: list[dict]) -> list[dict]:
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
    data = await acall_with_fallback(prompt=user, models=models, system_prompt=prompt)
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
