"""Tree-structure helpers: leaf collection, quality filtering, section hints."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

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
        found_leaf: dict | None = leaf_by_title.get(section)
        if found_leaf:
            depth = node_depths.get(found_leaf.get("node_id", ""), 2)
            weight = _tree_position_weight(found_leaf, depth)
            # Blend with existing confidence
            existing = c.get("confidence", 1.0)
            c["confidence"] = round(existing * weight, 3)


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
