"""Structure-first retrieval using PageIndex tree.

Adapted from PageIndex (https://github.com/vectify-ai/pageindex).
Original code Copyright (c) 2025 Vectify AI, MIT License.

Implements the PageIndex retrieval approach:
1. Read tree skeleton (summaries without text)
2. LLM reasons about which sections are relevant to a question
3. Load content on-demand from the relevant sections

This avoids sending the full document (truncated) and instead uses
the tree structure to guide targeted content retrieval.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from drbrain.extractor.llm_client import acall_with_fallback
from drbrain.parser.pageindex_parser import get_document_structure_json, get_node_content

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a document retrieval assistant. Your job is to identify which "
    "sections of a document are relevant to a given question. Analyze the "
    "document structure carefully and return only the node_id values of "
    "the most relevant sections. Always return a valid JSON array."
)

_SECTION_SELECTION_PROMPT = """Given a document structure (tree of sections with summaries) and a question, identify which sections are most likely to contain the answer.

Return a JSON array of node_id strings for the relevant sections. Only include leaf nodes (sections with actual content). Return at most 5 node IDs.

Output schema (no markdown, no extra text):
["node_id_1", "node_id_2", ...]

Document Structure:
{structure_json}

Question: {question}"""


async def _ask_llm_for_relevant_nodes(
    question: str,
    structure_json: str,
    models: list[dict],
) -> list[str]:
    """Ask the LLM to identify relevant section node_ids from the tree skeleton."""
    prompt = _SECTION_SELECTION_PROMPT.format(
        structure_json=structure_json,
        question=question,
    )
    result = await acall_with_fallback(
        prompt=prompt,
        models=models,
        system_prompt=_SYSTEM_PROMPT,
        max_tokens=256,
    )
    if result is None:
        return []
    # Result should be a list of strings (node_ids)
    if isinstance(result, list):
        return [str(nid) for nid in result[:5]]
    # If LLM wrapped it in a dict, try common keys
    if isinstance(result, dict):
        for key in ("node_ids", "ids", "sections", "relevant"):
            if key in result and isinstance(result[key], list):
                return [str(nid) for nid in result[key][:5]]
    return []


def _get_node_title(structure: list[dict], node_id: str) -> str:
    """Find a node's title by node_id in the tree structure (recursive search)."""
    for node in structure:
        if node.get("node_id") == node_id:
            return node.get("title", "")
        if node.get("nodes"):
            result = _get_node_title(node["nodes"], node_id)
            if result:
                return result
    return ""


async def query_by_structure(
    question: str,
    paper_dir: Path,
    models: list[dict],
) -> list[dict] | None:
    """PageIndex-style retrieval: read tree skeleton, reason, load content.

    Args:
        question: The query/question to find relevant content for.
        paper_dir: Path to the per-paper directory containing tree.json and raw.md.
        models: LLM model configs for section selection.

    Returns:
        List of dicts with node_id, title, content for each relevant section,
        or None if retrieval fails.
    """
    tree_path = paper_dir / "tree.json"
    md_path = paper_dir / "raw.md"

    if not tree_path.exists() or not md_path.exists():
        log.warning(f"Missing tree.json or raw.md in {paper_dir}")
        return None

    tree_data = json.loads(tree_path.read_text(encoding="utf-8"))
    structure = tree_data.get("structure", [])
    if not structure:
        return None

    # Get tree skeleton for LLM context
    structure_json = get_document_structure_json(structure)

    # LLM decides which sections are relevant
    relevant_ids = await _ask_llm_for_relevant_nodes(question, structure_json, models)
    if not relevant_ids:
        log.warning("LLM identified no relevant sections")
        return None

    # Load content on-demand
    sections = []
    for nid in relevant_ids:
        content = get_node_content(md_path, structure, nid)
        if content and content.strip():
            title = _get_node_title(structure, nid)
            sections.append({"node_id": nid, "title": title or "", "content": content.strip()})

    if not sections:
        return None

    return sections
