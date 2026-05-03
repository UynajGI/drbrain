"""Structure-first retrieval using PageIndex tree with iterative search.

Adapted from PageIndex (https://github.com/vectify-ai/pageindex).
Original code Copyright (c) 2025 Vectify AI, MIT License.

Implements the PageIndex retrieval approach:
1. Read tree skeleton (summaries without text) — token-efficient
2. LLM iteratively navigates the tree: pick candidates → read → decide if more needed
3. Load content on-demand only for sections that survived the filter

This avoids sending the full document and simulates how human experts
navigate complex documents through tree search.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from drbrain.extractor.llm_client import acall_with_fallback
from drbrain.parser.pageindex_parser import get_document_structure_json, get_node_content

log = logging.getLogger(__name__)

_DEFAULT_MAX_ROUNDS = 2
_DEFAULT_PER_ROUND = 3

_SYSTEM_PROMPT = (
    "You are a document retrieval assistant using tree-search to find relevant sections. "
    "Your job is to navigate a document's hierarchical structure like a human researcher: "
    "scan the table of contents, identify promising sections, read them, then decide if "
    "you need more. Always return valid JSON."
)

_ROUND1_PROMPT = """You are searching a document for content relevant to a question.

Step 1: Read the document structure below (section titles with summaries).
Step 2: Identify the {per_round} most promising leaf sections that likely contain the answer.
Step 3: Return their node_ids.

Document Structure:
{structure_json}

Question: {question}

Return STRICT JSON (no markdown):
{{"node_ids": ["id1", "id2", ...], "reasoning": "one sentence about why you chose these"}}"""

_ROUND2_PROMPT = """You previously selected these sections and read their content:

{previous_content}

Based on what you've read, decide if you need more sections from the remaining structure.

Remaining sections (not yet read):
{remaining_structure}

Question: {question}

Return STRICT JSON:
{{"node_ids": ["id1", "id2", ...], "done": true_or_false, "reasoning": "one sentence"}}

If the content you've already read sufficiently answers the question, return "done": true with empty node_ids.
If you need more sections, return "done": false with up to {per_round} additional node_ids."""


async def _ask_llm_for_relevant_nodes(
    question: str,
    structure_json: str,
    models: list[dict],
) -> list[str]:
    """Legacy: one-shot section selection. Kept for backward compatibility."""
    prompt = _ROUND1_PROMPT.format(
        structure_json=structure_json,
        question=question,
        per_round=5,
    )
    result = await acall_with_fallback(
        prompt=prompt, models=models, system_prompt=_SYSTEM_PROMPT, max_tokens=1024,
    )
    if result is None:
        return []
    if isinstance(result, list):
        return [str(nid) for nid in result[:5]]
    if isinstance(result, dict):
        ids = result.get("node_ids", [])
        if isinstance(ids, list):
            return [str(nid) for nid in ids[:5]]
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


def _collect_all_leaf_ids(structure: list[dict]) -> set[str]:
    """Collect all leaf node IDs from the tree."""
    ids: set[str] = set()
    for node in structure:
        children = node.get("nodes", [])
        if not children:
            ids.add(node.get("node_id", ""))
        else:
            ids.update(_collect_all_leaf_ids(children))
    ids.discard("")
    return ids


def _build_remaining_structure(structure: list[dict], read_ids: set[str]) -> str:
    """Build a structure JSON string with only unread leaf nodes."""
    import copy

    def _filter_read(nodes):
        result = []
        for node in nodes:
            children = node.get("nodes", [])
            if children:
                filtered_children = _filter_read(children)
                if filtered_children:
                    node_copy = dict(node)
                    node_copy["nodes"] = filtered_children
                    result.append(node_copy)
            elif node.get("node_id", "") not in read_ids:
                result.append(dict(node))
        return result

    filtered = _filter_read(structure)
    if not filtered:
        return "{}"
    return json.dumps(filtered, ensure_ascii=False, indent=2)


async def query_by_structure(
    question: str,
    paper_dir: Path,
    models: list[dict],
    max_rounds: int = _DEFAULT_MAX_ROUNDS,
    per_round: int = _DEFAULT_PER_ROUND,
) -> list[dict] | None:
    """PageIndex iterative tree-search retrieval.

    1. Show LLM tree skeleton → pick initial candidate sections
    2. Load content for those sections
    3. LLM reviews content + remaining tree → decides if more needed
    4. If yes, load additional sections and repeat (up to max_rounds)

    This simulates how human experts navigate documents:
    scan TOC → read promising sections → decide if research is complete.
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

    all_leaf_ids = _collect_all_leaf_ids(structure)
    structure_json = get_document_structure_json(structure)

    # ── Round 1: Initial selection from tree skeleton ──
    r1_prompt = _ROUND1_PROMPT.format(
        structure_json=structure_json,
        question=question,
        per_round=per_round,
    )
    r1 = await acall_with_fallback(
        prompt=r1_prompt, models=models, system_prompt=_SYSTEM_PROMPT, max_tokens=1024,
    )
    if r1 is None:
        return None

    round1_ids = []
    if isinstance(r1, dict):
        round1_ids = r1.get("node_ids", [])
    if not round1_ids:
        return None

    round1_ids = [str(nid) for nid in round1_ids[:per_round]]

    # Load round 1 content
    collected: dict[str, dict] = {}
    for nid in round1_ids:
        content = get_node_content(md_path, structure, nid)
        if content and content.strip():
            title = _get_node_title(structure, nid)
            collected[nid] = {"node_id": nid, "title": title or "", "content": content.strip()}

    if not collected:
        return None

    # ── Round 2: Review + expand ──
    for round_num in range(2, max_rounds + 1):
        read_ids = set(collected.keys())
        remaining_ids = all_leaf_ids - read_ids
        if not remaining_ids:
            break

        # Build context from already-read content
        prev_parts = []
        for nid, sec in collected.items():
            prev_parts.append(f"### {sec['title']} ({nid})\n{sec['content'][:500]}...")
        previous_text = "\n\n".join(prev_parts)

        remaining_json = _build_remaining_structure(structure, read_ids)

        r2_prompt = _ROUND2_PROMPT.format(
            previous_content=previous_text,
            remaining_structure=remaining_json,
            question=question,
            per_round=per_round,
        )
        r2 = await acall_with_fallback(
            prompt=r2_prompt, models=models, system_prompt=_SYSTEM_PROMPT, max_tokens=512,
        )
        if r2 is None or not isinstance(r2, dict):
            break

        if r2.get("done", False):
            break

        additional_ids = r2.get("node_ids", [])
        if not additional_ids:
            break

        for nid in additional_ids:
            nid = str(nid)
            if nid in collected:
                continue
            content = get_node_content(md_path, structure, nid)
            if content and content.strip():
                title = _get_node_title(structure, nid)
                collected[nid] = {"node_id": nid, "title": title or "", "content": content.strip()}

    if not collected:
        return None
    return list(collected.values())
