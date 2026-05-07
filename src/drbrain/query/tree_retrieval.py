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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from drbrain.config import EmbedConfig

from drbrain.extractor.llm_client import acall_with_fallback
from drbrain.parser.pageindex_parser import get_document_structure_json, get_node_content
from drbrain.storage.paths import raw_md_path, tree_json_path

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
        prompt=prompt,
        models=models,
        system_prompt=_SYSTEM_PROMPT,
        max_tokens=1024,
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


_MAX_SKELETON_CHARS = 8000  # If tree skeleton exceeds this, use top-level navigation


def _build_top_level_structure(structure: list[dict], depth: int = 2) -> list[dict]:
    """Collapse tree to show only top N levels. Deeper nodes replaced with child count."""
    result = []
    for node in structure:
        item = {
            "title": node.get("title", ""),
            "node_id": node.get("node_id", ""),
        }
        children = node.get("nodes", [])
        if children:
            if depth <= 1:
                item["child_count"] = len(children)
                item["children"] = []
            else:
                item["children"] = _build_top_level_structure(children, depth - 1)
        result.append(item)
    return result


def _expand_branch(structure: list[dict], selected_ids: set[str]) -> list[dict]:
    """Expand selected branches one level deeper. Returns leaf nodes under selected branches."""
    result = []
    for node in structure:
        nid = node.get("node_id", "")
        if nid in selected_ids:
            children = node.get("nodes", [])
            if children:
                result.extend(children)  # Expand: show children of selected nodes
            else:
                result.append(node)  # Already a leaf, include as-is
    return result


_NON_LEAF_SELECTION_PROMPT = """You are navigating a document's hierarchical structure to find relevant content.

The structure below shows ONLY the top-level sections (not leaf content yet).
Pick the {per_round} most relevant sections to explore further.

Document Structure:
{structure_json}

Question: {question}

Return STRICT JSON (no markdown):
{{"node_ids": ["id1", "id2"], "reasoning": "one sentence"}}"""

_LEAF_SELECTION_PROMPT = """These are the expanded sections under your previously selected branches.
Pick up to {per_round} leaf nodes that are most relevant to the question.

Expanded Sections:
{expanded_json}

Question: {question}

Return STRICT JSON (no markdown):
{{"node_ids": ["id1", "id2"], "reasoning": "one sentence"}}"""


async def query_by_structure(
    question: str,
    paper_dir: Path,
    models: list[dict],
    max_rounds: int = _DEFAULT_MAX_ROUNDS,
    per_round: int = _DEFAULT_PER_ROUND,
) -> list[dict] | None:
    """PageIndex iterative tree-search retrieval with adaptive depth.

    If tree skeleton fits in context → one-shot selection from full tree.
    If tree skeleton too large → top-level navigation: show headings first,
    LLM picks branches → expand selected branches → LLM picks leaves.
    """
    tree_path = tree_json_path(paper_dir)
    md_path = raw_md_path(paper_dir)

    if not tree_path.exists() or not md_path.exists():
        log.warning(f"Missing tree.json or raw.md in {paper_dir}")
        return None

    tree_data = json.loads(tree_path.read_text(encoding="utf-8"))
    structure = tree_data.get("structure", [])
    if not structure:
        return None

    all_leaf_ids = _collect_all_leaf_ids(structure)
    full_skeleton = get_document_structure_json(structure)
    use_navigation = len(full_skeleton) > _MAX_SKELETON_CHARS

    # ── Adaptive tree navigation ──
    selected_ids: list[str] = []

    if use_navigation:
        # Step 1: Show top-level structure only
        top_structure = _build_top_level_structure(structure, depth=2)
        top_json = json.dumps(top_structure, ensure_ascii=False, indent=2)

        nav1 = await acall_with_fallback(
            prompt=_NON_LEAF_SELECTION_PROMPT.format(
                structure_json=top_json,
                question=question,
                per_round=per_round,
            ),
            models=models,
            system_prompt=_SYSTEM_PROMPT,
            max_tokens=512,
        )
        if nav1 and isinstance(nav1, dict):
            branch_ids = set(nav1.get("node_ids", []))

            # Step 2: Expand selected branches to show their children
            expanded = _expand_branch(structure, branch_ids)
            if expanded:
                expanded_json = json.dumps(expanded, ensure_ascii=False, indent=2)

                # Step 3: If expanded structure is still large, navigate deeper
                if len(expanded_json) > _MAX_SKELETON_CHARS:
                    # Recurse: show expanded as new top-level
                    nav2 = await acall_with_fallback(
                        prompt=_NON_LEAF_SELECTION_PROMPT.format(
                            structure_json=json.dumps(
                                _build_top_level_structure(expanded, depth=2),
                                ensure_ascii=False,
                                indent=2,
                            ),
                            question=question,
                            per_round=per_round,
                        ),
                        models=models,
                        system_prompt=_SYSTEM_PROMPT,
                        max_tokens=512,
                    )
                    if nav2 and isinstance(nav2, dict):
                        selected_ids = [str(n) for n in nav2.get("node_ids", [])[:per_round]]
                else:
                    # Show expanded children, let LLM pick leaves
                    nav2 = await acall_with_fallback(
                        prompt=_LEAF_SELECTION_PROMPT.format(
                            expanded_json=expanded_json,
                            question=question,
                            per_round=per_round,
                        ),
                        models=models,
                        system_prompt=_SYSTEM_PROMPT,
                        max_tokens=512,
                    )
                    if nav2 and isinstance(nav2, dict):
                        selected_ids = [str(n) for n in nav2.get("node_ids", [])[:per_round]]
    else:
        # Small tree: one-shot selection
        r1 = await acall_with_fallback(
            prompt=_ROUND1_PROMPT.format(
                structure_json=full_skeleton,
                question=question,
                per_round=per_round,
            ),
            models=models,
            system_prompt=_SYSTEM_PROMPT,
            max_tokens=1024,
        )
        if r1 and isinstance(r1, dict):
            selected_ids = [str(n) for n in r1.get("node_ids", [])[:per_round]]

    if not selected_ids:
        return None

    # ── Load content for selected leaves ──
    collected: dict[str, dict] = {}
    for nid in selected_ids:
        nid = str(nid)
        content = get_node_content(md_path, structure, nid)
        if content and content.strip():
            title = _get_node_title(structure, nid)
            collected[nid] = {"node_id": nid, "title": title or "", "content": content.strip()}

    if not collected:
        return None

    # ── Review round: check if more content needed ──
    read_ids = set(collected.keys())
    remaining_ids = all_leaf_ids - read_ids
    if remaining_ids and max_rounds > 1:
        prev_parts = []
        for nid, sec in list(collected.items())[:3]:
            prev_parts.append(f"### {sec['title']} ({nid})\n{sec['content'][:400]}...")
        previous_text = "\n\n".join(prev_parts)
        remaining_json = _build_remaining_structure(structure, read_ids)

        r2 = await acall_with_fallback(
            prompt=_ROUND2_PROMPT.format(
                previous_content=previous_text,
                remaining_structure=remaining_json,
                question=question,
                per_round=per_round,
            ),
            models=models,
            system_prompt=_SYSTEM_PROMPT,
            max_tokens=512,
        )
        if r2 and isinstance(r2, dict) and not r2.get("done") and r2.get("node_ids"):
            for nid in r2["node_ids"]:
                nid = str(nid)
                if nid in collected:
                    continue
                content = get_node_content(md_path, structure, nid)
                if content and content.strip():
                    title = _get_node_title(structure, nid)
                    collected[nid] = {
                        "node_id": nid,
                        "title": title or "",
                        "content": content.strip(),
                    }

    if not collected:
        return None
    return list(collected.values())


# ── Layer 4: Collapsed tree, cross-paper, hybrid scoring ────────────────────


def _hybrid_score(
    bm25_results: list[dict],
    vector_results: list[dict],
    bm25_key: str = "bm25_score",
    alpha: float = 0.5,
) -> list[dict]:
    """Weighted sum fusion of BM25 and vector retrieval results.

    Args:
        bm25_results: BM25 results with score in bm25_key field.
        vector_results: Vector results with score in "score" field.
        bm25_key: Field name for BM25 score.
        alpha: Weight for BM25 (1-alpha for vector).

    Returns:
        Merged list sorted by combined score descending.
    """

    # Normalize BM25 scores to [0, 1]
    max_bm25 = max((r.get(bm25_key, 0.0) for r in bm25_results), default=1.0)
    bm25_map: dict[str, float] = {}
    for r in bm25_results:
        rid = r.get("id", r.get("node_id", r.get("label", "")))
        raw = r.get(bm25_key, 0.0)
        bm25_map[rid] = raw / max_bm25 if max_bm25 > 0 else 0.0

    # Vector scores are already cosine ∈ [-1, 1], no normalization needed
    vec_map: dict[str, float] = {}
    for r in vector_results:
        rid = r.get("id", r.get("node_id", ""))
        vec_map[rid] = max(r.get("score", 0.0), 0.0)

    # Merge
    all_ids: set[str] = set(bm25_map.keys()) | set(vec_map.keys())
    merged: list[dict] = []
    for rid in all_ids:
        bm25_norm = bm25_map.get(rid, 0.0)
        vec_norm = vec_map.get(rid, 0.0)
        combined = alpha * bm25_norm + (1.0 - alpha) * vec_norm

        merged.append(
            {
                "id": rid,
                "bm25_score": bm25_map.get(rid, 0.0),
                "vector_score": vec_map.get(rid, 0.0),
                "combined_score": combined,
            }
        )

    merged.sort(key=lambda x: x["combined_score"], reverse=True)
    return merged


def _rrf_score(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: combine multiple ranked lists.

    RRF score = sum over lists of 1/(k + rank).
    """
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def query_cross_paper(
    query: str,
    db_path: Path,
    top_k: int = 10,
    cfg: EmbedConfig | None = None,
    paper_ids: list[str] | None = None,
) -> list[dict]:
    """Cross-paper collapsed tree retrieval.

    Embeds query and searches all tree_vectors (PageIndex + RAPTOR nodes)
    across all papers using cosine similarity.

    Args:
        query: Natural language query.
        db_path: SQLite database path.
        top_k: Number of results.
        cfg: Embedding config.
        paper_ids: Optional filter to specific papers.

    Returns:
        List of {node_id, paper_id, score, tree_layer}.
    """
    from drbrain.services.embedding import search_tree

    results = search_tree(query, db_path, top_k=top_k, cfg=cfg)
    if paper_ids:
        results = [r for r in results if r["paper_id"] in paper_ids]
    return results


async def query_by_structure_hybrid(
    question: str,
    paper_dir: Path,
    db_path: Path,
    models: list[dict],
    cfg: EmbedConfig | None = None,
    top_k: int = 5,
) -> list[dict] | None:
    """LLM-primary tree retrieval with optional vector pre-filtering.

    LLM navigation is the PRIMARY reasoning path. When vectors are
    available, they pre-filter candidate nodes to narrow the search
    space before LLM evaluation. When vectors are unavailable,
    pure LLM navigation (the default, not a fallback).

    Args:
        question: Natural language question.
        paper_dir: Paper directory.
        db_path: SQLite database path.
        models: LLM model list.
        cfg: Embedding config (None → no vectors, pure LLM).
        top_k: Number of sections to return.

    Returns:
        List of {node_id, title, content, source} or None.
    """
    import json

    from drbrain.parser.pageindex_parser import get_document_structure_json, get_node_content
    from drbrain.storage.paths import tree_json_path

    tree_path = tree_json_path(paper_dir)
    if not tree_path.exists():
        return None

    tree_data = json.loads(tree_path.read_text(encoding="utf-8"))
    structure = tree_data.get("structure", [])
    if not structure:
        return None

    paper_id = paper_dir.name
    md_path = paper_dir / "raw.md"

    # ── Step 1: LLM navigation (PRIMARY) ──
    # Always run LLM reasoning on the tree structure
    full_skeleton = get_document_structure_json(structure)
    all_leaf_ids = _collect_all_leaf_ids(structure)

    # LLM navigation: send document structure + question
    prompt = _ROUND1_PROMPT.format(
        structure_json=full_skeleton,
        question=question,
        per_round=top_k * 2,  # get more candidates, narrow later
    )

    llm_response = await acall_with_fallback(
        prompt=prompt,
        models=models,
        system_prompt=_SYSTEM_PROMPT,
        max_tokens=1024,
    )

    if not llm_response:
        return None

    # Parse LLM-selected node IDs
    llm_selected: set[str] = set()
    try:
        data = json.loads(llm_response) if isinstance(llm_response, str) else llm_response
        if isinstance(data, dict):
            node_ids = data.get("node_ids", [])
            if isinstance(node_ids, list):
                for nid in node_ids:
                    nid = str(nid)
                    if nid in all_leaf_ids:
                        llm_selected.add(nid)
    except (json.JSONDecodeError, TypeError):
        # Fallback: extract node_ids from text
        for line in llm_response.strip().split("\n"):
            for leaf_id in all_leaf_ids:
                if leaf_id in line:
                    llm_selected.add(leaf_id)

    # ── Step 2: Vector augmentation (AUXILIARY) ──
    vector_candidates: set[str] = set()
    if cfg is not None:
        try:
            from drbrain.services.embedding import _embed_provider, search_tree

            provider = _embed_provider(cfg)
            if provider != "none":
                vec_results = search_tree(question, db_path, top_k=top_k * 2, cfg=cfg)
                # Filter to this paper
                vector_candidates = {r["node_id"] for r in vec_results if r["paper_id"] == paper_id}
        except Exception:
            pass  # vector failure is non-blocking

    # ── Step 3: Merge — LLM-selected nodes take priority ──
    # LLM picks are primary; vectors add candidates LLM might have missed
    merged_ids: list[str] = list(llm_selected)
    for vid in vector_candidates:
        if vid not in llm_selected:
            merged_ids.append(vid)

    if not merged_ids:
        return None

    # ── Step 4: Fetch content for merged nodes ──
    sections = []
    for node_id in merged_ids[:top_k]:
        try:
            content = get_node_content(md_path, structure, node_id)
        except Exception:
            content = ""
        title = _get_node_title(structure, node_id)
        source = (
            "llm+vector"
            if node_id in vector_candidates and node_id in llm_selected
            else "llm"
            if node_id in llm_selected
            else "vector"
        )
        sections.append(
            {
                "node_id": node_id,
                "title": title or "",
                "content": content.strip() if content else "",
                "source": source,
            }
        )

    return sections
