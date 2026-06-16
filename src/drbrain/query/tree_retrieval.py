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
    from drbrain.extractor.cache import ApiCache

import sqlite3

import numpy as np

from drbrain.extractor.llm_client import acall_with_fallback
from drbrain.parser.pageindex_parser import get_document_structure_json, get_node_content
from drbrain.storage.connection import connect_wal
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
    *,
    _cache: ApiCache | None = None,
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
        _cache=_cache,
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
    *,
    _cache: ApiCache | None = None,
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
    log.info(
        "[tree-retrieval] %s — %d leaves, skeleton=%d chars (navigation=%s)",
        paper_dir.name,
        len(all_leaf_ids),
        len(full_skeleton),
        use_navigation,
    )

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
            _cache=_cache,
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
                        _cache=_cache,
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
                        _cache=_cache,
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
            _cache=_cache,
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
            _cache=_cache,
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
        log.info("[tree-retrieval] no sections matched for question: %.60s", question)
        return None
    log.info("[tree-retrieval] retrieved %d sections", len(collected))
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


# ── RAPTOR Figure 2: Two-stage tree traversal (layer-by-layer + collapsed fallback) ───


def tree_traversal_search(
    query: str,
    db_path: Path,
    top_k: int = 5,
    min_results: int = 3,
    cfg: EmbedConfig | None = None,
) -> list[dict]:
    """RAPTOR Figure 2: layer-by-layer tree traversal with collapsed fallback.

    Stage 1 (tree traversal):
      Start at root layer (highest RAPTOR layer). For each layer, compute
      cosine similarity for nodes in that layer only. Keep top-k nodes.
      Descend to their children (via tree_summaries.source_node_ids) in the
      next layer. Repeat until pageindex leaf layer or no children remain.

    Stage 2 (collapsed tree fallback):
      If Stage 1 returns fewer than ``min_results``, fall back to the
      existing collapsed tree search (flat cosine across all layers) via
      ``search_tree()``.

    Compared to ``collapsed_tree_retrieval`` (which scans ALL tree_vectors),
    this is more token-efficient because:
      - Root layer: only root-layer nodes are compared
      - Deeper layers: only children of selected nodes are compared
      - Low-scoring branches are pruned early

    Args:
        query: Natural language query text.
        db_path: SQLite database path.
        top_k: Number of nodes to keep per layer during traversal.
        min_results: Minimum results before triggering Stage 2 fallback.
        cfg: Optional EmbedConfig. ``provider=none`` disables all vectors.

    Returns:
        List of {node_id, paper_id, score, tree_layer}, sorted by score descending.
    """
    from drbrain.services.embedding import _embed_batch, _embed_provider, search_tree

    provider = _embed_provider(cfg)
    if provider == "none":
        return []

    if not db_path.exists():
        return []

    conn = connect_wal(db_path)
    try:
        # Discover available layers
        layer_rows = conn.execute(
            "SELECT DISTINCT tree_layer FROM tree_vectors ORDER BY tree_layer"
        ).fetchall()
        all_layers = [r[0] for r in layer_rows]

        if not all_layers:
            return []

        # Sort RAPTOR layers by layer number (raptor_L1, raptor_L2, ...)
        raptor_layers = sorted(
            [lyr for lyr in all_layers if lyr.startswith("raptor_L")],
            key=lambda x: int(x.split("L")[1]),
        )
        has_pageindex = "pageindex" in all_layers

        # If no RAPTOR layers, short-circuit: search pageindex directly
        if not raptor_layers:
            if has_pageindex:
                page_rows = conn.execute(
                    "SELECT node_id, paper_id, embedding, tree_layer "
                    "FROM tree_vectors WHERE tree_layer = 'pageindex'"
                ).fetchall()
            else:
                return []

            # Embed query
            query_vec = _embed_batch([query], cfg)
            if not query_vec:
                return []
            qv = np.asarray(query_vec[0], dtype="float32")
            query_dim = len(query_vec[0])

            scored = _score_nodes(page_rows, qv, query_dim)
            return scored[:top_k]

        # ── Stage 1: Layer-by-layer traversal ──
        query_vec = _embed_batch([query], cfg)
        if not query_vec:
            return []
        qv = np.asarray(query_vec[0], dtype="float32")
        query_dim = len(query_vec[0])

        # Start at root (highest RAPTOR layer) and descend
        # Traverse RAPTOR layers from highest → lowest
        current_candidates: list[str] | None = None  # None = search all in layer
        leaf_candidates: list[str] | None = None  # Final pageindex candidates

        for layer_idx in range(len(raptor_layers) - 1, -1, -1):
            layer_name = raptor_layers[layer_idx]

            # Query this layer's vectors
            layer_rows = _query_layer_vectors(conn, layer_name, current_candidates)

            if not layer_rows:
                break

            scored = _score_nodes(layer_rows, qv, query_dim)
            top = scored[:top_k]

            if not top:
                break

            # Collect children for the next layer down
            children = _collect_children(conn, [r["node_id"] for r in top])

            if layer_idx == 0:
                # At lowest RAPTOR layer: children point to pageindex nodes
                leaf_candidates = children
            else:
                # At intermediate RAPTOR layer: children point to next RAPTOR layer
                current_candidates = children if children else None

        # ── Search pageindex using filtered candidates ──
        final_results: list[dict] = []

        if has_pageindex:
            if leaf_candidates:
                # Search only children of selected RAPTOR nodes
                placeholders = ",".join("?" for _ in leaf_candidates)
                page_rows = conn.execute(
                    f"SELECT node_id, paper_id, embedding, tree_layer "
                    f"FROM tree_vectors "
                    f"WHERE node_id IN ({placeholders}) AND tree_layer = 'pageindex'",
                    leaf_candidates,
                ).fetchall()
            else:
                # No traversal candidates: fall back to all pageindex nodes
                page_rows = conn.execute(
                    "SELECT node_id, paper_id, embedding, tree_layer "
                    "FROM tree_vectors WHERE tree_layer = 'pageindex'"
                ).fetchall()

            if page_rows:
                scored = _score_nodes(page_rows, qv, query_dim)
                final_results = scored[:top_k]
        else:
            # No pageindex: return best from last RAPTOR layer
            if raptor_layers:
                last_rows = _query_layer_vectors(conn, raptor_layers[0], current_candidates)
                if last_rows:
                    scored = _score_nodes(last_rows, qv, query_dim)
                    final_results = scored[:top_k]

        # ── Stage 2: Collapsed tree fallback ──
        if len(final_results) < min_results:
            fallback = search_tree(query, db_path, top_k=top_k, cfg=cfg)
            if fallback:
                existing_ids = {r["node_id"] for r in final_results}
                for r in fallback:
                    if r["node_id"] not in existing_ids:
                        final_results.append(r)
                        existing_ids.add(r["node_id"])
                final_results.sort(key=lambda x: x["score"], reverse=True)
                final_results = final_results[:top_k]

        return final_results

    finally:
        conn.close()


def _query_layer_vectors(
    conn: sqlite3.Connection,
    layer_name: str,
    candidate_ids: list[str] | None,
) -> list[tuple]:
    """Fetch tree_vectors rows for a layer, optionally filtered by candidate IDs."""
    if candidate_ids:
        placeholders = ",".join("?" for _ in candidate_ids)
        return conn.execute(
            f"SELECT node_id, paper_id, embedding, tree_layer "
            f"FROM tree_vectors "
            f"WHERE node_id IN ({placeholders}) AND tree_layer = ?",
            [*candidate_ids, layer_name],
        ).fetchall()
    else:
        return conn.execute(
            "SELECT node_id, paper_id, embedding, tree_layer "
            "FROM tree_vectors WHERE tree_layer = ?",
            (layer_name,),
        ).fetchall()


def _score_nodes(
    rows: list[tuple],
    query_vec: np.ndarray,
    query_dim: int,
) -> list[dict]:
    """Compute cosine similarity for a list of tree_vectors rows.

    Assumes stored embeddings are normalized float32 blobs.
    Skips rows with dimension mismatch.  Uses vectorized matrix multiply
    instead of per-row struct.unpack + np.dot for ~10-50x speedup.
    """
    # Filter to rows with matching dimension
    valid = []
    for row in rows:
        blob = row[2]
        if len(blob) // 4 == query_dim:
            valid.append(row)
        else:
            log.warning(
                "Dimension mismatch in tree_vectors node_id=%s: stored=%s query=%s",
                row[0],
                len(blob) // 4,
                query_dim,
            )
    if not valid:
        return []

    # Concatenate all blobs into a single matrix and compute similarities
    all_blobs = b"".join(row[2] for row in valid)
    mat = np.frombuffer(all_blobs, dtype=np.float32).reshape(len(valid), query_dim)
    sims = mat @ query_vec  # (N,) vector of dot products

    results = [
        {
            "node_id": valid[i][0],
            "paper_id": valid[i][1],
            "score": float(sims[i]),
            "tree_layer": valid[i][3],
        }
        for i in range(len(valid))
    ]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def _collect_children(
    conn: sqlite3.Connection,
    parent_ids: list[str],
) -> list[str]:
    """Collect child node IDs from tree_summaries.source_node_ids.

    Returns deduplicated list of all child node IDs across parents.
    """
    if not parent_ids:
        return []
    placeholders = ",".join("?" for _ in parent_ids)
    rows = conn.execute(
        f"SELECT source_node_ids FROM tree_summaries WHERE node_id IN ({placeholders})",
        parent_ids,
    ).fetchall()

    children: list[str] = []
    seen: set[str] = set()
    for (src_json,) in rows:
        try:
            ids = json.loads(src_json)
            for cid in ids:
                cid = str(cid)
                if cid not in seen:
                    seen.add(cid)
                    children.append(cid)
        except (json.JSONDecodeError, TypeError):
            pass
    return children


async def query_by_structure_hybrid(
    question: str,
    paper_dir: Path,
    db_path: Path,
    models: list[dict],
    cfg: EmbedConfig | None = None,
    top_k: int = 5,
    *,
    _cache: ApiCache | None = None,
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
        _cache=_cache,
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
            log.warning(
                "Vector augmentation failed, continuing with LLM-only results", exc_info=True
            )

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
