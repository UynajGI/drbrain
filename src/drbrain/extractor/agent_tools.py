"""Shared tool definitions and handlers for graph-reasoning agents.

Extracted from ReasonerAgent so SessionAgent can reuse the same
tool definitions and execution logic without duplication.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# -- Tool definitions (OpenAI function-calling format) --

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_concepts",
            "description": "Search concepts in the knowledge graph by keyword",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keyword"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_neighbors",
            "description": "Get neighbors of a concept node in the graph",
            "parameters": {
                "type": "object",
                "properties": {
                    "node": {"type": "string"},
                    "hops": {"type": "integer", "default": 1},
                    "direction": {
                        "type": "string",
                        "enum": ["forward", "backward", "both"],
                    },
                },
                "required": ["node"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_path",
            "description": "Find shortest path between two concepts in the graph",
            "parameters": {
                "type": "object",
                "properties": {
                    "src": {"type": "string"},
                    "dst": {"type": "string"},
                },
                "required": ["src", "dst"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_structure",
            "description": (
                "Get the section tree skeleton for a paper (titles + node_ids only, no content)"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "Paper local_id"},
                },
                "required": ["paper_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_section_content",
            "description": "Get the full text content of a specific section within a paper",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "Paper local_id"},
                    "node_id": {
                        "type": "string",
                        "description": "Tree node ID from document structure",
                    },
                },
                "required": ["paper_id", "node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_tree",
            "description": (
                "Search across all paper sections by semantic similarity (collapsed tree retrieval)"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for finding relevant sections",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_raptor_summaries",
            "description": (
                "Get RAPTOR cross-section summaries for a paper. "
                "Returns hierarchical summaries that capture themes across multiple sections."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "Paper local_id"},
                },
                "required": ["paper_id"],
            },
        },
    },
]


# -- Tool handler functions (stateless, take explicit dependencies) --


def search_concepts(db, query: str, limit: int = 5) -> list[dict]:
    """Search concepts in the knowledge graph by keyword via BM25."""
    if db is None:
        return []
    from drbrain.query.bm25 import build_bm25_index

    bm25 = build_bm25_index(db)
    results = bm25.search(query, limit=limit)
    return [{"label": r["label"], "type": r["type"], "score": r["score"]} for r in results]


def get_neighbors(graph, node: str, hops: int = 1, direction: str = "both") -> list[dict]:
    """Get neighbors of a concept node in the graph."""
    if graph is None:
        return []
    results = graph.traverse(start_nodes={node}, hops=hops, direction=direction)
    return [
        {
            "target": r.target,
            "source": r.source,
            "distance": r.distance,
            "path": [{"src": s.src, "relation": s.relation, "dst": s.dst} for s in r.path],
        }
        for r in results
    ]


def find_path(graph, src: str, dst: str) -> dict | None:
    """Find shortest path between two concepts in the graph."""
    if graph is None or src not in graph.graph or dst not in graph.graph:
        return None
    import networkx as nx

    ug = graph.graph.to_undirected()
    try:
        node_path = nx.shortest_path(ug, source=src, target=dst)
        return {"path": node_path, "length": len(node_path) - 1}
    except (nx.NetworkXNoPath, nx.NetworkXError):
        return None


def get_document_structure(papers_dir: Path | None, paper_id: str) -> list[dict]:
    """Return the tree skeleton for a paper (titles + node_ids, no content)."""
    if papers_dir is None:
        return []

    from drbrain.storage.paths import tree_json_path

    tree_path = tree_json_path(papers_dir / paper_id)
    if not tree_path.exists():
        return []

    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    structure = tree.get("structure", [])

    def _extract(nodes: list[dict]) -> list[dict]:
        result = []
        for n in nodes:
            item = {
                "node_id": n.get("node_id", ""),
                "title": n.get("title", ""),
            }
            child_nodes = n.get("nodes", [])
            if child_nodes:
                item["children"] = _extract(child_nodes)
            result.append(item)
        return result

    return _extract(structure)


def get_section_content(papers_dir: Path | None, paper_id: str, node_id: str) -> str:
    """Return the raw text content for a tree node."""
    if papers_dir is None:
        return ""

    from drbrain.parser.pageindex_parser import get_node_content
    from drbrain.storage.paths import raw_md_path, tree_json_path

    paper_dir = papers_dir / paper_id
    tree_path = tree_json_path(paper_dir)
    md_path = raw_md_path(paper_dir)
    if not tree_path.exists() or not md_path.exists():
        return ""

    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    structure = tree.get("structure", [])
    try:
        return get_node_content(md_path, structure, node_id) or ""
    except Exception:
        return ""


def search_tree(db, query: str) -> list[dict]:
    """Cross-paper collapsed tree search (vector + BM25 hybrid)."""
    if db is None:
        return []
    from drbrain.query.tree_retrieval import query_cross_paper

    return query_cross_paper(query, db.path)


def get_raptor_summaries(db, paper_id: str) -> list[dict]:
    """Return RAPTOR cross-section summaries for a paper."""
    if db is None:
        return []

    rows = db.conn.execute(
        "SELECT node_id, paper_id, summary_text, source_node_ids, tree_layer "
        "FROM tree_summaries WHERE paper_id = ? ORDER BY tree_layer",
        (paper_id,),
    ).fetchall()

    return [
        {
            "node_id": r[0],
            "paper_id": r[1],
            "summary_text": r[2],
            "source_node_ids": json.loads(r[3]) if r[3] else [],
            "tree_layer": r[4],
        }
        for r in rows
    ]


# -- Tool dispatch map --

TOOL_HANDLERS: dict[str, Any] = {
    "search_concepts": search_concepts,
    "get_neighbors": get_neighbors,
    "find_path": find_path,
    "get_document_structure": get_document_structure,
    "get_section_content": get_section_content,
    "search_tree": search_tree,
    "get_raptor_summaries": get_raptor_summaries,
}


def execute_tool(
    name: str, args: dict, db=None, graph=None, papers_dir: Path | None = None
) -> list | dict | str | None:
    """Execute a single tool by name with the provided arguments.

    Maps tool name to handler and resolves the correct arguments for each.
    """
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return []

    if name in ("search_concepts",):
        return handler(db, **args)
    elif name in ("get_neighbors", "find_path"):
        return handler(graph, **args)
    elif name in ("get_document_structure", "get_section_content"):
        return handler(papers_dir, **args)
    elif name in ("search_tree", "get_raptor_summaries"):
        return handler(db, **args)
    return []
