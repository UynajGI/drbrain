"""Knowledge genealogy — concept lineage trees from graph traversal."""

from __future__ import annotations

from collections import deque

from drbrain.graph.engine import GraphEngine
from drbrain.storage.database import Database


def evolve_concept(
    graph: GraphEngine,
    db: Database,
    concept_label: str,
    direction: str = "both",
    max_depth: int = 3,
) -> list[dict]:
    """Build a concept lineage tree via BFS graph traversal.

    Returns a list of root nodes. Each node is:
        {"label": str, "local_id": str, "year": int|None,
         "relation": str, "children": [...]}

    direction: "ancestors" (incoming edges), "descendants" (outgoing), "both"
    """
    # Find matching concept nodes
    rows = db.conn.execute(
        "SELECT label, local_id, type FROM concepts WHERE label = ?", (concept_label,)
    ).fetchall()

    if not rows:
        return []

    # Track which edges to follow
    follow_relations = {"extends", "refines", "applies"}

    trees: list[dict] = []

    for label, local_id, ctype in rows:
        # Get paper year
        year_row = db.conn.execute(
            "SELECT year FROM papers WHERE local_id = ?", (local_id,)
        ).fetchone()
        year = year_row[0] if year_row else None

        root = {
            "label": label,
            "type": ctype,
            "local_id": local_id,
            "year": year,
            "relation": None,  # root has no incoming relation
            "children": [],
        }

        if direction in ("descendants", "both"):
            _bfs_descendants(graph, db, root, follow_relations, max_depth)
        if direction in ("ancestors", "both"):
            _bfs_ancestors(graph, db, root, follow_relations, max_depth)

        trees.append(root)

    if direction in ("ancestors", "both"):
        trees = _reroot_with_ancestors(trees)

    return trees


def _bfs_descendants(graph, db, parent, relations, max_depth):
    """BFS outward from parent, following outgoing edges."""
    visited = {(parent["label"], parent["local_id"])}
    queue = deque([(parent, 0)])

    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue

        if node["label"] not in graph.graph:
            continue

        for n in graph.graph.neighbors(node["label"]):
            for edge_data in graph.graph[node["label"]][n].values():
                rel = edge_data.get("relation", "")
                if rel not in relations:
                    continue

                # Get paper info for child concept
                concept_row = db.conn.execute(
                    "SELECT local_id, type FROM concepts WHERE label = ? LIMIT 1", (n,)
                ).fetchone()
                if not concept_row:
                    continue

                child_local_id, child_type = concept_row
                child_key = (n, child_local_id)
                if child_key in visited:
                    continue
                visited.add(child_key)

                year_row = db.conn.execute(
                    "SELECT year FROM papers WHERE local_id = ?", (child_local_id,)
                ).fetchone()
                year = year_row[0] if year_row else None

                child = {
                    "label": n,
                    "type": child_type,
                    "local_id": child_local_id,
                    "year": year,
                    "relation": rel,
                    "children": [],
                }
                node.setdefault("children", []).append(child)
                queue.append((child, depth + 1))


def _bfs_ancestors(graph, db, parent, relations, max_depth):
    """BFS inward toward parent from incoming edges."""
    visited = {(parent["label"], parent["local_id"])}
    queue = deque([(parent, 0)])

    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue

        if node["label"] not in graph.graph:
            continue

        for pred in graph.graph.predecessors(node["label"]):
            for edge_data in graph.graph[pred][node["label"]].values():
                rel = edge_data.get("relation", "")
                if rel not in relations:
                    continue

                concept_row = db.conn.execute(
                    "SELECT local_id, type FROM concepts WHERE label = ? LIMIT 1", (pred,)
                ).fetchone()
                if not concept_row:
                    continue

                pred_local_id, pred_type = concept_row
                pred_key = (pred, pred_local_id)
                if pred_key in visited:
                    continue
                visited.add(pred_key)

                year_row = db.conn.execute(
                    "SELECT year FROM papers WHERE local_id = ?", (pred_local_id,)
                ).fetchone()
                year = year_row[0] if year_row else None

                ancestor = {
                    "label": pred,
                    "type": pred_type,
                    "local_id": pred_local_id,
                    "year": year,
                    "relation": rel,
                    "children": [node],
                }
                node.setdefault("_ancestors", []).append(ancestor)
                queue.append((ancestor, depth + 1))


def _reroot_with_ancestors(nodes: list[dict]) -> list[dict]:
    """Replace each node with its deepest ancestors as new tree roots.

    Nodes with _ancestors are replaced by their ancestors (recursively),
    so the returned list contains only nodes with no ancestors.  The
    original tree structure is preserved through each ancestor's
    ``children`` field, which already points toward the matched concept.
    """
    roots: list[dict] = []
    for node in nodes:
        ancestors = node.pop("_ancestors", [])
        if ancestors:
            roots.extend(_reroot_with_ancestors(ancestors))
        else:
            roots.append(node)
    return roots


def trace_descendants(
    db: Database,
    graph: GraphEngine,
    local_id: str,
    generations: int = 3,
) -> dict | None:
    """Trace a paper's academic offspring via concept graph edges.

    Returns a tree dict rooted at the given paper, with children being
    papers that extend, refine, apply, challenge, or cite its concepts.

    Returns None if the paper is not found.
    """
    paper = db.get_paper(local_id)
    if not paper:
        return None

    root = {
        "label": paper.get("title", local_id),
        "local_id": local_id,
        "year": paper.get("year"),
        "relation": None,
        "children": [],
    }

    follow_relations = {"extends", "refines", "applies", "challenges", "cites"}
    visited_papers: set[str] = {local_id}

    # BFS: (parent_node, depth) — depth 0 = root paper
    queue = deque([(root, 0)])

    while queue:
        parent, depth = queue.popleft()
        if depth >= generations:
            continue

        concepts = db.conn.execute(
            "SELECT label FROM concepts WHERE local_id = ?", (parent["local_id"],)
        ).fetchall()

        for (concept_label,) in concepts:
            if concept_label not in graph.graph:
                continue

            for dst_label in graph.graph.neighbors(concept_label):
                for edge_data in graph.graph[concept_label][dst_label].values():
                    rel = edge_data.get("relation", "")
                    if rel not in follow_relations:
                        continue

                    child_rows = db.conn.execute(
                        "SELECT DISTINCT local_id FROM concepts WHERE label = ?",
                        (dst_label,),
                    ).fetchall()

                    for (child_local_id,) in child_rows:
                        if child_local_id in visited_papers:
                            continue
                        visited_papers.add(child_local_id)

                        child_paper = db.get_paper(child_local_id)
                        if not child_paper:
                            continue

                        child = {
                            "label": child_paper.get("title", child_local_id),
                            "local_id": child_local_id,
                            "year": child_paper.get("year"),
                            "relation": rel,
                            "children": [],
                        }
                        parent.setdefault("children", []).append(child)
                        queue.append((child, depth + 1))

    return root


def format_tree(nodes: list[dict], indent: str = "", mermaid: bool = False) -> str:
    """Format lineage tree as text or Mermaid diagram."""
    if mermaid:
        return _to_mermaid(nodes)
    return _to_text_tree(nodes)


def _to_text_tree(nodes: list[dict], prefix: str = "") -> str:
    """Render as indented text tree with box-drawing characters."""
    lines: list[str] = []
    for i, node in enumerate(nodes):
        is_last = i == len(nodes) - 1
        connector = "└─ " if is_last else "├─ "

        year_str = f" ({node['year']})" if node.get("year") else ""
        rel_str = f" — {node['relation']}" if node.get("relation") else ""
        type_str = f" [{node.get('type', '')}]" if node.get("type") else ""

        lines.append(f"{prefix}{connector}{node['label']}{type_str}{year_str}{rel_str}")

        children = node.get("children", [])
        if children:
            child_prefix = prefix + ("   " if is_last else "│  ")
            lines.append(_to_text_tree(children, child_prefix))

    return "\n".join(lines)


def _to_mermaid(nodes: list[dict]) -> str:
    """Render as Mermaid flowchart."""
    lines = ["graph TD"]
    _mermaid_nodes(lines, nodes, None)
    return "\n".join(lines)


def _mermaid_nodes(lines: list[str], nodes: list[dict], parent_id: str | None):
    """Recursively add Mermaid nodes and edges."""
    for node in nodes:
        nid = node["label"].replace(" ", "_")[:50]
        year_str = f" ({node['year']})" if node.get("year") else ""
        lines.append(f'    {nid}["{node["label"]}{year_str}"]')
        if parent_id:
            rel = node.get("relation", "")
            lines.append(f"    {parent_id} -->|{rel}| {nid}")
        children = node.get("children", [])
        if children:
            _mermaid_nodes(lines, children, nid)
