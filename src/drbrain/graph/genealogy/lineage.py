"""Concept lineage: evolve_concept, BFS traversal, trace_descendants."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from drbrain.graph.engine import GraphEngine
from drbrain.storage.database import Database

if TYPE_CHECKING:
    pass

from drbrain.graph.genealogy.paradigm import (
    _format_provenance,
    _get_concept_provenance,
)


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

        section, node_id, _ = _get_concept_provenance(db, label, ctype)
        root = {
            "label": label,
            "type": ctype,
            "local_id": local_id,
            "year": year,
            "section": section,
            "node_id": node_id,
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


def _collect_reachable_labels(
    g, start: str, relations: set[str], direction: str, max_depth: int
) -> set[str]:
    """Collect all node labels reachable from *start* within *max_depth* hops.

    *direction* is ``"out"`` (successors) or ``"in"`` (predecessors).
    Does NOT include *start* itself.
    """
    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(start, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        if node not in g:
            continue
        neighbors = g.successors(node) if direction == "out" else g.predecessors(node)
        for nb in neighbors:
            # Edge direction differs: successors have g[node][nb], predecessors
            # have g[nb][node]. Access the correct adjacency to avoid KeyError.
            adj = g[node][nb] if direction == "out" else g[nb][node]
            for edge_data in adj.values():
                if edge_data.get("relation", "") in relations:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append((nb, depth + 1))
    return visited


def _preload_concept_info(db, labels: set[str]) -> dict[str, tuple[str | None, str, int | None]]:
    """Batch-load concept (local_id, type) and paper year for a set of labels.

    Returns ``{label: (local_id, type, year)}``.
    """
    if not labels:
        return {}
    ph = ",".join("?" for _ in labels)
    rows = db.conn.execute(
        f"SELECT c.label, c.local_id, c.type, p.year "
        f"FROM concepts c LEFT JOIN papers p ON c.local_id = p.local_id "
        f"WHERE c.label IN ({ph})",
        tuple(labels),
    ).fetchall()
    return {r[0]: (r[1], r[2], r[3]) for r in rows}


def _bfs_descendants(graph, db, parent, relations, max_depth):
    """BFS outward from parent, following outgoing edges."""
    # Pre-load concept info for all nodes reachable from parent
    reachable_labels = _collect_reachable_labels(
        graph.graph, parent["label"], relations, "out", max_depth
    )
    info_map = _preload_concept_info(db, reachable_labels)

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

                info = info_map.get(n)
                if not info:
                    continue

                child_local_id, child_type, year = info
                child_key = (n, child_local_id)
                if child_key in visited:
                    continue
                visited.add(child_key)

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
    # Pre-load concept info for all nodes reachable from parent
    reachable_labels = _collect_reachable_labels(
        graph.graph, parent["label"], relations, "in", max_depth
    )
    info_map = _preload_concept_info(db, reachable_labels)

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

                info = info_map.get(pred)
                if not info:
                    continue

                pred_local_id, pred_type, year = info
                pred_key = (pred, pred_local_id)
                if pred_key in visited:
                    continue
                visited.add(pred_key)

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

    # ── Batch preloads to eliminate N+1 queries ──
    # 1a. local_id -> set of concept labels (for parent paper lookups)
    local_id_to_labels: dict[str, set[str]] = {}
    # 1b. label -> set of local_ids (for child paper lookups)
    label_to_local_ids: dict[str, set[str]] = {}
    for label, lid in db.conn.execute("SELECT label, local_id FROM concepts").fetchall():
        local_id_to_labels.setdefault(lid, set()).add(label)
        label_to_local_ids.setdefault(label, set()).add(lid)

    # 2. local_id -> paper dict
    all_papers = db.get_all_papers()
    paper_map: dict[str, dict] = {p["local_id"]: p for p in all_papers}

    # 3. label -> (section, node_id, local_id) provenance
    #    ORDER BY confidence DESC so first match wins (mirrors _get_concept_provenance).
    provenance_map: dict[str, tuple[str, str, str]] = {}
    for label, section, node_id, lid in db.conn.execute(
        "SELECT label, section, node_id, local_id FROM concepts ORDER BY confidence DESC"
    ).fetchall():
        key = label.lower()
        if key not in provenance_map and section:
            provenance_map[key] = (section, node_id or "", lid or "")

    # BFS: (parent_node, depth) — depth 0 = root paper
    queue = deque([(root, 0)])

    while queue:
        parent, depth = queue.popleft()
        if depth >= generations:
            continue

        parent_concepts = local_id_to_labels.get(parent["local_id"], set())

        for concept_label in parent_concepts:
            if concept_label not in graph.graph:
                continue

            for dst_label in graph.graph.neighbors(concept_label):
                for edge_data in graph.graph[concept_label][dst_label].values():
                    rel = edge_data.get("relation", "")
                    if rel not in follow_relations:
                        continue

                    child_ids = label_to_local_ids.get(dst_label, set())

                    for child_local_id in child_ids:
                        if child_local_id in visited_papers:
                            continue
                        visited_papers.add(child_local_id)

                        child_paper = paper_map.get(child_local_id)
                        if not child_paper:
                            continue

                        prov = provenance_map.get(dst_label.lower(), ("", "", ""))
                        section, node_id, paper_id = prov
                        provenance = _format_provenance(
                            section, node_id, paper_id or child_local_id
                        )
                        child = {
                            "label": child_paper.get("title", child_local_id),
                            "local_id": child_local_id,
                            "year": child_paper.get("year"),
                            "relation": rel,
                            "via_concept": dst_label,
                            "via_section": section,
                            "via_provenance": provenance,
                            "children": [],
                        }
                        parent.setdefault("children", []).append(child)
                        queue.append((child, depth + 1))

    return root
