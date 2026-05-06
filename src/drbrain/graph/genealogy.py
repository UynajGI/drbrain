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


def detect_paradigm_shifts(
    graph: GraphEngine,
    db: Database,
    concept: str | None = None,
    paper_ids: list[str] | None = None,
    decline_threshold: float = 0.5,
    growth_threshold: int = 3,
    explosion_threshold: int = 8,
    descendant_threshold: int = 3,
    cascade_threshold: int = 1,
) -> list[dict]:
    """Detect paradigm shifts: replacement, explosion, cross-domain.

    Returns list of shift dicts, each with: type, description, concepts involved.

    Args:
        concept: Optional concept label to filter explosion detection.
        paper_ids: Optional list of paper IDs to scope detection to a workspace.
        decline_threshold: Fraction decline to flag replacement (default 0.5 = 50%).
        growth_threshold: Min new-method papers for replacement.
        explosion_threshold: Min total papers for explosion detection.
        descendant_threshold: Min descendant concepts for explosion (PRD: 3+).
        cascade_threshold: Min cascaded concepts for cross-domain detection.
    """
    results: list[dict] = []

    # Type 1: Replacement -- find challenges edges where old is declining
    challenge_edges = db.conn.execute(
        "SELECT src_id, dst_id, weight FROM edges WHERE relation = 'challenges'"
    ).fetchall()

    for src_id, dst_id, conf in challenge_edges:
        # Count papers per year for old concept (dst_id = being challenged)
        old_years = db.conn.execute(
            "SELECT year, COUNT(*) FROM papers p JOIN concepts c ON c.local_id = p.local_id "
            "WHERE c.label = ? AND p.year IS NOT NULL GROUP BY p.year ORDER BY p.year",
            (dst_id,),
        ).fetchall()

        # Count papers per year for new concept
        new_years = db.conn.execute(
            "SELECT year, COUNT(*) FROM papers p JOIN concepts c ON c.local_id = p.local_id "
            "WHERE c.label = ? AND p.year IS NOT NULL GROUP BY p.year ORDER BY p.year",
            (src_id,),
        ).fetchall()

        if len(old_years) >= 2 and len(new_years) >= 2:
            max_old_year = max(y for y, _ in old_years)
            old_recent = sum(c for y, c in old_years if y >= max_old_year - 2)
            old_old = sum(c for y, c in old_years if y < max_old_year - 2)

            if old_old > 0 and old_recent / old_old <= (1 - decline_threshold):
                new_total = sum(c for _, c in new_years)
                if new_total >= growth_threshold:
                    results.append(
                        {
                            "type": "replacement",
                            "old_concept": dst_id,
                            "new_concept": src_id,
                            "description": f"{src_id} is replacing {dst_id} (decline: {old_recent}/{old_old}, new: {new_total})",
                            "confidence": conf,
                        }
                    )

    # Type 2: Explosion -- concept with rapid growth + descendants
    if concept:
        concept_labels = [concept]
    elif paper_ids:
        # Filter concepts to only those appearing in workspace papers
        placeholders = ",".join("?" for _ in paper_ids)
        concept_labels = [
            r[0]
            for r in db.conn.execute(
                f"SELECT DISTINCT c.label FROM concepts c "
                f"WHERE c.local_id IN ({placeholders}) AND c.type IN ('Method', 'Problem')",
                paper_ids,
            ).fetchall()
        ]
    else:
        concept_labels = [
            r[0]
            for r in db.conn.execute(
                "SELECT DISTINCT label FROM concepts WHERE type IN ('Method', 'Problem')"
            ).fetchall()
        ]

    for clabel in concept_labels:
        year_counts = db.conn.execute(
            "SELECT year, COUNT(*) FROM papers p JOIN concepts c ON c.local_id = p.local_id "
            "WHERE c.label = ? AND p.year IS NOT NULL GROUP BY p.year ORDER BY p.year",
            (clabel,),
        ).fetchall()

        total = sum(c for _, c in year_counts)
        if total >= explosion_threshold and len(year_counts) <= 2:
            # Check for descendant concepts via graph
            descendants = []
            if clabel in graph.graph:
                for n in graph.graph.neighbors(clabel):
                    for edge_data in graph.graph[clabel][n].values():
                        rel = edge_data.get("relation", "")
                        if rel in ("extends", "refines", "applies"):
                            descendants.append(n)

            if len(descendants) >= descendant_threshold:
                results.append(
                    {
                        "type": "explosion",
                        "concept": clabel,
                        "paper_count": total,
                        "descendants": descendants[:10],
                        "description": f"{clabel} exploded to {total} papers with {len(descendants)} descendant concepts",
                    }
                )

    # Type 3: Cross-domain invasion -- applies edges with cascading
    applies_edges = db.conn.execute(
        "SELECT src_id, dst_id, weight FROM edges WHERE relation = 'applies'"
    ).fetchall()

    for src_id, dst_id, conf in applies_edges:
        # Check if dst has further descendants (cascade in new domain)
        cascade = []
        visited = {src_id, dst_id}
        queue = [dst_id]
        while queue and len(cascade) <= 5:
            node = queue.pop(0)
            if node not in graph.graph:
                continue
            for n in graph.graph.neighbors(node):
                if n in visited:
                    continue
                visited.add(n)
                for edge_data in graph.graph[node][n].values():
                    rel = edge_data.get("relation", "")
                    if rel in ("extends", "refines", "applies"):
                        cascade.append(n)
                        queue.append(n)

        if len(cascade) >= cascade_threshold:
            results.append(
                {
                    "type": "cross_domain",
                    "source_concept": src_id,
                    "target_concept": dst_id,
                    "cascade": cascade[:10],
                    "description": f"{src_id} crossed domains via {dst_id}, spawned {len(cascade)} concepts",
                    "confidence": conf,
                }
            )

    return results


def landscape_workspace(
    db: Database,
    workspace_path: str | None = None,
    paper_ids: list[str] | None = None,
) -> dict:
    """Generate a domain landscape: timeline, gaps, debates.

    Returns dict with keys: timeline, gaps, debates.
    """
    result: dict[str, list] = {"timeline": [], "gaps": [], "debates": []}

    # Determine paper IDs
    if paper_ids is None:
        if workspace_path:
            from drbrain.storage.workspace import load_workspace_papers

            try:
                paper_ids = load_workspace_papers(workspace_path)
            except (FileNotFoundError, OSError):
                paper_ids = []
        else:
            return {"error": "No workspace or paper_ids provided"}

    if not paper_ids:
        return result

    # Build timeline: papers ordered by year
    placeholders = ",".join("?" for _ in paper_ids)
    rows = db.conn.execute(
        f"SELECT local_id, title, year FROM papers "
        f"WHERE local_id IN ({placeholders}) AND year IS NOT NULL "
        f"ORDER BY year, local_id",
        paper_ids,
    ).fetchall()

    for row in rows:
        concepts = db.conn.execute(
            "SELECT label, type FROM concepts WHERE local_id = ? LIMIT 5",
            (row[0],),
        ).fetchall()

        result["timeline"].append(
            {
                "local_id": row[0],
                "title": row[1][:80] if row[1] else "",
                "year": row[2],
                "key_concepts": [{"label": c[0], "type": c[1]} for c in concepts],
            }
        )

    # Detect persistent gaps and debate zones via seed analysis
    try:
        from drbrain.graph.engine import GraphEngine

        graph = GraphEngine()
        graph.load_from_db(db)
        seeds = graph.detect_research_seeds(db)

        for s in seeds:
            seed_type = s.get("type", "")
            if seed_type == "unaddressed_gap":
                result["gaps"].append(
                    {
                        "description": s.get("description", ""),
                        "concept": s.get("concept", ""),
                    }
                )
            elif seed_type == "debate_zone":
                result["debates"].append(
                    {
                        "description": s.get("description", ""),
                        "concept": s.get("concept", ""),
                    }
                )
    except Exception:
        pass  # Seed detection requires data; skip if insufficient

    return result


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
