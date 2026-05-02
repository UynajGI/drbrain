"""Graph query subcommands: neighbors, path."""

from __future__ import annotations

import json

import typer

from drbrain.cli.commands import _resolve_node_type, _resolve_workspace_papers, load_config
from drbrain.graph.engine import GraphEngine
from drbrain.storage.database import Database

graph_app = typer.Typer(help="Direct graph queries without BM25 text search")


@graph_app.command("neighbors")
def neighbors_cmd(
    node_label: str = typer.Argument(..., help="Concept label or paper ID"),
    hops: int = typer.Option(1, "--hops", "-n", help="Number of hops"),
    relation: str = typer.Option(
        None,
        "--relation",
        "-R",
        help="Comma-separated relation types (e.g. addresses,extends)",
    ),
    direction: str = typer.Option(
        "both",
        "--direction",
        "-D",
        help="Traversal direction: forward, backward, or both",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
    workspace: str = typer.Option(None, "--workspace", "-w", help="Limit to workspace"),
):
    """Traverse graph from a node, showing neighbors with path info."""
    cfg = load_config()
    db = Database(cfg["db"]["path"])

    # Parse relation filter
    _relations: set[str] | None = None
    if relation is not None:
        _relations = {r.strip() for r in relation.split(",") if r.strip()}
        valid_relations = {
            "addresses",
            "leaves_open",
            "points_to",
            "proposes",
            "extends",
            "replaces",
            "solves",
            "supports",
            "challenges",
            "limits",
            "constrains",
            "affiliated_with",
        }
        invalid = _relations - valid_relations
        if invalid:
            typer.echo(f"Invalid relation(s): {', '.join(sorted(invalid))}", err=True)
            typer.echo(f"Valid relations: {', '.join(sorted(valid_relations))}", err=True)
            raise typer.Exit(1)

    if direction not in ("forward", "backward", "both"):
        typer.echo(
            f"Invalid direction '{direction}'. Must be: forward, backward, or both",
            err=True,
        )
        raise typer.Exit(1)

    graph = GraphEngine()
    paper_ids = _resolve_workspace_papers(workspace)
    graph.load_from_db(db, paper_ids=paper_ids)

    # Check node exists
    if node_label not in graph.graph:
        typer.echo(f"Node '{node_label}' not found in graph.")
        db.close()
        raise typer.Exit(1)

    results: list[dict] = []
    trs = graph.traverse(
        start_nodes={node_label},
        hops=hops,
        relations=_relations,
        direction=direction,
    )

    seen_ids = {node_label}
    for tr in trs:
        if tr.target in seen_ids:
            continue
        seen_ids.add(tr.target)

        node_type, paper = _resolve_node_type(db, tr.target)

        if node_type == "Paper" and paper:
            label = paper["title"]
            text = paper.get("abstract", "")
            year = paper.get("year")
        else:
            label = tr.target
            text = ""
            year = None

        results.append(
            {
                "local_id": tr.target,
                "type": node_type,
                "label": label,
                "text": text,
                "year": year,
                "_via_graph": True,
                "_source_seed": tr.source,
                "_distance": tr.distance,
                "_path": [
                    {"src": s.src, "relation": s.relation, "dst": s.dst, "hop": s.hop}
                    for s in tr.path
                ],
            }
        )

    graph.graph = None

    if json_output:
        typer.echo(json.dumps(results, indent=2, ensure_ascii=False, default=str))
        db.close()
        return

    if not results:
        typer.echo(f"No neighbors found for '{node_label}'.")
        db.close()
        return

    seed_type, _ = _resolve_node_type(db, node_label)
    db.close()

    typer.echo(f"Neighbors of {node_label} ({seed_type}):")
    for r in results:
        path_parts = [node_label]
        for step in r.get("_path", []):
            path_parts.append(step["relation"])
            path_parts.append(step["dst"])
        path_str = " -> ".join(path_parts)
        typer.echo(f"  {r['local_id']} ({r['type']})")
        typer.echo(f"    graph: {path_str}")


@graph_app.command("path")
def path_cmd(
    src_label: str = typer.Argument(..., help="Source node label"),
    dst_label: str = typer.Argument(..., help="Destination node label"),
    max_length: int = typer.Option(6, "--max-length", help="Maximum path length (BFS cutoff)"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
    workspace: str = typer.Option(None, "--workspace", "-w", help="Limit to workspace"),
):
    """Find shortest path between two nodes in the knowledge graph."""
    import networkx as nx

    cfg = load_config()
    db = Database(cfg["db"]["path"])

    graph = GraphEngine()
    paper_ids = _resolve_workspace_papers(workspace)
    graph.load_from_db(db, paper_ids=paper_ids)

    g = graph.graph

    # Check nodes exist
    if src_label not in g:
        typer.echo(f"Source node '{src_label}' not found in graph.", err=True)
        db.close()
        raise typer.Exit(1)
    if dst_label not in g:
        typer.echo(f"Target node '{dst_label}' not found in graph.", err=True)
        db.close()
        raise typer.Exit(1)
    if src_label == dst_label:
        typer.echo("Source and target are the same node.")
        db.close()
        return

    # Shortest path on undirected copy for pathfinding
    ug = g.to_undirected()
    try:
        node_path = nx.shortest_path(ug, source=src_label, target=dst_label)
    except (nx.NetworkXNoPath, nx.NetworkXError):
        typer.echo(
            f"No path found between '{src_label}' and '{dst_label}' (max length: {max_length})"
        )
        db.close()
        return

    # Check max_length (cutoff applied by nx.shortest_path)
    path_len = len(node_path) - 1
    if path_len > max_length:
        typer.echo(
            f"No path found between '{src_label}' and '{dst_label}' (max length: {max_length})"
        )
        db.close()
        return

    # Recover edge data from original directed graph
    path_steps: list[dict] = []
    for i in range(len(node_path) - 1):
        u, v = node_path[i], node_path[i + 1]
        # Check both directions in the directed graph
        if g.has_edge(u, v):
            edge_data = list(g[u][v].values())[0]  # first edge's data
            path_steps.append(
                {
                    "src": u,
                    "relation": edge_data["relation"],
                    "dst": v,
                    "direction": "forward",
                }
            )
        elif g.has_edge(v, u):
            edge_data = list(g[v][u].values())[0]
            path_steps.append(
                {
                    "src": v,
                    "relation": edge_data["relation"],
                    "dst": u,
                    "direction": "backward",
                }
            )

    db.close()

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "src": src_label,
                    "dst": dst_label,
                    "length": path_len,
                    "path": path_steps,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    # Terminal output
    parts: list[str] = []
    for step in path_steps:
        if step["direction"] == "forward":
            parts.append(f"{step['src']} --{step['relation']}--> {step['dst']}")
        else:
            parts.append(f"{step['dst']} --{step['relation']}--> {step['src']} (reversed)")

    typer.echo(f"Path from {src_label} to {dst_label} ({path_len} hops):")
    if parts:
        typer.echo("  " + " -> ".join(parts))
    else:
        typer.echo("  (direct)")


@graph_app.command("related")
def related_cmd(
    paper_id: list[str] = typer.Argument(..., help="Two or more paper local_id values"),
    mode: str = typer.Option(
        "concepts",
        "--mode",
        "-m",
        help="Analysis mode: concepts, graph, or edges",
    ),
    min_shared: int = typer.Option(
        2,
        "--min-shared",
        help="Minimum number of papers a concept/edge must appear in to be shown",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
    workspace: str = typer.Option(None, "--workspace", "-w", help="Limit to workspace"),
):
    """Analyze shared concepts and connections across multiple papers."""
    cfg = load_config()
    db = Database(cfg["db"]["path"])

    if len(paper_id) < 2:
        typer.echo("At least 2 paper IDs required.", err=True)
        db.close()
        raise typer.Exit(1)

    if mode not in ("concepts", "graph", "edges"):
        typer.echo(f"Invalid mode '{mode}'. Must be: concepts, graph, or edges", err=True)
        db.close()
        raise typer.Exit(1)

    for pid in paper_id:
        paper = db.get_paper(pid)
        if not paper:
            typer.echo(f"Paper '{pid}' not found in database.", err=True)
            db.close()
            raise typer.Exit(1)

    if mode == "concepts":
        _related_concepts(db, paper_id, min_shared, json_output)
    elif mode == "graph":
        _related_graph(db, paper_id, min_shared, json_output, workspace)
    elif mode == "edges":
        _related_edges(db, paper_id, min_shared, json_output)

    db.close()


def _related_concepts(db, paper_ids: list[str], min_shared: int, json_output: bool):
    """Mode: concepts — SQL intersection of concept labels across papers."""
    placeholders = ",".join("?" for _ in paper_ids)
    rows = db.conn.execute(
        f"SELECT label, type, COUNT(DISTINCT local_id) as paper_count "
        f"FROM concepts WHERE local_id IN ({placeholders}) "
        f"GROUP BY label, type HAVING paper_count >= ? "
        f"ORDER BY paper_count DESC, type, label",
        (*paper_ids, min_shared),
    ).fetchall()

    shared = [{"label": r[0], "type": r[1], "paper_count": r[2]} for r in rows]

    coverage: list[dict] = []
    for pid in paper_ids:
        total = db.conn.execute(
            "SELECT COUNT(*) FROM concepts WHERE local_id = ?", (pid,)
        ).fetchone()[0]
        shared_count = db.conn.execute(
            f"SELECT COUNT(DISTINCT label) FROM concepts "
            f"WHERE local_id = ? AND label IN ("
            f"  SELECT label FROM concepts WHERE local_id IN ({placeholders}) "
            f"  GROUP BY label HAVING COUNT(DISTINCT local_id) >= ?"
            f")",
            (pid, *paper_ids, min_shared),
        ).fetchone()[0]
        coverage.append({"paper_id": pid, "total_concepts": total, "shared_concepts": shared_count})

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "mode": "concepts",
                    "papers": paper_ids,
                    "min_shared": min_shared,
                    "shared": shared,
                    "coverage": coverage,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if not shared:
        typer.echo(f"No shared concepts found (min-shared: {min_shared}).")
        return

    typer.echo(f"Shared concepts across {len(paper_ids)} papers (min-shared: {min_shared}):")
    for s in shared:
        typer.echo(f"  {s['label']} ({s['type']})  {s['paper_count']} papers")

    typer.echo()
    typer.echo("Coverage:")
    for c in coverage:
        typer.echo(
            f"  {c['paper_id']}: {c['total_concepts']} concepts, {c['shared_concepts']} shared"
        )


def _related_edges(db, paper_ids: list[str], min_shared: int, json_output: bool):
    """Mode: edges — SQL query for shared (relation, target) edge patterns."""
    placeholders = ",".join("?" for _ in paper_ids)
    rows = db.conn.execute(
        f"SELECT relation, dst_id, COUNT(DISTINCT source_paper) as paper_count "
        f"FROM edges WHERE source_paper IN ({placeholders}) "
        f"GROUP BY relation, dst_id HAVING paper_count >= ? "
        f"ORDER BY paper_count DESC, relation, dst_id",
        (*paper_ids, min_shared),
    ).fetchall()

    shared_edges = [{"relation": r[0], "target": r[1], "paper_count": r[2]} for r in rows]

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "mode": "edges",
                    "papers": paper_ids,
                    "min_shared": min_shared,
                    "shared_edges": shared_edges,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if not shared_edges:
        typer.echo(f"No shared edge patterns found (min-shared: {min_shared}).")
        return

    typer.echo(f"Shared edge patterns across {len(paper_ids)} papers (min-shared: {min_shared}):")
    for e in shared_edges:
        typer.echo(f"  {e['relation']} -> {e['target']}  {e['paper_count']} papers")


def _related_graph(
    db, paper_ids: list[str], min_shared: int, json_output: bool, workspace: str | None
):
    """Mode: graph — load graph, traverse from each paper's concepts, intersect neighbors."""
    graph = GraphEngine()
    paper_ids_set = _resolve_workspace_papers(workspace)
    graph.load_from_db(db, paper_ids=paper_ids_set)

    # Collect each paper's concept labels
    paper_concepts: dict[str, set[str]] = {}
    for pid in paper_ids:
        rows = db.conn.execute("SELECT label FROM concepts WHERE local_id = ?", (pid,)).fetchall()
        labels = {r[0] for r in rows}
        if labels:
            paper_concepts[pid] = labels

    if not paper_concepts:
        typer.echo("No concepts found for any of the given papers.")
        return

    # For each paper, collect 1-hop neighbors with path info
    paper_neighbors: dict[str, set[str]] = {}
    paper_paths: dict[str, dict[str, list[dict]]] = {}
    for pid, labels in paper_concepts.items():
        if not labels:
            paper_neighbors[pid] = set()
            paper_paths[pid] = {}
            continue
        trs = graph.traverse(start_nodes=labels, hops=1, direction="both")
        paper_neighbors[pid] = {tr.target for tr in trs} | labels
        paper_paths[pid] = {}
        for tr in trs:
            if tr.target not in paper_paths[pid]:
                paper_paths[pid][tr.target] = [
                    {"src": s.src, "relation": s.relation, "dst": s.dst} for s in tr.path
                ]

    graph.graph = None

    # Find concepts shared by >= min_shared papers
    concept_paper_map: dict[str, set[str]] = {}
    for pid, neighbors in paper_neighbors.items():
        for concept in neighbors:
            if concept not in concept_paper_map:
                concept_paper_map[concept] = set()
            concept_paper_map[concept].add(pid)

    shared_concepts = {
        c: papers for c, papers in concept_paper_map.items() if len(papers) >= min_shared
    }

    # Build connections from pre-computed paths
    connections: list[dict] = []
    for concept, papers in sorted(shared_concepts.items()):
        paths_per_paper: list[dict] = []
        for pid in sorted(papers):
            if pid in paper_paths and concept in paper_paths[pid]:
                paths_per_paper.append({"paper_id": pid, "path": paper_paths[pid][concept]})
        connections.append(
            {
                "concept": concept,
                "paper_count": len(papers),
                "paths": paths_per_paper,
            }
        )

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "mode": "graph",
                    "papers": paper_ids,
                    "connections": connections,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if not connections:
        typer.echo("No shared graph connections found.")
        return

    for conn in connections:
        typer.echo(f"\n  {conn['concept']} — shared by {conn['paper_count']} papers:")
        for p in conn["paths"]:
            path_str = " -> ".join(f"{s['src']} --{s['relation']}--> {s['dst']}" for s in p["path"])
            typer.echo(f"    {p['paper_id']}:  {path_str}")
