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
