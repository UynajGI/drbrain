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
