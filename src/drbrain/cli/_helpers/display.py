"""Shared helper functions for CLI commands."""

from __future__ import annotations

import typer

from drbrain.graph.engine import GraphEngine
from drbrain.storage.database import Database


def _apply_mined_rules(graph, mined_rules: list[dict]) -> list[dict]:
    """Apply mined path rules to the graph, returning inferred edges.

    Each mined rule has `body_path` (list of relations) and `head` (inferred relation).
    Matches the path pattern in the graph and infers direct edges with the head relation.
    """
    if not mined_rules:
        return []

    inferred: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for rule in mined_rules:
        body = rule["body_path"]
        head = rule["head"]
        confidence = rule.get("confidence", 0.5)

        if len(body) < 2:
            continue

        # Build pattern matches: for each 2-hop path matching body_path,
        # infer the head relation between source and target.
        # body_path: [r_i, r_j] means: src -[r_i]-> mid -[r_j]-> dst => src -[head]-> dst
        # Convert to (relation, direction) pattern for matching
        pattern = [(rel, "forward") for rel in body]

        matches = _match_pattern(graph, pattern)
        for src, dst in matches:
            edge_key = (src, dst, head)
            if edge_key not in seen:
                seen.add(edge_key)
                rule_name = f"mined:{head}"
                inferred.append(
                    {
                        "src": src,
                        "dst": dst,
                        "relation": head,
                        "via": rule_name,
                        "confidence": round(float(confidence), 4),
                    }
                )

    return inferred


def _match_pattern(graph, pattern: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Find all node pairs matching a relation path pattern.

    Pattern is a list of (relation, direction) steps where direction is
    always "forward" (src → dst along the relation edge).
    """
    from collections import defaultdict

    if len(pattern) < 2:
        return []

    # Build adjacency indices for each relation in the pattern
    rel_indices = []
    for rel, direction in pattern:
        idx: dict[str, set[str]] = defaultdict(set)
        for u, v, data in graph.graph.edges(data=True):
            if data["relation"] == rel:
                if direction == "forward":
                    idx[v].add(u)  # given v, find u where u→v
                else:
                    idx[u].add(v)
        rel_indices.append(idx)

    first_idx = rel_indices[0]
    results: list[tuple[str, str]] = []
    visited_edges: set[tuple[str, str, str]] = set()

    for middle_node, prev_nodes in first_idx.items():
        for prev in prev_nodes:
            end_nodes = _extend_chain(graph, rel_indices[1:], middle_node)
            for end in end_nodes:
                edge_key = (prev, end, pattern[0][0])
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    results.append((prev, end))

    return results


def _extend_chain(graph, remaining_indices: list[dict[str, set[str]]], current: str) -> set[str]:
    """Recursively extend a chain through remaining relation indices."""
    if not remaining_indices:
        return {current}

    idx = remaining_indices[0]
    next_nodes = idx.get(current, set())
    if not remaining_indices[1:]:
        return next_nodes

    result: set[str] = set()
    for node in next_nodes:
        result |= _extend_chain(graph, remaining_indices[1:], node)
    return result


def _export_paper_to_meta(db: Database, local_id: str) -> dict:
    """Build export-ready metadata dict from DB."""
    paper = db.get_paper(local_id)
    if not paper:
        return {}

    authors = db.conn.execute(
        "SELECT GROUP_CONCAT(a.variant, ' and ') "
        "FROM concepts c JOIN aliases a ON a.canonical_id = c.label "
        "WHERE c.local_id = ? AND c.type = 'Actor'",
        (local_id,),
    ).fetchone()

    author_list = authors[0] if authors and authors[0] else ""
    first_author = author_list.split(" and ")[0].strip() if author_list else ""
    from drbrain.storage.export import _extract_lastname

    lastname = _extract_lastname(first_author)

    return {
        "local_id": local_id,
        "title": paper.get("title", ""),
        "year": paper.get("year"),
        "doi": paper.get("doi", ""),
        "arxiv": paper.get("arxiv", ""),
        "authors": author_list,
        "first_author_lastname": lastname,
        "paper_type": paper.get("paper_type", "paper"),
        "abstract": paper.get("abstract", ""),
        "journal": paper.get("journal", ""),
        "publisher": paper.get("publisher", ""),
        "citation_count": paper.get("citation_count", 0),
        "volume": paper.get("volume", ""),
        "pages": paper.get("pages", ""),
    }


def _enrich_tree_with_sections(tree: dict, graph: GraphEngine, db: Database) -> None:
    """Recursively enrich a genealogy tree with section provenance."""
    labels: list[str] = []

    def _collect(node: dict) -> None:
        for key in ("concept", "label"):
            if key in node:
                labels.append(str(node[key]))
        for child in node.get("children", []):
            _collect(child)

    _collect(tree)
    if not labels:
        return

    section_map = graph.get_section_contexts_batch(db.conn, labels)

    def _enrich(node: dict) -> None:
        for key in ("concept", "label"):
            if key in node and node[key] in section_map:
                node["section"] = section_map[node[key]]["section"]
                node["node_id"] = section_map[node[key]]["node_id"]
        for child in node.get("children", []):
            _enrich(child)

    _enrich(tree)


def _show_actor(cfg: dict, author_id: str) -> None:
    """Show detailed info for a single actor."""
    db = Database(cfg["db"]["path"])

    # Get display names from aliases
    aliases = db.conn.execute(
        "SELECT variant FROM aliases WHERE canonical_id = ?", (author_id,)
    ).fetchall()
    display_names = [a[0] for a in aliases]

    # Get papers
    papers = db.conn.execute(
        "SELECT DISTINCT c.local_id, p.title, p.year "
        "FROM concepts c JOIN papers p ON c.local_id = p.local_id "
        "WHERE c.type = 'Actor' AND c.label = ? ORDER BY p.year",
        (author_id,),
    ).fetchall()
    paper_ids = [p[0] for p in papers]

    # Get shared_actor connections (edges between papers of this author and other papers)
    connected_papers: list[str] = []
    if paper_ids:
        placeholders = ",".join("?" for _ in paper_ids)
        rows = db.conn.execute(
            f"SELECT DISTINCT e.dst_id FROM edges e "
            f"WHERE e.relation = 'shared_actor' AND e.src_id IN ({placeholders})",
            paper_ids,
        ).fetchall()
        connected_papers = [r[0] for r in rows if r[0] not in paper_ids]

    db.close()

    if not papers:
        typer.echo(f"Actor '{author_id}' has no associated papers.")
        return

    typer.echo(f"\nAuthor: {author_id}")
    if display_names:
        typer.echo(f"Display: {', '.join(display_names)}")
    typer.echo(f"Papers: {len(papers)}")
    for title, year in [(p[1], p[2]) for p in papers]:
        year_str = f" ({year})" if year else ""
        typer.echo(f"  - {title}{year_str}")

    if connected_papers:
        typer.echo(f"\nShared actor connections ({len(connected_papers)}):")
        # Resolve connected papers to their actors
        db2 = Database(cfg["db"]["path"])
        for pid in connected_papers:
            paper = db2.get_paper(pid)
            title = paper["title"][:80] if paper else pid
            connected_actors = db2.conn.execute(
                "SELECT DISTINCT c.label FROM concepts c WHERE c.type = 'Actor' AND c.local_id = ?",
                (pid,),
            ).fetchall()
            actor_names = []
            for (aid,) in connected_actors:
                alias_row = db2.conn.execute(
                    "SELECT variant FROM aliases WHERE canonical_id = ? LIMIT 1", (aid,)
                ).fetchone()
                actor_names.append(alias_row[0] if alias_row else aid)
            typer.echo(f"  - [{', '.join(actor_names)}] {title}")
        db2.close()


def _print_analyze_report(report: dict) -> None:
    """Print a formatted analysis report."""
    if "error" in report:
        typer.echo(f"Error: {report['error']}", err=True)
        return

    if report.get("executive_summary"):
        typer.echo("\n[bold]Executive Summary[/bold]")
        typer.echo(f"  {report['executive_summary']}")

    p = report["paper"]
    s = report["summary"]
    typer.echo(f"\n[bold]Knowledge Frontier: {p['title']} ({p['year']})[/bold]")

    if report.get("cross_paper_insights"):
        insights = report["cross_paper_insights"]
        typer.echo(f"\n[bold]── Cross-paper Insights ({len(insights)})[/bold]")
        for ins in insights[:5]:
            typer.echo(f"  Method '{ins['method']}' ({ins['method_paper']})")
            typer.echo(f"    → could address Problem '{ins['problem']}' ({ins['problem_paper']})")
            typer.echo(f"    (similarity: {ins['similarity']})")

    typer.echo(f"\n[bold]── Research Seeds ({s['seeds']})[/bold]")
    for seed in report.get("seeds", []):
        typer.echo(
            f"  [{seed.get('type', '?')}] {seed.get('concept', '?')}: {seed.get('description', '?')}"
        )
        if seed.get("suggested_solutions"):
            typer.echo(f"    → {seed['suggested_solutions']}")

    typer.echo(f"\n[bold]── Causal Chains ({s['causal_chains']})[/bold]")
    for chain in report.get("causal_chains", []):
        typer.echo(f"  {chain['source']} → {chain['target']} (via: {chain['via']})")

    typer.echo(f"\n[bold]── Inferred Edges ({s['inferred_edges']})[/bold]")

    if report.get("critical_nodes"):
        typer.echo(f"\n[bold]── Critical Nodes ({s['critical_nodes']})[/bold]")
        for node in report["critical_nodes"]:
            typer.echo(f"  {node}")

    if report.get("hypotheses"):
        typer.echo(f"\n[bold]── Hypotheses ({s['hypotheses']})[/bold]")
        for hyp in report["hypotheses"]:
            typer.echo(f"  [{hyp['type']}] {hyp['description']} ({hyp['confidence']:.2f})")

    if report.get("isomorphisms"):
        typer.echo(f"\n[bold]── Isomorphisms ({s['isomorphisms']})[/bold]")
        for iso in report["isomorphisms"]:
            typer.echo(f"  {iso['pattern']} ({iso['similarity']:.2f})")

    typer.echo()


def _render_landscape(result: dict, top_n: int):
    """Render landscape as ASCII timeline."""
    timeline = result.get("timeline", [])
    if not timeline:
        typer.echo("No papers found.")
        return

    typer.echo("\nLandscape")
    typer.echo("=" * 60)

    current_year = None
    for entry in timeline:
        year = entry["year"]
        title = entry["title"]

        if year != current_year:
            current_year = year
            typer.echo(f"\n  {year}  ", nl=False)
        else:
            typer.echo("        ", nl=False)

        typer.echo(f"{title}")

        for concept in entry.get("key_concepts", [])[:top_n]:
            typer.echo(f"        +- {concept['label']} [{concept['type']}]")

    gaps = result.get("gaps", [])
    if gaps:
        typer.echo(f"\nPersistent gaps ({len(gaps)}):")
        for g in gaps[:top_n]:
            provenance = g.get("provenance", "")
            typer.echo(f"  * {g['description'][:120]} ({g.get('concept', '')})")
            if provenance:
                typer.echo(f"        {provenance}")

    debates = result.get("debates", [])
    if debates:
        typer.echo(f"\nDebates ({len(debates)}):")
        for d in debates[:top_n]:
            provenance = d.get("provenance", "")
            typer.echo(f"  * {d['description'][:120]} ({d.get('concept', '')})")
            if provenance:
                typer.echo(f"        {provenance}")


def _build_closure_context(
    graph,
    seed_labels: list[str],
    top_k: int = 5,
) -> str:
    """Build a context string from closure-inferred edges for seed concept labels.

    Runs ``closure_incremental`` scoped to the given seed labels, sorts by
    confidence (descending), and returns lines in the format::

        --[inferred: <relation>]--> <dst> (confidence: X.XX, via: <via>)

    Args:
        graph: GraphEngine instance loaded from DB.
        seed_labels: Concept labels that were matched by BM25/search.
        top_k: Maximum number of inferred edges to include.

    Returns:
        Formatted multi-line string, or empty string if no edges inferred.
    """
    if not seed_labels or graph.graph.number_of_edges() == 0:
        return ""

    inferred = graph.closure_incremental(set(seed_labels))
    if not inferred:
        return ""

    # Sort by confidence descending (default 1.0 if missing)
    sorted_edges = sorted(
        inferred,
        key=lambda e: e.get("confidence", 1.0),
        reverse=True,
    )

    lines: list[str] = []
    for edge in sorted_edges[:top_k]:
        relation = edge["relation"].replace("_", " ")
        conf = edge.get("confidence", 1.0)
        via = edge.get("via", "")
        # Build annotation
        annotation_parts = [f"confidence: {conf:.2f}"]
        if via:
            annotation_parts.append(f"via: {via}")
        annotation = ", ".join(annotation_parts)
        lines.append(f"  --[inferred: {relation}]--> {edge['dst']} ({annotation})")

    return "\n".join(lines)
