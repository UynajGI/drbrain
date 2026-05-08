"""Analysis and reasoning commands."""

from __future__ import annotations

import asyncio
import json

import typer

from drbrain.cli._common import (
    _enrich_tree_with_sections,
    _render_landscape,
    _resolve_workspace_papers,
)
from drbrain.graph.engine import GraphEngine
from drbrain.storage.database import Database


def reason_cmd(
    ctx: typer.Context,
    question: str = typer.Argument(..., help="Question to reason about using the knowledge graph"),
    bidirectional: bool = typer.Option(
        False,
        "--bidirectional",
        "-b",
        help="Use bidirectional LLM-KG iterative reasoning loop (validates hypotheses against graph constraints)",
    ),
    max_rounds: int = typer.Option(
        3,
        "--max-rounds",
        "-r",
        help="Maximum hypothesis-revision rounds for bidirectional mode",
    ),
):
    """LLM agent that reasons over the knowledge graph using tool-calling."""
    from drbrain.extractor.reasoner import ReasonerAgent

    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])
    graph = GraphEngine()
    graph.load_from_db(db)

    models = cfg.get("llm", {}).get("models", [])
    if not models:
        typer.echo("No LLM models configured. Run: drbrain setup", err=True)
        db.close()
        raise typer.Exit(1)

    agent = ReasonerAgent(db=db, graph_engine=graph, models=models)

    if bidirectional:
        typer.echo(f"Bidirectional reasoning: {question}\n")
        result = asyncio.run(agent.reason_bidirectional(question, max_rounds=max_rounds))
        if "error" in result:
            typer.echo(f"Error: {result['error']}", err=True)
        else:
            typer.echo(f"Answer (round {result['rounds']}): {result['answer']}\n")
            typer.echo(f"Hypotheses explored: {len(result['hypotheses'])}")
            for i, (h, v) in enumerate(zip(result["hypotheses"], result["kg_validations"]), 1):
                typer.echo(
                    f"  Round {i}: consistent={v['consistent']}, "
                    f"violations={len(v['violations'])}, patterns={len(v['patterns'])}"
                )
    else:
        typer.echo(f"Reasoning: {question}\n")
        answer = asyncio.run(agent.reason(question))
        typer.echo(answer)

    db.close()


def ask_cmd(
    ctx: typer.Context,
    question: list[str] = typer.Argument(
        ..., help="Natural language question about the knowledge graph"
    ),
    top_k: int = typer.Option(5, "--top", "-k", help="Number of graph concepts to retrieve"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Ask a question in natural language — searches the KG and returns an answer.

    Example: drbrain ask "Is attention better than CNN for NLP?"
    """
    import asyncio

    question_text = " ".join(question)
    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])

    from drbrain.query.bm25 import build_bm25_index

    idx = build_bm25_index(db)
    results = idx.search(question_text, limit=top_k)

    if not results:
        db.close()
        typer.echo("No relevant concepts found in the knowledge graph.")
        return

    graph = GraphEngine()
    graph.load_from_db(db)
    context_parts = []
    for r in results[:top_k]:
        label = r.get("label", "")
        ctype = r.get("type", "")
        paper_id = r.get("local_id", "")
        paper_info = db.get_paper(paper_id) if paper_id else {}
        paper_title = paper_info.get("title", paper_id) if paper_info else paper_id
        context_parts.append(f"- {label} ({ctype}) from {paper_title}")
        if label in graph.graph:
            neighbors = graph.traverse({label}, hops=1, direction="both")[:5]
            for n in neighbors:
                rel = n.path[0].relation.replace("_", " ") if n.path else "related to"
                context_parts.append(f"  --{rel}--> {n.target}")

    context = "\n".join(context_parts[:50])

    models = cfg.llm.models if hasattr(cfg, "llm") else cfg.get("llm", {}).get("models", [])
    if not models:
        db.close()
        typer.echo("No LLM models configured. Showing graph context only:\n\n" + context)
        return

    from drbrain.extractor.llm_client import acall_text_with_fallback

    prompt = (
        f"Answer this research question using the knowledge graph context below.\n\n"
        f"Question: {question_text}\n\n"
        f"Knowledge Graph Context:\n{context}\n\n"
        f"Answer concisely in 2-4 sentences. If the context doesn't contain "
        f"enough information, say so. Cite specific concepts and relations."
    )

    answer = asyncio.run(acall_text_with_fallback(prompt, models, max_tokens=300))
    db.close()

    if json_output:
        import json as _json

        typer.echo(
            _json.dumps(
                {"question": question_text, "answer": answer, "context": context},
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    typer.echo(f"\nQ: {question_text}\n")
    typer.echo(f"A: {answer}\n")
    typer.echo(f"(based on {len(results)} graph concepts)")


def evolve_cmd(
    ctx: typer.Context,
    concept: str = typer.Argument(..., help="Concept label to trace evolution of"),
    direction: str = typer.Option(
        "both",
        "--direction",
        "-d",
        help="Traversal direction: ancestors, descendants, both",
    ),
    max_depth: int = typer.Option(3, "--max-depth", "-n", help="Max traversal depth"),
    mermaid: bool = typer.Option(False, "--mermaid", help="Output as Mermaid diagram"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Show how a concept evolved — its ancestors and descendants in the knowledge graph."""
    if direction not in ("ancestors", "descendants", "both"):
        typer.echo("--direction must be: ancestors, descendants, both", err=True)
        raise typer.Exit(1)

    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])
    graph = GraphEngine()
    graph.load_from_db(db)

    from drbrain.graph.genealogy import evolve_concept, format_tree

    trees = evolve_concept(graph, db, concept, direction=direction, max_depth=max_depth)

    if not trees:
        db.close()
        typer.echo(f"No concept found matching: {concept}")
        raise typer.Exit(0)

    if json_output:
        typer.echo(json.dumps(trees, indent=2, ensure_ascii=False, default=str))
    elif mermaid:
        typer.echo(format_tree(trees, mermaid=True))
    else:
        typer.echo(f"\nEvolution of: {concept}\n")
        for root in trees:
            typer.echo(format_tree([root]))

    db.close()


def descendants_cmd(
    ctx: typer.Context,
    paper_id: str = typer.Argument(..., help="Paper local_id to trace descendants of"),
    generations: int = typer.Option(
        3, "--generations", "-g", help="Number of generations to trace"
    ),
    mermaid: bool = typer.Option(False, "--mermaid", help="Output as Mermaid diagram"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    sections: bool = typer.Option(
        False, "--sections", help="Show section provenance for each concept"
    ),
):
    """Trace a paper's academic offspring — who cited, extended, or refined it."""
    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])
    graph = GraphEngine()
    graph.load_from_db(db)

    from drbrain.graph.genealogy import format_tree, trace_descendants

    tree = trace_descendants(db, graph, paper_id, generations=generations)

    if tree is None:
        db.close()
        typer.echo(f"Paper not found: {paper_id}", err=True)
        raise typer.Exit(1)

    if sections and tree:
        _enrich_tree_with_sections(tree, graph, db)

    if json_output:
        typer.echo(json.dumps(tree, indent=2, ensure_ascii=False, default=str))
    elif mermaid:
        typer.echo(format_tree([tree], mermaid=True))
    else:
        typer.echo(f"\nDescendants of: {paper_id}\n")
        typer.echo(format_tree([tree]))

    db.close()


def landscape_cmd(
    ctx: typer.Context,
    workspace: str = typer.Argument(None, help="Workspace name or path"),
    top_n: int = typer.Option(5, "--top-n", help="Top concepts/items to show"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Show a domain landscape -- timeline, persistent gaps, and debates."""
    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])

    # Resolve workspace
    if isinstance(workspace, typer.models.OptionInfo):
        workspace = workspace.default
    paper_ids = None
    if workspace:
        from drbrain.storage.workspace import load_workspace_papers

        try:
            paper_ids = load_workspace_papers(workspace)
        except (FileNotFoundError, OSError):
            paper_ids = []

    from drbrain.graph.genealogy import landscape_workspace

    result = landscape_workspace(db, workspace_path=workspace, paper_ids=paper_ids)

    if json_output:
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    elif "error" in result:
        typer.echo(f"Error: {result['error']}", err=True)
    else:
        _render_landscape(result, top_n)

    db.close()


def paradigm_cmd(
    ctx: typer.Context,
    concept: str = typer.Argument(None, help="Concept to check for paradigm shifts"),
    workspace: str = typer.Option(None, "--workspace", "-w", help="Scan entire workspace"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Detect paradigm shifts -- replacement, explosion, or cross-domain invasion."""
    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])
    graph = GraphEngine()
    graph.load_from_db(db)

    # Resolve workspace
    if isinstance(workspace, typer.models.OptionInfo):
        workspace = workspace.default
    paper_ids = None
    if workspace:
        from drbrain.storage.workspace import load_workspace_papers

        try:
            paper_ids = load_workspace_papers(workspace)
        except (FileNotFoundError, OSError):
            paper_ids = []

    from drbrain.graph.genealogy import detect_paradigm_shifts

    results = detect_paradigm_shifts(graph, db, concept=concept, paper_ids=paper_ids)

    if json_output:
        typer.echo(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    else:
        if not results:
            typer.echo("No paradigm shifts detected.")
        else:
            type_labels = {
                "replacement": "Replacement",
                "explosion": "Explosion",
                "cross_domain": "Cross-Domain",
            }
            for r in results:
                typer.echo(f"\n[{type_labels.get(r['type'], r['type'])}] {r['description']}")
                prov = r.get("provenance") or r.get("source_provenance") or ""
                if prov:
                    typer.echo(f"        {prov}")
                old_prov = r.get("old_provenance", "")
                new_prov = r.get("new_provenance", "")
                if old_prov and new_prov:
                    typer.echo(f"        old: {old_prov}")
                    typer.echo(f"        new: {new_prov}")

    db.close()


def transfers_cmd(
    ctx: typer.Context,
    from_ws: str = typer.Option(None, "--from", help="Source workspace (methods)"),
    to_ws: str = typer.Option(None, "--to", help="Target workspace (problems)"),
    auto: bool = typer.Option(False, "--auto", help="Auto-detect domains"),
    min_confidence: float = typer.Option(
        0.3, "--min-confidence", help="Minimum transfer confidence"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    history: bool = typer.Option(False, "--history", help="Show historical transfer timeline"),
    sections: bool = typer.Option(
        False, "--sections", help="Show section provenance for transferred concepts"
    ),
):
    """Discover cross-domain method migration opportunities."""
    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])
    graph = GraphEngine()
    graph.load_from_db(db)

    if sections:
        # Collect concept labels for section enrichment
        labels: set[str] = set()
        if history:
            from drbrain.graph.genealogy import find_transfer_history

            results = find_transfer_history(db, graph)
        else:
            from drbrain.graph.genealogy import (
                find_transfer_opportunities,
                find_transfer_opportunities_auto,
            )

            if auto:
                results = find_transfer_opportunities_auto(db, graph, min_confidence=min_confidence)
            elif from_ws and to_ws:
                src_papers = _resolve_workspace_papers(from_ws)
                tgt_papers = _resolve_workspace_papers(to_ws)
                results = find_transfer_opportunities(
                    db,
                    graph,
                    source_paper_ids=list(src_papers) if src_papers else [],
                    target_paper_ids=list(tgt_papers) if tgt_papers else [],
                    min_confidence=min_confidence,
                )
            else:
                typer.echo(
                    "Use --from/--to for explicit workspaces, --auto for automatic detection, "
                    "or --history for historical transfers.",
                    err=True,
                )
                db.close()
                raise typer.Exit(1)

        for r in results or []:
            if "source_method" in r:
                labels.add(str(r["source_method"]))
            if "target_problem" in r:
                labels.add(str(r["target_problem"]))

        section_map = graph.get_section_contexts_batch(db.conn, list(labels))

        # Enrich results
        for r in results or []:
            if "source_method" in r and r["source_method"] in section_map:
                r["source_section"] = section_map[r["source_method"]]["section"]
            if "target_problem" in r and r["target_problem"] in section_map:
                r["target_section"] = section_map[r["target_problem"]]["section"]

        if json_output:
            typer.echo(json.dumps(results, indent=2, ensure_ascii=False, default=str))
        else:
            if not results:
                typer.echo("No transfers found.")
            else:
                for r in results:
                    parts = [
                        f"{r.get('source_concept', '?')}",
                    ]
                    if r.get("source_section"):
                        parts.append(f"[{r['source_section']}]")
                    parts.append("→")
                    parts.append(f"{r.get('target_concept', '?')}")
                    if r.get("target_section"):
                        parts.append(f"[{r['target_section']}]")
                    if r.get("confidence"):
                        parts.append(f"({r['confidence']:.2f})")
                    typer.echo(" ".join(parts))
        db.close()
        return

    if history:
        from drbrain.graph.genealogy import find_transfer_history

        results = find_transfer_history(db, graph)
        if json_output:
            typer.echo(json.dumps(results, indent=2, ensure_ascii=False, default=str))
        else:
            if not results:
                typer.echo("No historical transfers found.")
            else:
                typer.echo("\nCross-Domain Transfer History")
                typer.echo("═" * 40)
                current_year = None
                for r in results:
                    year = r.get("year", "?")
                    if year != current_year:
                        current_year = year
                        typer.echo(f"\n{year}  ", nl=False)
                    else:
                        typer.echo("      ", nl=False)
                    typer.echo(
                        f"{r['source_concept']} → {r['target_concept']} ({r['confidence']:.2f})"
                    )
        db.close()
        return

    from drbrain.graph.genealogy import (
        find_transfer_opportunities,
        find_transfer_opportunities_auto,
    )

    if auto:
        results = find_transfer_opportunities_auto(db, graph, min_confidence=min_confidence)
    elif from_ws and to_ws:
        src_papers = _resolve_workspace_papers(from_ws)
        tgt_papers = _resolve_workspace_papers(to_ws)
        results = find_transfer_opportunities(
            db,
            graph,
            source_paper_ids=list(src_papers) if src_papers else [],
            target_paper_ids=list(tgt_papers) if tgt_papers else [],
            min_confidence=min_confidence,
        )
    else:
        typer.echo(
            "Use --from/--to for explicit workspaces, or --auto for automatic detection.", err=True
        )
        db.close()
        raise typer.Exit(1)

    if json_output:
        typer.echo(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    elif not results:
        typer.echo("No transfer opportunities found.")
    else:
        for r in results:
            typer.echo(f"  {r['source_method']} -> {r['target_problem']} ({r['confidence']:.2f})")

    db.close()


def isomorphism_cmd(
    ctx: typer.Context,
    concept: str = typer.Argument(None, help="Concept to find isomorphic patterns for"),
    min_confidence: float = typer.Option(
        0.5, "--min-confidence", help="Minimum confidence threshold"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Find structurally isomorphic subgraphs — concepts with similar relation patterns."""
    from drbrain.extractor.isomorphism import (
        enrich_isomorphisms_with_raptor,
        find_isomorphic_patterns,
    )

    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])
    graph = GraphEngine()
    graph.load_from_db(db)

    mappings = find_isomorphic_patterns(graph)
    mappings = enrich_isomorphisms_with_raptor(mappings, db)

    if concept:
        mappings = [m for m in mappings if m.source_domain == concept or m.target_domain == concept]
    if min_confidence > 0:
        mappings = [m for m in mappings if m.confidence >= min_confidence]

    if json_output:
        result = [
            {
                "source": m.source_domain,
                "target": m.target_domain,
                "shared_structure": m.shared_structure,
                "confidence": m.confidence,
                "raptor_source_context": m.raptor_source_context,
                "raptor_target_context": m.raptor_target_context,
            }
            for m in mappings
        ]
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        if not mappings:
            typer.echo("No isomorphic patterns found.")
        else:
            typer.echo(f"\nIsomorphic patterns ({len(mappings)}):")
            for m in mappings:
                typer.echo(
                    f"  {m.source_domain} ↔ {m.target_domain} "
                    f"({m.confidence:.2f}) [{m.shared_structure}]"
                )

    db.close()


def difficulty_cmd(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Show a difficulty map — gaps classified by source section type."""
    from drbrain.graph.genealogy import analyze_difficulty

    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])
    result = analyze_difficulty(db)

    if json_output:
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        total = sum(len(v) for v in result.values())
        if total == 0:
            typer.echo("No gaps found. Run: drbrain build first.")
        else:
            typer.echo(f"\nDifficulty map ({total} gaps)")
            typer.echo("=" * 50)
            for cat, label in [
                ("limitation", "Limitation gaps"),
                ("future_work", "Future work gaps"),
                ("discussion", "Discussion gaps"),
                ("uncategorized", "Uncategorized gaps"),
            ]:
                items = result.get(cat, [])
                if items:
                    typer.echo(f"\n{label} ({len(items)}):")
                    for g in items:
                        typer.echo(f"  * {g['label']}")
                        typer.echo(f"        {g['provenance']}")

    db.close()


def frontier_cmd(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Show knowledge frontier — active gaps, debates, and paradigm shifts."""
    from drbrain.graph.genealogy import analyze_frontier

    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])
    result = analyze_frontier(db)

    if json_output:
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        typer.echo("\nKnowledge Frontier")
        typer.echo("=" * 50)
        typer.echo(f"\n{result['summary']}\n")

        active = result.get("active_gaps", [])
        if active:
            typer.echo(f"Active gaps ({len(active)}):")
            for g in active[:10]:
                typer.echo(f"  * {g['label']} ({g['year']})")
                typer.echo(f"        {g['provenance']}")

        debates = result.get("debates", [])
        if debates:
            typer.echo(f"\nActive debates ({len(debates)}):")
            for d in debates[:5]:
                typer.echo(f"  * {d['description'][:120]}")

        shifts = result.get("paradigm_shifts", [])
        if shifts:
            typer.echo(f"\nParadigm shifts ({len(shifts)}):")
            for s in shifts[:5]:
                typer.echo(f"  [{s['type']}] {s['description'][:120]}")

    db.close()
