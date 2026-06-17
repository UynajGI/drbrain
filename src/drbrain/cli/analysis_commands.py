"""Analysis and reasoning commands."""

from __future__ import annotations

import asyncio
import json

import typer

from drbrain.cli._common import (
    _build_closure_context,
    _enrich_tree_with_sections,
    _render_landscape,
    _resolve_workspace_papers,
)
from drbrain.graph.engine import GraphEngine
from drbrain.storage.database import Database


def _print_evolution_stats(db: Database, concept: str) -> None:
    """Print temporal evolution signal and year-by-year counts for a concept."""
    signal = db.get_concept_signal(concept)
    evolution = db.get_concept_evolution(concept)

    if signal:
        sig = signal["signal"]
        emoji = {
            "emerging": "🆕",
            "established": "✅",
            "declining": "📉",
            "contested": "⚔️",
            "resurging": "🔄",
        }.get(sig, "")
        typer.echo(
            f"\n  {emoji} Signal: {sig}  "
            f"({signal['paper_count']} papers, "
            f"avg confidence {signal['avg_confidence']}, "
            f"{signal['first_seen']}–{signal['last_seen']})"
        )

    if evolution:
        typer.echo("  Year-by-year:")
        for entry in evolution:
            bar = "█" * entry["count"]
            trend_tag = {"growing": "↑", "declining": "↓", "first_appeared": "·"}.get(
                entry["trend"], " "
            )
            typer.echo(
                f"    {entry['year']}  {bar} {entry['count']} "
                f"(conf {entry['avg_conf']}) {trend_tag}"
            )


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
    session_id: str = typer.Option(
        None,
        "--session",
        "-s",
        help="Use persistent session. 'new' to create, or existing session ID.",
    ),
    workflow: str = typer.Option(
        None,
        "--workflow",
        "-w",
        help="Use a structured reasoning workflow: causal|contradiction|temporal|hypothesis",
    ),
    list_workflows_flag: bool = typer.Option(
        False,
        "--list-workflows",
        help="List available reasoning workflows and exit.",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable workflow-level result caching.",
    ),
    visualize: bool = typer.Option(
        False,
        "--visualize",
        "-v",
        help="Show workflow pipeline diagram and step-by-step result summary.",
    ),
):
    """LLM agent that reasons over the knowledge graph using tool-calling."""
    if isinstance(list_workflows_flag, typer.models.OptionInfo):
        list_workflows_flag = list_workflows_flag.default

    if list_workflows_flag:
        from drbrain.reasoning import list_workflows as _list_wfs

        wfs = _list_wfs()
        typer.echo("Available reasoning workflows:")
        for w in wfs:
            typer.echo(f"  {w['name']:15s}  {w['description']}")
        return

    # Normalize typer OptionInfo objects when called directly (not via CLI)
    if isinstance(bidirectional, typer.models.OptionInfo):
        bidirectional = bidirectional.default
    if isinstance(max_rounds, typer.models.OptionInfo):
        max_rounds = max_rounds.default
    if isinstance(session_id, typer.models.OptionInfo):
        session_id = session_id.default
    if isinstance(workflow, typer.models.OptionInfo):
        workflow = workflow.default
    if isinstance(no_cache, typer.models.OptionInfo):
        no_cache = no_cache.default
    if isinstance(visualize, typer.models.OptionInfo):
        visualize = visualize.default

    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])
    graph = GraphEngine()
    graph.load_from_db(db)

    models = cfg.get("llm", {}).get("models", [])
    if not models:
        typer.echo("No LLM models configured. Run: drbrain setup", err=True)
        db.close()
        raise typer.Exit(1)

    # ── Workflow path (structured pipeline) ──
    if workflow:
        from drbrain.reasoning import WorkflowContext, get_workflow

        try:
            wf = get_workflow(workflow)
        except ValueError as e:
            typer.echo(str(e), err=True)
            db.close()
            raise typer.Exit(1)

        from drbrain.reasoning import WorkflowVisualizer

        vz = WorkflowVisualizer(wf)

        if visualize:
            typer.echo(vz.text_flowchart())

        typer.echo(f"Workflow [{wf.name}]: {question}\n")
        if no_cache:
            wf_ctx = WorkflowContext(db=db, graph=graph, models=models, question=question)
        else:
            from drbrain.extractor.cache import ApiCache

            _cache_dir = cfg.get("dirs", {}).get("cache", "data/cache") + "/workflows"
            wf_ctx = WorkflowContext(
                db=db,
                graph=graph,
                models=models,
                question=question,
                cache=ApiCache(_cache_dir),
            )
        results = wf.execute(wf_ctx)

        # Output results
        if visualize:
            # Detailed step-by-step summary
            typer.echo(vz.summarize_results(wf_ctx))
            # Mermaid diagram (can be rendered in Markdown viewers)
            typer.echo(vz.mermaid_flowchart())
        else:
            # Default: show only LLM synthesis results
            for step_name, result in results.items():
                if result is None:
                    continue
                step = next((s for s in wf.steps if s.name == step_name), None)
                if step and step.requires_llm:
                    typer.echo(f"\n{'─' * 60}")
                    typer.echo(f"Result [{step_name}]:")
                    typer.echo(f"{'─' * 60}")
                    typer.echo(result)
                elif isinstance(result, dict):
                    typer.echo(f"[{step_name}] {result}")
                elif isinstance(result, list) and result:
                    typer.echo(f"[{step_name}] {len(result)} items")
                    for item in result[:3]:
                        typer.echo(f"  - {item}")

        db.close()
        return

    # ── Original reasoning paths (tool-calling agent) ──
    # Compute closure-inferred edges for initial seed concepts
    closure_ctx = ""
    if graph.graph.number_of_edges() > 0:
        from drbrain.query.bm25 import build_bm25_index

        idx = build_bm25_index(db)
        initial_results = idx.search(question, limit=5)
        seed_labels = [r.get("label", "") for r in initial_results if r.get("label")]
        if seed_labels:
            closure_ctx = _build_closure_context(graph, seed_labels, top_k=5)

    models = cfg.get("llm", {}).get("models", [])
    if not models:
        typer.echo("No LLM models configured. Run: drbrain setup", err=True)
        db.close()
        raise typer.Exit(1)

    if session_id:
        # ── SessionAgent path (persistent multi-turn reasoning) ──
        from drbrain.extractor.session_agent import SessionAgent

        agent = SessionAgent()
        if session_id == "new":
            sid = agent.create_session(db, title="reason", models=models)
            if closure_ctx:
                agent.inject_context(closure_ctx, label="closure")
        else:
            ok = agent.load_session(
                db, session_id, graph=graph, models=models, closure_context=closure_ctx
            )
            if not ok:
                typer.echo(f"Session not found: {session_id}", err=True)
                db.close()
                raise typer.Exit(1)
            sid = session_id

        agent.graph = graph

        if bidirectional:
            typer.echo(f"Bidirectional reasoning (session): {question}\n")
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
            typer.echo(f"Reasoning (session): {question}\n")
            answer = asyncio.run(agent.ask(question))
            typer.echo(answer)

        typer.echo(f"\n[Session: {sid}]")
    else:
        # ── ReasonerAgent path (stateless, original behavior) ──
        from drbrain.extractor.reasoner import ReasonerAgent

        agent = ReasonerAgent(db=db, graph_engine=graph, models=models, closure_context=closure_ctx)

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

    # Normalize typer OptionInfo objects when called directly (not via CLI)
    if isinstance(top_k, typer.models.OptionInfo):
        top_k = top_k.default
    if isinstance(json_output, typer.models.OptionInfo):
        json_output = json_output.default

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

    # Inject closure-inferred edges for BM25-matched concept labels
    seed_labels = [r.get("label", "") for r in results[:top_k] if r.get("label")]
    if seed_labels:
        closure_ctx = _build_closure_context(graph, seed_labels, top_k=top_k)
        if closure_ctx:
            context_parts.append("\nInferred relations (logical closure):")
            context_parts.append(closure_ctx)

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
    stats: bool = typer.Option(
        False, "--stats", help="Show temporal evolution signal and year-by-year counts"
    ),
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
        result = {"trees": trees}
        if stats:
            signal = db.get_concept_signal(concept)
            evolution = db.get_concept_evolution(concept)
            result["stats"] = {"signal": signal, "evolution": evolution}
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    elif mermaid:
        typer.echo(format_tree(trees, mermaid=True))
    else:
        typer.echo(f"\nEvolution of: {concept}\n")
        for root in trees:
            typer.echo(format_tree([root]))
        if stats:
            _print_evolution_stats(db, concept)

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
