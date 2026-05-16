"""Build and embed pipeline commands."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from drbrain.graph.engine import GraphEngine
from drbrain.storage.database import Database
from drbrain.storage.paths import raw_md_path, tree_json_path


def translate_cmd(
    ctx: typer.Context,
    local_id: str = typer.Argument(..., help="Paper local_id"),
    target_lang: str = typer.Option(
        "zh", "--lang", "-l", help="Target language code: zh, en, ja, etc."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-translation even if output exists"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Translate a paper's markdown via LLM."""
    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])

    paper = db.get_paper(local_id)
    db.close()

    if not paper:
        typer.echo(f"Paper not found: {local_id}", err=True)
        raise typer.Exit(1)

    papers_dir = Path(cfg.get("dirs", {}).get("papers", "data/papers"))
    paper_dir = papers_dir / local_id

    if not raw_md_path(paper_dir).exists():
        typer.echo(f"No raw.md found for {local_id}. Run 'drbrain ingest' first.", err=True)
        raise typer.Exit(1)

    llm_models = cfg.get("llm", {}).get("models", [])
    if not llm_models:
        typer.echo("No LLM models configured.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Translating: {paper['title']} (→ {target_lang})")

    from drbrain.services.translate import translate_paper

    result = translate_paper(
        paper_dir,
        models=llm_models,
        target_lang=target_lang,
        force=force,
    )

    if not result.ok:
        if result.partial:
            msg = f"Partial translation ({result.completed_chunks}/{result.total_chunks} chunks) — re-run to resume"
        elif result.skip_reason:
            msg = f"Translation skipped: {result.skip_reason}"
        else:
            msg = "Translation failed."
        if json_output:
            typer.echo(
                json.dumps(
                    {"error": msg, "partial": result.partial, "skip_reason": result.skip_reason},
                    ensure_ascii=False,
                )
            )
        else:
            typer.echo(msg, err=True)
        if not result.partial:
            raise typer.Exit(1)
        return

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "paper": local_id,
                    "output": str(result.path),
                    "completed_chunks": result.completed_chunks,
                    "total_chunks": result.total_chunks,
                },
                ensure_ascii=False,
            )
        )
    else:
        typer.echo(f"Translated: {result.path}")


def build_cmd(
    ctx: typer.Context,
    paper_id: list[str] = typer.Argument(
        None, help="Paper IDs to build graph for. Omit for all unprocessed."
    ),
    all_papers: bool = typer.Option(
        False, "--all", help="Build graph for all papers in the database"
    ),
    skip_refine: bool = typer.Option(
        False, "--skip-refine", help="Skip iterative refinement stage"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """Build knowledge graph from ingested papers using 5-stage LLM extraction."""
    import time as _time

    from loguru import logger as _build_log

    from drbrain.extractor.concept import build_graph_from_tree

    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])

    # Select papers to process
    if all_papers:
        papers = db.get_all_papers()
    elif paper_id:
        papers = []
        for pid in paper_id:
            p = db.get_paper(pid)
            if p:
                papers.append(p)
            else:
                typer.echo(f"Paper not found: {pid}", err=True)
    else:
        all_papers = db.get_all_papers()
        papers = [p for p in all_papers if p.get("status") == "uploaded"]

    _build_log.info(f"[build] starting: {len(papers)} papers, skip_refine={skip_refine}")

    if not papers:
        typer.echo("No papers to build. Run: drbrain ingest first")
        db.close()
        return

    llm_models = cfg.get("llm", {}).get("models", [])
    if not llm_models:
        typer.echo("No LLM models configured. Run: drbrain setup", err=True)
        db.close()
        raise typer.Exit(1)

    papers_dir = Path(cfg.get("dirs", {}).get("papers", "data/papers"))
    all_results = []

    for paper in papers:
        pid = paper["local_id"]
        _t_paper = _time.monotonic()
        _build_log.info(f"[build] paper={pid} title={paper['title'][:60]}")
        typer.echo(f"\n{pid}: {paper['title'][:80]}")

        tree_path = tree_json_path(papers_dir / pid)
        md_path = raw_md_path(papers_dir / pid)

        # Retry tree generation if raw.md exists but tree.json is missing
        if not tree_path.exists() and md_path.exists():
            typer.echo("  Tree missing, retrying...")
            try:
                from drbrain.parser.pageindex_parser import TreeConfig, md_to_tree

                pageindex_cfg = TreeConfig(
                    if_add_node_summary=True,
                    if_add_doc_description=True,
                    if_add_node_text=False,
                    if_add_node_id=True,
                    max_node_tokens=10000,
                    min_token_threshold=5000,
                )
                doc_tree = asyncio.run(
                    md_to_tree(str(md_path), config=pageindex_cfg, models=llm_models)
                )
                tree_path.write_text(doc_tree.to_json(), encoding="utf-8")
                typer.echo(f"  Tree regenerated: {len(doc_tree.structure)} sections")
            except Exception as e:
                typer.echo(f"  Tree regeneration failed: {e}")
                continue
        elif not md_path.exists():
            typer.echo("  No raw.md — ingest this paper first")
            continue

        import json as _json

        tree = _json.loads(tree_path.read_text(encoding="utf-8"))
        structure = tree.get("structure", [])
        if not structure:
            typer.echo("  Empty tree structure — skipping")
            continue

        # Run 5-stage pipeline
        typer.echo("  Stage 1: Ontology...")
        result = asyncio.run(
            build_graph_from_tree(md_path, structure, llm_models, skip_refine=skip_refine)
        )

        concepts = result.get("concepts", [])
        relations = result.get("relations", [])
        merges = result.get("merges", [])
        corrections = result.get("corrections", [])

        typer.echo(f"  Stage 2: Entities...   {len(concepts)} concepts")
        typer.echo(f"  Stage 3: Relations...  {len(relations)} edges")
        typer.echo(f"  Stage 4: Coreference... {len(merges)} merges")
        if not skip_refine:
            typer.echo(f"  Stage 5: Refine...     {len(corrections)} corrections")
        _build_log.info(
            f"[build] extracted paper={pid} concepts={len(concepts)} relations={len(relations)} "
            f"merges={len(merges)} corrections={len(corrections)}"
        )

        # Validate and insert concepts
        valid_types = {"Problem", "Method", "Conclusion", "Debate", "Gap", "Actor"}
        valid_count = 0
        rejected = 0
        for c in concepts:
            ctype = c.get("type", "")
            label = c.get("label", "")
            conf = c.get("confidence", 0.5)
            if ctype not in valid_types or not label:
                rejected += 1
                continue
            db.insert_concept(
                pid, ctype, label, conf, section=c.get("section", ""), node_id=c.get("node_id", "")
            )
            valid_count += 1

        # Insert relations
        for r in relations:
            head = r.get("head", "")
            rel = r.get("rel", "")
            tail = r.get("tail", "")
            if head and rel and tail:
                try:
                    db.insert_edge(
                        head,
                        tail,
                        rel,
                        pid,
                        node_id=r.get("node_id", ""),
                        section=r.get("section", ""),
                    )
                except Exception:
                    _build_log.debug(f"duplicate or invalid edge: {head} --[{rel}]--> {tail}")
                    pass  # duplicate edge or invalid reference

        # Mark as extracted
        db.conn.execute("UPDATE papers SET status = 'extracted' WHERE local_id = ?", (pid,))
        db.commit()

        _t_done = _time.monotonic() - _t_paper
        _build_log.info(
            f"[build] paper={pid} done in {_t_done:.1f}s — inserted={valid_count} rejected={rejected}"
        )
        typer.echo(f"  Valid: {valid_count} | Rejected: {rejected} ({_t_done:.1f}s)")
        all_results.append({"paper_id": pid, "concepts": valid_count, "relations": len(relations)})

    if json_output:
        typer.echo(json.dumps({"results": all_results}, indent=2, ensure_ascii=False))
    elif all_results:
        total_c = sum(r["concepts"] for r in all_results)
        total_r = sum(r["relations"] for r in all_results)
        # Cross-paper concept deduplication
        from drbrain.extractor.concept import dedup_concepts_by_label

        merged = dedup_concepts_by_label(db)
        if merged:
            typer.echo(f"  Dedup: {merged} duplicate concepts merged")

        typer.echo(
            f"\nBuild complete: {total_c} concepts, {total_r} relations across {len(all_results)} papers"
        )

    db.close()


def embed_cmd(
    ctx: typer.Context,
    dim: int = typer.Option(128, "--dim", help="Embedding dimension"),
    epochs: int = typer.Option(100, "--epochs", help="Training epochs"),
    retrain: bool = typer.Option(False, "--retrain", help="Force retrain"),
    tree: bool = typer.Option(
        False, "--tree", help="Generate tree node text embeddings (PageIndex + RAPTOR)"
    ),
):
    """Train TransE graph embeddings. Use --tree for text embeddings."""
    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])

    # --tree mode: text embeddings for tree nodes (Layer 2)
    if tree:
        import asyncio

        from drbrain.config import EmbedConfig

        embed_cfg = cfg.get("embed", EmbedConfig())
        if isinstance(embed_cfg, dict):
            embed_cfg = EmbedConfig(**embed_cfg)

        if getattr(embed_cfg, "provider", "local") == "none":
            typer.echo("embed.provider=none; tree vector generation is disabled")
            db.close()
            return

        papers_dir = Path(cfg["dirs"]["papers"])
        llm_models_raw = cfg.get("llm", {})
        llm_models = (
            llm_models_raw.get("models", [])
            if hasattr(llm_models_raw, "get")
            else getattr(llm_models_raw, "models", [])
        )
        bridge_mod = __import__("drbrain.services.embedding", fromlist=["build_paper_tree_vectors"])
        total = 0
        for paper_path in sorted(papers_dir.iterdir()):
            if not paper_path.is_dir():
                continue
            count = asyncio.run(
                bridge_mod.build_paper_tree_vectors(paper_path, db.path, embed_cfg, llm_models)
            )
            if count:
                typer.echo(f"  {paper_path.name}: {count} vectors+summaries")
            total += count

        typer.echo(f"Tree vectors+summaries: {total} total")
        db.close()
        return
    graph = GraphEngine()
    graph.load_from_db(db)

    if graph.graph.number_of_nodes() == 0:
        typer.echo("No graph data. Run: drbrain build first", err=True)
        db.close()
        raise typer.Exit(1)

    # Load existing embeddings for incremental training
    existing = db.load_embeddings()
    init_ents = existing if existing and not retrain else None
    init_rels = None  # relations are re-learned each time (fewer, changes matter)

    db.clear_embeddings()
    from drbrain.graph.embedding import TransE

    t = TransE(dim=dim, epochs=epochs)
    typer.echo(
        f"Training embeddings (dim={dim}, epochs={epochs}, "
        f"nodes={graph.graph.number_of_nodes()}"
        f"{', incremental' if init_ents else ', from scratch'})..."
    )
    t.train(graph.graph, init_entities=init_ents, init_relations=init_rels)

    for label, vec in t.entities.items():
        db.save_embedding(label, vec, dim)
    for label, vec in t.relations.items():
        db.save_embedding(f"__rel__{label}", vec, dim)
    db.commit()
    typer.echo(f"Trained {len(t.entities)} entities, {len(t.relations)} relations")
    db.close()
