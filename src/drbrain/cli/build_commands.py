"""Build and embed pipeline commands."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import typer

from drbrain.cli._common import open_db
from drbrain.graph.engine import GraphEngine
from drbrain.storage.database import Database
from drbrain.storage.paths import paper_dir as resolve_paper_dir
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
    with open_db(cfg) as db:
        paper = db.get_paper(local_id)

    if not paper:
        typer.echo(f"Paper not found: {local_id}", err=True)
        raise typer.Exit(1)

    papers_dir = Path(cfg.get("dirs", {}).get("papers", "data/papers"))
    paper_dir = resolve_paper_dir(papers_dir, local_id)

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


def _build_extraction_summary(
    paper_id: str,
    concepts: list[dict],
    relations: list[dict],
    merges: list[dict],
    corrections: list[dict],
) -> str:
    """Format extraction results as a structured text summary for session context."""
    lines = [f"Extraction results for paper {paper_id}:"]

    # Concepts summary
    if concepts:
        by_type: dict[str, list[str]] = {}
        for c in concepts:
            t = c.get("type", "Unknown")
            by_type.setdefault(t, []).append(c.get("label", ""))
        lines.append(f"\nConcepts ({len(concepts)} total):")
        for t, labels in sorted(by_type.items()):
            top = labels[:10]
            suffix = f" ... +{len(labels) - 10} more" if len(labels) > 10 else ""
            lines.append(f"  {t}: {', '.join(top)}{suffix}")

    # Relations summary
    if relations:
        lines.append(f"\nRelations ({len(relations)} total):")
        for r in relations[:15]:
            lines.append(f"  {r.get('head', '?')} --[{r.get('rel', '?')}]--> {r.get('tail', '?')}")
        if len(relations) > 15:
            lines.append(f"  ... +{len(relations) - 15} more")

    # Merges summary
    if merges:
        lines.append(f"\nCoreference merges ({len(merges)} total):")
        for m in merges[:10]:
            lines.append(f"  {m.get('canonical', '?')} <- {m.get('variants', [])}")

    # Corrections summary
    if corrections:
        lines.append(f"\nRefinement corrections ({len(corrections)} total):")
        for c in corrections[:5]:
            lines.append(f"  {c.get('description', str(c)[:120])}")

    return "\n".join(lines)


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
    session_id: str = typer.Option(
        None,
        "--session",
        "-s",
        help="Save extraction context to session. 'new' to create, or existing session ID.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """Build knowledge graph from ingested papers using 5-stage LLM extraction."""
    import time as _time

    from loguru import logger as _build_log

    from drbrain.extractor.cache import ApiCache
    from drbrain.extractor.concept import build_graph_from_tree

    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])

    # LLM response cache (deduplicate retries across stages)
    from drbrain.security import configured_secret_values, safe_error

    secrets = configured_secret_values(cfg)
    cache = ApiCache("data/spool/llm_cache", secrets=secrets)

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
        # Incremental default: build papers that are either (a) not yet
        # extracted (status == 'uploaded') or (b) extracted but touched since
        # the last build run (e.g. rebuilt via 'drbrain build PID' after a
        # re-ingest). Falls back to pure status filter when no last_run is set
        # or when the db helper is unavailable (keeps test mocks working).
        all_paper_rows = db.get_all_papers()
        base = [p for p in all_paper_rows if p.get("status") == "uploaded"]
        last_build = db.get_last_run("build") if hasattr(db, "get_last_run") else None
        if last_build:
            for p in all_paper_rows:
                if p.get("status") == "extracted":
                    ts = db.get_paper_timestamp(p["local_id"])
                    if ts and ts > last_build:
                        base.append(p)
        papers = base

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
    failed = 0

    for paper in papers:
        pid = paper["local_id"]
        _t_paper = _time.monotonic()
        _build_log.info(f"[build] paper={pid} title={paper['title'][:60]}")
        typer.echo(f"\n{pid}: {paper['title'][:80]}")

        try:
            paper_path = resolve_paper_dir(papers_dir, pid)
        except (OSError, ValueError) as exc:
            message = safe_error(exc, secrets=secrets)
            db.upsert_paper_artifact(pid, "kg", "failed", error=message)
            db.commit()
            typer.echo(f"  Paper directory unavailable: {message}")
            failed += 1
            continue
        tree_path = tree_json_path(paper_path)
        md_path = raw_md_path(paper_path)
        db.upsert_paper_artifact(pid, "kg", "running")
        db.commit()

        # Retry tree generation when raw.md exists but tree.json is missing or
        # malformed.  A half-written tree must not block recovery on the next
        # build run.
        existing_tree = None
        tree_invalid = False
        if tree_path.exists():
            try:
                existing_tree = json.loads(tree_path.read_text(encoding="utf-8"))
                tree_invalid = not isinstance(existing_tree, dict) or not isinstance(
                    existing_tree.get("structure"), list
                )
            except (OSError, UnicodeError, ValueError):
                tree_invalid = True
        if (not tree_path.exists() or tree_invalid) and md_path.exists():
            typer.echo("  Tree missing, retrying...")
            try:
                from drbrain.parser.pageindex.sdk_backend import configure_tree_backend
                from drbrain.parser.pageindex_parser import TreeConfig, md_to_tree

                pageindex_cfg = TreeConfig(
                    if_add_node_summary=True,
                    if_add_doc_description=True,
                    if_add_node_text=False,
                    if_add_node_id=True,
                    max_node_tokens=10000,
                    min_token_threshold=5000,
                )
                configure_tree_backend(pageindex_cfg, cfg.get("pageindex"))
                doc_tree = asyncio.run(
                    md_to_tree(str(md_path), config=pageindex_cfg, models=llm_models)
                )
                tree_path.write_text(doc_tree.to_json(), encoding="utf-8")
                existing_tree = json.loads(tree_path.read_text(encoding="utf-8"))
                db.upsert_paper_artifact(
                    pid,
                    "tree",
                    "ready",
                    fingerprint=hashlib.sha256(tree_path.read_bytes()).hexdigest(),
                    metadata_json=json.dumps({"nodes": len(doc_tree.structure)}),
                )
                db.commit()
                typer.echo(f"  Tree regenerated: {len(doc_tree.structure)} sections")
            except Exception as e:
                db.upsert_paper_artifact(pid, "tree", "degraded", error=str(e))
                db.upsert_paper_artifact(pid, "kg", "skipped", error="tree unavailable")
                db.commit()
                typer.echo(f"  Tree regeneration failed: {safe_error(e, secrets=secrets)}")
                failed += 1
                continue
        elif not md_path.exists():
            db.upsert_paper_artifact(pid, "tree", "skipped", error="raw.md missing")
            db.upsert_paper_artifact(pid, "kg", "skipped", error="raw.md missing")
            db.commit()
            typer.echo("  No raw.md — ingest this paper first")
            failed += 1
            continue

        try:
            tree = existing_tree or json.loads(tree_path.read_text(encoding="utf-8"))
            structure = tree.get("structure", []) if isinstance(tree, dict) else []
            if not isinstance(structure, list):
                structure = []
        except (OSError, UnicodeError, ValueError) as exc:
            message = safe_error(exc, secrets=secrets)
            db.upsert_paper_artifact(pid, "tree", "failed", error=message)
            db.upsert_paper_artifact(pid, "kg", "skipped", error="tree.json unreadable")
            db.commit()
            typer.echo(f"  Tree read failed: {message}")
            failed += 1
            continue
        if not structure:
            db.upsert_paper_artifact(pid, "tree", "degraded", error="empty tree")
            db.upsert_paper_artifact(pid, "kg", "skipped", error="empty tree")
            db.commit()
            typer.echo("  Empty tree structure — skipping")
            failed += 1
            continue

        # Run 5-stage pipeline
        typer.echo("  Stage 1: Ontology...")
        try:
            result = asyncio.run(
                build_graph_from_tree(
                    md_path, structure, llm_models, skip_refine=skip_refine, cache=cache
                )
            )
        except Exception as exc:
            db.upsert_paper_artifact(pid, "kg", "failed", error=safe_error(exc, secrets=secrets))
            db.commit()
            failed += 1
            typer.echo(f"  Extraction failed: {safe_error(exc, secrets=secrets)}")
            continue

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

        # Mark as extracted (set_paper_status also bumps updated_at)
        db.set_paper_status(pid, "extracted")
        db.upsert_paper_artifact(
            pid,
            "tree",
            "ready",
            fingerprint=hashlib.sha256(tree_path.read_bytes()).hexdigest(),
            metadata_json=json.dumps({"nodes": len(structure)}),
        )
        db.upsert_paper_artifact(
            pid,
            "kg",
            "ready" if valid_count or relations else "degraded",
            metadata_json=json.dumps(
                {"concepts": valid_count, "relations": len(relations), "rejected": rejected}
            ),
            error="no valid concepts or relations" if not (valid_count or relations) else "",
        )
        db.set_last_run("build")
        db.commit()

        _t_done = _time.monotonic() - _t_paper
        _build_log.info(
            f"[build] paper={pid} done in {_t_done:.1f}s — inserted={valid_count} rejected={rejected}"
        )
        typer.echo(f"  Valid: {valid_count} | Rejected: {rejected} ({_t_done:.1f}s)")
        all_results.append({"paper_id": pid, "concepts": valid_count, "relations": len(relations)})

        # Inject extraction results into session if requested
        if session_id:
            from drbrain.extractor.session_agent import SessionAgent

            sess_agent = SessionAgent()
            if session_id == "new":
                sid = sess_agent.create_session(db, title=f"build:{pid}", models=llm_models)
                session_id = sid  # reuse for subsequent papers
            else:
                sess_agent.load_session(db, session_id, models=llm_models)

            summary = _build_extraction_summary(pid, concepts, relations, merges, corrections)
            sess_agent.inject_context(summary, label=f"build:{pid}")
            typer.echo(f"  Session: {sess_agent.session_id}")

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
    if failed:
        typer.echo(f"\n{failed} paper(s) failed — see errors above", err=True)

    db.close()
    if failed:
        raise typer.Exit(1)


def embed_cmd(
    ctx: typer.Context,
    dim: int = typer.Option(128, "--dim", help="Embedding dimension"),
    epochs: int = typer.Option(100, "--epochs", help="Training epochs"),
    retrain: bool = typer.Option(False, "--retrain", help="Force retrain"),
    tree: bool = typer.Option(
        False, "--tree", help="Generate tree node text embeddings (PageIndex + RAPTOR)"
    ),
    papers: str = typer.Option(
        "", "--papers", help="Comma-separated paper IDs to embed (default: all)"
    ),
    db_path: str = typer.Option(
        "", "--db", help="Override db path (shard databases; default cfg db.path)"
    ),
):
    """Train TransE graph embeddings. Use --tree for text embeddings."""
    cfg = ctx.obj["config"]
    db = Database(db_path or cfg["db"]["path"])

    # --tree mode: text embeddings for tree nodes (Layer 2)
    if tree:
        import asyncio

        from drbrain.config import EmbedConfig

        embed_cfg = cfg.get("embed", EmbedConfig())
        if isinstance(embed_cfg, dict):
            embed_cfg = EmbedConfig(**embed_cfg)

        papers_dir = Path(cfg["dirs"]["papers"])
        paper_filter = {p.strip() for p in papers.split(",") if p.strip()} if papers else None

        def paper_specs() -> list[tuple[str, Path]]:
            """Resolve paper IDs from the DB, retaining a legacy dir fallback."""
            try:
                rows = db.get_all_papers()
            except Exception:  # noqa: BLE001
                rows = []
            specs: list[tuple[str, Path]] = []
            for row in rows or []:
                pid = row.get("local_id") if isinstance(row, dict) else None
                if not pid:
                    continue
                pid = str(pid)
                if paper_filter is not None and pid not in paper_filter:
                    continue
                specs.append((pid, resolve_paper_dir(papers_dir, pid)))
            if specs:
                return specs
            # Old shard databases may not contain papers rows yet.  Keep the
            # directory scan as a read-only compatibility fallback.
            return [
                (path.name, path)
                for path in sorted(papers_dir.iterdir())
                if path.is_dir() and (paper_filter is None or path.name in paper_filter)
            ]

        if getattr(embed_cfg, "provider", "local") == "none":
            typer.echo("embed.provider=none; tree vector generation is disabled")
            for pid, _paper_path in paper_specs():
                db.upsert_paper_artifact(pid, "pageindex", "skipped", error="embedding disabled")
                db.upsert_paper_artifact(pid, "raptor", "skipped", error="embedding disabled")
            db.commit()
            db.close()
            return

        llm_models_raw = cfg.get("llm", {})
        llm_models = (
            llm_models_raw.get("models", [])
            if hasattr(llm_models_raw, "get")
            else getattr(llm_models_raw, "models", [])
        )
        bridge_mod = __import__("drbrain.services.embedding", fromlist=["build_paper_tree_vectors"])
        from drbrain.storage.node_projection import collect_tree_node_records

        total = 0
        failed = 0
        for pid, paper_path in paper_specs():
            db.upsert_paper_artifact(pid, "pageindex", "running")
            db.upsert_paper_artifact(pid, "raptor", "pending")
            db.commit()
            try:
                count = asyncio.run(
                    bridge_mod.build_paper_tree_vectors(paper_path, db.path, embed_cfg, llm_models)
                )
                node_count = len(collect_tree_node_records(paper_path, paper_id=pid))
                pageindex_count = int(
                    db.conn.execute(
                        "SELECT COUNT(*) FROM tree_vectors WHERE paper_id = ? AND tree_layer = ?",
                        (pid, "pageindex"),
                    ).fetchone()[0]
                )
                raptor_count = int(
                    db.conn.execute(
                        "SELECT COUNT(*) FROM tree_vectors WHERE paper_id = ? AND tree_layer LIKE ?",
                        (pid, "raptor_%"),
                    ).fetchone()[0]
                )
                page_status = "ready" if node_count and pageindex_count else "degraded"
                db.upsert_paper_artifact(
                    pid,
                    "pageindex",
                    page_status,
                    metadata_json=json.dumps({"nodes": node_count, "vectors": pageindex_count}),
                    error="no vectors created" if page_status == "degraded" else "",
                )
                if not llm_models:
                    db.upsert_paper_artifact(pid, "raptor", "skipped", error="no LLM models")
                elif not node_count:
                    db.upsert_paper_artifact(
                        pid, "raptor", "skipped", error="PageIndex unavailable"
                    )
                elif raptor_count:
                    db.upsert_paper_artifact(
                        pid,
                        "raptor",
                        "ready",
                        metadata_json=json.dumps({"summaries": raptor_count}),
                    )
                else:
                    db.upsert_paper_artifact(
                        pid, "raptor", "degraded", error="insufficient nodes or no summaries"
                    )
                db.commit()
            except Exception as exc:
                failed += 1
                db.upsert_paper_artifact(pid, "pageindex", "failed", error=str(exc))
                db.upsert_paper_artifact(pid, "raptor", "skipped", error="PageIndex failed")
                db.commit()
                typer.echo(f"  {pid}: embedding failed: {exc}", err=True)
                continue
            if count:
                typer.echo(f"  {paper_path.name}: {count} vectors+summaries")
            total += count

        typer.echo(f"Tree vectors+summaries: {total} total")
        db.close()
        if failed:
            raise typer.Exit(1)
        return
    graph = GraphEngine()
    graph.load_from_db(db)

    if graph.graph.number_of_nodes() == 0:
        typer.echo("No graph data. Run: drbrain build first", err=True)
        db.close()
        raise typer.Exit(1)

    # Load existing embeddings. Relations are stored with a __rel__ prefix.
    existing = db.load_embeddings()
    init_ents: dict | None = None
    init_rels: dict | None = None
    if existing and not retrain:
        init_ents = {k: v for k, v in existing.items() if not k.startswith("__rel__")}
        init_rels = {k[len("__rel__") :]: v for k, v in existing.items() if k.startswith("__rel__")}

    from drbrain.graph.embedding import TransE

    t = TransE(dim=dim, epochs=epochs)

    if init_ents and init_rels:
        # ── Incremental path: only train on edges from dirty papers ──
        last_embed = db.get_last_run("embed")
        if last_embed is not None:
            changed = db.get_papers_since(last_embed)
        else:
            changed = None  # no watermark -> fall back to full
        if changed:
            placeholders = ",".join("?" * len(changed))
            new_edge_rows = db.conn.execute(
                f"SELECT src_id, relation, dst_id FROM edges WHERE source_paper IN ({placeholders})",
                changed,
            ).fetchall()
            new_edges = [(r[0], r[2], r[1]) for r in new_edge_rows]
            typer.echo(
                f"Training embeddings incremental ({len(new_edges)} new/changed edges "
                f"from {len(changed)} papers, dim={dim})..."
            )
            t.train_incremental(
                graph.graph,
                new_edges,
                init_entities=init_ents,
                init_relations=init_rels,
            )
            # Seed t with all entities/relations so save loop persists everything
            for k, v in init_ents.items():
                if k not in t.entities:
                    t.entities[k] = v
            for k, v in init_rels.items():
                if k not in t.relations:
                    t.relations[k] = v
        else:
            typer.echo(
                f"Training embeddings full (dim={dim}, epochs={epochs}, "
                f"nodes={graph.graph.number_of_nodes()})..."
            )
            t.train(graph.graph, init_entities=init_ents, init_relations=init_rels)
    else:
        # ── Full path (first run, --retrain, or no existing vectors) ──
        typer.echo(
            f"Training embeddings (dim={dim}, epochs={epochs}, "
            f"nodes={graph.graph.number_of_nodes()}, from scratch)..."
        )
        t.train(graph.graph, init_entities=init_ents, init_relations=init_rels)

    # Persist: save_embedding uses INSERT OR REPLACE, so no need to clear first.
    for label, vec in t.entities.items():
        db.save_embedding(label, vec, dim)
    for label, vec in t.relations.items():
        db.save_embedding(f"__rel__{label}", vec, dim)
    db.set_last_run("embed")
    db.commit()
    typer.echo(f"Trained {len(t.entities)} entities, {len(t.relations)} relations")
    db.close()
