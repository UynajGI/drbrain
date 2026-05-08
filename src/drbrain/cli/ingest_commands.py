"""Ingest pipeline commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from drbrain.cli._common import (
    _apply_mined_rules,
    _fetch_citations_interested,
    _ingest_single_paper,
    _resolve_workspace_papers,
)
from drbrain.dedup.resolver import DedupEngine
from drbrain.graph.engine import GraphEngine
from drbrain.services.fetch import _resolve_identifier, fetch_paper
from drbrain.storage.database import Database

console = Console()


def ingest_cmd(
    ctx: typer.Context,
    paths: list[str] = typer.Argument(
        None, help="PDF file(s) or directory. Defaults to data/spool/inbox/."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output machine-readable JSON to stdout"
    ),
):
    """Ingest pipeline: parse -> identify -> tree -> paper record.

    Accepts single file, multiple files, or a directory of PDFs.
    Defaults to data/spool/inbox/ when no paths provided.
    """
    cfg = ctx.obj["config"]
    if not paths:
        # Expose API tokens to libraries that read them from environment
        import os as _os

        _dx_token = cfg.get("api", {}).get("deepxiv_token", "")
        if _dx_token and "DEEPXIV_TOKEN" not in _os.environ:
            _os.environ["DEEPXIV_TOKEN"] = _dx_token
        inbox_path = cfg.get("dirs", {}).get("inbox", "data/spool/inbox")
        paths = [inbox_path]

    pdf_files: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            pdf_files.extend(sorted(path.glob("*.pdf")))
        elif path.is_file():
            pdf_files.append(path)
        else:
            if not json_output:
                typer.echo(f"File not found: {p}", err=True)

    if not pdf_files:
        if json_output:
            typer.echo(json.dumps({"error": "No PDF files found"}))
        else:
            typer.echo("No PDF files found.", err=True)
        raise typer.Exit(1)

    db = Database(cfg["db"]["path"])
    dedup = DedupEngine(db)

    results = []
    for i, pdf_path in enumerate(pdf_files, 1):
        if not json_output and len(pdf_files) > 1:
            typer.echo(f"\n{'=' * 60}")
            typer.echo(f"[{i}/{len(pdf_files)}] {pdf_path}")
            typer.echo(f"{'=' * 60}")

        result = _ingest_single_paper(
            pdf_path,
            cfg,
            db,
            dedup,
            json_mode=json_output,
        )
        results.append(result)

    if json_output:
        output = {
            "ingested": len(results),
            "successful": sum(1 for r in results if r.get("ok")),
            "failed": sum(1 for r in results if not r.get("ok")),
            "papers": [r.get("report", {}) for r in results if r.get("ok")],
            "errors": [
                r.get("error", str(pdf_files[i])) for i, r in enumerate(results) if not r.get("ok")
            ],
        }
        typer.echo(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    else:
        if len(pdf_files) > 1:
            typer.echo(f"\n{'=' * 60}")
            typer.echo(f"Batch complete: {len(results)} papers ingested")
            success = sum(1 for r in results if r.get("ok"))
            typer.echo(f"  Successful: {success}, Failed: {len(results) - success}")

    db.close()


def fetch_cmd(
    ctx: typer.Context,
    identifier: str = typer.Argument(..., help="DOI, title, or arXiv ID to fetch"),
    arxiv: bool = typer.Option(False, "--arxiv", help="Treat identifier as arXiv ID"),
):
    """Fetch a paper: find PDF from open access sources -> download -> ingest."""
    # Normalize typer params when called directly (not through CLI)
    if isinstance(arxiv, typer.models.OptionInfo):
        arxiv = arxiv.default

    cfg = ctx.obj["config"]

    doi, title, arxiv_id = _resolve_identifier(identifier, is_arxiv=arxiv)

    fetch_cfg = cfg.get("fetch", {})

    typer.echo(f"Fetching: {identifier}")
    result = fetch_paper(doi=doi, title=title, arxiv_id=arxiv_id, fetch_config=fetch_cfg)

    if not result:
        typer.echo("Could not find a downloadable PDF from any source.", err=True)
        raise typer.Exit(1)

    typer.echo(f"  Downloaded: {result['pdf_path']}")

    # Ingest the downloaded paper
    pdf_path = Path(result["pdf_path"])
    db = Database(cfg["db"]["path"])
    dedup = DedupEngine(db)
    ingest_result = _ingest_single_paper(pdf_path, cfg, db, dedup, json_mode=False)
    db.close()

    if ingest_result.get("ok"):
        typer.echo(f"  Ingested: {ingest_result.get('local_id')}")
        typer.echo(f"  Next: drbrain build {ingest_result.get('local_id')}")
    else:
        typer.echo(f"  Ingest failed: {ingest_result.get('error', 'unknown error')}", err=True)
        raise typer.Exit(1)


def citations_cmd(
    ctx: typer.Context,
    local_id: str = typer.Argument(..., help="Paper local_id"),
    ctype: str = typer.Option(
        "all", "--type", "-t", help="Query type: refs, citing, shared-refs, all"
    ),
    limit: int = typer.Option(200, "--limit", "-l", help="Max results per type"),
    sort: str = typer.Option(
        "cited_by_count:desc",
        "--sort",
        "-s",
        help="Sort: cited_by_count:desc, publication_date:desc, relevance_score:desc",
    ),
    workspace: str = typer.Option(None, "--workspace", "-w", help="Limit to workspace"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
    fetch_interested: bool = typer.Option(
        False, "--fetch-interested", help="Interactively select and fetch placeholder papers"
    ),
):
    """Query citation graph for a paper: refs, citing, shared-refs."""
    # Normalize typer params when called directly (not through CLI)
    if isinstance(ctype, typer.models.OptionInfo):
        ctype = ctype.default
    if isinstance(workspace, typer.models.OptionInfo):
        workspace = workspace.default
    if isinstance(json_output, typer.models.OptionInfo):
        json_output = json_output.default
    if isinstance(fetch_interested, typer.models.OptionInfo):
        fetch_interested = fetch_interested.default

    if ctype not in ("refs", "citing", "shared-refs", "all"):
        typer.echo("Type must be: refs, citing, shared-refs, all", err=True)
        raise typer.Exit(1)

    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])

    paper = db.get_paper(local_id)
    if not paper:
        db.close()
        typer.echo(f"Paper not found: {local_id}", err=True)
        raise typer.Exit(1)

    from drbrain.storage.citation_graph import query_citation_graph

    # Auto-expand citations if none stored yet
    existing = db.conn.execute(
        "SELECT COUNT(*) FROM citation_cache WHERE source_paper = ?", (local_id,)
    ).fetchone()[0]
    if existing == 0:
        typer.echo("  Expanding citations (OpenAlex + S2 + CrossRef)...")
        from drbrain.extractor.citation import expand_citations_multi

        refs_added, citing_added = expand_citations_multi(db, local_id, limit=limit, sort=sort)
        typer.echo(f"  Found {refs_added} references, {citing_added} citing")

    result = query_citation_graph(local_id, db.conn, ctype=ctype)

    if workspace:
        paper_ids = _resolve_workspace_papers(workspace)
        if paper_ids and result.get("refs"):
            result["refs"] = [r for r in result["refs"] if r.get("local_id", "") in paper_ids]

    db.close()

    if json_output:
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return

    p = result["paper"]
    c = result.get("counts", {})
    typer.echo(f"\nCitation Graph: {p['title']} ({p['year']})")
    typer.echo(f"  References: {c.get('references', 0)} | Cited by: {c.get('citing', 0)}")

    if result.get("refs"):
        typer.echo("\nReferences:")
        for r in result["refs"]:
            year_str = f" ({r['year']})" if r.get("year") else ""
            typer.echo(f"  - {r['title']}{year_str}")

    if result.get("citing"):
        typer.echo("\nCited by:")
        for cit in result["citing"]:
            year_str = f" ({cit['year']})" if cit.get("year") else ""
            typer.echo(f"  - {cit['title']}{year_str}")

    if result.get("shared_refs"):
        typer.echo("\nShared References:")
        for sr in result["shared_refs"]:
            tag = " [unlinked]" if sr["status"] == "unlinked" else ""
            typer.echo(f"  - {sr['shared_with_title']} ({sr['shared_count']} shared){tag}")

    # --fetch-interested: interactive selection and batch fetch
    if fetch_interested:
        _fetch_citations_interested(ctx, result)


def check_citations_cmd(
    ctx: typer.Context,
    text: str = typer.Argument(None, help="Text to check citations in"),
    file: str = typer.Option(None, "--file", "-f", help="Read text from file"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Verify in-text citations against local library."""
    # Normalize typer params when called directly (not through CLI)
    if isinstance(text, typer.models.ArgumentInfo):
        text = text.default
    if isinstance(file, typer.models.OptionInfo):
        file = file.default
    if isinstance(json_output, typer.models.OptionInfo):
        json_output = json_output.default

    if file:
        text = Path(file).read_text(encoding="utf-8")

    if not text:
        typer.echo("Provide text or use --file", err=True)
        raise typer.Exit(1)

    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])

    from drbrain.extractor.citation_check import extract_citations, match_citations

    citations = extract_citations(text)
    citations = match_citations(citations, db)
    db.close()

    if json_output:
        result = [
            {
                "author": c.author,
                "year": c.year,
                "raw": c.raw,
                "found": c.found,
                "matched_id": c.matched_id,
                "matched_title": c.matched_title,
            }
            for c in citations
        ]
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if not citations:
        typer.echo("No citations found in text.")
        return

    typer.echo(f"Found {len(citations)} citations:")
    for c in citations:
        if c.found:
            typer.echo(f'  ✓ {c.author} ({c.year}) → {c.matched_id} "{c.matched_title}"')
        else:
            typer.echo(f"  ✗ {c.author} ({c.year}) → no match")


def report_cmd(
    ctx: typer.Context,
    local_id: str,
    json_output: bool = typer.Option(False, "--json", help="Output full report JSON to stdout"),
):
    """Display single-paper report."""
    cfg = ctx.obj["config"]
    report_dir = Path(cfg["dirs"]["reports"])
    report_path = report_dir / f"{local_id}.json"
    if not report_path.exists():
        msg = {"error": f"No report found for {local_id}"}
        if json_output:
            typer.echo(json.dumps(msg))
        else:
            typer.echo(
                f"No report found for {local_id}. Run: drbrain ingest or drbrain expand", err=True
            )
        raise typer.Exit(1)

    data = json.loads(report_path.read_text())

    if json_output:
        typer.echo(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        return

    s = data["summary"]
    typer.echo(f"\nReport: {data['paper']['title']}")
    typer.echo(f"  Status: {data['paper']['status']}")
    typer.echo(f"  Coverage: {s['graph_coverage']:.1%}")
    typer.echo(f"  References in graph: {s['refs_in_graph']}/{s['total_refs']}")
    typer.echo(f"  Citations in graph: {s['cits_in_graph']}/{s['total_cits']}")

    concepts = data["concepts"]
    for ctype in ["problems", "methods", "conclusions", "debates", "gaps", "actors"]:
        if concepts.get(ctype):
            typer.echo(f"  {ctype}: {len(concepts[ctype])}")

    if data["boundary_alert"].get("low_coverage"):
        typer.echo(
            "  [bold yellow]Alert: Low coverage - consider expanding citation network[/bold yellow]"
        )


def closure_cmd(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Output inferred edges but do not persist to database"
    ),
    rule: list[str] = typer.Option(
        None, "--rule", help="Run only the named rule(s). Repeatable. Omit for all."
    ),
    workspace: str = typer.Option(None, "--workspace", "-w", help="Limit to workspace"),
    mode: str = typer.Option("symbolic", "--mode", help="Inference mode: symbolic or hybrid"),
    mine_rules: bool = typer.Option(
        False, "--mine-rules", help="Mine path rules from TransE embeddings"
    ),
    min_confidence: float = typer.Option(
        0.6, "--min-confidence", help="Minimum confidence for mined rules (0.0-1.0)"
    ),
    ground: bool = typer.Option(
        False, "--ground", help="Ground transitive rules as concrete triples (t-norm)"
    ),
):
    """Run rule-based closure on the full graph."""
    # Normalize typer OptionInfo objects when calling directly (not via CLI)
    if isinstance(rule, typer.models.OptionInfo):
        rule = rule.default
    if isinstance(dry_run, typer.models.OptionInfo):
        dry_run = dry_run.default
    if isinstance(json_output, typer.models.OptionInfo):
        json_output = json_output.default
    if isinstance(mode, typer.models.OptionInfo):
        mode = mode.default
    if isinstance(mine_rules, typer.models.OptionInfo):
        mine_rules = mine_rules.default
    if isinstance(min_confidence, typer.models.OptionInfo):
        min_confidence = min_confidence.default
    if isinstance(ground, typer.models.OptionInfo):
        ground = ground.default

    valid_rules = {
        "creates_debate",
        "gap_addressed",
        "indirect_evolution",
        "gap_to_debate",
        "shared_actor",
        "transitive_closure",
        "asymmetric_violations",
        "method_supersedes_problem",
        "challenge_chain",
        "gap_inheritance",
        "indirect_support",
    }
    if rule is not None:
        invalid = set(rule) - valid_rules
        if invalid:
            typer.echo(f"Invalid rule(s): {', '.join(sorted(invalid))}", err=True)
            typer.echo(f"Valid rules: {', '.join(sorted(valid_rules))}", err=True)
            raise typer.Exit(1)

    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])
    graph = GraphEngine()
    paper_ids = _resolve_workspace_papers(workspace)
    graph.load_from_db(db, paper_ids=paper_ids)

    inferred = graph.closure(mode=mode)

    # ── Embedding-driven rule mining ───────────────────────────────────
    if mine_rules:
        from drbrain.extractor.rule_miner import mine_path_rules

        mined_rules = mine_path_rules(graph, db, min_confidence=min_confidence, top_k=20)
        mined_edges = _apply_mined_rules(graph, mined_rules)
        inferred.extend(mined_edges)
        if not json_output:
            typer.echo(
                f"Mined {len(mined_rules)} path rules from embeddings -> {len(mined_edges)} inferred edges"
            )

    # ── Rule grounding (t-norm transitive closure) ──────────────────────
    if ground:
        grounded = graph.ground_rules(min_confidence=min_confidence)
        if grounded:
            inferred.extend(grounded)
            if not json_output:
                typer.echo(f"Grounded {len(grounded)} transitive rule instances (t-norm)")

    if rule is not None:
        rule_set = set(rule)
        inferred = [e for e in inferred if e["relation"] in rule_set]

    if not dry_run:
        for edge in inferred:
            db.insert_edge(edge["src"], edge["dst"], edge["relation"], "closure")
        db.commit()
    db.close()

    if json_output:
        typer.echo(
            json.dumps(
                {"inferred": inferred, "count": len(inferred)},
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
        return

    typer.echo(f"Inferred edges: {len(inferred)}")
    for edge in inferred:
        typer.echo(
            f"  {edge['src']} --[{edge['relation']}]--> {edge['dst']} (via {edge.get('via', 'unknown')})"
        )
