"""Ingest pipeline commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console

from drbrain.cli._common import (
    _apply_mined_rules,
    _fetch_citations_interested,
    _ingest_single_paper,
    _resolve_workspace_papers,
    open_db,
)
from drbrain.dedup.resolver import DedupEngine
from drbrain.graph.engine import GraphEngine
from drbrain.services.fetch import _resolve_identifier, fetch_paper

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

    with open_db(cfg) as db:
        dedup = DedupEngine(db)

        logger.info("[ingest] batch start — %d PDF(s)", len(pdf_files))
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
                    r.get("error", str(pdf_files[i]))
                    for i, r in enumerate(results)
                    if not r.get("ok")
                ],
            }
            typer.echo(json.dumps(output, indent=2, ensure_ascii=False, default=str))
        else:
            if len(pdf_files) > 1:
                typer.echo(f"\n{'=' * 60}")
                typer.echo(f"Batch complete: {len(results)} papers ingested")
                success = sum(1 for r in results if r.get("ok"))
                typer.echo(f"  Successful: {success}, Failed: {len(results) - success}")

        success = sum(1 for r in results if r.get("ok"))
        logger.info("[ingest] batch done — %d/%d papers ingested", success, len(results))


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
    with open_db(cfg) as db:
        dedup = DedupEngine(db)
        ingest_result = _ingest_single_paper(pdf_path, cfg, db, dedup, json_mode=False)

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
    with open_db(cfg) as db:
        paper = db.get_paper(local_id)
        if not paper:
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
    with open_db(cfg) as db:
        from drbrain.extractor.citation_check import extract_citations, match_citations

        citations = extract_citations(text)
        citations = match_citations(citations, db)

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
    with open_db(cfg) as db:
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


def ingest_link_cmd(
    ctx: typer.Context,
    urls: list[str] = typer.Argument(..., help="Web URL(s) to ingest"),
    pdf: bool = typer.Option(None, "--pdf/--no-pdf", help="Force PDF extraction mode"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview only — extract, don't save"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """Ingest web URLs by extracting rendered content via external web extractor.

    Depends on an external qt-web-extractor service (default: http://127.0.0.1:8766).
    Set WEBEXTRACT_URL env var to configure.
    """
    from drbrain.providers.webtools import (
        _slugify_title,
        check_webextract_service,
        extract_web,
    )

    if dry_run:
        typer.echo(f"[dry-run] Will extract and ingest {len(urls)} link(s):")
        for u in urls:
            typer.echo(f"  - {u}")
        return

    # Check service availability
    if not check_webextract_service(timeout=3.0):
        typer.echo(
            "Web extraction service not reachable.\n"
            "  Install qt-web-extractor and ensure it's running on http://127.0.0.1:8766\n"
            "  Or set WEBEXTRACT_URL to point to your extractor instance.",
            err=True,
        )
        raise typer.Exit(1)

    cfg = ctx.obj["config"]
    papers_dir = Path(cfg.get("dirs", {}).get("papers", "data/papers"))
    with open_db(cfg) as db:
        results: list[dict] = []
        for i, url in enumerate(u for u in urls if u.strip()):
            if not json_output:
                typer.echo(f"[{i + 1}/{len(urls)}] Extracting {url} ...")

            extracted = extract_web(url.strip(), pdf=pdf)
            title = extracted.get("title", "")
            text = extracted.get("text", "")
            error = extracted.get("error", "")

            if error and not text:
                typer.echo(f"  Extraction failed: {error}", err=True)
                results.append({"url": url, "status": "error", "error": error})
                continue

            # Generate a local_id from title
            slug = _slugify_title(title, url)
            local_id = slug
            paper_dir = papers_dir / local_id

            # Handle duplicates
            if paper_dir.exists():
                base = slug
                for n in range(2, 100):
                    local_id = f"{base}-{n}"
                    paper_dir = papers_dir / local_id
                    if not paper_dir.exists():
                        break

            paper_dir.mkdir(parents=True, exist_ok=True)

            # Write markdown
            md_content = _render_extracted_markdown(title, url, text)
            (paper_dir / "raw.md").write_text(md_content, encoding="utf-8")

            # Register in DB
            db.insert_paper(
                local_id=local_id,
                title=title or url,
                year=None,
                status="uploaded",
            )

            results.append(
                {
                    "url": url,
                    "local_id": local_id,
                    "title": title,
                    "status": "ok",
                }
            )

            if not json_output:
                typer.echo(f"  -> {local_id}  ({len(text)} chars)")

        db.commit()

    if json_output:
        typer.echo(json.dumps(results, ensure_ascii=False, indent=2))
        return

    ok_count = sum(1 for r in results if r["status"] == "ok")
    err_count = sum(1 for r in results if r["status"] == "error")
    typer.echo(f"\nIngested {ok_count} link(s)" + (f", {err_count} error(s)" if err_count else ""))


def patent_search_cmd(
    ctx: typer.Context,
    query: list[str] = typer.Argument(..., help="Search query terms"),
    application: str = typer.Option(
        None, "--application", "-a", help="Lookup by application number"
    ),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
    source: str = typer.Option(
        "ppubs", "--source", "-s", help="Search source: ppubs (free) or odp (API key)"
    ),
    api_key: str = typer.Option(None, "--api-key", help="USPTO ODP API key (for --source odp)"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """Search USPTO patents."""
    import os as _os

    # Application number lookup (ODP only)
    if application:
        if source == "ppubs":
            typer.echo(
                "Use --source odp for application-number lookup; ODP API key required.", err=True
            )
            raise typer.Exit(1)

        key = api_key or _os.environ.get("USPTO_ODP_API_KEY", "")
        if not key:
            typer.echo(
                "USPTO ODP API key required. Set --api-key or USPTO_ODP_API_KEY env var.", err=True
            )
            typer.echo("Register: https://data.uspto.gov/apis/getting-started", err=True)
            raise typer.Exit(1)

        from drbrain.providers.uspto_odp import USPTOAPIError, get_patent_by_application_number

        try:
            result = get_patent_by_application_number(application, api_key=key)
        except USPTOAPIError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1)

        if not result:
            typer.echo(f"No patent found for application number: {application}")
            raise typer.Exit(1)
        if json_output:
            typer.echo(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            return
        _print_patent_odp(result)
        return

    # Search mode
    query_str = " ".join(query) if query else ""
    if not query_str:
        typer.echo("Provide search terms or use --application.", err=True)
        raise typer.Exit(1)

    if source == "odp":
        key = api_key or _os.environ.get("USPTO_ODP_API_KEY", "")
        if not key:
            typer.echo(
                "USPTO ODP API key required. Set --api-key or USPTO_ODP_API_KEY env var.", err=True
            )
            typer.echo("Register: https://data.uspto.gov/apis/getting-started", err=True)
            raise typer.Exit(1)

        from drbrain.providers.uspto_odp import USPTOAPIError
        from drbrain.providers.uspto_odp import search_patents as odp_search

        try:
            results = odp_search(query_str, api_key=key, limit=limit)
        except USPTOAPIError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1)

        if json_output:
            typer.echo(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
            return
        if not results:
            typer.echo(f"No patent results for '{query_str}'.")
            return
        typer.echo(f"\nFound {len(results)} USPTO patent record(s):")
        for i, p in enumerate(results, 1):
            _print_patent_odp(p, idx=i)
        return

    # Default: PPUBS (no auth)
    from drbrain.providers.uspto_ppubs import PpubsError
    from drbrain.providers.uspto_ppubs import search_patents as ppubs_search

    try:
        results = ppubs_search(query_str, limit=limit)
    except PpubsError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)

    if json_output:
        typer.echo(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
        return
    if not results:
        typer.echo(f"No patent results for '{query_str}'.")
        return
    typer.echo(f"\nFound {len(results)} USPTO patent record(s):")
    for i, ppub in enumerate(results, 1):
        _print_patent_ppubs(ppub, idx=i)


def _print_patent_odp(p, idx: int | None = None):
    prefix = f"[{idx}] " if idx else ""
    typer.echo(f"\n{prefix}{p.title}")
    typer.echo(f"    Application: {p.application_number}")
    if p.publication_number:
        typer.echo(f"    Publication: {p.publication_number}")
    if p.inventors:
        typer.echo(f"    Inventors: {', '.join(p.inventors[:3])}")
    if p.filing_date:
        typer.echo(f"    Filing: {p.filing_date}")
    if p.application_status:
        typer.echo(f"    Status: {p.application_status}")


def _print_patent_ppubs(ppub, idx: int | None = None):
    prefix = f"[{idx}] " if idx else ""
    typer.echo(f"\n{prefix}{ppub.title}")
    if ppub.publication_number:
        typer.echo(f"    Publication: {ppub.publication_number}")
    if ppub.inventors:
        typer.echo(f"    Inventors: {', '.join(ppub.inventors[:3])}")
    if ppub.assignees:
        typer.echo(f"    Assignees: {', '.join(ppub.assignees[:2])}")
    if ppub.filing_date:
        typer.echo(f"    Filing: {ppub.filing_date}")
    if ppub.publication_date:
        typer.echo(f"    Published: {ppub.publication_date}")


def pipeline_cmd(
    ctx: typer.Context,
    preset: str = typer.Option(None, "--preset", "-p", help="Preset: full, quick, embed"),
    steps: str = typer.Option(None, "--steps", "-s", help="Comma-separated step names"),
    list_steps_flag: bool = typer.Option(False, "--list", help="List available steps and presets"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview steps without executing"),
):
    """Chain multiple processing steps in sequence (ingest → build → embed → closure)."""
    from drbrain.services.pipeline import list_steps_info, resolve_steps

    if list_steps_flag:
        steps_info, presets_info = list_steps_info()
        typer.echo("Available steps:")
        for s in steps_info:
            typer.echo(f"  {s['name']:<10} [{s['scope']:<7}]  {s['description']}")
        typer.echo("\nAvailable presets:")
        for p in presets_info:
            typer.echo(f"  {p['name']:<10} = {', '.join(p['steps'])}")
        return

    try:
        step_names = resolve_steps(preset=preset, steps_str=steps)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)

    if dry_run:
        typer.echo(f"[dry-run] Would execute {len(step_names)} step(s): {', '.join(step_names)}")
        return

    typer.echo(f"Pipeline: {' -> '.join(step_names)}")
    typer.echo()

    import subprocess as _sp
    import sys as _sys

    for i, name in enumerate(step_names, 1):
        typer.echo(f"[{i}/{len(step_names)}] {name} ...")
        if name == "ingest":
            _sp.run([_sys.executable, "-m", "drbrain.cli.main", "ingest"], check=False)
        elif name == "build":
            _sp.run([_sys.executable, "-m", "drbrain.cli.main", "build", "--all"], check=False)
        elif name == "embed":
            _sp.run(
                [_sys.executable, "-m", "drbrain.cli.main", "embed", "--tree"],
                check=False,
            )
        elif name == "closure":
            _sp.run(
                [_sys.executable, "-m", "drbrain.cli.main", "closure"],
                check=False,
            )

    typer.echo(f"\nPipeline complete: {', '.join(step_names)}")


def proceedings_cmd(
    ctx: typer.Context,
    list_flag: bool = typer.Option(False, "--list", "-l", help="List all proceedings"),
    create: str = typer.Option(None, "--create", help="Create proceeding: 'Name Year [Venue]'"),
    show: str = typer.Option(None, "--show", help="Show proceeding by ID"),
    add: tuple[str, str] = typer.Option(
        (None, None), "--add", help="Add paper to proceeding: PROCEEDING_ID PAPER_ID"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """Manage conference proceedings."""
    from drbrain.storage.proceedings import (
        DEFAULT_PATH,
        add_paper,
        create_proceeding,
        get_proceeding,
        list_proceedings,
    )

    store_path = DEFAULT_PATH

    if create:
        parts = create.rsplit(maxsplit=1)
        if len(parts) == 2 and parts[1].isdigit():
            name, year_str = parts[0], parts[1]
            year = int(year_str)
            venue = ""
        else:
            all_parts = create.split()
            name = " ".join(all_parts[:-1]) if len(all_parts) > 1 else all_parts[0]
            year = int(all_parts[-1]) if all_parts[-1].isdigit() else 2024
            venue = ""
        p = create_proceeding(store_path, name, year, venue=venue)
        typer.echo(f"Created: [{p['id']}] {name} ({year})")
        return

    if add and add[0] and add[1]:
        try:
            add_paper(store_path, add[0], add[1])
        except ValueError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1)
        typer.echo(f"Added {add[1]} to proceeding {add[0]}")
        return

    if show:
        p = get_proceeding(store_path, show)
        if not p:
            typer.echo(f"Proceeding not found: {show}", err=True)
            raise typer.Exit(1)
        if json_output:
            typer.echo(json.dumps(p, ensure_ascii=False, indent=2))
            return
        typer.echo(f"[{p['id']}] {p['name']} ({p['year']})")
        if p.get("venue"):
            typer.echo(f"  Venue: {p['venue']}")
        typer.echo(f"  Papers: {len(p.get('papers', []))}")
        for paper_id in p.get("papers", []):
            typer.echo(f"    - {paper_id}")
        return

    # Default: list
    proceedings = list_proceedings(store_path)
    if json_output:
        typer.echo(json.dumps(proceedings, ensure_ascii=False, indent=2))
        return
    if not proceedings:
        typer.echo("No proceedings. Create one with: drbrain proceedings --create 'NeurIPS 2024'")
        return
    typer.echo(f"Proceedings ({len(proceedings)}):")
    for p in proceedings:
        pc = len(p.get("papers", []))
        venue_str = f" — {p.get('venue', '')}" if p.get("venue") else ""
        typer.echo(f"  [{p['id']}] {p['name']} ({p['year']}){venue_str} — {pc} paper(s)")


def explore_cmd(
    ctx: typer.Context,
    list_flag: bool = typer.Option(False, "--list", "-l", help="List all explore silos"),
    create: str = typer.Option(None, "--create", help="Create a new explore silo"),
    delete: str = typer.Option(None, "--delete", help="Delete an explore silo"),
    name: str = typer.Option(None, "--name", "-n", help="Silo name for --search or --show"),
    search: str = typer.Option(None, "--search", "-s", help="Search papers in a silo"),
    show: bool = typer.Option(False, "--show", help="Show silo papers"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """Manage explore silos — lightweight literature discovery collections."""
    from pathlib import Path as _Path

    from drbrain.storage.explore import (
        create_explore_silo,
        delete_explore_silo,
        get_silo_papers,
        list_explore_silos,
        search_silo,
    )

    root = _Path("data/explore")

    if create:
        try:
            silo = create_explore_silo(root, create)
        except ValueError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1)
        typer.echo(f"Created explore silo: {silo['name']}")
        return

    if delete:
        try:
            delete_explore_silo(root, delete)
        except Exception as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1)
        typer.echo(f"Deleted: {delete}")
        return

    if search and name:
        try:
            results = search_silo(root, name, search)
        except ValueError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1)
        if json_output:
            typer.echo(json.dumps(results, ensure_ascii=False, indent=2))
            return
        if not results:
            typer.echo(f"No results for '{search}' in silo '{name}'.")
        typer.echo(f"Results ({len(results)}):")
        for i, r in enumerate(results, 1):
            authors = ", ".join(r.get("authors", [])[:2])
            year = f" ({r.get('year', '?')})"
            typer.echo(f"  [{i}] {r.get('title', '?')}{year} — {authors}")
        return

    if show and name:
        try:
            papers = get_silo_papers(root, name)
        except ValueError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1)
        if json_output:
            typer.echo(json.dumps(papers, ensure_ascii=False, indent=2))
            return
        typer.echo(f"Silo '{name}' — {len(papers)} paper(s):")
        for i, r in enumerate(papers, 1):
            authors = ", ".join(r.get("authors", [])[:2])
            year = f" ({r.get('year', '?')})"
            typer.echo(f"  [{i}] {r.get('title', '?')}{year} — {authors}")
        return

    # Default: list
    silos = list_explore_silos(root)
    if json_output:
        typer.echo(json.dumps(silos, ensure_ascii=False, indent=2))
        return
    if not silos:
        typer.echo("No explore silos. Create one with: drbrain explore --create <name>")
        return
    typer.echo(f"Explore silos ({len(silos)}):")
    for s in silos:
        desc = f" — {s.get('description', '')}" if s.get("description") else ""
        typer.echo(f"  {s['name']}: {s.get('paper_count', 0)} papers{desc}")


def _render_extracted_markdown(title: str, source_url: str, body: str) -> str:
    parts = [
        f"# {title}",
        "",
        f"Source URL: {source_url}",
        "",
    ]
    body_text = (body or "").strip()
    if body_text:
        parts.append(body_text)
    return "\n".join(parts).rstrip() + "\n"
