"""System check and maintenance commands."""

from __future__ import annotations

import asyncio
import importlib
import json
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from drbrain.cli._common import (
    _print_analyze_report,
    _resolve_workspace_papers,
)
from drbrain.config import load_config
from drbrain.graph.engine import GraphEngine
from drbrain.storage.database import Database


def check_cmd(ctx: typer.Context):
    """Check dependencies, configuration, and environment variables."""
    console = Console()
    warnings = []
    errors = []

    cfg = ctx.obj["config"]

    console.print("\n[bold]DrBrain — Dependency & Configuration Check[/bold]\n")

    # -- Python packages --
    console.print("[bold]Python Packages[/bold]")
    table = Table(show_header=False, box=None, padding=(0, 2))
    required_packages = [
        ("pymupdf", "fitz"),
        ("litellm", "litellm"),
        ("typer", "typer"),
        ("rich", "rich"),
        ("pyyaml", "yaml"),
        ("pydantic", "pydantic"),
    ]
    for pkg_name, import_name in required_packages:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "installed")
            table.add_row(f"  {pkg_name}", f"[green]OK[/green] ({version})")
        except ImportError:
            table.add_row(f"  {pkg_name}", "[red]MISSING[/red]")
            errors.append(f"Missing Python package: {pkg_name}")
    console.print(table)

    # -- External CLI tools --
    console.print("\n[bold]External Tools[/bold]")
    table2 = Table(show_header=False, box=None, padding=(0, 2))
    mineru_found = shutil.which("mineru-open-api")
    pymupdf_found = False
    try:
        importlib.import_module("fitz")
        pymupdf_found = True
    except ImportError:
        pass

    cli_tools = {
        "mineru-open-api": ("MinerU PDF parser CLI", mineru_found),
        "PyMuPDF (fitz)": ("PDF fallback parser", pymupdf_found),
    }
    for tool, (desc, found) in cli_tools.items():
        if found:
            label = f"  {tool}"
            if isinstance(found, str):
                table2.add_row(label, f"[green]OK[/green] ({found}) — {desc}")
            else:
                table2.add_row(label, f"[green]OK[/green] — {desc}")
        else:
            table2.add_row(f"  {tool}", "[yellow]NOT FOUND[/yellow]", f"{desc} (optional)")
            if tool == "mineru-open-api":
                warnings.append("mineru-open-api not found — using PyMuPDF fallback")

    console.print(table2)
    # Parser path
    console.print("\n[bold]Parser Path[/bold]")
    if mineru_found:
        console.print("  MinerU CLI → [green]PyMuPDF[/green]")
    elif pymupdf_found:
        console.print("  [yellow]MinerU not found[/yellow], using: [green]PyMuPDF[/green]")
    else:
        console.print("  [red]No PDF parser available[/red] — install pymupdf")

    # -- Config files --
    console.print("\n[bold]Configuration[/bold]")
    table3 = Table(show_header=False, box=None, padding=(0, 2))

    base_config = Path("config.yaml")
    local_config = Path("config.local.yaml")

    if base_config.exists():
        table3.add_row("  config.yaml", "[green]Found[/green]")
    else:
        table3.add_row("  config.yaml", "[red]Missing[/red]")
        errors.append("config.yaml not found")

    if local_config.exists():
        table3.add_row("  config.local.yaml", "[green]Found (overrides base)[/green]")
    else:
        table3.add_row("  config.local.yaml", "[yellow]Not found[/yellow]")
        warnings.append(
            "No config.local.yaml — using base config values (env var placeholders unresolved)"
        )

    # Check key values in config
    try:
        console.print("\n[bold]API Keys & Tokens[/bold]")
        table4 = Table(show_header=False, box=None, padding=(0, 2))

        # LLM models
        models = cfg.get("llm", {}).get("models", [])
        for i, m in enumerate(models):
            label = f"  LLM [{i}] {m.get('provider', '?')}/{m.get('model', '?')}"
            api_key = m.get("api_key", "")
            if api_key and not api_key.startswith("${"):
                masked = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
                table4.add_row(label, f"[green]Set[/green] ({masked})")
            else:
                table4.add_row(label, "[yellow]Not configured[/yellow]")
                warnings.append(f"LLM model {m.get('provider')}/{m.get('model')} has no API key")

        # MinerU token
        mineru_token = cfg.get("mineru", {}).get("token", "")
        if mineru_token and not mineru_token.startswith("${"):
            table4.add_row("  MinerU token", "[green]Set[/green]")
        else:
            table4.add_row(
                "  MinerU token", "[yellow]Not set[/yellow]", "(flash mode will be used)"
            )
            warnings.append("MinerU token not configured — flash extraction mode only")

        # CrossRef email
        crossref_email = cfg.get("api", {}).get("crossref_email", "")
        if crossref_email and not crossref_email.startswith("${"):
            table4.add_row("  CrossRef email", f"[green]Set[/green] ({crossref_email})")
        else:
            table4.add_row(
                "  CrossRef email", "[yellow]Not set[/yellow]", "(optional, polite pool)"
            )
            warnings.append("CrossRef email not set — will use anonymous polite pool access")

        # OpenAlex token
        oa_token = cfg.get("api", {}).get("openalex_token", "")
        if oa_token and not oa_token.startswith("${"):
            table4.add_row("  OpenAlex token", "[green]Set[/green]")
        else:
            table4.add_row(
                "  OpenAlex token",
                "[yellow]Not set[/yellow]",
                "(anonymous access, lower rate limit)",
            )
            warnings.append("OpenAlex token not set — using anonymous access with lower rate limit")

        console.print(table4)

    except FileNotFoundError as e:
        console.print(f"  [red]{e}[/red]")

    # -- Directories --
    console.print("\n[bold]Directories[/bold]")
    table5 = Table(show_header=False, box=None, padding=(0, 2))
    try:
        dirs_config = cfg.get("dirs", {})
        dir_paths = (
            list(dirs_config.values())
            if dirs_config
            else [
                "data/spool/inbox",
                "data/spool/pending",
                "data/papers",
                "data/reports",
                "data/cache",
                "data/logs",
            ]
        )
        for dir_path in dir_paths:
            p = Path(dir_path)
            if p.exists():
                table5.add_row(f"  {dir_path}", "[green]Exists[/green]")
            else:
                p.mkdir(parents=True, exist_ok=True)
                table5.add_row(f"  {dir_path}", "[green]Created[/green]")
    except Exception:
        for d in ["data/spool/inbox", "data/spool/pending", "data/papers"]:
            p = Path(d)
            if p.exists():
                table5.add_row(f"  {d}", "[green]Exists[/green]")
            else:
                table5.add_row(f"  {d}", "[yellow]Missing[/yellow]")
    console.print(table5)

    # -- Database --
    console.print("\n[bold]Database[/bold]")
    table6 = Table(show_header=False, box=None, padding=(0, 2))
    try:
        db_path = cfg.get("db", {}).get("path", "data/drbrain.db")
        p = Path(db_path)
        if p.exists():
            table6.add_row(f"  {db_path}", "[green]Exists[/green]")
        else:
            table6.add_row(
                f"  {db_path}",
                "[yellow]Not yet created[/yellow]",
                "(run `drbrain ingest` to initialize)",
            )
    except Exception:
        table6.add_row("  (config unavailable)", "[yellow]Unknown[/yellow]")
    console.print(table6)

    # -- Paper count --
    console.print("\n[bold]Library[/bold]")
    table_lib = Table(show_header=False, box=None, padding=(0, 2))
    try:
        db = Database(cfg["db"]["path"])
        paper_count = db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        concept_count = db.conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
        table_lib.add_row("  Papers", f"[green]{paper_count}[/green]")
        table_lib.add_row("  Concepts", f"[green]{concept_count}[/green]")
        db.close()
    except Exception:
        table_lib.add_row("  (db unavailable)", "[yellow]Unknown[/yellow]")
    console.print(table_lib)

    # -- Disk space --
    console.print("\n[bold]Disk Space[/bold]")
    table_disk = Table(show_header=False, box=None, padding=(0, 2))
    data_path = Path("data")
    if data_path.exists():
        usage = shutil.disk_usage(data_path)
        free_gb = usage.free / (1024**3)
        total_gb = usage.total / (1024**3)
        if free_gb < 1:
            table_disk.add_row(
                "  data/ free space",
                f"[red]{free_gb:.1f} GB[/red]",
                f"(total {total_gb:.0f} GB) — critically low",
            )
            warnings.append(f"Low disk space: {free_gb:.1f} GB free on data/ partition")
        elif free_gb < 10:
            table_disk.add_row(
                "  data/ free space",
                f"[yellow]{free_gb:.1f} GB[/yellow]",
                f"(total {total_gb:.0f} GB)",
            )
        else:
            table_disk.add_row(
                "  data/ free space",
                f"[green]{free_gb:.1f} GB[/green]",
                f"(total {total_gb:.0f} GB)",
            )
    console.print(table_disk)

    # -- MinerU API connectivity --
    console.print("\n[bold]API Connectivity[/bold]")
    table_api = Table(show_header=False, box=None, padding=(0, 2))
    try:
        mineru_token = cfg.get("mineru", {}).get("token", "")
        if mineru_token and not mineru_token.startswith("${"):
            import urllib.request as _urllib

            try:
                req = _urllib.request.Request(
                    "https://api.mineru.com/api/v1/status",
                    headers={"Authorization": f"Bearer {mineru_token}"},
                )
                _urllib.request.urlopen(req, timeout=5)
                table_api.add_row("  MinerU API", "[green]Reachable[/green]")
            except Exception:
                table_api.add_row(
                    "  MinerU API", "[yellow]Unreachable[/yellow]", "(check token/network)"
                )
                warnings.append("MinerU API unreachable (token-tier may not work)")
        else:
            table_api.add_row(
                "  MinerU API", "[yellow]Not configured[/yellow]", "(flash mode will be used)"
            )
    except Exception:
        table_api.add_row("  MinerU API", "[yellow]Unknown[/yellow]")

    # -- MinerU CLI --
    mineru_cli_available = False
    try:
        import shutil as _shutil

        cli = _shutil.which("mineru-open-api")
        if cli:
            mineru_cli_available = True
            table_api.add_row("  MinerU CLI", f"[green]Found[/green] ({cli})")
        else:
            table_api.add_row(
                "  MinerU CLI", "[yellow]Not found[/yellow]", "(install: npm i -g mineru-open-api)"
            )
    except Exception:
        table_api.add_row("  MinerU CLI", "[yellow]Unknown[/yellow]")

    # Only warn about PyMuPDF fallback if no MinerU path works
    if not mineru_cli_available:
        warnings.append("MinerU unavailable — PDF parsing will use PyMuPDF fallback")

    # -- DeepXiv connectivity --
    try:
        dx_token = cfg.get("api", {}).get("deepxiv_token", "")
        if dx_token and not dx_token.startswith("${"):
            try:
                from deepxiv_sdk import Reader as _dxReader

                r = _dxReader(token=dx_token)
                r.brief("1706.03762")
                table_api.add_row("  DeepXiv", "[green]Reachable[/green]")
            except Exception:
                table_api.add_row(
                    "  DeepXiv", "[yellow]Unreachable[/yellow]", "(check token at data.rag.ac.cn)"
                )
        else:
            table_api.add_row(
                "  DeepXiv",
                "[yellow]Not configured[/yellow]",
                "(register at https://data.rag.ac.cn/register)",
            )
    except Exception:
        table_api.add_row("  DeepXiv", "[yellow]Unknown[/yellow]")

    # -- LLM API connectivity --
    try:
        llm_models = cfg.get("llm", {}).get("models", [])
        for i, m in enumerate(llm_models):
            label = f"  LLM [{i}] {m.get('provider', '?')}/{m.get('model', '?')}"
            api_key = m.get("api_key", "")
            if api_key and api_key.startswith("${"):
                table_api.add_row(label, "[yellow]Env var not set[/yellow]")
                continue
            try:
                import litellm as _llm

                name = f"{m['provider']}/{m['model']}"
                kwargs = {
                    "model": name,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 5,
                    "timeout": 10,
                }
                if m.get("api_key"):
                    kwargs["api_key"] = m["api_key"]
                if m.get("base_url"):
                    kwargs["api_base"] = m["base_url"]
                _llm.completion(**kwargs)
                table_api.add_row(label, "[green]Reachable[/green]")
            except Exception as e:
                err_msg = str(e)[:60]
                table_api.add_row(label, "[yellow]Unreachable[/yellow]", f"({err_msg})")
                warnings.append(f"LLM [{i}] {m.get('model', '?')} unreachable")
    except Exception:
        table_api.add_row("  LLM", "[yellow]Not configured[/yellow]", "(run `drbrain setup`)")

    console.print(table_api)

    # -- Summary --
    console.print("\n[bold]Summary[/bold]")
    if errors:
        console.print(f"\n[bold red]Errors ({len(errors)}):[/bold red]")
        for e in errors:
            console.print(f"  [red]✗[/red] {e}")
    if warnings:
        console.print(f"\n[bold yellow]Warnings ({len(warnings)}):[/bold yellow]")
        for w in warnings:
            console.print(f"  [yellow]![/yellow] {w}")
    if not errors and not warnings:
        console.print("\n[bold green]All checks passed![/bold green]")
    elif not errors:
        console.print(
            f"\n[bold green]Ready to use[/bold green] ({len(warnings)} optional warnings)"
        )
    else:
        console.print(f"\n[bold red]Not ready[/bold red] — fix {len(errors)} error(s) above")

    if errors:
        raise typer.Exit(1)


def analyze_cmd(
    ctx: typer.Context,
    local_id: str = typer.Argument(None, help="Paper local_id (single paper mode)"),
    papers: str = typer.Option(None, "--papers", help="Comma-separated paper IDs"),
    query: str = typer.Option(None, "--query", help="BM25 search query to select papers"),
    discover: str = typer.Option(None, "--discover", help="LLM graph discovery question"),
    workspace: str = typer.Option(None, "--workspace", "-w", help="Workspace boundary scan"),
    full: bool = typer.Option(False, "--full", "-f", help="Full analysis (slower)"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Analyze knowledge frontier: seeds, causal chains, hypotheses, and more.

    Paper selection (mutually exclusive, first match wins):
    - <local_id>: single paper
    - --papers p1,p2,...: specific papers
    - --query "text": BM25 search then analyze matches
    - --discover "question": LLM graph exploration to find relevant papers
    - --workspace myws: all papers in workspace
    - (none): error — specify one of the above
    """
    # Normalize typer OptionInfo objects when calling directly (not via CLI)
    if isinstance(papers, typer.models.OptionInfo):
        papers = papers.default
    if isinstance(query, typer.models.OptionInfo):
        query = query.default
    if isinstance(discover, typer.models.OptionInfo):
        discover = discover.default
    if isinstance(workspace, typer.models.OptionInfo):
        workspace = workspace.default
    if isinstance(full, typer.models.OptionInfo):
        full = full.default
    if isinstance(json_output, typer.models.OptionInfo):
        json_output = json_output.default

    from drbrain.report.analyzer import analyze_paper

    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])
    graph = GraphEngine()
    llm_models = cfg.get("llm", {}).get("models", [])

    # ── Paper selection ──
    selected: list[dict] = []

    if local_id:
        p = db.get_paper(local_id)
        if p:
            selected = [p]
        else:
            typer.echo(f"Paper not found: {local_id}", err=True)
            db.close()
            raise typer.Exit(1)
    elif papers:
        for pid in papers.split(","):
            pid = pid.strip()
            p = db.get_paper(pid)
            if p:
                selected.append(p)
            else:
                typer.echo(f"Paper not found: {pid}", err=True)
    elif query:
        from drbrain.query.bm25 import build_bm25_index

        bm25 = build_bm25_index(db)
        results = bm25.search(query, limit=20)
        seen = set()
        for r in results:
            pid = r["local_id"]
            if pid not in seen:
                seen.add(pid)
                p = db.get_paper(pid)
                if p:
                    selected.append(p)
        typer.echo(f"Query '{query}': {len(selected)} papers matched")
    elif discover:
        from drbrain.extractor.reasoner import ReasonerAgent

        agent = ReasonerAgent(db=db, graph_engine=graph, models=llm_models)
        typer.echo(f"Discovering papers for: {discover}")
        answer = asyncio.run(
            agent.reason(
                f"Find papers in the knowledge graph relevant to: {discover}. "
                "Search concepts and explore neighbors. Return ONLY a comma-separated "
                "list of the most relevant paper IDs (like pe211dc,p6a321e). Max 10."
            )
        )
        import re as _re

        ids = _re.findall(r"p[a-f0-9]{6}", answer)
        for pid in ids[:10]:
            p = db.get_paper(pid)
            if p:
                selected.append(p)
        typer.echo(f"Discovered: {len(selected)} papers")
    elif workspace:
        paper_ids = _resolve_workspace_papers(workspace)
        all_papers = db.get_all_papers()
        selected = [p for p in all_papers if paper_ids and p["local_id"] in paper_ids]
    else:
        typer.echo(
            "Specify a paper ID, --papers, --query, --discover, or --workspace.",
            err=True,
        )
        db.close()
        raise typer.Exit(1)

    if not selected:
        typer.echo("No papers to analyze.")
        db.close()
        raise typer.Exit(1)

    # Load full graph for seed detection and cross-paper insights
    graph.load_from_db(db)

    # ── Run analysis ──
    reports = [
        analyze_paper(db, graph, p["local_id"], full=full, models=llm_models) for p in selected
    ]

    # Add cross-paper insights for multi-paper analysis
    if len(reports) > 1:
        from drbrain.report.analyzer import add_cross_paper_insights

        reports = add_cross_paper_insights(reports, db=db)

    db.close()

    if json_output:
        typer.echo(json.dumps(reports, indent=2, ensure_ascii=False, default=str))
    elif len(reports) == 1:
        _print_analyze_report(reports[0])
    else:
        typer.echo(f"Analysis: {len(reports)} papers\n")
        for r in reports:
            _print_analyze_report(r)


def clean_cmd(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
    config_path: str = typer.Option("config.yaml", "--config", "-c", help="Config file path"),
) -> None:
    """Clear data directories (db, cache, logs, papers, reports). Keeps inbox (PDFs) intact."""
    cfg = load_config(config_path)
    dirs = cfg.get("dirs", {})

    targets = [
        cfg.get("db", {}).get("path", "data/drbrain.db"),
        "data/metrics.db",
        dirs.get("cache", "data/cache"),
        dirs.get("logs", "data/logs"),
        dirs.get("papers", "data/papers"),
        dirs.get("reports", "data/reports"),
    ]

    existing = [t for t in targets if Path(t).exists()]
    if not existing:
        typer.echo("Nothing to clean — data directories are already empty.")
        return

    if not force:
        typer.echo("Will clear these directories:")
        for d in existing:
            count = sum(1 for _ in Path(d).rglob("*") if _.is_file())
            typer.echo(f"  {d}/ ({count} files)")
        confirm = typer.confirm("Proceed?", default=False)
        if not confirm:
            typer.echo("Cancelled.")
            return

    if force:
        from drbrain.auth import has_password, verify_password

        if has_password(cfg):
            pw = typer.prompt("Admin password", hide_input=True)
            if not verify_password(pw, cfg["admin"]["password_hash"]):
                typer.echo("Wrong password.", err=True)
                raise typer.Exit(1)

    for d in existing:
        p = Path(d)
        if p.is_file():
            p.unlink()
            typer.echo(f"  Removed {d}")
        elif p.is_dir():
            for item in p.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            typer.echo(f"  Cleared {d}/")

    # Recreate directories (skip file paths)
    for t in targets:
        p = Path(t)
        if not p.suffix:  # directory path (no extension)
            p.mkdir(parents=True, exist_ok=True)

    typer.echo("Done. Inbox (PDFs) untouched.")


# -- Workspace commands --
