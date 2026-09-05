"""Ingest pipeline commands."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import click
import typer
from loguru import logger
from rich.console import Console

from drbrain.cli._common import (
    _apply_mined_rules,
    _fetch_citations_interested,
    _ingest_single_paper,
    _resolve_workspace_papers,
    open_db,
    runtime_data_path,
)
from drbrain.dedup.resolver import DedupEngine
from drbrain.graph.engine import GraphEngine
from drbrain.security import configured_secret_values, redact_sensitive, safe_error
from drbrain.services.fetch import (  # noqa: F401
    _resolve_identifier,
    download_pdf,
    fetch_paper,
    resolve_pdf_url,
)
from drbrain.storage.inbox import first_symlink_component, scan_inbox
from drbrain.storage.paths import paper_dir as resolve_paper_dir_path
from drbrain.storage.paths import paper_fs_key, writable_artifact_path

console = Console()


def _write_paper_text_atomically(paper_path: Path, filename: str, content: str) -> Path:
    """Publish a paper text artifact without following a stale symlink."""
    destination = writable_artifact_path(paper_path, filename)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{filename}.", dir=str(paper_path), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return destination


def _runtime_data_path(ctx: typer.Context, value: str | Path, *, label: str) -> Path:
    """Resolve a command-owned mutable path inside the active runtime root."""
    return runtime_data_path(ctx, value, label=label)


@contextmanager
def _scoped_deepxiv_token(cfg) -> Iterator[None]:
    """Expose the configured DeepXiv token only for the current command.

    The DeepXiv client currently reads its token from the process environment.
    Preserve a caller-provided value, restore it on every exit path, and never
    include the token in command output or logs.
    """

    api = cfg.get("api", {}) if hasattr(cfg, "get") else getattr(cfg, "api", {})
    token = (
        api.get("deepxiv_token", "") if hasattr(api, "get") else getattr(api, "deepxiv_token", "")
    )
    had_previous = "DEEPXIV_TOKEN" in os.environ
    previous = os.environ.get("DEEPXIV_TOKEN")
    if token and not had_previous:
        os.environ["DEEPXIV_TOKEN"] = str(token)
    try:
        yield
    finally:
        if had_previous:
            # ``previous`` is non-None for a present environment entry in the
            # normal POSIX environment mapping; assign defensively anyway.
            os.environ["DEEPXIV_TOKEN"] = previous or ""
        else:
            os.environ.pop("DEEPXIV_TOKEN", None)


def _resolve_report_path(report_dir: Path, local_id: str) -> Path:
    """Resolve canonical reports with safe compatibility fallbacks.

    Older report writers used ``report_dir/<local_id>.json`` directly, which
    made slash-containing DOI IDs nested directories.  Read those layouts only
    when the candidate remains beneath ``report_dir`` and fail closed when
    multiple legacy candidates exist.
    """
    canonical = report_dir / f"{paper_fs_key(local_id)}.json"
    root = report_dir.resolve()
    if canonical.is_file():
        try:
            canonical.resolve().relative_to(root)
        except (OSError, ValueError) as exc:
            raise ValueError(f"report path escapes report root: {canonical}") from exc
        return canonical

    candidates: list[Path] = []
    value = str(local_id)
    if "/" in value:
        parts = value.split("/")
        if all(part and part not in {".", ".."} for part in parts):
            candidates.extend(
                [
                    report_dir.joinpath(*parts[:-1], f"{parts[-1]}.json"),
                    report_dir / f"{value.replace('/', '_')}.json",
                    report_dir.joinpath(parts[0], f"{'_'.join(parts[1:])}.json"),
                ]
            )
    elif value and "/" not in value and "\\" not in value and "\x00" not in value:
        # Non-DOI IDs may still have been written before canonical encoding
        # (e.g. a Unicode or boundary-dot local_id).
        candidates.append(report_dir / f"{value}.json")

    safe_existing: list[Path] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            candidate.resolve().relative_to(root)
        except (OSError, ValueError) as exc:
            raise ValueError(f"report path escapes report root: {candidate}") from exc
        if candidate not in safe_existing:
            safe_existing.append(candidate)
    if len(safe_existing) > 1:
        raise ValueError(f"ambiguous report layouts for local_id {local_id!r}")
    return safe_existing[0] if safe_existing else canonical


def _runtime_value(runtime, *names):
    """Read a value from either a RuntimeContext mapping or dataclass.

    The root CLI callback owns the concrete RuntimeContext type.  Keeping this
    small adapter here lets the pipeline command remain compatible with older
    callers that only provide ``ctx.obj["config"]``.
    """
    if runtime is None:
        return None
    for name in names:
        if isinstance(runtime, dict):
            value = runtime.get(name)
        else:
            value = getattr(runtime, name, None)
        if isinstance(value, (str, Path)) and str(value):
            return str(value)
    return None


def _runtime_path(ctx: typer.Context, value: str | Path) -> Path:
    """Resolve a command-local path inside the active runtime namespace."""
    obj = getattr(ctx, "obj", None)
    runtime = obj.get("runtime") if isinstance(obj, dict) else None
    resolver = getattr(runtime, "resolve_path", None)
    if callable(resolver):
        resolved = resolver(value)
        if isinstance(resolved, Path):
            return resolved
    return Path(value).expanduser()


def _runtime_lexical_path(ctx: typer.Context, value: str | Path) -> Path:
    """Return an input path before runtime resolution dereferences symlinks."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    obj = getattr(ctx, "obj", None)
    runtime = obj.get("runtime") if isinstance(obj, dict) else None
    root = getattr(runtime, "root", None)
    if root:
        return Path(root) / path
    return Path.cwd() / path


def _configured_papers_root(cfg: dict) -> str | Path:
    """Read the normalized papers directory from typed or dict config."""
    dirs = cfg.get("dirs", {})
    if hasattr(dirs, "get"):
        value = dirs.get("papers")
        if value:
            return value
    return "data/papers"


def _fetch_staging_key(identifier: str) -> str:
    """Return a bounded, traversal-safe directory key for a fetch request."""
    return f"fetch-{hashlib.sha256(identifier.encode('utf-8')).hexdigest()[:24]}"


def _pipeline_runtime_paths(ctx: typer.Context) -> tuple[str | None, str | None]:
    """Resolve the config file and repository root for pipeline children.

    Newer CLI callbacks put these values in ``ctx.obj["runtime"]``.  The
    parent Click context and environment fallbacks preserve compatibility with
    direct command invocations and older wrappers.
    """
    obj = getattr(ctx, "obj", None)
    runtime = obj.get("runtime") if isinstance(obj, dict) else None

    config_path = _runtime_value(runtime, "config_path", "config_file", "config")
    root = _runtime_value(runtime, "root", "repo_root", "worktree_root")

    if isinstance(obj, dict):
        config_path = config_path or _runtime_value(
            obj, "config_path", "config_file", "config_overlay"
        )
        root = root or _runtime_value(obj, "root", "repo_root", "worktree_root")

    # Click stores root options on the root context.  Inspect only real dicts so
    # MagicMock-based direct command tests do not manufacture false values.
    current = ctx
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        params = getattr(current, "params", None)
        if isinstance(params, dict):
            config_path = config_path or _runtime_value(
                params, "config", "config_path", "config_file"
            )
            root = root or _runtime_value(params, "root", "repo_root", "worktree_root")
        parent = getattr(current, "parent", None)
        # Avoid following dynamically-created attributes on MagicMock/direct
        # test contexts forever; real Click parents are typer.Context objects.
        current = parent if isinstance(parent, click.Context) else None

    if not config_path:
        if "DRBRAIN_CONFIG" in os.environ:
            config_path = os.environ["DRBRAIN_CONFIG"]
        elif "DRBRAIN_CONFIG_PATH" in os.environ:
            config_path = os.environ["DRBRAIN_CONFIG_PATH"]
    if not root:
        if "DRBRAIN_ROOT" in os.environ:
            root = os.environ["DRBRAIN_ROOT"]
        elif "DRBRAIN_RUNTIME_ROOT" in os.environ:
            root = os.environ["DRBRAIN_RUNTIME_ROOT"]
    return config_path, root


def _pipeline_exit_code(value) -> int:
    """Return a valid process exit code for Typer/Click."""
    try:
        code = int(value)
    except (TypeError, ValueError):
        return 1
    return code if 0 < code <= 255 else 1


def _pipeline_child_env(ctx: typer.Context, root: str | None) -> dict[str, str]:
    """Build a child environment from RuntimeContext when available."""
    obj = getattr(ctx, "obj", None)
    runtime = obj.get("runtime") if isinstance(obj, dict) else None
    child_env = None
    child_env_factory = getattr(runtime, "child_env", None)
    if callable(child_env_factory):
        try:
            candidate = child_env_factory()
        except (OSError, TypeError, ValueError) as exc:
            # A failed context export is an isolation failure.  Falling back
            # to the parent environment can retain a stale temp/config root
            # from another worktree and make child stages write there.
            raise ValueError(
                f"unable to construct pipeline child environment: {safe_error(exc)}"
            ) from exc
        if not isinstance(candidate, dict):
            raise ValueError("RuntimeContext.child_env() must return an environment mapping")
        child_env = {str(key): str(value) for key, value in candidate.items()}
    if child_env is None:
        if root:
            # Direct/library callers may not have a callback-created context;
            # construct one so every selector is derived from the same root.
            from drbrain.runtime import RuntimeContext

            try:
                child_env = RuntimeContext.create(root).child_env()
            except (OSError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"unable to construct pipeline child environment: {safe_error(exc)}"
                ) from exc
        else:
            child_env = os.environ.copy()
    if root:
        from drbrain.runtime import RuntimeContext

        # Normalize the root written into argv/env and overwrite every runtime
        # selector, rather than leaving a caller's stale alias in place.
        effective_root = RuntimeContext.create(root).root
        child_env["DRBRAIN_ROOT"] = str(effective_root)
        child_env["DRBRAIN_RUNTIME_ROOT"] = str(effective_root)
    return child_env


def _run_pipeline_step(
    name: str,
    args: list[str],
    *,
    root: str | None,
    env: dict[str, str] | None = None,
) -> None:
    """Run one child command and fail closed on any execution error."""
    import subprocess as _sp

    child_env = env if env is not None else os.environ.copy()
    if root:
        child_env["DRBRAIN_ROOT"] = root

    run_kwargs = {
        "check": True,
        "env": child_env,
    }
    if root:
        run_kwargs["cwd"] = root

    try:
        result = _sp.run(args, **run_kwargs)
    except _sp.CalledProcessError as exc:
        code = _pipeline_exit_code(exc.returncode)
        typer.echo(f"Pipeline failed at step '{name}' (exit {code}).", err=True)
        raise typer.Exit(code) from exc
    except _sp.SubprocessError as exc:
        typer.echo(
            f"Pipeline failed at step '{name}': {safe_error(exc)}",
            err=True,
        )
        raise typer.Exit(1) from exc
    except OSError as exc:
        typer.echo(
            f"Pipeline failed at step '{name}': {safe_error(exc)}",
            err=True,
        )
        raise typer.Exit(1) from exc

    # A mocked runner (or a custom subprocess adapter) may return a non-zero
    # CompletedProcess even when it ignores ``check=True``.  Check explicitly
    # so the command contract remains fail-fast in that case too.
    return_code = getattr(result, "returncode", 0)
    if return_code:
        code = _pipeline_exit_code(return_code)
        typer.echo(f"Pipeline failed at step '{name}' (exit {code}).", err=True)
        raise typer.Exit(code)


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
    with _scoped_deepxiv_token(cfg):
        return _ingest_cmd_impl(ctx, paths, json_output)


def _ingest_cmd_impl(
    ctx: typer.Context,
    paths: list[str] = typer.Argument(
        None, help="PDF file(s) or directory. Defaults to data/spool/inbox/."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output machine-readable JSON to stdout"
    ),
):
    """Implementation for :func:`ingest_cmd` under the scoped token context."""
    cfg = ctx.obj["config"]
    config_secrets = configured_secret_values(cfg)
    if not paths:
        inbox_path = cfg.get("dirs", {}).get("inbox", "data/spool/inbox")
        paths = [inbox_path]

    pdf_files: list[Path] = []
    for p in paths:
        lexical_path = _runtime_lexical_path(ctx, p)
        symlink_component = first_symlink_component(lexical_path)
        if symlink_component is not None:
            if not json_output:
                typer.echo(f"Skipping symlink input: {p}", err=True)
            continue

        path = _runtime_path(ctx, p)
        if path.is_dir():
            pdf_files.extend(scan_inbox(path))
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

            try:
                result = _ingest_single_paper(
                    pdf_path,
                    cfg,
                    db,
                    dedup,
                    json_mode=json_output,
                )
                # Keep the batch contract explicit even when an alternate
                # ingestion adapter returns an unexpected value.
                if not isinstance(result, dict):
                    result = {
                        "ok": False,
                        "local_id": None,
                        "error": "ingest returned a non-object result",
                    }
            except Exception as exc:  # noqa: BLE001 - isolate one bad PDF
                # A failed paper may have opened a transaction before raising.
                # Roll it back so the following paper starts from a clean DB
                # state, while preserving already committed papers.
                try:
                    db.conn.rollback()
                except Exception:  # noqa: BLE001 - retain the original error
                    pass
                result = {
                    "ok": False,
                    "local_id": None,
                    "error": safe_error(
                        f"{type(exc).__name__}: {exc}",
                        secrets=config_secrets,
                    ),
                }
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
            typer.echo(
                json.dumps(
                    redact_sensitive(output),
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )
        else:
            if len(pdf_files) > 1:
                typer.echo(f"\n{'=' * 60}")
                typer.echo(f"Batch complete: {len(results)} papers ingested")
                success = sum(1 for r in results if r.get("ok"))
                typer.echo(f"  Successful: {success}, Failed: {len(results) - success}")

        success = sum(1 for r in results if r.get("ok"))
        logger.info("[ingest] batch done — %d/%d papers ingested", success, len(results))

    # A batch can continue to collect independent results, but callers must
    # still be able to detect that at least one paper failed.
    if any(not r.get("ok") for r in results):
        raise typer.Exit(1)


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
    result = fetch_paper(
        doi=doi,
        title=title,
        arxiv_id=arxiv_id,
        fetch_config=fetch_cfg,
        papers_root=_configured_papers_root(cfg),
    )

    if not result:
        typer.echo("Could not find a downloadable PDF from any source.", err=True)
        raise typer.Exit(1)

    typer.echo(f"  Downloaded: {result['pdf_path']}")

    # Ingest the downloaded paper
    pdf_path = Path(result["pdf_path"])
    with open_db(cfg) as db:
        dedup = DedupEngine(db)
        ingest_result = _ingest_single_paper(
            pdf_path, cfg, db, dedup, json_mode=False, override_metadata=result
        )

    if ingest_result.get("ok"):
        typer.echo(f"  Ingested: {ingest_result.get('local_id')}")
        typer.echo(f"  Next: drbrain build {ingest_result.get('local_id')}")
    else:
        typer.echo(
            "  Ingest failed: "
            + safe_error(
                ingest_result.get("error", "unknown error"), secrets=configured_secret_values(cfg)
            ),
            err=True,
        )
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
        text = text.default or ""
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
    report_dir = _runtime_data_path(ctx, cfg["dirs"]["reports"], label="reports directory")
    report_path = _resolve_report_path(report_dir, local_id)
    if not report_path.is_file():
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
    incremental: bool = typer.Option(
        True,
        "--incremental/--full",
        help="Incremental: only run rules near papers changed since last closure run. "
        "Use --full to scan the whole graph.",
    ),
):
    """Run rule-based closure on the graph.

    By default incremental: only applies rules to the 2-hop neighborhood of
    concepts touched by papers modified since the last closure run. Use --full
    to scan the entire graph (previous behavior).
    """
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
        min_confidence = float(min_confidence.default or 0.6)
    if isinstance(ground, typer.models.OptionInfo):
        ground = ground.default
    if isinstance(incremental, typer.models.OptionInfo):
        incremental = incremental.default

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

        # Decide incremental vs full. Incremental falls back to full when there
        # is no last_run watermark (first run) or no changed papers.
        use_incremental = incremental
        seed_nodes: set[str] = set()
        if incremental:
            last_run = db.get_last_run("closure")
            if last_run is None:
                use_incremental = False  # first run -> full
            else:
                changed_papers = db.get_papers_since(last_run)
                if changed_papers:
                    # Collect concept labels from changed papers as seeds
                    placeholders = ",".join("?" * len(changed_papers))
                    rows = db.conn.execute(
                        f"SELECT DISTINCT label FROM concepts WHERE local_id IN ({placeholders})",
                        changed_papers,
                    ).fetchall()
                    seed_nodes = {r[0] for r in rows}
                if not seed_nodes:
                    # Nothing changed since last closure -> no-op
                    if json_output:
                        typer.echo(json.dumps({"inferred": [], "count": 0, "skipped": True}))
                    else:
                        typer.echo("Closure: no changes since last run, skipping")
                    db.set_last_run("closure")
                    db.commit()
                    return

        if use_incremental and seed_nodes:
            if not json_output:
                typer.echo(
                    f"Closure: incremental ({len(seed_nodes)} seed nodes from changed papers)"
                )
            inferred = graph.closure_incremental(seed_nodes)
        else:
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
            db.set_last_run("closure")
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


def batch_fetch_cmd(
    ctx: typer.Context,
    input_file: str = typer.Argument(..., help="File containing one DOI or URL per line"),
    output_dir: str | None = typer.Option(None, "--output", "-o"),
    delay: float = typer.Option(1.0, "--delay", "-d", help="Delay between fetches (seconds)"),
    skip_existing: bool = typer.Option(True, "--skip-existing", help="Skip DOIs already in DB"),
):
    """Batch fetch papers from a DOI/URL list file.

    Reads a text file with one DOI or URL per line.
    Resolves open-access PDF URLs via arXiv/OpenAlex/Unpaywall.
    Downloads PDFs to the inbox for subsequent ingest.

    Lines starting with # are ignored as comments.
    """
    import time

    # resolve_pdf_url and download_pdf are imported at module top

    if isinstance(output_dir, typer.models.OptionInfo):
        output_dir = output_dir.default
    cfg = ctx.obj["config"]
    config_secrets = configured_secret_values(cfg)
    fetch_cfg = cfg.get("fetch", {})

    # Validate input file
    input_path = _runtime_path(ctx, input_file)
    if not input_path.exists():
        typer.echo(f"Input file not found: {input_file}", err=True)
        raise typer.Exit(1)

    # Read and parse lines
    raw_lines = input_path.read_text(encoding="utf-8").splitlines()
    entries: list[str] = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        entries.append(stripped)

    if not entries:
        typer.echo("No entries found in input file.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Batch fetch: {len(entries)} entries from {input_file}")

    # Ensure output directory exists inside the selected runtime.
    if not output_dir:
        dirs = cfg.get("dirs", {})
        output_dir = dirs.get("inbox") if hasattr(dirs, "get") else None
        output_dir = output_dir or "data/spool/inbox"
    out = _runtime_data_path(ctx, output_dir, label="batch-fetch output")
    out.mkdir(parents=True, exist_ok=True)

    fetched = 0
    skipped = 0
    failed = 0
    errors: list[str] = []

    with open_db(cfg) as db:
        for idx, entry in enumerate(entries, 1):
            typer.echo(f"[{idx}/{len(entries)}] {entry}")

            # Resolve identifier type (DOI / arXiv ID / arXiv URL / title)
            doi, title, arxiv_id = _resolve_identifier(entry)

            # Check if already in DB (try both doi and arxiv)
            if skip_existing:
                ext_key = "doi" if doi else ("arxiv" if arxiv_id else None)
                ext_val = doi or arxiv_id
                if ext_key and ext_val:
                    existing_id = db.get_paper_by_external_id(ext_key, ext_val)
                    if existing_id:
                        typer.echo(f"  Skipped (already in DB as {existing_id})")
                        skipped += 1
                        continue

            # Resolve PDF URL through multi-stage fallback
            try:
                pdf_url = resolve_pdf_url(
                    doi=doi, title=title, arxiv_id=arxiv_id, fetch_config=fetch_cfg
                )
            except Exception as exc:
                logger.warning(
                    "resolve_pdf_url failed for {}: {}",
                    entry,
                    safe_error(exc, secrets=config_secrets),
                )
                pdf_url = None

            # If entry looks like a direct PDF URL, use it directly
            if not pdf_url and entry.startswith(("http://", "https://")) and entry.endswith(".pdf"):
                pdf_url = entry

            if not pdf_url:
                msg = f"No PDF URL found for: {entry}"
                typer.echo(f"  FAILED: {msg}")
                errors.append(msg)
                failed += 1
                continue

            # Download PDF
            dest_dir = out / _fetch_staging_key(entry)
            dest_dir.mkdir(parents=True, exist_ok=True)
            try:
                pdf_path = download_pdf(pdf_url, dest_dir, fetch_config=fetch_cfg)
            except Exception as exc:
                msg = f"Download failed for {entry}: {safe_error(exc, secrets=config_secrets)}"
                typer.echo(f"  FAILED: {msg}")
                errors.append(msg)
                failed += 1
                continue

            if not pdf_path:
                msg = f"Download returned empty for: {entry}"
                typer.echo(f"  FAILED: {msg}")
                errors.append(msg)
                failed += 1
                continue

            typer.echo(f"  OK: {pdf_path}")
            fetched += 1

            # Polite delay between fetches
            if delay > 0 and idx < len(entries):
                time.sleep(delay)

    # Summary
    sep_line = "=" * 40
    typer.echo(f"\n{sep_line}")
    typer.echo(f"Batch fetch complete: {len(entries)} entries")
    typer.echo(f"  Fetched: {fetched}, Skipped: {skipped}, Failed: {failed}")
    if errors:
        typer.echo("\nFailures:")
        for e in errors:
            typer.echo(f"  - {safe_error(e, secrets=config_secrets)}")

    logger.info(
        "[batch-fetch] done - %d fetched, %d skipped, %d failed",
        fetched,
        skipped,
        failed,
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
    from drbrain.dedup.resolver import PaperIDs, canonical_paper_id
    from drbrain.providers.webtools import (
        canonical_web_url,
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
    config_secrets = configured_secret_values(cfg)
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
                safe_extraction_error = safe_error(error, secrets=config_secrets)
                typer.echo(f"  Extraction failed: {safe_extraction_error}", err=True)
                results.append({"url": url, "status": "error", "error": safe_extraction_error})
                continue

            ids = PaperIDs(
                doi=extracted.get("doi"),
                arxiv=extracted.get("arxiv"),
                s2_id=extracted.get("s2_id") or extracted.get("semantic_scholar_id"),
                openalex_id=extracted.get("openalex_id"),
            ).normalized()
            local_id = canonical_paper_id(
                ids,
                title=title,
                source_key=canonical_web_url(url),
            )
            paper_dir = resolve_paper_dir_path(papers_dir, local_id)

            # A pre-existing DB row means this URL was already ingested.  Do
            # not mint a suffix based on the current title: that made retries
            # create duplicate papers whenever extraction metadata changed.
            if db.get_paper(local_id) is not None:
                results.append(
                    {"url": url, "local_id": local_id, "title": title, "status": "skipped"}
                )
                continue

            paper_dir.mkdir(parents=True, exist_ok=True)

            # Write markdown
            md_content = _render_extracted_markdown(title, url, text)
            _write_paper_text_atomically(paper_dir, "raw.md", md_content)

            # Register in DB
            db.insert_paper(
                local_id=local_id,
                title=title or url,
                year=None,
                status="uploaded",
            )
            db.insert_paper_ids(
                local_id,
                doi=ids.doi,
                arxiv=ids.arxiv,
                s2_id=ids.s2_id,
                openalex_id=ids.openalex_id,
                strict=True,
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
        typer.echo(
            json.dumps(
                redact_sensitive(results),
                ensure_ascii=False,
                indent=2,
            )
        )
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

    cfg = ctx.obj["config"]
    config_secrets = configured_secret_values(cfg)

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
            typer.echo(safe_error(e, secrets=(*config_secrets, key)), err=True)
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
            typer.echo(safe_error(e, secrets=(*config_secrets, key)), err=True)
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
        results: list = ppubs_search(query_str, limit=limit)  # type: ignore[assignment,no-redef]  # PpubsPatent redef  # type: ignore[assignment]
    except PpubsError as e:
        typer.echo(safe_error(e, secrets=config_secrets), err=True)
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
    full: bool = typer.Option(
        False,
        "--full",
        help="Force full (non-incremental) processing on every step. Default is incremental.",
    ),
):
    """Chain multiple processing steps in sequence (ingest → build → embed → closure).

    By default each step runs in incremental mode: build only touches papers
    not yet extracted (or touched since last build), closure only scans the
    neighborhood of recently-changed concepts, embed only trains on new edges.
    Use --full to force a complete rebuild across every step.
    """
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
        typer.echo(safe_error(e), err=True)
        raise typer.Exit(1)

    if dry_run:
        typer.echo(f"[dry-run] Would execute {len(step_names)} step(s): {', '.join(step_names)}")
        typer.echo(f"[dry-run] Mode: {'full' if full else 'incremental'}")
        return

    mode_label = "full" if full else "incremental"
    typer.echo(f"Pipeline ({mode_label}): {' -> '.join(step_names)}")
    typer.echo()

    import sys as _sys

    config_path, root = _pipeline_runtime_paths(ctx)
    child_env = _pipeline_child_env(ctx, root)

    def child_command(step: str, *step_args: str) -> list[str]:
        args = [_sys.executable, "-m", "drbrain.cli.main"]
        if config_path:
            args.extend(["--config", config_path])
        # ``--root`` is supplied by the root CLI callback in the isolated
        # runtime.  Keep it conditional so older direct callers still work.
        if root:
            args.extend(["--root", root])
        args.extend([step, *step_args])
        return args

    for i, name in enumerate(step_names, 1):
        typer.echo(f"[{i}/{len(step_names)}] {name} ...")
        if name == "ingest":
            args = child_command("ingest")
        elif name == "build":
            # Incremental (default): no --all → builds only dirty papers.
            # Full: --all → rebuilds every paper.
            args = child_command("build")
            if full:
                args.append("--all")
        elif name == "embed":
            # Pipeline runs tree-embedding (PageIndex/RAPTOR) which is already
            # content-hash incremental. Standalone 'drbrain embed' (no --tree)
            # does TransE and has its own incremental path; pipeline does not
            # invoke TransE to match prior behavior.
            args = child_command("embed", "--tree")
        elif name == "closure":
            # Incremental (default): --incremental → 2-hop neighborhood.
            # Full: --full flag on closure_cmd → whole-graph scan.
            args = child_command("closure")
            if full:
                args.append("--full")
        else:  # resolve_steps currently prevents this; keep the invariant local.
            typer.echo(f"Pipeline failed: unsupported step '{name}'.", err=True)
            raise typer.Exit(1)

        _run_pipeline_step(name, args, root=root, env=child_env)

    typer.echo(f"\nPipeline complete ({mode_label}): {', '.join(step_names)}")


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

    store_path = _runtime_data_path(ctx, DEFAULT_PATH, label="proceedings store")

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
            typer.echo(safe_error(e), err=True)
            raise typer.Exit(1)
        typer.echo(f"Added {add[1]} to proceeding {add[0]}")
        return

    if show:
        proc: dict | None = get_proceeding(store_path, show)
        if not proc:
            typer.echo(f"Proceeding not found: {show}", err=True)
            raise typer.Exit(1)
        if json_output:
            typer.echo(json.dumps(proc, ensure_ascii=False, indent=2))
            return
        typer.echo(f"[{proc['id']}] {proc['name']} ({proc['year']})")
        if proc.get("venue"):
            typer.echo(f"  Venue: {proc['venue']}")
        typer.echo(f"  Papers: {len(proc.get('papers', []))}")
        for paper_id in proc.get("papers", []):
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
    from drbrain.storage.explore import (
        create_explore_silo,
        delete_explore_silo,
        get_silo_papers,
        list_explore_silos,
        search_silo,
    )

    root = _runtime_data_path(ctx, "data/explore", label="explore data")

    if create:
        try:
            silo = create_explore_silo(root, create)
        except ValueError as e:
            typer.echo(safe_error(e), err=True)
            raise typer.Exit(1)
        typer.echo(f"Created explore silo: {silo['name']}")
        return

    if delete:
        try:
            delete_explore_silo(root, delete)
        except Exception as e:
            typer.echo(safe_error(e), err=True)
            raise typer.Exit(1)
        typer.echo(f"Deleted: {delete}")
        return

    if search and name:
        try:
            results = search_silo(root, name, search)
        except ValueError as e:
            typer.echo(safe_error(e), err=True)
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
            typer.echo(safe_error(e), err=True)
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
