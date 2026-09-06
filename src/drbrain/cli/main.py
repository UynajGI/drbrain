"""DrBrain CLI — ingest, query, expand, and more."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from loguru import logger

from drbrain.cli._helpers.security import redact_cli_args
from drbrain.cli.analysis_commands import (
    ask_cmd,
    descendants_cmd,
    difficulty_cmd,
    evolve_cmd,
    frontier_cmd,
    isomorphism_cmd,
    landscape_cmd,
    paradigm_cmd,
    reason_cmd,
    survey_cmd,
    transfers_cmd,
)
from drbrain.cli.autoresearch_commands import autoresearch_app
from drbrain.cli.build_commands import (
    build_cmd,
    embed_cmd,
    translate_cmd,
)
from drbrain.cli.check_commands import (
    analyze_cmd,
    check_cmd,
    clean_cmd,
)
from drbrain.cli.concept_graph_commands import cg_app
from drbrain.cli.export_commands import (
    backup_cmd,
    delete_cmd,
    document_cmd,
    export_cmd,
    export_okf_cmd,
    lineage_cmd,
    metrics_cmd,
    queue_cmd,
    queue_resolve_all_cmd,
    queue_resolve_cmd,
    restore_cmd,
    style_cmd,
)
from drbrain.cli.graph_commands import graph_app
from drbrain.cli.ingest_commands import (
    batch_fetch_cmd,
    check_citations_cmd,
    citations_cmd,
    closure_cmd,
    explore_cmd,
    fetch_cmd,
    ingest_cmd,
    ingest_link_cmd,
    patent_search_cmd,
    pipeline_cmd,
    proceedings_cmd,
    report_cmd,
)
from drbrain.cli.query_commands import (
    fsearch_cmd,
    hybrid_cmd,
    index_cmd,
    list_cmd,
    query_cmd,
    search_cmd,
    seed_cmd,
    show_cmd,
    stats_cmd,
)
from drbrain.cli.rag_commands import rag_app
from drbrain.cli.repair_commands import (
    enrich_cmd,
    import_cmd,
    repair_cmd,
)
from drbrain.cli.session_commands import session_app
from drbrain.cli.setup import setup_cmd
from drbrain.cli.webui_commands import webui_cmd
from drbrain.cli.ws_commands import ws_app
from drbrain.log import setup_logging
from drbrain.runtime import RuntimeContext
from drbrain.security import configured_secret_values, safe_error
from drbrain.services.audit import audit_cmd

app = typer.Typer(help="DrBrain — Academic Knowledge Graph System")


@app.callback()
def _main_callback(
    ctx: typer.Context,
    config: str = typer.Option(
        "", "--config", help="Override config file (e.g. config.embed1.yaml)"
    ),
    root: str = typer.Option(
        "",
        "--root",
        envvar="DRBRAIN_ROOT",
        help="Runtime/worktree root for relative data paths",
    ),
) -> None:
    """Called before every command. Sets up logging and loads config."""
    from drbrain.config import Config, load_config
    from drbrain.log import get_session_id

    ctx.ensure_object(dict)
    try:
        import os as _os

        # Keep validation strict for an explicitly scoped invocation, while
        # avoiding process-global state leaking between repeated CliRunner
        # calls (or library users embedding the Typer app).
        explicit_root = bool(root) or (
            "DRBRAIN_ROOT" in _os.environ or "DRBRAIN_RUNTIME_ROOT" in _os.environ
        )
        parameter_source = None
        get_parameter_source = getattr(ctx, "get_parameter_source", None)
        if callable(get_parameter_source):
            parameter_source = get_parameter_source("config")
        config_from_command_line = getattr(parameter_source, "name", "") == "COMMANDLINE"
        root_parameter_source = None
        if callable(get_parameter_source):
            root_parameter_source = get_parameter_source("root")
        root_from_command_line = getattr(root_parameter_source, "name", "") == "COMMANDLINE"
        if root_from_command_line and (not isinstance(root, str) or not root.strip()):
            raise ValueError("Runtime root must not be empty")
        if config_from_command_line:
            # Keep an explicitly supplied empty value distinguishable from the
            # Option default.  Empty selectors must fail closed instead of
            # silently selecting a lower-precedence environment alias.
            overlay_spec = config
        elif config:
            overlay_spec = config
        elif "DRBRAIN_CONFIG" in _os.environ:
            # An empty primary selector is explicit and must not fall through
            # to the legacy CONFIG_PATH alias.
            overlay_spec = _os.environ["DRBRAIN_CONFIG"]
        elif "DRBRAIN_CONFIG_PATH" in _os.environ:
            overlay_spec = _os.environ["DRBRAIN_CONFIG_PATH"]
        else:
            overlay_spec = None
        if overlay_spec is not None and (
            not isinstance(overlay_spec, str) or not overlay_spec.strip()
        ):
            raise ValueError("Config overlay must not be empty")
        # Build the base context first so the compatibility ``DRBRAIN_CONFIG``
        # value can be distinguished from a real overlay.  Shell launchers and
        # ``RuntimeContext.child_env`` historically use that variable for the
        # base file; loading it a second time as an overlay breaks ``setup`` in
        # a fresh runtime and can duplicate configuration layers.
        setup_requested = ctx.invoked_subcommand == "setup"
        runtime = RuntimeContext.create(
            root or None,
            create_root=setup_requested and explicit_root,
        )
        if overlay_spec:
            overlay_candidate = runtime.assert_within_root(overlay_spec, label="config overlay")
            if overlay_candidate == runtime.base_config_path.resolve():
                overlay_spec = None
            else:
                runtime = RuntimeContext.create(
                    root or None,
                    overlay_path=overlay_candidate,
                    create_root=setup_requested and explicit_root,
                )
        # ``setup`` is the one command whose purpose is to initialize a new
        # runtime root.  It must be able to start before a base config exists;
        # all other commands fail closed before opening any database.
        setup_without_base = (
            ctx.invoked_subcommand == "setup"
            and not runtime.base_config_path.exists()
            and not runtime.base_config_path.is_symlink()
            and not overlay_spec
        )
        base_config = runtime.validate_config_file(
            runtime.base_config_path,
            label="base config",
            required=not setup_without_base,
        )
        local_config = runtime.root / "config.local.yaml"
        if local_config.exists() or local_config.is_symlink():
            runtime.validate_config_file(
                local_config,
                label="local config",
                required=True,
            )
        # Config.from_yaml/load_config discover the active runtime through the
        # environment for compatibility with direct library callers.  An
        # explicit ``--root`` must take precedence over an inherited selector
        # while these layers are read; otherwise a NEW root can be validated
        # against OLD and fail (or, worse, load OLD's local config).  Restore
        # the caller's selectors immediately after loading so the later
        # command-scoped environment export still has the original baseline.
        selector_keys = ("DRBRAIN_ROOT", "DRBRAIN_RUNTIME_ROOT")
        previous_selectors = {key: _os.environ.get(key) for key in selector_keys}
        selector_active = any(previous_selectors.values())
        selector_overridden = bool(root)
        if selector_overridden:
            _os.environ["DRBRAIN_ROOT"] = str(runtime.root)
            _os.environ["DRBRAIN_RUNTIME_ROOT"] = str(runtime.root)
        try:
            if overlay_spec:
                # ``RuntimeContext.create`` checks the resolved overlay
                # boundary; this second check rejects a final symlink before
                # it is opened.
                overlay_config = runtime.validate_config_file(
                    runtime.overlay_path,
                    label="config overlay",
                    required=True,
                )
                # ``overlay_path`` preserves base -> config.local -> explicit
                # layering when a command-specific config is requested.
                loaded = Config.from_yaml(
                    base_config,
                    overlay_path=overlay_config,
                )
            elif setup_without_base:
                loaded = Config()
            else:
                loaded = load_config(base_config)
        finally:
            if selector_overridden:
                for key, value in previous_selectors.items():
                    if value is None:
                        _os.environ.pop(key, None)
                    else:
                        _os.environ[key] = value
        # The CLI is itself a write-capable entrypoint.  Even when no selector
        # was supplied, refuse mutable absolute paths that point outside this
        # checkout; otherwise a copied config.local.yaml could make this
        # worktree write into another branch's corpus.  Library callers that
        # intentionally share a resource can still use Config.from_yaml
        # directly and opt into their own policy.
        # Legacy in-process callers may intentionally provide shared absolute
        # paths.  Once a runtime selector/explicit root is active, enforce the
        # containment policy so a selected worktree cannot escape its data
        # namespace.
        if explicit_root or selector_active:
            runtime.validate_config(loaded)

        # Child processes and setup helpers need the same namespace while the
        # command is running.  Restore the caller's environment when Click
        # closes the context so one in-process invocation cannot retarget the
        # next one accidentally.
        runtime_env = {
            "DRBRAIN_TEMP_ROOT": str(runtime.temp_root),
            "DRBRAIN_RUN_ID": runtime.run_id,
            "DRBRAIN_CONFIG": str(runtime.overlay_path or runtime.base_config_path),
            "DRBRAIN_CONFIG_PATH": str(runtime.overlay_path or runtime.base_config_path),
        }
        if explicit_root:
            # Publishing the root activates fail-closed checks in lower-level
            # connection factories.  Keep that contract for explicitly
            # isolated invocations; legacy invocations may still provide a
            # deliberately shared absolute path in their config.
            runtime_env.update(
                {
                    "DRBRAIN_ROOT": str(runtime.root),
                    "DRBRAIN_RUNTIME_ROOT": str(runtime.root),
                }
            )
        previous_env = {key: _os.environ.get(key) for key in runtime_env}
        _os.environ.update(runtime_env)

        def _restore_runtime_env() -> None:
            for key, value in previous_env.items():
                if value is None:
                    _os.environ.pop(key, None)
                else:
                    _os.environ[key] = value

        # ``Context.call_on_close`` is part of Click's public API and also
        # runs when a command exits through ``typer.Exit``.
        ctx.call_on_close(_restore_runtime_env)
        ctx.obj["runtime"] = runtime
        # A context is created for every invocation so relative defaults have a
        # single anchor, but only an explicitly selected root is an isolation
        # boundary.  Legacy invocations may still target an explicit restore
        # directory outside the checkout; scoped invocations must stay inside
        # their runtime root.
        ctx.obj["runtime_isolated"] = explicit_root
        ctx.obj["config"] = runtime.apply_config(loaded)
        dirs = ctx.obj["config"].get("dirs", {})
        log_dir = dirs.get("logs") if hasattr(dirs, "get") else None
        setup_logging(
            log_path=Path(log_dir or runtime.root / "data" / "logs") / "drbrain.log",
            secrets=configured_secret_values(ctx.obj["config"]),
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        # Fail before any command opens a database or touches an artifact.
        # A malformed command section is still rejected before any I/O, but it
        # is a command-level configuration failure rather than an invocation
        # /path usage error.  Preserve the CLI's established exit-code split.
        command_config_error = ctx.invoked_subcommand == "autoresearch" and str(exc).startswith(
            "configured section 'autoresearch'"
        )
        # Keep the established human-facing capitalization while preserving
        # the detailed, redacted reason produced by RuntimeContext.
        detail = safe_error(exc)
        if detail:
            detail = detail[0].upper() + detail[1:]
        if command_config_error:
            typer.echo(f"[autoresearch] invalid config: {detail}", err=True)
        else:
            typer.echo(f"Runtime/config error: {detail}", err=True)
        raise typer.Exit(1 if command_config_error else 2) from exc

    cmd = redact_cli_args(sys.argv[1:]) if len(sys.argv) > 1 else "(no args)"
    logger.info("CLI invoked [{}]: {}", get_session_id(), cmd)


app.command("setup")(setup_cmd)
app.command("ingest")(ingest_cmd)
app.command("ingest-link")(ingest_link_cmd)
app.command("patent-search")(patent_search_cmd)
app.command("pipeline")(pipeline_cmd)
app.command("proceedings")(proceedings_cmd)
app.command("explore")(explore_cmd)
app.command("batch-fetch")(batch_fetch_cmd)
app.command("fetch")(fetch_cmd)
app.command("citations")(citations_cmd)
app.command("check-citations")(check_citations_cmd)
app.command("report")(report_cmd)
app.command("closure")(closure_cmd)
app.command("seed")(seed_cmd)
app.command("list")(list_cmd)
app.command("stats")(stats_cmd)
app.command("webui")(webui_cmd)
app.command("show")(show_cmd)
app.command("index")(index_cmd)
app.command("query")(query_cmd)
app.command("fsearch")(fsearch_cmd)
app.command("search")(search_cmd)
app.command("hybrid")(hybrid_cmd)
app.command("export")(export_cmd)
app.command("export-okf")(export_okf_cmd)
app.command("queue")(queue_cmd)
app.command("queue resolve")(queue_resolve_cmd)
app.command("queue resolve-all")(queue_resolve_all_cmd)
app.command("delete")(delete_cmd)
app.command("lineage")(lineage_cmd)
app.command("ask")(ask_cmd)
app.command("check")(check_cmd)
app.command("audit")(audit_cmd)
app.command("style")(style_cmd)
app.command("document")(document_cmd)
app.command("metrics")(metrics_cmd)
app.command("clean")(clean_cmd)
app.command("backup")(backup_cmd)
app.command("restore")(restore_cmd)
app.command("analyze")(analyze_cmd)
app.command("repair")(repair_cmd)
app.command("enrich")(enrich_cmd)
app.command("import")(import_cmd)
app.command("translate")(translate_cmd)
app.command("build")(build_cmd)
app.command("embed")(embed_cmd)
app.command("evolve")(evolve_cmd)
app.command("descendants")(descendants_cmd)
app.command("landscape")(landscape_cmd)
app.command("paradigm")(paradigm_cmd)
app.command("transfers")(transfers_cmd)
app.command("isomorphism")(isomorphism_cmd)
app.command("difficulty")(difficulty_cmd)
app.command("frontier")(frontier_cmd)
app.command("survey")(survey_cmd)
app.command("reason")(reason_cmd)

# Sub-apps
app.add_typer(session_app, name="session")
app.add_typer(graph_app, name="graph")
app.add_typer(ws_app, name="ws")
app.add_typer(cg_app, name="cg")
app.add_typer(rag_app, name="rag")
app.add_typer(autoresearch_app, name="autoresearch")

if __name__ == "__main__":
    app()
