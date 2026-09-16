"""``drbrain index`` — prepare, inspect and verify the searchable index.

Main line: ``ingest → index build → search / ask``.  ``index build`` fills the
enabled retrieval legs from the canonical store — the incremental lexical BM25
index, canonical FTS, shared leaf/region vectors and the unified tree hierarchy
— and publishes one queryable tree generation.  ``index status`` separates the
three states a corpus can be in (ingested / indexed / retrievable) and reports
per-leg readiness, versions, backlog and failure reasons.  ``index verify``
re-checks exactly what ``search`` and ``ask`` actually read; the wider storage
audit stays in ``drbrain storage audit``.

Bare ``drbrain index`` keeps the historical incremental BM25 contract: the
group callback runs it when no subcommand is given, so existing callers, exit
codes and ``--json`` shapes stay valid.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from drbrain.cli._common import open_db, runtime_data_path
from drbrain.security import redact_sensitive
from drbrain.services.index_report import (
    build_index_status as _core_index_status,
)
from drbrain.services.index_report import (
    build_index_verify as _core_index_verify,
)

DEFAULT_TREE_STORAGE = "data/tree"

index_app = typer.Typer(
    help="Prepare, inspect and verify the searchable index (lexical + FTS + vectors + tree)",
    invoke_without_command=True,
    no_args_is_help=False,
)


def _runtime_option(value: Any, default: Any) -> Any:
    """Normalize a Typer ``OptionInfo`` for direct (non-CLI) callers."""
    if isinstance(value, typer.models.OptionInfo):
        return default if value.default is None else value.default
    return value


def _tree_storage_root(ctx: typer.Context, cfg: Any, override: str = "") -> Path:
    """Resolve the unified tree storage root the same way ``index build`` does."""
    from drbrain.rag.config import get_llamaindex_config

    li = get_llamaindex_config(cfg)
    configured = (
        str(override or "").strip()
        or str(getattr(li, "tree_storage", "") or "").strip()
        or DEFAULT_TREE_STORAGE
    )
    return Path(runtime_data_path(ctx, configured, label="tree storage"))


# ── the historical incremental BM25 stage (bare ``drbrain index``) ───────────


def rebuild_lexical_index(
    cfg: Any,
    *,
    rebuild: bool = False,
    notify: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Rebuild the lexical BM25 index (the pre-``index build`` behavior).

    Incremental by default: skips the rebuild when no paper changed since the
    last successful run.  Returns the historical JSON shape — ``{"documents",
    "indexed"}`` (plus ``"up_to_date": True`` on the skip path) — so both the
    legacy command and ``index build`` report the same lexical stage result.
    """
    from drbrain.services.index_build import rebuild_lexical

    with open_db(cfg) as db:
        return rebuild_lexical(db, rebuild=rebuild, notify=notify)


def emit_lexical_result(payload: dict[str, Any], *, json_output: bool) -> None:
    """Print the lexical stage result in the historical (pre-refactor) format."""
    if json_output:
        typer.echo(json.dumps(payload))
    elif payload.get("up_to_date"):
        typer.echo(
            f"Index up to date ({payload['documents']} documents, no changes since last run)"
        )
    else:
        typer.echo(f"Indexed {payload['documents']} documents")


@index_app.callback()
def index_callback(
    ctx: typer.Context,
    rebuild: bool = typer.Option(False, "--rebuild", help="Force full rebuild"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
) -> None:
    """Prepare, inspect and verify the searchable index — the main-line index entry.

    ``drbrain index build`` prepares every enabled retrieval leg (lexical BM25
    + canonical FTS + shared vectors + unified tree, and the LlamaIndex
    generation when ``rag_engine: llamaindex``) and publishes what search/ask
    read; ``drbrain index status`` reports ingested / indexed / retrievable
    per leg, and ``drbrain index verify`` re-checks exactly that.

    A bare invocation keeps the historical incremental BM25 rebuild: it skips
    the rebuild when no paper changed since the last successful run; use
    --rebuild to force a full rebuild.
    """
    if ctx.invoked_subcommand is not None:
        # Options declared on the group belong to the bare command only.
        return
    cfg = ctx.obj["config"]
    payload = rebuild_lexical_index(
        cfg,
        rebuild=bool(_runtime_option(rebuild, False)),
        notify=None if json_output else typer.echo,
    )
    emit_lexical_result(payload, json_output=bool(_runtime_option(json_output, False)))


# ── index build ─────────────────────────────────────────────────────────────


@index_app.command("build")
def index_build_cmd(
    ctx: typer.Context,
    force: bool = typer.Option(False, "--force", "-f", help="Force a full rebuild of every stage"),
    tree_storage: str = typer.Option(
        "",
        "--tree-storage",
        help="Storage root for unified tree generations (default: data/tree)",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
) -> None:
    """Prepare every enabled retrieval leg and publish a queryable generation.

    The lexical BM25 index, canonical FTS, the shared vector collection and the
    unified tree hierarchy are filled incrementally from the canonical store;
    a tree generation is published only when something changed.  Exit code 1
    when any stage failed — an unfinished or failed stage is never reported as
    ready.

    This command serves the main corpus.  A persisted LlamaIndex generation
    (``llamaindex.rag_engine: llamaindex``) is still prepared by the
    compatibility command ``drbrain rag index``, which the missing-index hint
    names for that engine; the shard pipelines keep their own legacy stage
    until the merge path understands the unified tables.
    """
    cfg = ctx.obj["config"]
    _force = bool(_runtime_option(force, False))
    _json = bool(_runtime_option(json_output, False))
    tree_storage = str(_runtime_option(tree_storage, "") or "")

    # The build itself (job slot, checkpoints, stages) lives in
    # ``drbrain.services.index_build`` so the WebUI runs the identical job.
    from drbrain.services.index_build import (
        IndexBuildBusyError,
        IndexBuildProfileUnavailableError,
        run_index_build,
    )

    try:
        with open_db(cfg) as db:
            payload = run_index_build(
                cfg,
                db=db,
                force=_force,
                tree_storage=_tree_storage_root(ctx, cfg, tree_storage),
                notify=None if _json else typer.echo,
            )
    except IndexBuildProfileUnavailableError as exc:
        typer.echo(f"[index] {exc}", err=True)
        raise typer.Exit(1) from exc
    except IndexBuildBusyError as exc:
        typer.echo(f"[index] {exc}", err=True)
        typer.echo(
            "[index] wait for it to finish, or inspect it with `drbrain index status`.", err=True
        )
        raise typer.Exit(1) from exc

    lexical = payload.get("lexical") or {}
    if _json:
        typer.echo(json.dumps(redact_sensitive(payload), indent=2, ensure_ascii=False, default=str))
    else:
        typer.echo(
            f"Index build ({'full' if _force else 'incremental'}): "
            f"{'ok' if payload.get('ok') else 'FAILED'}"
        )
        typer.echo(
            f"  lexical:    {lexical.get('documents', 0)} documents, indexed={lexical.get('indexed')}"
        )
        for stage in ("fts", "vectors", "hierarchy", "publication"):
            typer.echo(f"  {stage + ':':<12} {(payload.get(stage) or {}).get('status', '')}")
        typer.echo(
            f"  changed={payload.get('changed')} published={payload.get('published') or 'none'} "
            f"duration_ms={payload.get('duration_ms')}"
        )
        if payload.get("failed_stages"):
            typer.echo(f"Failed stages: {', '.join(payload['failed_stages'])}", err=True)
    if not payload.get("ok"):
        raise typer.Exit(code=1)


# ── index status ────────────────────────────────────────────────────────────


def build_index_status(ctx: typer.Context, cfg: Any) -> dict[str, Any]:
    """The three-state readiness report (ingested / indexed / retrievable).

    Thin CLI wrapper: the report itself lives in
    :mod:`drbrain.services.index_report`, so the WebUI reads the same numbers
    through ``app/service.py`` instead of growing a second implementation of
    "is this searchable".
    """
    return _core_index_status(cfg, tree_storage=_tree_storage_root(ctx, cfg))


def _emit_status(report: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps(redact_sensitive(report), indent=2, ensure_ascii=False, default=str))
        return
    typer.echo(f"Index status: {report['status']} (engine: {report['engine']})")
    states = report["states"]
    typer.echo(
        f"  Ingested:    {'yes' if states['ingested']['ready'] else 'no'}"
        f" — {states['ingested']['papers']} papers,"
        f" {states['ingested']['blocks']} blocks, documents {states['ingested']['documents']}"
    )
    typer.echo(f"  Indexed:     {'yes' if states['indexed']['ready'] else 'no'}")
    typer.echo(
        f"  Retrievable: {'yes' if states['retrievable']['ready'] else 'no'}"
        f" (generation: {report['generation'] or 'none'})"
    )
    for name, leg in report["legs"].items():
        mark = "ready" if leg["ready"] else "not ready"
        detail = ", ".join(
            f"{key}={value}"
            for key, value in leg.items()
            if key
            in {
                "documents",
                "indexed",
                "ready_count",
                "nodes",
                "pending",
                "generation",
                "leaves",
                "regions",
                "leaves_missing_parent",
            }
            and value is not None
        )
        typer.echo(f"  [{name}] {mark} ({detail})")
        for reason in leg["reasons"]:
            typer.echo(f"      reason: {reason}")
    if report["reasons"]:
        typer.echo("Reasons:")
        for reason in report["reasons"]:
            typer.echo(f"  - {reason}")


@index_app.command("status")
def index_status_cmd(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
) -> None:
    """Report ingested / indexed / retrievable state per retrieval leg.

    Read-only: it never embeds, builds or writes.  ``ready`` for a leg means the
    next ``search``/``ask`` can read it as-is; ``pending`` counts the work
    ``index build`` would still do.  Exit code 0 — the report itself is the
    result.
    """
    cfg = ctx.obj["config"]
    report = build_index_status(ctx, cfg)
    _emit_status(report, json_output=bool(_runtime_option(json_output, False)))


# ── index verify ────────────────────────────────────────────────────────────


def build_index_verify(ctx: typer.Context, cfg: Any) -> dict[str, Any]:
    """Re-check exactly what ``search``/``ask`` read (same code path as the WebUI).

    Thin CLI wrapper around :mod:`drbrain.services.index_report`.
    """
    return _core_index_verify(cfg, tree_storage=_tree_storage_root(ctx, cfg))


@index_app.command("verify")
def index_verify_cmd(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
) -> None:
    """Verify what ``search``/``ask`` read: FTS, vectors, membership, generation.

    Read-only.  Exit 1 when any check reports an error, 0 otherwise.  The broad
    storage-integrity audit remains ``drbrain storage audit``; this command only
    covers retrieval readiness and never duplicates that audit.
    """
    cfg = ctx.obj["config"]
    report = build_index_verify(ctx, cfg)
    if bool(_runtime_option(json_output, False)):
        typer.echo(json.dumps(redact_sensitive(report), indent=2, ensure_ascii=False, default=str))
    else:
        typer.echo(
            f"Index verify: {'ok' if report['ok'] else 'failed'}"
            f" (generation: {report['generation'] or 'none'})"
        )
        for check in report["checks"]:
            if check["ok"]:
                mark = "ok"
            else:
                mark = "warn" if check.get("severity") == "warning" else "FAIL"
            suffix = f" — {check['reason']}" if check["reason"] else ""
            typer.echo(f"  [{mark}] {check['name']}{suffix}")
        for warning in report["warnings"]:
            typer.echo(f"  [warn] {warning}", err=True)
    if not report["ok"]:
        raise typer.Exit(code=1)


__all__ = ["build_index_status", "build_index_verify", "index_app", "rebuild_lexical_index"]
