"""Operator entry points for durable autoresearch runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NoReturn

import typer

from drbrain.cli._common import open_db
from drbrain.config import AutoresearchConfig
from drbrain.loop import ResearchDirector, RunGovernance
from drbrain.loop.policy import ToolPolicy
from drbrain.loop.store import RunLedger

autoresearch_app = typer.Typer(help="Durable autoresearch operations")


def _settings(cfg: Any) -> AutoresearchConfig:
    """Read typed settings while accepting legacy dict-like CLI test config."""
    raw = cfg.get("autoresearch", {})
    if isinstance(raw, AutoresearchConfig):
        settings = raw
    elif isinstance(raw, dict):
        try:
            settings = AutoresearchConfig(**raw)
        except TypeError as exc:
            raise ValueError(f"invalid autoresearch settings: {exc}") from exc
    else:
        raise ValueError("autoresearch settings must be a mapping")
    if not isinstance(settings.mcp_servers, list):
        raise ValueError("autoresearch.mcp_servers must be a list")
    if not isinstance(settings.step_capabilities, dict):
        raise ValueError("autoresearch.step_capabilities must be a mapping")
    if not isinstance(settings.budget, dict):
        raise ValueError("autoresearch.budget must be a mapping")
    return settings


def _control(cfg: Any) -> RunGovernance:
    """Open an existing ledger without creating a misleading empty operator view."""
    settings = _settings(cfg)
    ledger_path = Path(settings.run_dir) / "ledger.sqlite3"
    if not ledger_path.is_file():
        raise FileNotFoundError(f"autoresearch ledger does not exist: {ledger_path}")
    return RunGovernance(RunLedger(ledger_path))


def _emit_operator_result(payload: dict[str, Any], *, json_output: bool, label: str) -> None:
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    typer.echo(
        f"Autoresearch {label}: topic={payload['topic']!r}; status={payload['status']}; "
        f"manual_review_steps={len(payload['manual_review_steps'])}"
    )


def _operator_error(exc: Exception) -> NoReturn:
    typer.echo(f"[autoresearch] operator command failed ({type(exc).__name__}): {exc}", err=True)
    raise typer.Exit(1) from exc


@autoresearch_app.command("run")
def run_cmd(
    ctx: typer.Context,
    topic: str = typer.Argument(..., help="Research topic; repeating it resumes its durable run"),
    max_cycles: int | None = typer.Option(
        None, "--max-cycles", help="Override autoresearch.max_cycles for this run"
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit the run summary as JSON"),
) -> None:
    """Run or resume the durable autoresearch loop for one topic."""
    cfg = ctx.obj["config"]
    try:
        settings = _settings(cfg)
    except ValueError as exc:
        typer.echo(f"[autoresearch] invalid config: {exc}", err=True)
        raise typer.Exit(1) from exc
    if not settings.enabled:
        typer.echo(
            "[autoresearch] disabled: set `autoresearch.enabled: true` in config.yaml",
            err=True,
        )
        raise typer.Exit(1)

    effective_max_cycles = settings.max_cycles if max_cycles is None else max_cycles
    tool_policy = (
        ToolPolicy(step_capabilities=settings.step_capabilities)
        if settings.plugins_dir or settings.mcp_servers
        else None
    )
    try:
        with open_db(cfg) as db:
            director = ResearchDirector(
                cfg,
                db=db,
                plugins_dir=settings.plugins_dir or None,
                mcp_servers=settings.mcp_servers,
                run_dir=settings.run_dir,
                n_critics=settings.n_critics,
                lease_seconds=settings.lease_seconds,
                tool_policy=tool_policy,
                require_rag_evidence=settings.require_rag_evidence,
            )
            state = director.run_sync(
                topic,
                max_cycles=effective_max_cycles,
                stagnation_cycles=settings.stagnation_cycles,
                max_adaptations=settings.max_adaptations,
                budget=dict(settings.budget),
            )
    except Exception as exc:  # noqa: BLE001 - CLI reports the durable-run failure
        typer.echo(f"[autoresearch] durable run failed ({type(exc).__name__}): {exc}", err=True)
        raise typer.Exit(1) from exc

    summary = {
        "topic": state.get("topic", topic),
        "cycles": state.get("cycles", 0),
        "champion": state.get("champion", []),
        "rejected": state.get("rejected", []),
        "workspace": settings.run_dir,
        "budget": dict(settings.budget),
    }
    if json_output:
        typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    typer.echo(
        f"Autoresearch returned: topic={summary['topic']!r}; cycles={summary['cycles']}; "
        f"champion={len(summary['champion'])}; workspace={summary['workspace']}; "
        f"budget={summary['budget']}"
    )


@autoresearch_app.command("status")
def status_cmd(
    ctx: typer.Context,
    identifier: str = typer.Argument(..., help="Durable run ID or topic"),
    json_output: bool = typer.Option(False, "--json", help="Emit the status as JSON"),
) -> None:
    """Inspect a durable run without starting it."""
    try:
        payload = _control(ctx.obj["config"]).status(identifier)
    except Exception as exc:  # noqa: BLE001 - CLI reports operator errors consistently
        _operator_error(exc)
    _emit_operator_result(payload, json_output=json_output, label="status")


@autoresearch_app.command("pause")
def pause_cmd(
    ctx: typer.Context,
    identifier: str = typer.Argument(..., help="Durable run ID or topic"),
    reason: str = typer.Option("operator_pause", "--reason", help="Recorded pause reason"),
    json_output: bool = typer.Option(False, "--json", help="Emit the status as JSON"),
) -> None:
    """Pause a run without deleting evidence or artifacts."""
    try:
        payload = _control(ctx.obj["config"]).pause(identifier, reason=reason)
    except Exception as exc:  # noqa: BLE001 - CLI reports operator errors consistently
        _operator_error(exc)
    _emit_operator_result(payload, json_output=json_output, label="pause")


@autoresearch_app.command("cancel")
def cancel_cmd(
    ctx: typer.Context,
    identifier: str = typer.Argument(..., help="Durable run ID or topic"),
    reason: str = typer.Option("operator_cancel", "--reason", help="Recorded cancellation reason"),
    json_output: bool = typer.Option(False, "--json", help="Emit the status as JSON"),
) -> None:
    """Cancel a run and revoke future durable work."""
    try:
        payload = _control(ctx.obj["config"]).cancel(identifier, reason=reason)
    except Exception as exc:  # noqa: BLE001 - CLI reports operator errors consistently
        _operator_error(exc)
    _emit_operator_result(payload, json_output=json_output, label="cancel")


@autoresearch_app.command("trace")
def trace_cmd(
    ctx: typer.Context,
    identifier: str = typer.Argument(..., help="Durable run ID or topic"),
) -> None:
    """Print the immutable event and tool-call trace as JSON."""
    try:
        payload = _control(ctx.obj["config"]).trace(identifier)
    except Exception as exc:  # noqa: BLE001 - CLI reports operator errors consistently
        _operator_error(exc)
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@autoresearch_app.command("audit")
def audit_cmd(
    ctx: typer.Context,
    identifier: str = typer.Argument(..., help="Durable run ID or topic"),
) -> None:
    """Print the aggregate audit summary as JSON."""
    try:
        payload = _control(ctx.obj["config"]).audit_summary(identifier)
    except Exception as exc:  # noqa: BLE001 - CLI reports operator errors consistently
        _operator_error(exc)
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@autoresearch_app.command("resolve-manual-review")
def resolve_manual_review_cmd(
    ctx: typer.Context,
    identifier: str = typer.Argument(..., help="Durable run ID or topic"),
    step_id: str = typer.Option(..., "--step", help="Manual-review step to abandon"),
    reason: str = typer.Option(..., "--reason", help="Recorded operator disposition"),
    json_output: bool = typer.Option(False, "--json", help="Emit the status as JSON"),
) -> None:
    """Abandon one reviewed step; run again explicitly to begin a new cycle."""
    try:
        payload = _control(ctx.obj["config"]).resolve_manual_review(
            identifier, step_id=step_id, reason=reason
        )
    except Exception as exc:  # noqa: BLE001 - CLI reports operator errors consistently
        _operator_error(exc)
    _emit_operator_result(payload, json_output=json_output, label="manual-review resolution")
