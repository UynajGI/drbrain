"""Operator entry points for durable autoresearch runs."""

from __future__ import annotations

import json
from typing import Any

import typer

from drbrain.cli._common import open_db
from drbrain.config import AutoresearchConfig
from drbrain.loop import ResearchDirector
from drbrain.loop.policy import ToolPolicy

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
