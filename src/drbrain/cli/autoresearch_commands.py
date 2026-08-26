"""Operator entry points for durable autoresearch runs."""

from __future__ import annotations

import json
from typing import Any

import typer

from drbrain.cli._common import open_db
from drbrain.config import AutoresearchConfig
from drbrain.loop import ResearchDirector

autoresearch_app = typer.Typer(help="Durable autoresearch operations")


def _settings(cfg: Any) -> AutoresearchConfig:
    """Read typed settings while accepting legacy dict-like CLI test config."""
    raw = cfg.get("autoresearch", {})
    if isinstance(raw, AutoresearchConfig):
        return raw
    return AutoresearchConfig(**raw) if isinstance(raw, dict) else AutoresearchConfig()


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
    settings = _settings(cfg)
    if not settings.enabled:
        typer.echo(
            "[autoresearch] disabled: set `autoresearch.enabled: true` in config.yaml",
            err=True,
        )
        raise typer.Exit(1)

    effective_max_cycles = settings.max_cycles if max_cycles is None else max_cycles
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
                require_rag_evidence=settings.require_rag_evidence,
            )
            state = director.run_sync(
                topic,
                max_cycles=effective_max_cycles,
                stagnation_cycles=settings.stagnation_cycles,
                max_adaptations=settings.max_adaptations,
            )
    except Exception as exc:  # noqa: BLE001 - CLI reports the durable-run failure
        typer.echo(f"[autoresearch] run failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    summary = {
        "topic": state.get("topic", topic),
        "cycles": state.get("cycles", 0),
        "champion": state.get("champion", []),
        "rejected": state.get("rejected", []),
        "workspace": settings.run_dir,
    }
    if json_output:
        typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    typer.echo(
        f"Autoresearch returned: topic={summary['topic']!r}; cycles={summary['cycles']}; "
        f"champion={len(summary['champion'])}; workspace={summary['workspace']}"
    )
