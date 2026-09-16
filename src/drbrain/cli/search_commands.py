"""``drbrain search`` — three-leg evidence retrieval without answer synthesis.

Main line: ``ingest → index build → search / ask``.  ``search`` runs the exact
retrieval chain ``ask`` uses (:func:`drbrain.rag.retrieval.retrieve_documents`)
and stops before synthesis: it returns the evidence rows with their source,
text locator, route and index generation, and never asks a chat model for an
answer.

``--paper`` scopes *every* leg to the given papers: the SQL legs filter on
``filters={"paper_ids": [...]}`` and the unified tree leg receives the same
``local_ids`` so its ANN entry search cannot silently read another paper.
``--source`` adds the configured external source (arXiv) as rows marked
``source="arxiv"`` that carry url/doi/arxiv_id and deliberately no local
locator — external hits are not local evidence.
"""

from __future__ import annotations

import json
from typing import Any

import typer

from drbrain.security import redact_sensitive
from drbrain.services.evidence_search import run_evidence_search

_SOURCES = ("local", "arxiv", "all")


def _option(value: Any, default: Any) -> Any:
    """Normalize a Typer ``OptionInfo`` for direct (non-CLI) callers."""
    if isinstance(value, typer.models.OptionInfo):
        return default if value.default is None else value.default
    return value


def _render_search(payload: dict[str, Any]) -> None:
    """Plain-text evidence listing (sources, locators, route, generation)."""
    route = payload["route"]
    typer.echo(f'Search: "{payload["query"]}" — {len(payload["evidence"])} evidence rows')
    typer.echo(
        f"  engine: {payload['engine']}  route: {','.join(route['legs']) or '(none)'}"
        f"  generation: {payload['generations']['result'] or 'none'}"
    )
    for leg in payload["legs"]:
        detail = f"{leg['count']} rows" if leg["status"] == "ok" else leg["status"]
        suffix = f" — {leg['reason']}" if leg["reason"] else ""
        typer.echo(f"  [{leg['source']}] {detail}{suffix}")
    for index, row in enumerate(payload["evidence"], 1):
        score = row.get("score")
        score_str = f"{score:.4f}" if isinstance(score, int | float) else "n/a"
        locator = row.get("node_id") or row.get("paper_id") or row.get("url") or ""
        typer.echo(
            f"  {index}. [{row.get('source', '')}] {row.get('title') or locator}"
            f" (score: {score_str}, paper: {row.get('paper_id', '')}, node: {locator})"
        )
        offsets = " ".join(
            f"{key}={row[key]}"
            for key in ("block_id", "char_start", "char_end")
            if row.get(key) not in (None, "")
        )
        if offsets:
            typer.echo(f"       {offsets}")
        text = str(row.get("text") or "").strip().replace("\n", " ")
        if text:
            typer.echo(f"       {text[:200]}")


def search_cmd(
    ctx: typer.Context,
    query: list[str] = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum evidence rows"),
    paper: list[str] = typer.Option(
        None,
        "--paper",
        help="Restrict local evidence to paper local_id (repeatable: --paper A --paper B)",
    ),
    source: str = typer.Option(
        "local",
        "--source",
        help="Evidence source: local | arxiv | all (external rows carry no local locator)",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """Retrieve evidence over the same chain as ``ask`` — without an answer.

    Returns the sources, text locators, route and index version of every
    retrieved row (bm25/vector/tree by default).  ``--paper`` scopes every leg
    to the given papers, ``--source arxiv|all`` adds external rows, and a
    missing index is reported with the command that prepares it.

    The payload itself is built by :mod:`drbrain.services.evidence_search`, so
    the WebUI retrieves evidence through the same code path.
    """
    _limit = int(_option(limit, 10))
    _json = bool(_option(json_output, False))
    _paper = [str(item) for item in (paper or [])]
    _source = str(_option(source, "local") or "local").strip().lower()
    if _source not in _SOURCES:
        raise typer.BadParameter(
            f"unknown source {_source!r}; expected one of {list(_SOURCES)}",
            param_hint="--source",
        )
    text = " ".join(str(item) for item in query).strip()
    if not text:
        raise typer.BadParameter("search requires a nonempty query", param_hint="QUERY")

    payload = run_evidence_search(
        ctx.obj["config"], text, limit=_limit, paper_ids=_paper or None, source=_source
    )
    if payload.get("status") == "source_unavailable":
        reason = next(
            (
                str(leg.get("reason") or "")
                for leg in payload.get("legs") or []
                if leg.get("reason")
            ),
            "",
        )
        typer.echo(f"[search] local evidence unavailable: {reason}", err=True)
        typer.echo("[search] prepare it with: drbrain index build", err=True)
    if _json:
        typer.echo(json.dumps(redact_sensitive(payload), indent=2, ensure_ascii=False, default=str))
    else:
        _render_search(payload)
    if payload.get("status") == "source_unavailable":
        raise typer.Exit(1)


__all__ = ["search_cmd"]
