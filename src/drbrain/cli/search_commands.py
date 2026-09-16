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
import time
from typing import Any

import typer

from drbrain.cli._common import open_db
from drbrain.security import redact_sensitive, safe_error

_SOURCES = ("local", "arxiv", "all")


def _option(value: Any, default: Any) -> Any:
    """Normalize a Typer ``OptionInfo`` for direct (non-CLI) callers."""
    if isinstance(value, typer.models.OptionInfo):
        return default if value.default is None else value.default
    return value


def _local_identity_sets(cfg: Any) -> tuple[set[str], set[str]]:
    """Lowercased DOIs / normalized arXiv ids already in the local library."""
    from pathlib import Path

    from drbrain.services.fsearch import _normalize_arxiv_ref

    db_path = cfg["db"]["path"]
    if not Path(db_path).exists():
        return set(), set()
    from drbrain.storage.connection import connect_wal

    conn = connect_wal(db_path)
    try:
        rows = conn.execute(
            "SELECT doi, arxiv FROM paper_ids WHERE doi != '' OR arxiv != ''"
        ).fetchall()
    finally:
        conn.close()
    dois: set[str] = set()
    arxiv_ids: set[str] = set()
    for doi, arxiv in rows:
        if doi:
            dois.add(str(doi).lower())
        if arxiv:
            arxiv_ids.add(_normalize_arxiv_ref(arxiv))
    return dois, arxiv_ids


def _arxiv_rows(cfg: Any, query: str, limit: int) -> list[dict[str, Any]]:
    """External source rows in the shared evidence shape (no local locator)."""
    from drbrain.services.fsearch import _merge_with_local_status, search_arxiv

    results = search_arxiv(query, max_results=max(1, int(limit)))
    dois, arxiv_ids = _local_identity_sets(cfg)
    annotated = _merge_with_local_status(results, dois, arxiv_ids)
    rows: list[dict[str, Any]] = []
    for item in annotated:
        arxiv_id = str(item.get("arxiv_id") or "")
        rows.append(
            {
                "source": "arxiv",
                "title": str(item.get("title") or ""),
                "text": str(item.get("summary") or "")[:500],
                "score": None,
                "authors": list(item.get("authors") or []),
                "year": item.get("year"),
                "doi": str(item.get("doi") or ""),
                "arxiv_id": arxiv_id,
                "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                "ingested": bool(item.get("ingested")),
            }
        )
    return rows


def _route_info(cfg: Any) -> dict[str, Any]:
    """The normalized retrieval route the search will run (T43 registry)."""
    from drbrain.rag.config import get_llamaindex_config
    from drbrain.rag.legs import normalize_legs

    li = get_llamaindex_config(cfg)
    requested = [str(item) for item in (li.retrievers or [])]
    try:
        normalized = normalize_legs(requested)
    except Exception as exc:  # noqa: BLE001 - an invalid route is reported, not hidden
        return {
            "requested": requested,
            "legs": [],
            "extras": [],
            "notes": [safe_error(exc)],
        }
    return {
        "requested": requested,
        "legs": list(normalized.legs),
        "extras": list(normalized.extras),
        "notes": list(normalized.notes),
    }


def _generations(cfg: Any, result: Any) -> dict[str, Any]:
    """Index versions this answer must be read against (design §status)."""
    generations: dict[str, Any] = {
        "result": str(getattr(result, "generation", "") or "") or None,
        "tree": None,
        "sql": None,
    }
    try:
        from drbrain.tree.leg import active_tree_generation

        generations["tree"] = active_tree_generation(cfg)
    except Exception:  # noqa: BLE001 - version reporting stays best-effort
        pass
    try:
        from drbrain.rag.index_generations import get_active_index_generation

        generations["sql"] = get_active_index_generation(cfg)
    except Exception:  # noqa: BLE001 - version reporting stays best-effort
        pass
    return generations


def _local_evidence(
    cfg: Any, query: str, limit: int, paper_ids: list[str] | None
) -> tuple[Any, dict[str, Any], str]:
    """Run the ask retrieval chain and return ``(rows, route, error)``."""
    from drbrain.rag.config import coerce_config
    from drbrain.rag.retrieval import retrieve_documents

    route = _route_info(cfg)
    filters = {"paper_ids": list(paper_ids)} if paper_ids else None
    typed = coerce_config(cfg)
    graph = None
    if "graph" in route["extras"]:
        # The graph leg is an explicit extra; only then is the KG loaded.
        from drbrain.graph.engine import GraphEngine

        with open_db(cfg) as db:
            graph = GraphEngine()
            graph.load_from_db(db)
            rows = retrieve_documents(typed, db, graph, query, filters=filters, top_k=int(limit))
        return rows, route, ""
    with open_db(cfg) as db:
        rows = retrieve_documents(typed, db, graph, query, filters=filters, top_k=int(limit))
    return rows, route, ""


def _legs_json(rows: Any) -> list[dict[str, Any]]:
    return [
        {
            "source": str(leg.source),
            "status": str(leg.status),
            "count": int(leg.count),
            "duration_ms": round(float(leg.duration_ms), 3),
            "reason": str(leg.reason),
        }
        for leg in (getattr(rows.result, "legs", None) or [])
    ]


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
    """
    _limit = int(_option(limit, 10))
    _json = bool(_option(json_output, False))
    _paper = list(paper or [])
    _source = str(_option(source, "local") or "local").strip().lower()
    if _source not in _SOURCES:
        raise typer.BadParameter(
            f"unknown source {_source!r}; expected one of {list(_SOURCES)}",
            param_hint="--source",
        )
    text = " ".join(str(item) for item in query).strip()
    if not text:
        raise typer.BadParameter("search requires a nonempty query", param_hint="QUERY")

    cfg = ctx.obj["config"]
    from drbrain.rag.config import get_llamaindex_config

    local_requested = _source in ("local", "all")
    external_requested = _source in ("arxiv", "all")

    evidence: list[dict[str, Any]] = []
    legs: list[dict[str, Any]] = []
    route = _route_info(cfg)
    generations: dict[str, Any] = {"result": None, "tree": None, "sql": None}
    local_error = ""
    if local_requested:
        started = time.monotonic()
        try:
            rows, route, local_error = _local_evidence(cfg, text, _limit, _paper or None)
        except Exception as exc:  # noqa: BLE001 - report the remedy instead of a traceback
            local_error = safe_error(exc)
            rows = None
        if local_error:
            legs.append(
                {
                    "source": "local",
                    "status": "unavailable",
                    "count": 0,
                    "duration_ms": round((time.monotonic() - started) * 1000, 3),
                    "reason": local_error,
                }
            )
        else:
            legs = _legs_json(rows)
            generations = _generations(cfg, rows.result)
            evidence.extend(dict(row) for row in rows)
    if external_requested:
        started = time.monotonic()
        try:
            arxiv_rows = _arxiv_rows(cfg, text, _limit)
            arxiv_error = ""
        except Exception as exc:  # noqa: BLE001 - a failed provider is a reported state
            arxiv_rows, arxiv_error = [], safe_error(exc)
        legs.append(
            {
                "source": "arxiv",
                "status": "unavailable" if arxiv_error else ("ok" if arxiv_rows else "empty"),
                "count": len(arxiv_rows),
                "duration_ms": round((time.monotonic() - started) * 1000, 3),
                "reason": arxiv_error or ("" if arxiv_rows else "no_results_or_unreachable"),
            }
        )
        evidence.extend(arxiv_rows)

    payload = {
        "query": text,
        "engine": str(getattr(get_llamaindex_config(cfg), "rag_engine", "llamaindex") or ""),
        "source": _source,
        "route": route,
        "generations": generations,
        "legs": legs,
        "evidence": evidence,
    }
    if local_error:
        typer.echo(f"[search] local evidence unavailable: {local_error}", err=True)
        typer.echo("[search] prepare it with: drbrain index build", err=True)
    if _json:
        typer.echo(json.dumps(redact_sensitive(payload), indent=2, ensure_ascii=False, default=str))
    else:
        _render_search(payload)
    if local_error:
        raise typer.Exit(1)


__all__ = ["search_cmd"]
