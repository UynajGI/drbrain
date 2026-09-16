"""Evidence retrieval shared by the CLI and the WebUI (04-arch A2).

``drbrain search`` runs the retrieval chain ``ask`` uses and stops before
synthesis, returning evidence rows with their source, text locator, route and
index generation.  The WebUI needs exactly those rows, so the payload builder
lives here and both callers adapt:

* the CLI keeps its option parsing, rendering and exit codes;
* ``app/service.py`` adds project scope and the display cap.

A missing index is a *reported state* (``status="source_unavailable"``) with the
command that prepares it — never an exception for the caller to guess at.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from drbrain.security import safe_error
from drbrain.storage.database import Database

SOURCES = ("local", "arxiv", "all")

#: Display cap for the WebUI (FR-S8: "top N", the retrieval layer has no total).
MAX_EVIDENCE_LIMIT = 100
DEFAULT_EVIDENCE_LIMIT = 20


@contextmanager
def _open_db(cfg: Any) -> Iterator[Any]:
    """Open the corpus database with the CLI's historical resolution."""
    db = Database(str(cfg["db"]["path"]))
    try:
        yield db
    finally:
        db.close()


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
    cfg: Any, query: str, limit: int, paper_ids: Sequence[str] | None
) -> tuple[Any, dict[str, Any], str]:
    """Run the ask retrieval chain and return ``(rows, route, error)``.

    ``paper_ids`` distinguishes "no restriction" (``None``) from "an explicit,
    possibly empty scope" (a sequence): an empty sequence is an empty filter —
    it must never be coerced into ``None``, which would read the whole corpus.
    """
    from drbrain.rag.config import coerce_config
    from drbrain.rag.retrieval import retrieve_documents

    route = _route_info(cfg)
    filters = {"paper_ids": [str(item) for item in paper_ids]} if paper_ids is not None else None
    typed = coerce_config(cfg)
    graph = None
    if "graph" in route["extras"]:
        # The graph leg is an explicit extra; only then is the KG loaded.
        from drbrain.graph.engine import GraphEngine

        with _open_db(cfg) as db:
            graph = GraphEngine()
            graph.load_from_db(db)
            rows = retrieve_documents(typed, db, graph, query, filters=filters, top_k=int(limit))
        return rows, route, ""
    with _open_db(cfg) as db:
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


def run_evidence_search(
    cfg: Any,
    query: str,
    *,
    limit: int = DEFAULT_EVIDENCE_LIMIT,
    paper_ids: Sequence[str] | None = None,
    source: str = "local",
) -> dict[str, Any]:
    """Build the ``drbrain search`` payload (same keys, same meanings).

    ``status`` is the structured outcome the WebUI renders: ``ok`` (rows
    found), ``empty`` (nothing matched), ``source_unavailable`` (a requested
    source could not run at all) or ``degraded`` (local rows fine, the external
    source failed).  ``hint`` says what to do about the non-ok ones.

    ``paper_ids=None`` means "no scope restriction"; an explicit sequence is
    enforced as a filter, and an *empty* sequence means "nothing is in scope",
    which returns immediately instead of being widened to the whole corpus.
    """
    text = str(query or "").strip()
    if not text:
        return {
            "query": "",
            "status": "empty_question",
            "hint": "请输入要检索的内容。",
            "engine": "",
            "source": source,
            "route": {},
            "generations": {"result": None, "tree": None, "sql": None},
            "legs": [],
            "evidence": [],
        }
    selected = str(source or "local").strip().lower()
    if selected not in SOURCES:
        selected = "local"
    from drbrain.rag.config import get_llamaindex_config

    if paper_ids is not None and not list(paper_ids):
        # An explicit empty scope is an empty result, never "no filter": the
        # request already says nothing is readable, so no retrieval runs.
        return {
            "query": text,
            "status": "empty",
            "hint": "检索范围为空：没有可检索的论文（项目里还没有论文，或所选论文不在项目内）。",
            "engine": str(getattr(get_llamaindex_config(cfg), "rag_engine", "llamaindex") or ""),
            "source": selected,
            "route": _route_info(cfg),
            "generations": {"result": None, "tree": None, "sql": None},
            "legs": [],
            "evidence": [],
            "scope_empty": True,
        }
    local_requested = selected in ("local", "all")
    external_requested = selected in ("arxiv", "all")

    evidence: list[dict[str, Any]] = []
    legs: list[dict[str, Any]] = []
    route = _route_info(cfg)
    generations: dict[str, Any] = {"result": None, "tree": None, "sql": None}
    local_error = ""
    if local_requested:
        started = time.monotonic()
        try:
            rows, route, local_error = _local_evidence(cfg, text, int(limit), paper_ids)
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
    external_error = ""
    if external_requested:
        started = time.monotonic()
        try:
            arxiv_rows = _arxiv_rows(cfg, text, int(limit))
        except Exception as exc:  # noqa: BLE001 - a failed provider is a reported state
            arxiv_rows, external_error = [], safe_error(exc)
        legs.append(
            {
                "source": "arxiv",
                "status": "unavailable" if external_error else ("ok" if arxiv_rows else "empty"),
                "count": len(arxiv_rows),
                "duration_ms": round((time.monotonic() - started) * 1000, 3),
                "reason": external_error or ("" if arxiv_rows else "no_results_or_unreachable"),
            }
        )
        evidence.extend(arxiv_rows)

    if local_error:
        status, hint = "source_unavailable", "运行 drbrain index build 准备检索索引。"
    elif evidence:
        status = "degraded" if external_error else "ok"
        hint = "外部来源本次不可用，下面是本地证据。" if external_error else ""
    else:
        status, hint = "empty", "换关键词，或先在索引页确认索引是否就绪。"
    return {
        "query": text,
        "status": status,
        "hint": hint,
        "engine": str(getattr(get_llamaindex_config(cfg), "rag_engine", "llamaindex") or ""),
        "source": selected,
        "route": route,
        "generations": generations,
        "legs": legs,
        "evidence": evidence,
    }


__all__ = [
    "DEFAULT_EVIDENCE_LIMIT",
    "MAX_EVIDENCE_LIMIT",
    "SOURCES",
    "run_evidence_search",
]
