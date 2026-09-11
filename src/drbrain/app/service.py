"""Service layer for the DrBrain WebUI.

Every function here wraps a capability that already exists behind the CLI
(``drbrain stats`` / ``search`` / ``ask`` / ``autoresearch`` / ``session``)
and returns plain JSON-serialisable data.  The HTTP layer is a thin router
over these functions, so they can be unit-tested without a socket.

Since M0 the service also owns the *request scope*: a project (stable
``project_id``) owns a corpus reference, conversation sessions and research
runs.  Callers pass an already-resolved ``project_id``/``session_id``; the
service verifies ownership before returning details, events, evidence or
exports, and it never mutates process-wide configuration for a request.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from drbrain.config import AutoresearchConfig
from drbrain.projects import (
    DEFAULT_PROJECT_ID,
    UNBOUND_SESSION_LABEL,
    new_project_id,
    normalize_project_id,
)
from drbrain.runtime import RuntimeContext
from drbrain.security import (
    configured_secret_values,
    redact_sensitive,
    redact_sensitive_text,
    safe_error,
)
from drbrain.storage.database import Database
from drbrain.storage.inbox import first_symlink_component

_LEDGER_FILE = "ledger.sqlite3"
MAX_PAGE_SIZE = 100
DEFAULT_PAGE_SIZE = 20


class ProjectNotFoundError(ValueError):
    """Raised when a request references an unknown project."""


class SessionNotFoundError(ValueError):
    """Raised when a request references an unknown conversation session."""


class RunNotFoundError(ValueError):
    """Raised when a request references an unknown or out-of-scope run."""


class PaperNotInProjectError(ValueError):
    """Raised when a paper is requested outside the current project scope."""


@dataclass(frozen=True)
class Scope:
    """A resolved request scope (never a process-wide configuration change)."""

    project_id: str = DEFAULT_PROJECT_ID
    session_id: str = ""

    @classmethod
    def resolve(cls, project_id: str | None = None, session_id: str | None = None) -> Scope:
        return cls(normalize_project_id(project_id), str(session_id or ""))


# ── config helpers ───────────────────────────────────────────────────────────


def autoresearch_settings(cfg: Any) -> AutoresearchConfig:
    """Return typed autoresearch settings from a Config or a plain dict."""
    raw = cfg.get("autoresearch", {}) if hasattr(cfg, "get") else {}
    if isinstance(raw, AutoresearchConfig):
        return raw
    if isinstance(raw, dict):
        return AutoresearchConfig(**raw)
    raise ValueError("autoresearch settings must be a mapping")


def _active_runtime() -> RuntimeContext | None:
    """Return the selected runtime, preserving fail-closed empty semantics."""
    if "DRBRAIN_ROOT" not in os.environ and "DRBRAIN_RUNTIME_ROOT" not in os.environ:
        return None
    return RuntimeContext.create()


def _runtime_path(value: str | Path, *, label: str) -> Path:
    """Resolve a service-owned path against the active runtime namespace.

    CLI callers pass an already-normalized config, but the service is also a
    public Python API.  Apply the same environment boundary for raw configs so
    a WebUI started in a data-only root cannot silently fall back to CWD.
    """
    raw = Path(value).expanduser()
    runtime = _active_runtime()
    if runtime is not None:
        lexical = raw if raw.is_absolute() else runtime.root / raw
    else:
        lexical = raw if raw.is_absolute() else Path.cwd() / raw
    link = first_symlink_component(lexical)
    if link is not None:
        raise ValueError(f"{label} must not contain a symlink: {link}")
    if runtime is not None:
        return runtime.assert_within_root(lexical, label=label)
    return lexical.resolve()


def ledger_path(cfg: Any) -> Path:
    return _runtime_path(
        Path(autoresearch_settings(cfg).run_dir) / _LEDGER_FILE,
        label="autoresearch ledger",
    )


def db_path(cfg: Any) -> str:
    value = cfg["db"]["path"]
    if str(value) == ":memory:":
        return ":memory:"
    return str(_runtime_path(value, label="database path"))


def workspace_root(cfg: Any) -> Path:
    """Return the workspace directory that the project switcher maps to.

    Mirrors ``drbrain.storage.workspace._default_workspace_root`` but keeps
    the runtime symlink/containment policy of the service layer.
    """
    return _runtime_path(Path("workspace"), label="workspace root")


@contextmanager
def _db(cfg: Any) -> Iterator[Database]:
    db = Database(db_path(cfg))
    try:
        yield db
    finally:
        db.close()


@contextmanager
def _ledger(cfg: Any) -> Iterator[sqlite3.Connection | None]:
    """Read-only connection to the autoresearch ledger, or ``None`` if absent."""
    path = ledger_path(cfg)
    # Never read a symlinked ledger: this endpoint is read-only, but following
    # an alias could disclose another worktree's research history.
    if path.is_symlink() or not path.is_file():
        yield None
        return
    # ``Path.as_uri`` escapes spaces and query characters before SQLite parses
    # the URI; string interpolation of a raw path can otherwise alter mode or
    # the database filename.
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _loads(value: Any, default: Any) -> Any:
    if not value:
        return default
    try:
        return redact_sensitive(json.loads(value))
    except (TypeError, ValueError):
        return default


def _count(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> int:
    try:
        row = conn.execute(sql, params).fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row[0]) if row and row[0] is not None else 0


# ── projects (M0) ────────────────────────────────────────────────────────────


def sync_workspace_projects(cfg: Any, db: Database) -> None:
    """Register existing workspaces as projects (idempotent).

    A workspace is the corpus reference a project points at.  The project id
    is generated once and stays stable when the workspace is renamed; only
    new workspace directories create rows here.
    """
    from drbrain.storage import workspace as workspace_store

    try:
        root = workspace_root(cfg)
    except (OSError, ValueError):
        return
    for name in workspace_store.list_workspaces(root):
        if db.find_project_by_workspace(name) is not None:
            continue
        info = workspace_store.get_workspace(name, root) or {}
        db.upsert_project(
            new_project_id(),
            name=name,
            description=str(info.get("description") or ""),
            workspace_name=name,
        )


def projects(cfg: Any, *, with_counts: bool = False) -> list[dict[str, Any]]:
    """List projects for the switcher; default project first."""
    with _db(cfg) as db:
        sync_workspace_projects(cfg, db)
        rows = db.list_projects()
        if with_counts:
            for row in rows:
                paper_ids = project_paper_ids(cfg, row["project_id"], db=db)
                row["papers"] = (
                    len(paper_ids)
                    if paper_ids is not None
                    else int(db.get_stats().get("papers", 0))
                )
                row["sessions"] = db.count_agent_sessions(row["project_id"])
        return rows


def resolve_project(cfg: Any, project_id: str | None) -> dict[str, Any]:
    """Return the project row or raise :class:`ProjectNotFound`."""
    pid = normalize_project_id(project_id)
    with _db(cfg) as db:
        sync_workspace_projects(cfg, db)
        row = db.get_project(pid)
        if row is None:
            raise ProjectNotFoundError(f"unknown project: {pid}")
        return row


def project_paper_ids(
    cfg: Any, project_id: str | None, *, db: Database | None = None
) -> list[str] | None:
    """Return the project's paper IDs, or ``None`` for the default project.

    ``None`` means "no membership filter": the default project is the whole
    imported library.  A workspace-backed project maps to that workspace's
    paper references, so the library itself is never duplicated.
    """
    pid = normalize_project_id(project_id)
    owns_db = db is None
    if owns_db:
        db = Database(db_path(cfg))
    try:
        row = db.get_project(pid)
        if row is None:
            raise ProjectNotFoundError(f"unknown project: {pid}")
        workspace_name = row.get("workspace_name")
        if not workspace_name:
            return None
        from drbrain.storage import workspace as workspace_store

        try:
            root = workspace_root(cfg)
        except (OSError, ValueError):
            return []
        return list(workspace_store.load_workspace_papers(str(workspace_name), root))
    finally:
        if owns_db:
            db.close()


# ── dashboard ────────────────────────────────────────────────────────────────


def dashboard(cfg: Any, project_id: str | None = None) -> dict[str, Any]:
    """KPI counters for the workbench page (database + ledger + plugins)."""
    pid = normalize_project_id(project_id)
    stats: dict[str, Any] = {}
    path = db_path(cfg)
    project_row: dict[str, Any] | None = None
    recent_sessions: list[dict[str, Any]] = []
    if path == ":memory:" or Path(path).is_file():
        with _db(cfg) as db:
            sync_workspace_projects(cfg, db)
            project_row = db.get_project(pid)
            if project_row is None:
                raise ProjectNotFoundError(f"unknown project: {pid}")
            paper_ids = project_paper_ids(cfg, pid, db=db)
            stats = db.get_stats(paper_ids)
            recent_sessions = [
                s for s in db.list_agent_sessions(pid, limit=5) if s["messages"] > 1
            ][:3]
    ledger: dict[str, Any] = {"runs": 0, "settlements": 0, "verified": 0, "events": 0}
    with _ledger(cfg) as conn:
        if conn is not None:
            ledger = {
                "runs": _count(
                    conn,
                    "SELECT COUNT(*) FROM research_runs WHERE project_id = ?",
                    (pid,),
                ),
                "settlements": _count(
                    conn,
                    """
                    SELECT COUNT(*) FROM research_claim_settlements s
                    JOIN research_runs r ON r.run_id = s.run_id WHERE r.project_id = ?
                    """,
                    (pid,),
                ),
                "verified": _count(
                    conn,
                    """
                    SELECT COUNT(*) FROM research_claim_settlements s
                    JOIN research_runs r ON r.run_id = s.run_id
                    WHERE s.verdict='keep' AND r.project_id = ?
                    """,
                    (pid,),
                ),
                "events": _count(
                    conn,
                    """
                    SELECT COUNT(*) FROM research_events e
                    JOIN research_runs r ON r.run_id = e.run_id WHERE r.project_id = ?
                    """,
                    (pid,),
                ),
            }
    return {
        "papers": int(stats.get("papers", 0)),
        "concepts": int(stats.get("concepts", 0)),
        "edges": int(stats.get("edges", 0)),
        "arguments": int(stats.get("arguments", 0)),
        "uploaded": int(stats.get("uploaded", 0)),
        "project": project_row,
        "ledger": ledger,
        "plugins": len(plugins(cfg)),
        "recent_runs": runs(cfg, project_id=pid)[:5],
        "recent_sessions": recent_sessions,
        "availability": availability(cfg),
    }


def availability(cfg: Any) -> dict[str, Any]:
    """Feature availability flags used by the first-run guide and settings."""
    with _db(cfg) as db:
        llm_configured = bool(cfg.get("llm", {}).get("models")) if hasattr(cfg, "get") else False
        rag_enabled = False
        try:
            from drbrain.rag.engine import resolve_engine

            rag_enabled = resolve_engine(cfg, "llamaindex") == "llamaindex"
        except Exception:  # noqa: BLE001 - availability must never break the page
            rag_enabled = False
        settings = autoresearch_settings(cfg)
        stats = db.get_stats()
    return {
        "llm": llm_configured,
        "rag": rag_enabled,
        "autoresearch": bool(settings.enabled),
        "papers": int(stats.get("papers", 0)),
    }


# ── search / ask ─────────────────────────────────────────────────────────────


def search(
    cfg: Any,
    query: str,
    limit: int = 10,
    type_filter: str | None = None,
    project_id: str | None = None,
) -> list[dict]:
    """BM25 keyword search — same engine as ``drbrain search``.

    With a workspace-backed project the results are filtered to the project
    membership after ranking; the default project searches the whole library.
    """
    query = query.strip()
    if not query:
        return []
    from drbrain.query.bm25 import build_bm25_index

    pid = normalize_project_id(project_id)
    with _db(cfg) as db:
        paper_ids = project_paper_ids(cfg, pid, db=db)
        bm25 = build_bm25_index(db)
        # Over-fetch so a post-filter can still fill the requested page; the
        # cap keeps one request from materialising the whole corpus.
        fetch = min(max(int(limit) * 20, int(limit)), 2000) if paper_ids is not None else int(limit)
        results = bm25.search(query, type_filter=type_filter, limit=fetch)
    if paper_ids is None:
        return [dict(r) for r in results]
    allowed = set(paper_ids)
    return [dict(r) for r in results if str(r["local_id"]) in allowed][: int(limit)]


def ask(cfg: Any, question: str, top_k: int = 5) -> dict[str, Any]:
    """Retrieval-augmented answer — same path as ``drbrain ask`` (non-streaming).

    Returns ``{"error": ..., "unavailable": True}`` when the LlamaIndex engine
    is not enabled / indexed, instead of raising, so the UI can explain.
    """
    question = question.strip()
    if not question:
        return {"error": "empty question"}
    from drbrain.rag.engine import ask_llamaindex, resolve_engine

    if resolve_engine(cfg, "llamaindex") != "llamaindex":
        return {
            "error": "llamaindex engine unavailable: set `llamaindex.enabled: true` "
            "and run `drbrain rag index`",
            "unavailable": True,
        }
    with _db(cfg) as db:
        result = ask_llamaindex(cfg, db, question, top_k=top_k, streaming=False)
    return dict(result)


# ── papers (literature page) ─────────────────────────────────────────────────


def papers(
    cfg: Any,
    project_id: str | None = None,
    *,
    query: str = "",
    status: str | None = None,
    page: int = 1,
    per_page: int = DEFAULT_PAGE_SIZE,
) -> dict[str, Any]:
    """Paged, project-scoped library listing."""
    pid = normalize_project_id(project_id)
    page = max(1, int(page))
    per_page = max(1, min(int(per_page), MAX_PAGE_SIZE))
    with _db(cfg) as db:
        sync_workspace_projects(cfg, db)
        if db.get_project(pid) is None:
            raise ProjectNotFoundError(f"unknown project: {pid}")
        paper_ids = project_paper_ids(cfg, pid, db=db)
        items, total = db.list_papers(
            paper_ids=paper_ids,
            query=query.strip(),
            status=status,
            limit=per_page,
            offset=(page - 1) * per_page,
        )
    return {
        "items": items,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": max(1, (total + per_page - 1) // per_page) if total else 1,
        "query": query.strip(),
    }


def paper_detail(cfg: Any, local_id: str, project_id: str | None = None) -> dict[str, Any]:
    """Metadata, concepts, arguments and tree outline for one paper."""
    pid = normalize_project_id(project_id)
    with _db(cfg) as db:
        sync_workspace_projects(cfg, db)
        paper_ids = project_paper_ids(cfg, pid, db=db)
        if paper_ids is not None and local_id not in set(paper_ids):
            raise PaperNotInProjectError(f"paper {local_id!r} is not in project {pid!r}")
        paper = db.get_paper(local_id)
        if paper is None:
            raise PaperNotInProjectError(f"unknown paper: {local_id}")
        concepts = db.get_concepts_by_paper(local_id)
        arguments = db.get_arguments_by_paper(local_id)
    outline: list[dict[str, Any]] = []
    try:
        papers_root = Path(cfg["dirs"]["papers"])
        path = _runtime_path(papers_root, label="papers root")
        outline = paper_outline(path, local_id)
    except Exception:  # noqa: BLE001 - outline is optional detail, never fatal
        outline = []
    return {
        "paper": redact_sensitive(dict(paper)),
        "concepts": [redact_sensitive(dict(c)) for c in concepts],
        "arguments": [redact_sensitive(dict(a)) for a in arguments],
        "outline": outline,
    }


def paper_outline(
    papers_root: Path, local_id: str, *, max_nodes: int = 300
) -> list[dict[str, Any]]:
    """Return a bounded section outline from the paper's PageIndex tree."""
    from drbrain.storage.paths import paper_dir, tree_json_path

    path = tree_json_path(paper_dir(papers_root, local_id))
    if not path.is_file():
        return []
    try:
        structure = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(structure, list):
        return []

    out: list[dict[str, Any]] = []

    def walk(nodes: list, depth: int) -> None:
        if depth > 4 or len(out) >= max_nodes:
            return
        for node in nodes:
            if not isinstance(node, dict) or len(out) >= max_nodes:
                continue
            out.append(
                {
                    "node_id": str(node.get("node_id") or ""),
                    "title": str(node.get("title") or node.get("summary") or ""),
                    "depth": depth,
                    "children": len(node.get("nodes") or []),
                }
            )
            walk(list(node.get("nodes") or []), depth + 1)

    walk(structure, 0)
    return out


# ── autoresearch ledger (read side) ──────────────────────────────────────────


def runs(
    cfg: Any,
    project_id: str | None = None,
    *,
    session_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    pid = normalize_project_id(project_id)
    clauses = ["r.project_id = ?"]
    params: list[Any] = [pid]
    if session_id is not None:
        clauses.append("r.session_id = ?")
        params.append(str(session_id))
    if limit is not None:
        params.append(max(1, int(limit)))
        limit_sql = "LIMIT ?"
    else:
        limit_sql = ""
    with _ledger(cfg) as conn:
        if conn is None:
            return []
        try:
            rows = conn.execute(
                f"""
                SELECT r.run_id, r.topic, r.status, r.created_at, r.updated_at, r.completed_at,
                       r.project_id, r.session_id,
                       (SELECT COUNT(*) FROM research_events e WHERE e.run_id = r.run_id) AS events,
                       (SELECT COUNT(*) FROM research_claim_settlements s WHERE s.run_id = r.run_id) AS settlements,
                       (SELECT COUNT(*) FROM research_claim_settlements s
                         WHERE s.run_id = r.run_id AND s.verdict = 'keep') AS verified
                FROM research_runs r WHERE {" AND ".join(clauses)}
                ORDER BY r.updated_at DESC {limit_sql}
                """,
                tuple(params),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [redact_sensitive(dict(r)) for r in rows]


def _run_row(cfg: Any, run_id: str) -> dict[str, Any] | None:
    with _ledger(cfg) as conn:
        if conn is None:
            return None
        try:
            row = conn.execute(
                """
                SELECT r.run_id, r.topic, r.status, r.created_at, r.updated_at, r.completed_at,
                       r.project_id, r.session_id, r.client_request_id,
                       r.config_json, r.budget_json
                FROM research_runs r WHERE r.run_id = ?
                """,
                (run_id,),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
    return redact_sensitive(dict(row)) if row is not None else None


def _assert_run_scope(row: dict[str, Any] | None, project_id: str | None) -> dict[str, Any]:
    pid = normalize_project_id(project_id)
    if row is None or str(row.get("project_id") or DEFAULT_PROJECT_ID) != pid:
        raise RunNotFoundError("unknown research run")
    return row


def run_detail(cfg: Any, run_id: str, project_id: str | None = None) -> dict[str, Any]:
    """Run identity, scope, counts and the derived display status."""
    row = _assert_run_scope(_run_row(cfg, run_id), project_id)
    claims = run_claims(cfg, run_id, project_id=project_id)
    experiments_list = experiments(cfg, run_id=run_id, project_id=project_id)
    event_count = 0
    with _ledger(cfg) as conn:
        if conn is not None:
            event_count = _count(
                conn, "SELECT COUNT(*) FROM research_events WHERE run_id = ?", (run_id,)
            )
    detail = dict(row)
    detail["events"] = event_count
    detail["claims"] = len(claims)
    detail["verified"] = sum(1 for c in claims if c.get("verdict") == "keep")
    detail["experiments"] = len(experiments_list)
    detail["display_status"] = display_run_status(cfg, row)
    detail["session_label"] = row.get("session_id") or UNBOUND_SESSION_LABEL
    return detail


def display_run_status(cfg: Any, row: dict[str, Any]) -> str:
    """Derive a truthful display status from durable state, not process memory.

    A durable ``running`` row without a live worker (for example after a
    restart) is shown as ``interrupted`` — the run is resumable, but it is not
    currently executing.
    """
    status = str(row.get("status") or "")
    if status == "running" and not run_manager().is_running(str(row.get("run_id") or "")):
        return "interrupted"
    return status


def run_events(
    cfg: Any,
    run_id: str,
    after: int = 0,
    limit: int = 200,
    project_id: str | None = None,
) -> list[dict[str, Any]]:
    _assert_run_scope(_run_row(cfg, run_id), project_id)
    with _ledger(cfg) as conn:
        if conn is None:
            return []
        try:
            rows = conn.execute(
                """
                SELECT event_seq, actor, event_type, payload_json, created_at
                FROM research_events WHERE run_id = ? AND event_seq > ?
                ORDER BY event_seq LIMIT ?
                """,
                (run_id, int(after), max(1, min(int(limit), 1000))),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [
        redact_sensitive(
            {
                "seq": r["event_seq"],
                "actor": r["actor"],
                "type": r["event_type"],
                "payload": _loads(r["payload_json"], {}),
                "created_at": r["created_at"],
            }
        )
        for r in rows
    ]


def run_claims(cfg: Any, run_id: str, project_id: str | None = None) -> list[dict[str, Any]]:
    """Proposals of a run joined with their critic reviews and settlement."""
    _assert_run_scope(_run_row(cfg, run_id), project_id)
    with _ledger(cfg) as conn:
        if conn is None:
            return []
        try:
            proposals = conn.execute(
                """
                SELECT proposal_id, claim_id, author, payload_json, status, review_score, created_at
                FROM research_proposals WHERE run_id = ? ORDER BY created_at
                """,
                (run_id,),
            ).fetchall()
            reviews = conn.execute(
                """
                SELECT rv.* FROM research_critic_reviews rv
                JOIN research_proposals p ON p.proposal_id = rv.proposal_id
                WHERE p.run_id = ?
                """,
                (run_id,),
            ).fetchall()
            settlements = conn.execute(
                "SELECT * FROM research_claim_settlements WHERE run_id = ?", (run_id,)
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    by_proposal: dict[str, list[dict[str, Any]]] = {}
    for rv in reviews:
        d = dict(rv)
        by_proposal.setdefault(str(d.get("proposal_id")), []).append(
            {k: d.get(k) for k in ("reviewer", "verdict", "score") if k in d}
        )
    settled = {s["claim_id"]: dict(s) for s in settlements}
    out = []
    for p in proposals:
        payload = _loads(p["payload_json"], {})
        st = settled.get(p["claim_id"])
        out.append(
            redact_sensitive(
                {
                    "proposal_id": p["proposal_id"],
                    "claim_id": p["claim_id"],
                    "author": p["author"],
                    "status": p["status"],
                    "review_score": p["review_score"],
                    "statement": payload.get("statement") or payload.get("hypothesis") or "",
                    "reviews": by_proposal.get(p["proposal_id"], []),
                    "verdict": st["verdict"] if st else None,
                    "reason": st["reason"] if st else None,
                    "evidence_ids": _loads(st["evidence_ids_json"], []) if st else [],
                    "created_at": p["created_at"],
                }
            )
        )
    return out


def experiments(
    cfg: Any,
    run_id: str | None = None,
    project_id: str | None = None,
) -> list[dict[str, Any]]:
    """Compute jobs recorded by the loop (plan / config / artifacts / verdict)."""
    pid = normalize_project_id(project_id)
    with _ledger(cfg) as conn:
        if conn is None:
            return []
        where = "WHERE r.project_id = ?"
        params: list[Any] = [pid]
        if run_id:
            where += " AND x.run_id = ?"
            params.append(run_id)
        try:
            rows = conn.execute(
                f"""
                SELECT x.experiment_id, x.run_id, x.claim_id, x.status, x.seed, x.created_at,
                       x.plan_json, x.config_json,
                       (SELECT COUNT(*) FROM research_artifacts a
                         WHERE a.experiment_id = x.experiment_id) AS artifacts,
                       s.verdict, s.reason, s.result_json
                FROM research_experiments x
                JOIN research_runs r ON r.run_id = x.run_id
                LEFT JOIN research_claim_settlements s ON s.experiment_id = x.experiment_id
                {where} ORDER BY x.created_at DESC
                """,
                tuple(params),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [
        redact_sensitive(
            {
                "experiment_id": r["experiment_id"],
                "run_id": r["run_id"],
                "claim_id": r["claim_id"],
                "status": r["status"],
                "seed": r["seed"],
                "artifacts": r["artifacts"],
                "verdict": r["verdict"],
                "reason": r["reason"],
                "plan": _loads(r["plan_json"], {}),
                "config": _loads(r["config_json"], {}),
                "result": _loads(r["result_json"], {}),
                "created_at": r["created_at"],
            }
        )
        for r in rows
    ]


def run_report(
    cfg: Any, run_id: str, project_id: str | None = None, *, fmt: str = "markdown"
) -> tuple[str, str, str]:
    """Build a downloadable run report.

    Returns ``(body, media_type, filename)``.  The report binds the claim
    verdicts to their evidence references so a downloaded conclusion stays
    traceable.
    """
    detail = run_detail(cfg, run_id, project_id)
    claims = run_claims(cfg, run_id, project_id=project_id)
    jobs = experiments(cfg, run_id=run_id, project_id=project_id)
    events = run_events(cfg, run_id=run_id, after=0, limit=1000, project_id=project_id)
    created = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(detail["created_at"])))
    if fmt == "json":
        payload = {
            "run": detail,
            "claims": claims,
            "experiments": jobs,
            "events": events,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        body = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        return body, "application/json; charset=utf-8", f"run-{run_id}.json"

    lines = [
        f"# 研究运行报告 — {detail['topic']}",
        "",
        f"- 运行 ID：`{run_id}`",
        f"- 项目：`{detail.get('project_id') or DEFAULT_PROJECT_ID}`",
        f"- 会话：`{detail.get('session_id') or UNBOUND_SESSION_LABEL}`",
        f"- 状态：{detail['status']}",
        f"- 创建时间：{created}",
        "",
        "## 裁决结论",
        "",
    ]
    if not claims:
        lines.append("_尚无已提交的裁决结论。_")
    for c in claims:
        verdict = c.get("verdict") or "待裁决"
        lines.append(f"### [{verdict}] {c.get('statement') or c.get('claim_id')}")
        lines.append("")
        lines.append(f"- claim_id：`{c.get('claim_id')}`")
        if c.get("reason"):
            lines.append(f"- 理由：{c['reason']}")
        evidence = c.get("evidence_ids") or []
        lines.append(f"- 证据：{', '.join(f'`{e}`' for e in evidence) if evidence else '（无）'}")
        lines.append("")
    lines.append("## 计算实验")
    lines.append("")
    if not jobs:
        lines.append("_无计算实验记录。_")
    for x in jobs:
        lines.append(
            f"- `{x['experiment_id']}`（{x['status']}，产物 {x['artifacts']}）"
            + (f" — 裁决：{x['verdict']}" if x.get("verdict") else "")
        )
    lines += [
        "",
        "## 事件（节选）",
        "",
    ]
    for e in events[:200]:
        lines.append(f"- #{e['seq']} `{e['actor']}` {e['type']}")
    if len(events) > 200:
        lines.append(f"- …（共 {len(events)} 条事件，此处截取前 200 条）")
    lines.append("")
    body = "\n".join(lines)
    return body, "text/markdown; charset=utf-8", f"run-{run_id}.md"


# ── sessions (M2a) ───────────────────────────────────────────────────────────


def sessions(
    cfg: Any,
    project_id: str | None = None,
    *,
    page: int = 1,
    per_page: int = DEFAULT_PAGE_SIZE,
) -> dict[str, Any]:
    pid = normalize_project_id(project_id)
    page = max(1, int(page))
    per_page = max(1, min(int(per_page), MAX_PAGE_SIZE))
    with _db(cfg) as db:
        sync_workspace_projects(cfg, db)
        if db.get_project(pid) is None:
            raise ProjectNotFoundError(f"unknown project: {pid}")
        items = db.list_agent_sessions(pid, limit=per_page, offset=(page - 1) * per_page)
        total = db.count_agent_sessions(pid)
    return {
        "items": items,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": max(1, (total + per_page - 1) // per_page) if total else 1,
    }


def create_session(
    cfg: Any,
    project_id: str | None,
    *,
    title: str = "",
    system_prompt: str = "",
) -> dict[str, Any]:
    pid = normalize_project_id(project_id)
    title = title.strip() or time.strftime("会话 %Y-%m-%d %H:%M")
    with _db(cfg) as db:
        if db.get_project(pid) is None:
            raise ProjectNotFoundError(f"unknown project: {pid}")
        from drbrain.extractor.session_agent import SessionAgent

        agent = SessionAgent()
        session_id = agent.create_session(db, title=title, system_prompt=system_prompt)
        db.set_session_project(session_id, pid)
        db.record_webui_audit("session_created", detail=f"{pid}:{session_id}")
        return db.get_agent_session(session_id) or {"session_id": session_id, "title": title}


def get_session(cfg: Any, session_id: str, project_id: str | None = None) -> dict[str, Any]:
    """Return a session (with its run list) after an ownership check."""
    pid = normalize_project_id(project_id)
    with _db(cfg) as db:
        row = db.get_agent_session(session_id)
        if row is None or row["status"] == "deleted" or row["project_id"] != pid:
            raise SessionNotFoundError("unknown session")
        messages = db.get_agent_messages(session_id, limit=500)
    return {
        "session": row,
        "messages": messages,
        "runs": runs(cfg, pid, session_id=session_id),
        "memory": session_memory(cfg, session_id, project_id=pid),
    }


def session_messages(
    cfg: Any, session_id: str, project_id: str | None = None, *, after_seq: int = -1
) -> list[dict[str, Any]]:
    pid = normalize_project_id(project_id)
    with _db(cfg) as db:
        row = db.get_agent_session(session_id)
        if row is None or row["status"] == "deleted" or row["project_id"] != pid:
            raise SessionNotFoundError("unknown session")
        return db.get_agent_messages(session_id, after_seq=after_seq)


def chat_in_session(
    cfg: Any,
    session_id: str,
    question: str,
    project_id: str | None = None,
    *,
    max_turns: int = 8,
) -> dict[str, Any]:
    """Ask one question inside a persistent session (reuses SessionAgent).

    The full tool-calling loop runs on the existing session machinery; this
    function only scopes the session and returns the new messages plus the
    final answer so the page can render sources from the transcript.
    """
    question = question.strip()
    if not question:
        raise ValueError("empty question")
    pid = normalize_project_id(project_id)
    before_seq = -1
    with _db(cfg) as db:
        row = db.get_agent_session(session_id)
        if row is None or row["status"] == "deleted" or row["project_id"] != pid:
            raise SessionNotFoundError("unknown session")
        before_seq = db.next_agent_message_seq(session_id) - 1
    models = cfg.get("llm", {}).get("models") or []
    if not models:
        return {
            "unavailable": True,
            "error": "LLM models are not configured; set llm.models in config.yaml",
        }

    from drbrain.extractor.session_agent import SessionAgent

    agent = SessionAgent()
    with _db(cfg) as db:
        if not agent.load_session(db, session_id, models=list(models)):
            raise SessionNotFoundError("unknown session")
        import asyncio

        answer = asyncio.run(agent.ask(question, max_turns=max_turns))
        db.touch_session(session_id)
        messages = db.get_agent_messages(session_id, after_seq=before_seq)
    return {"answer": answer, "messages": messages, "session_id": session_id}


# ── session memory (M2a) ─────────────────────────────────────────────────────


def session_memory(
    cfg: Any, session_id: str | None = None, project_id: str | None = None
) -> dict[str, Any]:
    """Memory view: project-layer rows plus the session's own rows (and runs)."""
    pid = normalize_project_id(project_id)
    with _db(cfg) as db:
        if db.get_project(pid) is None:
            raise ProjectNotFoundError(f"unknown project: {pid}")
        project_rows = db.list_session_memory(pid, layers=("project",))
        session_rows = db.list_session_memory(pid, session_id=session_id) if session_id else []
    # Project-layer rows are inherited by every session; session rows are its own.
    sessions_specific = [r for r in session_rows if r["layer"] != "project"]
    return {
        "project": project_rows,
        "session": sessions_specific,
        "runs": [r for r in session_rows if r["layer"] == "run"],
    }


def record_run_memory(cfg: Any, run_id: str, project_id: str | None = None) -> int:
    """Write settled claims of one run into session memory (idempotent).

    Replaying the same settlement writes nothing new: the dedup key is
    ``run:<run_id>:<claim_id>``.  Called by the run page after a run reaches a
    settled state, and safe to call repeatedly.
    """
    pid = normalize_project_id(project_id)
    row = _assert_run_scope(_run_row(cfg, run_id), pid)
    session_id = str(row.get("session_id") or "")
    claims = run_claims(cfg, run_id, project_id=pid)
    written = 0
    with _db(cfg) as db:
        for claim in claims:
            verdict = claim.get("verdict")
            if not verdict:
                continue
            content = str(claim.get("statement") or claim.get("claim_id") or "")
            if not content:
                continue
            evidence = ", ".join(str(e) for e in (claim.get("evidence_ids") or []))
            created = db.insert_session_memory(
                memory_id=f"mem-{uuid.uuid4().hex[:12]}",
                project_id=pid,
                session_id=session_id,
                layer="run",
                kind="claim",
                content=f"[{verdict}] {content}",
                run_id=run_id,
                source_ref=f"claim:{claim.get('claim_id')}",
                dedup_key=f"run:{run_id}:{claim.get('claim_id')}",
            )
            if created:
                written += 1
            if evidence and created and session_id:
                db.insert_session_memory(
                    memory_id=f"mem-{uuid.uuid4().hex[:12]}",
                    project_id=pid,
                    session_id=session_id,
                    layer="run",
                    kind="evidence",
                    content=evidence,
                    run_id=run_id,
                    source_ref=f"claim:{claim.get('claim_id')}:evidence",
                    dedup_key=f"run:{run_id}:{claim.get('claim_id')}:evidence",
                )
    return written


def promote_memory(
    cfg: Any,
    session_id: str,
    memory_id: str,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Explicitly promote one session/run memory entry to the project layer."""
    pid = normalize_project_id(project_id)
    with _db(cfg) as db:
        row = db.get_agent_session(session_id)
        if row is None or row["status"] == "deleted" or row["project_id"] != pid:
            raise SessionNotFoundError("unknown session")
        entries = db.list_session_memory(pid, session_id=session_id)
        source = next((e for e in entries if e["memory_id"] == memory_id), None)
        if source is None:
            raise ValueError(f"unknown memory entry: {memory_id}")
        created = db.insert_session_memory(
            memory_id=f"mem-{uuid.uuid4().hex[:12]}",
            project_id=pid,
            session_id=session_id,
            layer="project",
            kind="promotion",
            content=source["content"],
            run_id=source.get("run_id") or "",
            source_ref=f"promoted:{source['memory_id']}",
            dedup_key=f"promote:{source['memory_id']}",
        )
        db.record_webui_audit("memory_promoted", detail=f"{pid}:{session_id}:{memory_id}")
        if not created:
            return {"promoted": False, "reason": "already promoted"}
    return {"promoted": True, "memory_id": memory_id}


# ── autoresearch (write side): durable launch + background worker ────────────


class RunManager:
    """Starts ``drbrain autoresearch run`` equivalents in daemon threads.

    The durable run row (ledger) is written *before* the worker starts, so a
    client always gets a persisted ``run_id`` back and a crash between request
    and worker cannot orphan the request.  One worker per run id; in-memory
    state only gates duplicate starts — the ledger stays the source of truth.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        self._run_threads: dict[str, threading.Thread] = {}
        self._errors: dict[str, str] = {}

    @staticmethod
    def _thread_key(project_id: str, session_id: str, topic: str) -> str:
        if project_id == DEFAULT_PROJECT_ID and not session_id:
            # Preserve the historic key for CLI-shaped callers/tests.
            return topic
        return f"{project_id}|{session_id}|{topic}"

    def start(
        self,
        cfg: Any,
        topic: str,
        max_cycles: int | None = None,
        *,
        project_id: str = DEFAULT_PROJECT_ID,
        session_id: str = "",
        client_request_id: str | None = None,
    ) -> dict[str, Any]:
        topic = topic.strip()
        if not topic:
            raise ValueError("empty research goal")
        project = normalize_project_id(project_id)
        session = str(session_id or "")
        settings = autoresearch_settings(cfg)
        if not settings.enabled:
            raise RuntimeError(
                "autoresearch disabled: set `autoresearch.enabled: true` in config.yaml"
            )
        # Validate every write-bearing path before creating the background
        # thread.  Read endpoints already go through ``ledger_path`` and
        # ``db_path``; without this normalization the director could still use
        # raw config values and write outside an active RuntimeContext.
        settings = replace(
            settings,
            run_dir=str(_runtime_path(settings.run_dir, label="autoresearch run directory")),
            plugins_dir=(
                str(
                    _runtime_path(
                        settings.plugins_dir,
                        label="autoresearch plugins directory",
                    )
                )
                if settings.plugins_dir
                else ""
            ),
        )
        # A Click invocation restores its process environment as soon as the
        # command returns, while this worker may continue for hours.  Capture
        # an absolute, validated config snapshot before starting the thread so
        # later DB/ledger access cannot fall back to the caller's CWD or an
        # unrelated runtime namespace.
        runtime_cfg = cfg
        runtime = _active_runtime()
        if runtime is not None:
            runtime.validate_config(cfg)
            runtime_cfg = runtime.apply_config(cfg)

        from drbrain.loop.store import RunLedger

        ledger = RunLedger(ledger_path(cfg))
        run = ledger.get_or_create_run(
            topic,
            project_id=project,
            session_id=session,
            client_request_id=client_request_id,
        )
        key = self._thread_key(project, session, topic)
        with self._lock:
            existing_thread = self._run_threads.get(run.run_id)
            if existing_thread is not None and existing_thread.is_alive():
                return self._launch_payload(run, started=False)
            self._errors.pop(key, None)
            thread = threading.Thread(
                target=self._run,
                args=(runtime_cfg, settings, topic, max_cycles, project, session, run.run_id),
                daemon=True,
            )
            self._threads[key] = thread
            self._run_threads[run.run_id] = thread
            thread.start()
        return self._launch_payload(run, started=True)

    @staticmethod
    def _launch_payload(run: Any, *, started: bool) -> dict[str, Any]:
        return {
            "run_id": run.run_id,
            "topic": run.topic,
            "status": run.status,
            "project_id": run.project_id,
            "session_id": run.session_id,
            "started": started,
            "detail_url": f"/api/runs/{run.run_id}",
            "events_url": f"/api/runs/{run.run_id}/events",
            "stream_url": f"/api/runs/{run.run_id}/stream",
        }

    def is_running(self, run_id: str) -> bool:
        thread = self._run_threads.get(run_id)
        return bool(thread and thread.is_alive())

    def status(self, topic: str) -> dict[str, Any]:
        t = self._threads.get(topic.strip())
        return {
            "topic": redact_sensitive_text(topic.strip()) or "",
            "alive": bool(t and t.is_alive()),
            "error": redact_sensitive_text(self._errors.get(topic.strip())),
        }

    def status_by_run(self, run_id: str) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "alive": self.is_running(run_id),
            "error": redact_sensitive_text(self._errors.get(run_id)),
        }

    def _run(
        self,
        cfg: Any,
        settings: AutoresearchConfig,
        topic: str,
        max_cycles: int | None,
        project_id: str,
        session_id: str,
        run_id: str,
    ) -> None:
        key = self._thread_key(project_id, session_id, topic)
        try:
            self._execute(cfg, settings, topic, max_cycles, project_id, session_id)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI via status()
            try:
                secrets = configured_secret_values(cfg)
            except Exception:  # noqa: BLE001 - reporting must never mask the failure
                secrets = ()
            message = safe_error(exc, secrets=secrets) or "internal error"
            self._errors[key] = f"{type(exc).__name__}: {message[:500]}"
            self._errors[run_id] = self._errors[key]

    def _execute(
        self,
        cfg: Any,
        settings: AutoresearchConfig,
        topic: str,
        max_cycles: int | None,
        project_id: str = DEFAULT_PROJECT_ID,
        session_id: str = "",
    ) -> None:
        """Mirror of ``drbrain autoresearch run`` (blocking; runs inside the thread)."""
        from drbrain.loop import ResearchDirector
        from drbrain.loop.policy import ToolPolicy

        tool_policy = (
            ToolPolicy(step_capabilities=settings.step_capabilities)
            if settings.plugins_dir or settings.mcp_servers
            else None
        )
        with _db(cfg) as db:
            director = ResearchDirector(
                cfg,
                db=db,
                plugins_dir=settings.plugins_dir or None,
                mcp_servers=settings.mcp_servers,
                run_dir=settings.run_dir,
                n_critics=settings.n_critics,
                single_agent=settings.single_agent,
                lease_seconds=settings.lease_seconds,
                tool_policy=tool_policy,
                require_rag_evidence=settings.require_rag_evidence,
                require_compute_tools=settings.require_compute_tools,
                compute_tool_names=list(settings.compute_tool_names) or None,
                step_timeout_seconds=settings.step_timeout_seconds,
            )
            director.run_sync(
                topic,
                max_cycles=settings.max_cycles if max_cycles is None else max_cycles,
                stagnation_cycles=settings.stagnation_cycles,
                max_adaptations=settings.max_adaptations,
                budget=dict(settings.budget),
                project_id=project_id,
                session_id=session_id,
            )


_RUN_MANAGER: RunManager | None = None


def run_manager() -> RunManager:
    """Process-wide run manager (one worker per run id)."""
    global _RUN_MANAGER
    if _RUN_MANAGER is None:
        _RUN_MANAGER = RunManager()
    return _RUN_MANAGER


def start_run(
    cfg: Any,
    topic: str,
    max_cycles: int | None = None,
    *,
    project_id: str | None = None,
    session_id: str = "",
    client_request_id: str | None = None,
) -> dict[str, Any]:
    """Resolve scope, persist the launch, then start one background worker."""
    pid = normalize_project_id(project_id)
    if session_id:
        with _db(cfg) as db:
            row = db.get_agent_session(session_id)
            if row is None or row["status"] == "deleted" or row["project_id"] != pid:
                raise SessionNotFoundError("unknown session")
    return run_manager().start(
        cfg,
        topic,
        max_cycles=max_cycles,
        project_id=pid,
        session_id=str(session_id or ""),
        client_request_id=client_request_id,
    )


# ── plugins / conformance (M3) ───────────────────────────────────────────────


def plugin_catalog(cfg: Any) -> list[dict[str, Any]]:
    """Discovered plugins with conformance report state (never faked as "online")."""
    plugins_dir = autoresearch_settings(cfg).plugins_dir
    discovered = plugins(cfg)
    if not plugins_dir:
        return discovered
    with _db(cfg) as db:
        path_ok = True
        try:
            path = _runtime_path(plugins_dir, label="autoresearch plugins directory")
            path_ok = path.is_dir() and first_symlink_component(path) is None
        except (OSError, ValueError):
            path_ok = False
        for item in discovered:
            report = None
            if path_ok:
                reports = db.list_plugin_conformance(item["name"], limit=1)
                report = reports[0] if reports else None
                if report and report.get("completed_at") is None and report["status"] == "pending":
                    report["stale"] = False
                elif report:
                    report["stale"] = report.get("plugin_version", "") != item.get("version", "")
            item["conformance"] = report
            item["conformance_state"] = _conformance_state(report)
            item["discovered_from"] = str(plugins_dir)
    return discovered


def _conformance_state(report: dict[str, Any] | None) -> str:
    if report is None:
        return "not_tested"
    status = str(report.get("status") or "")
    if status == "pending":
        return "checking"
    if report.get("stale"):
        return "stale"
    return status  # passed | failed


def start_conformance(cfg: Any, plugin_name: str) -> dict[str, Any]:
    """Queue a conformance check for one discovered plugin.

    The endpoint is explicit: opening the plugin list never runs checks.  The
    check itself is a small, truthful ABI/loadability probe; a full compliance
    suite arrives with plugin v2.
    """
    catalog = {p["name"]: p for p in plugin_catalog(cfg)}
    item = catalog.get(plugin_name)
    if item is None:
        raise ValueError(f"unknown plugin: {plugin_name}")
    check_id = f"conf-{uuid.uuid4().hex[:12]}"
    fingerprint = f"{item.get('version', '')}:{item.get('backend', '')}"
    with _db(cfg) as db:
        db.insert_plugin_conformance(
            check_id,
            plugin_name,
            str(item.get("version") or ""),
            fingerprint,
            "pending",
            "[]",
        )
    thread = threading.Thread(
        target=_run_conformance,
        args=(cfg, plugin_name, check_id, fingerprint),
        daemon=True,
    )
    thread.start()
    return {"check_id": check_id, "plugin": plugin_name, "status": "checking"}


def _run_conformance(cfg: Any, plugin_name: str, check_id: str, fingerprint: str) -> None:
    """Run the existing static conformance suite and store its per-plugin slice.

    The suite itself is ``drbrain.plugins.conformance`` (the same one plugin
    authors run); this function only bounds it to one plugin and persists the
    outcome.  A plugin whose checks cannot be isolated falls back to the whole
    directory report, which is still truthful — it is labeled with the plugin
    name and fingerprint.
    """
    from drbrain.plugins.conformance import run_conformance

    checks: list[dict[str, Any]] = []
    status = "failed"
    try:
        settings = autoresearch_settings(cfg)
        directory = _runtime_path(settings.plugins_dir, label="autoresearch plugins directory")
        report = run_conformance(directory)
        all_checks = [
            {"name": c.name, "passed": bool(c.passed), "detail": c.detail} for c in report.checks
        ]
        selected = [c for c in all_checks if plugin_name in c["name"].split(".")]
        if not selected:
            selected = [c for c in all_checks if c["name"].startswith(f"{plugin_name}.")]
        if not selected:
            selected = all_checks
        checks = selected
        status = "passed" if checks and all(c["passed"] for c in checks) else "failed"
    except Exception as exc:  # noqa: BLE001 - infrastructure failure is the report
        checks = [
            {
                "name": "conformance",
                "passed": False,
                "detail": safe_error(exc, limit=300),
            }
        ]
        status = "failed"
    with _db(cfg) as db:
        db.insert_plugin_conformance(
            check_id,
            plugin_name,
            "",
            fingerprint,
            status,
            json.dumps(checks, ensure_ascii=False),
            completed_at=time.time(),
        )


def conformance_report(cfg: Any, plugin_name: str, check_id: str) -> dict[str, Any]:
    with _db(cfg) as db:
        row = db.get_plugin_conformance(check_id)
    if row is None or row["plugin_name"] != plugin_name:
        raise ValueError(f"unknown conformance check: {check_id}")
    return row


def plugins(cfg: Any) -> list[dict[str, Any]]:
    """Model-as-Tool plugins discovered from ``autoresearch.plugins_dir``."""
    plugins_dir = autoresearch_settings(cfg).plugins_dir
    if not plugins_dir:
        return []
    plugin_path = _runtime_path(plugins_dir, label="autoresearch plugins directory")
    if first_symlink_component(plugin_path) is not None or not plugin_path.is_dir():
        return []
    from drbrain.plugins.registry import PluginRegistry

    registry = PluginRegistry()
    try:
        registry.discover(plugin_path)
    except Exception:  # noqa: BLE001 - a broken plugin dir must not take the UI down
        return []
    return [
        redact_sensitive(
            {
                "name": p.name,
                "type": p.plugin_type,
                "backend": p.backend,
                "version": p.version,
                "description": p.description,
                "resource": p.resource,
            }
        )
        for p in registry.list_plugins()
    ]


def settings_view(cfg: Any) -> dict[str, Any]:
    """Redacted effective configuration + feature availability for /settings."""
    from dataclasses import asdict, is_dataclass

    if is_dataclass(cfg) and not isinstance(cfg, type):
        raw = asdict(cfg)
    elif isinstance(cfg, dict):
        raw = dict(cfg)
    else:  # pragma: no cover - defensive for exotic config objects
        raw = {}
    safe = redact_sensitive(raw)
    with _db(cfg) as db:
        sessions = db.list_webui_sessions(limit=5)
        audits = db.list_webui_audit(limit=10)
    return {
        "config": safe,
        "availability": availability(cfg),
        "login_sessions": sessions,
        "audit": audits,
    }


def assets(cfg: Any) -> dict[str, Any]:
    """Where the data lives: database, ledger, plugins, export entry points."""
    dbp = db_path(cfg)
    lp = ledger_path(cfg)
    settings = autoresearch_settings(cfg)

    def size(p: str | Path) -> int | None:
        path = Path(p)
        return path.stat().st_size if path.is_file() else None

    return {
        "database": {"path": dbp, "bytes": size(dbp) if dbp != ":memory:" else None},
        "ledger": {"path": str(lp), "bytes": size(lp), "run_dir": settings.run_dir},
        "plugins_dir": settings.plugins_dir or None,
        "plugins": plugins(cfg),
        "exports": [
            {"label": "BibTeX", "command": "drbrain export --format bibtex"},
            {"label": "GraphML", "command": "drbrain export --format graphml"},
            {"label": "OKF markdown", "command": "drbrain export-okf"},
            {"label": "Backup", "command": "drbrain backup"},
        ],
    }
