"""Service layer for the DrBrain WebUI.

Every function here wraps a capability that already exists behind the CLI
(``drbrain stats`` / ``search`` / ``ask`` / ``autoresearch``) and returns plain
JSON-serialisable data. The HTTP layer in :mod:`drbrain.app.server` is a thin
router over these functions, so they can be unit-tested without a socket.

The UI starts empty: nothing is pre-loaded, every page reflects the live
database, the autoresearch ledger and the plugin directory of the current
configuration.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from drbrain.config import AutoresearchConfig
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


# ── dashboard ────────────────────────────────────────────────────────────────


def dashboard(cfg: Any) -> dict[str, Any]:
    """KPI counters for the workbench page (database + ledger + plugins)."""
    stats: dict[str, Any] = {}
    path = db_path(cfg)
    if path == ":memory:" or Path(path).is_file():
        with _db(cfg) as db:
            stats = db.get_stats()
    ledger: dict[str, Any] = {"runs": 0, "settlements": 0, "verified": 0, "events": 0}
    with _ledger(cfg) as conn:
        if conn is not None:
            ledger = {
                "runs": _count(conn, "SELECT COUNT(*) FROM research_runs"),
                "settlements": _count(conn, "SELECT COUNT(*) FROM research_claim_settlements"),
                "verified": _count(
                    conn,
                    "SELECT COUNT(*) FROM research_claim_settlements WHERE verdict='keep'",
                ),
                "events": _count(conn, "SELECT COUNT(*) FROM research_events"),
            }
    return {
        "papers": int(stats.get("papers", 0)),
        "concepts": int(stats.get("concepts", 0)),
        "edges": int(stats.get("edges", 0)),
        "arguments": int(stats.get("arguments", 0)),
        "ledger": ledger,
        "plugins": len(plugins(cfg)),
        "recent_runs": runs(cfg)[:5],
    }


# ── search / ask ─────────────────────────────────────────────────────────────


def search(cfg: Any, query: str, limit: int = 10, type_filter: str | None = None) -> list[dict]:
    """BM25 keyword search — same engine as ``drbrain search``."""
    query = query.strip()
    if not query:
        return []
    from drbrain.query.bm25 import build_bm25_index

    with _db(cfg) as db:
        bm25 = build_bm25_index(db)
        results = bm25.search(query, type_filter=type_filter, limit=limit)
    return [dict(r) for r in results]


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


# ── autoresearch ledger (read side) ──────────────────────────────────────────


def runs(cfg: Any) -> list[dict[str, Any]]:
    with _ledger(cfg) as conn:
        if conn is None:
            return []
        try:
            rows = conn.execute(
                """
                SELECT r.run_id, r.topic, r.status, r.created_at, r.updated_at, r.completed_at,
                       (SELECT COUNT(*) FROM research_events e WHERE e.run_id = r.run_id) AS events,
                       (SELECT COUNT(*) FROM research_claim_settlements s WHERE s.run_id = r.run_id) AS settlements,
                       (SELECT COUNT(*) FROM research_claim_settlements s
                         WHERE s.run_id = r.run_id AND s.verdict = 'keep') AS verified
                FROM research_runs r ORDER BY r.updated_at DESC
                """
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [redact_sensitive(dict(r)) for r in rows]


def run_events(cfg: Any, run_id: str, after: int = 0, limit: int = 200) -> list[dict[str, Any]]:
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
                (run_id, int(after), int(limit)),
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


def run_claims(cfg: Any, run_id: str) -> list[dict[str, Any]]:
    """Proposals of a run joined with their critic reviews and settlement."""
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


def experiments(cfg: Any, run_id: str | None = None) -> list[dict[str, Any]]:
    """Compute jobs recorded by the loop (plan / config / artifacts / verdict)."""
    with _ledger(cfg) as conn:
        if conn is None:
            return []
        where, params = ("WHERE x.run_id = ?", (run_id,)) if run_id else ("", ())
        try:
            rows = conn.execute(
                f"""
                SELECT x.experiment_id, x.run_id, x.claim_id, x.status, x.seed, x.created_at,
                       x.plan_json, x.config_json,
                       (SELECT COUNT(*) FROM research_artifacts a
                         WHERE a.experiment_id = x.experiment_id) AS artifacts,
                       s.verdict, s.reason, s.result_json
                FROM research_experiments x
                LEFT JOIN research_claim_settlements s ON s.experiment_id = x.experiment_id
                {where} ORDER BY x.created_at DESC
                """,
                params,
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


# ── autoresearch (write side): start a durable run in a background thread ──


class RunManager:
    """Starts ``drbrain autoresearch run`` equivalents in daemon threads.

    One thread per topic; repeating a topic resumes its durable run exactly
    like the CLI does. State is kept in memory only.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        self._errors: dict[str, str] = {}

    def start(self, cfg: Any, topic: str, max_cycles: int | None = None) -> dict[str, Any]:
        topic = topic.strip()
        if not topic:
            raise ValueError("empty research goal")
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
        with self._lock:
            t = self._threads.get(topic)
            if t is not None and t.is_alive():
                return {"topic": topic, "status": "running", "started": False}
            self._errors.pop(topic, None)
            t = threading.Thread(
                target=self._run, args=(runtime_cfg, settings, topic, max_cycles), daemon=True
            )
            self._threads[topic] = t
            t.start()
        return {"topic": topic, "status": "starting", "started": True}

    def status(self, topic: str) -> dict[str, Any]:
        t = self._threads.get(topic)
        return {
            "topic": redact_sensitive_text(topic) or "",
            "alive": bool(t and t.is_alive()),
            "error": redact_sensitive_text(self._errors.get(topic)),
        }

    def _run(
        self, cfg: Any, settings: AutoresearchConfig, topic: str, max_cycles: int | None
    ) -> None:
        try:
            self._execute(cfg, settings, topic, max_cycles)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI via status()
            try:
                secrets = configured_secret_values(cfg)
            except Exception:  # noqa: BLE001 - reporting must never mask the failure
                secrets = ()
            message = safe_error(exc, secrets=secrets) or "internal error"
            self._errors[topic] = f"{type(exc).__name__}: {message[:500]}"

    def _execute(
        self, cfg: Any, settings: AutoresearchConfig, topic: str, max_cycles: int | None
    ) -> None:
        """Mirror of ``drbrain autoresearch run`` (blocking; runs inside the thread)."""
        if True:
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
                )


# ── plugins / assets ─────────────────────────────────────────────────────────


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
