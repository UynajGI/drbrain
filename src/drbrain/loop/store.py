"""SQLite storage primitives for durable autoresearch runs.

This is deliberately a small, per-director ledger.  It does not alter the
knowledge-graph database or replace the existing workspace files; those files
remain compatibility projections owned by :mod:`drbrain.loop.director`.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drbrain.loop.state import RUN_CREATED
from drbrain.storage.connection import connect_wal

LEDGER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LedgerRun:
    """A durable research-run identity and lifecycle snapshot."""

    run_id: str
    topic: str
    status: str
    last_projected_event: int


@dataclass(frozen=True)
class LedgerEvent:
    """An append-only event in one research run."""

    run_id: str
    event_seq: int
    actor: str
    event_type: str
    payload: dict[str, Any]
    trace_id: str | None
    created_at: float


def _as_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def _from_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


class RunLedger:
    """Versioned SQLite ledger for autoresearch lifecycle facts.

    A ledger operation always uses a short ``BEGIN IMMEDIATE`` transaction.
    That serializes event-sequence allocation and makes a cycle's state update
    and event append indivisible before the compatibility projector touches any
    Markdown or JSONL file.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a connection with the current ledger schema inside a write tx."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = connect_wal(self.path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            self._ensure_schema(conn)
            # ``_ensure_schema`` may insert the version row, which opens
            # sqlite3's implicit transaction. Finish it before taking the
            # explicit IMMEDIATE writer lease used for lifecycle mutations.
            conn.commit()
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except Exception:
                conn.rollback()
                raise
            else:
                conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS ledger_schema_versions (
                version INTEGER PRIMARY KEY,
                applied_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS research_runs (
                run_id TEXT PRIMARY KEY,
                topic TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                config_json TEXT NOT NULL DEFAULT '{}',
                budget_json TEXT NOT NULL DEFAULT '{}',
                last_projected_event INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                completed_at REAL,
                CHECK (status IN ('created', 'running', 'paused', 'succeeded', 'failed', 'cancelled'))
            );

            CREATE TABLE IF NOT EXISTS research_steps (
                step_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES research_runs(run_id),
                step_name TEXT NOT NULL,
                status TEXT NOT NULL,
                dependencies_json TEXT NOT NULL DEFAULT '[]',
                lease_owner TEXT,
                lease_expires_at REAL,
                input_ref_json TEXT NOT NULL DEFAULT '{}',
                output_ref_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                CHECK (status IN (
                    'pending', 'ready', 'claimed', 'running', 'waiting_approval',
                    'succeeded', 'failed', 'timed_out', 'unknown', 'reconciling', 'manual_review'
                ))
            );

            CREATE INDEX IF NOT EXISTS idx_research_steps_run_status
                ON research_steps(run_id, status);

            CREATE TABLE IF NOT EXISTS research_attempts (
                attempt_id TEXT PRIMARY KEY,
                step_id TEXT NOT NULL REFERENCES research_steps(step_id),
                attempt_no INTEGER NOT NULL,
                status TEXT NOT NULL,
                failure_category TEXT,
                checkpoint_ref TEXT,
                environment_json TEXT NOT NULL DEFAULT '{}',
                seed INTEGER,
                started_at REAL NOT NULL,
                completed_at REAL,
                UNIQUE(step_id, attempt_no)
            );

            CREATE TABLE IF NOT EXISTS research_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES research_runs(run_id),
                event_seq INTEGER NOT NULL,
                actor TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                trace_id TEXT,
                created_at REAL NOT NULL,
                UNIQUE(run_id, event_seq)
            );

            CREATE INDEX IF NOT EXISTS idx_research_events_run_seq
                ON research_events(run_id, event_seq);
            """
        )
        row = conn.execute("SELECT MAX(version) AS version FROM ledger_schema_versions").fetchone()
        current = int(row["version"] or 0)
        if current > LEDGER_SCHEMA_VERSION:
            raise RuntimeError(
                f"ledger schema {current} is newer than supported {LEDGER_SCHEMA_VERSION}"
            )
        if current < LEDGER_SCHEMA_VERSION:
            conn.execute(
                "INSERT INTO ledger_schema_versions(version, applied_at) VALUES (?, ?)",
                (LEDGER_SCHEMA_VERSION, time.time()),
            )

    def get_run(self, topic: str) -> LedgerRun | None:
        with self.transaction() as conn:
            row = conn.execute(
                """
                SELECT run_id, topic, status, last_projected_event
                FROM research_runs WHERE topic = ?
                """,
                (topic,),
            ).fetchone()
            return self._run_from_row(row)

    def get_or_create_run(
        self,
        topic: str,
        *,
        config: Mapping[str, Any] | None = None,
        budget: Mapping[str, Any] | None = None,
        legacy_snapshot: Mapping[str, Any] | None = None,
    ) -> LedgerRun:
        """Return the stable run for ``topic``, importing legacy state once."""
        with self.transaction() as conn:
            row = conn.execute(
                """
                SELECT run_id, topic, status, last_projected_event
                FROM research_runs WHERE topic = ?
                """,
                (topic,),
            ).fetchone()
            if row is not None:
                existing = self._run_from_row(row)
                assert existing is not None
                return existing

            now = time.time()
            run_id = uuid.uuid4().hex
            conn.execute(
                """
                INSERT INTO research_runs(
                    run_id, topic, status, schema_version, config_json, budget_json,
                    last_projected_event, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    run_id,
                    topic,
                    RUN_CREATED,
                    LEDGER_SCHEMA_VERSION,
                    _as_json(dict(config or {})),
                    _as_json(dict(budget or {})),
                    now,
                    now,
                ),
            )
            self.append_event(
                conn,
                run_id,
                actor="director",
                event_type="run_created",
                payload={"topic": topic, "schema_version": LEDGER_SCHEMA_VERSION},
            )
            if legacy_snapshot is not None:
                self.append_event(
                    conn,
                    run_id,
                    actor="migration",
                    event_type="legacy_snapshot_imported",
                    payload={"state": dict(legacy_snapshot)},
                )
            return LedgerRun(run_id, topic, RUN_CREATED, 0)

    def record_resume(
        self,
        run_id: str,
        *,
        config: Mapping[str, Any],
        budget: Mapping[str, Any],
    ) -> LedgerEvent:
        """Append the effective configuration for one resumed director session.

        ``research_runs.config_json`` and ``budget_json`` preserve the original
        run specification.  A resume can intentionally use a different budget,
        so every later invocation is recorded as its own append-only audit
        event instead of silently rewriting the original settings.
        """
        with self.transaction() as conn:
            row = conn.execute("SELECT 1 FROM research_runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown research run: {run_id}")
            conn.execute(
                "UPDATE research_runs SET updated_at = ? WHERE run_id = ?",
                (time.time(), run_id),
            )
            return self.append_event(
                conn,
                run_id,
                actor="director",
                event_type="run_resumed",
                payload={
                    "session_id": uuid.uuid4().hex,
                    "config": dict(config),
                    "budget": dict(budget),
                },
            )

    def append_event(
        self,
        conn: sqlite3.Connection,
        run_id: str,
        *,
        actor: str,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> LedgerEvent:
        """Append an event while the caller owns the ledger transaction."""
        row = conn.execute(
            "SELECT COALESCE(MAX(event_seq), 0) + 1 AS next_seq FROM research_events WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        event_seq = int(row["next_seq"])
        created_at = time.time()
        event_payload = dict(payload or {})
        conn.execute(
            """
            INSERT INTO research_events(
                run_id, event_seq, actor, event_type, payload_json, trace_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                event_seq,
                actor,
                event_type,
                _as_json(event_payload),
                trace_id,
                created_at,
            ),
        )
        return LedgerEvent(
            run_id, event_seq, actor, event_type, event_payload, trace_id, created_at
        )

    def pending_projection_events(self, run_id: str) -> list[LedgerEvent]:
        """Return committed cycle snapshots that have not reached file projections."""
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT last_projected_event FROM research_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                return []
            rows = conn.execute(
                """
                SELECT run_id, event_seq, actor, event_type, payload_json, trace_id, created_at
                FROM research_events
                WHERE run_id = ?
                  AND event_seq > ?
                  AND event_type IN ('cycle_completed', 'state_snapshot')
                ORDER BY event_seq
                """,
                (run_id, row["last_projected_event"]),
            ).fetchall()
            return [self._event_from_row(event) for event in rows]

    def events(self, run_id: str) -> list[LedgerEvent]:
        """Read the ordered audit trail for one run without exposing a write path."""
        with self.transaction() as conn:
            rows = conn.execute(
                """
                SELECT run_id, event_seq, actor, event_type, payload_json, trace_id, created_at
                FROM research_events
                WHERE run_id = ?
                ORDER BY event_seq
                """,
                (run_id,),
            ).fetchall()
            return [self._event_from_row(event) for event in rows]

    def mark_projected(self, run_id: str, event_seq: int) -> None:
        """Advance the compatibility-projection cursor after a complete file write."""
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE research_runs
                SET last_projected_event = MAX(last_projected_event, ?), updated_at = ?
                WHERE run_id = ?
                """,
                (event_seq, time.time(), run_id),
            )

    @staticmethod
    def _run_from_row(row: sqlite3.Row | None) -> LedgerRun | None:
        if row is None:
            return None
        return LedgerRun(
            run_id=str(row["run_id"]),
            topic=str(row["topic"]),
            status=str(row["status"]),
            last_projected_event=int(row["last_projected_event"]),
        )

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> LedgerEvent:
        return LedgerEvent(
            run_id=str(row["run_id"]),
            event_seq=int(row["event_seq"]),
            actor=str(row["actor"]),
            event_type=str(row["event_type"]),
            payload=_from_json(row["payload_json"], {}),
            trace_id=row["trace_id"],
            created_at=float(row["created_at"]),
        )
