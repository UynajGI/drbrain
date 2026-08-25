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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from drbrain.loop.state import RUN_CREATED
from drbrain.storage.connection import connect_wal

LEDGER_SCHEMA_VERSION = 6


@dataclass(frozen=True)
class LedgerRun:
    """A durable research-run identity and lifecycle snapshot."""

    run_id: str
    topic: str
    status: str
    last_projected_event: int
    # Default preserves direct construction by older callers.
    config: dict[str, Any] = field(default_factory=dict)


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


@dataclass(frozen=True)
class LedgerCheckpoint:
    """One JSON-only workflow checkpoint bound to a durable attempt."""

    checkpoint_id: str
    run_id: str
    step_id: str
    attempt_id: str
    checkpoint_seq: int
    step_name: str
    context_payload: dict[str, Any]
    workflow_state: dict[str, Any]
    manifest: dict[str, Any]
    created_at: float


@dataclass(frozen=True)
class LedgerToolCall:
    """A durable tool proposal, intent, and final observation."""

    tool_call_id: str
    run_id: str
    step_id: str
    attempt_id: str
    node_name: str
    tool_name: str
    source: str
    side_effect: str
    status: str
    idempotency_key: str | None
    proposal: dict[str, Any]
    observation: dict[str, Any]
    created_at: float
    updated_at: float


class LeaseOwnershipError(RuntimeError):
    """Raised when a worker attempts to write after losing its SQLite lease."""


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

            CREATE TABLE IF NOT EXISTS research_checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES research_runs(run_id),
                step_id TEXT NOT NULL REFERENCES research_steps(step_id),
                attempt_id TEXT NOT NULL REFERENCES research_attempts(attempt_id),
                checkpoint_seq INTEGER NOT NULL,
                step_name TEXT NOT NULL,
                context_json TEXT NOT NULL,
                workflow_state_json TEXT NOT NULL DEFAULT '{}',
                manifest_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                UNIQUE(step_id, attempt_id, checkpoint_seq)
            );

            CREATE INDEX IF NOT EXISTS idx_research_checkpoints_step_seq
                ON research_checkpoints(step_id, checkpoint_seq DESC);

            CREATE INDEX IF NOT EXISTS idx_research_checkpoints_run_created
                ON research_checkpoints(run_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS research_attempt_progress (
                step_id TEXT PRIMARY KEY REFERENCES research_steps(step_id),
                attempt_id TEXT NOT NULL REFERENCES research_attempts(attempt_id),
                node_name TEXT NOT NULL,
                boundary_kind TEXT NOT NULL CHECK (boundary_kind IN ('started', 'checkpointed')),
                checkpoint_id TEXT REFERENCES research_checkpoints(checkpoint_id),
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS research_tool_calls (
                tool_call_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES research_runs(run_id),
                step_id TEXT NOT NULL REFERENCES research_steps(step_id),
                attempt_id TEXT NOT NULL REFERENCES research_attempts(attempt_id),
                node_name TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                source TEXT NOT NULL,
                side_effect TEXT NOT NULL,
                status TEXT NOT NULL,
                idempotency_key TEXT,
                proposal_json TEXT NOT NULL DEFAULT '{}',
                observation_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                CHECK (status IN (
                    'intent', 'succeeded', 'failed', 'timed_out', 'unknown',
                    'denied', 'waiting_approval'
                ))
            );

            CREATE INDEX IF NOT EXISTS idx_research_tool_calls_run_created
                ON research_tool_calls(run_id, created_at);

            CREATE INDEX IF NOT EXISTS idx_research_tool_calls_idempotency
                ON research_tool_calls(run_id, step_id, idempotency_key, created_at DESC);

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
        run_columns = {
            str(column["name"])
            for column in conn.execute("PRAGMA table_info(research_runs)").fetchall()
        }
        if "config_json" not in run_columns:
            conn.execute(
                "ALTER TABLE research_runs ADD COLUMN config_json TEXT NOT NULL DEFAULT '{}'"
            )
        if current < 5:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_front_half_node_specs (
                    run_id TEXT NOT NULL REFERENCES research_runs(run_id),
                    node_name TEXT NOT NULL,
                    input_schema_json TEXT NOT NULL,
                    output_schema_json TEXT NOT NULL,
                    allowed_tools_json TEXT NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    retry_class TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (run_id, node_name)
                );

                CREATE TABLE IF NOT EXISTS research_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES research_runs(run_id),
                    claim_id TEXT NOT NULL,
                    author TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    review_score REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE (run_id, claim_id),
                    CHECK (status IN ('proposed', 'discussion_pending', 'critiqued', 'discarded'))
                );

                CREATE INDEX IF NOT EXISTS idx_research_proposals_run_created
                    ON research_proposals(run_id, created_at);

                CREATE TABLE IF NOT EXISTS research_critic_reviews (
                    review_id TEXT PRIMARY KEY,
                    proposal_id TEXT NOT NULL REFERENCES research_proposals(proposal_id),
                    reviewer TEXT NOT NULL,
                    score REAL NOT NULL,
                    verdict TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE (proposal_id, reviewer)
                );

                CREATE TABLE IF NOT EXISTS research_queue_items (
                    queue_item_id TEXT PRIMARY KEY,
                    proposal_id TEXT NOT NULL UNIQUE REFERENCES research_proposals(proposal_id),
                    run_id TEXT NOT NULL REFERENCES research_runs(run_id),
                    status TEXT NOT NULL,
                    score REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    CHECK (status IN ('pending_review', 'ready', 'discarded'))
                );

                CREATE INDEX IF NOT EXISTS idx_research_queue_items_run_status
                    ON research_queue_items(run_id, status, created_at);
                """
            )
        if current < 6:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_execution_node_specs (
                    run_id TEXT NOT NULL REFERENCES research_runs(run_id),
                    node_name TEXT NOT NULL,
                    input_schema_json TEXT NOT NULL,
                    output_schema_json TEXT NOT NULL,
                    allowed_tools_json TEXT NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    retry_class TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (run_id, node_name)
                );

                CREATE TABLE IF NOT EXISTS research_experiments (
                    experiment_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES research_runs(run_id),
                    proposal_id TEXT,
                    claim_id TEXT NOT NULL,
                    producer_attempt_id TEXT,
                    plan_json TEXT NOT NULL,
                    environment_json TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    seed INTEGER,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE (run_id, claim_id),
                    CHECK (status IN ('planned', 'computed', 'settled'))
                );

                CREATE INDEX IF NOT EXISTS idx_research_experiments_run_created
                    ON research_experiments(run_id, created_at);

                CREATE TABLE IF NOT EXISTS research_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES research_runs(run_id),
                    experiment_id TEXT NOT NULL REFERENCES research_experiments(experiment_id),
                    producer_attempt_id TEXT,
                    tool_call_id TEXT,
                    kind TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    uri TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    byte_size INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE (experiment_id, kind, sha256, uri)
                );

                CREATE INDEX IF NOT EXISTS idx_research_artifacts_experiment
                    ON research_artifacts(experiment_id, created_at);

                CREATE TABLE IF NOT EXISTS research_claim_settlements (
                    settlement_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES research_runs(run_id),
                    experiment_id TEXT NOT NULL REFERENCES research_experiments(experiment_id),
                    claim_id TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    expected_champion_version INTEGER,
                    champion_version INTEGER,
                    created_at REAL NOT NULL,
                    UNIQUE (run_id, claim_id),
                    CHECK (verdict IN ('keep', 'discard', 'insufficient'))
                );

                CREATE INDEX IF NOT EXISTS idx_research_claim_settlements_run
                    ON research_claim_settlements(run_id, created_at);

                CREATE TABLE IF NOT EXISTS research_champion_versions (
                    run_id TEXT PRIMARY KEY REFERENCES research_runs(run_id),
                    version INTEGER NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
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
                SELECT run_id, topic, status, last_projected_event, config_json
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
                SELECT run_id, topic, status, last_projected_event, config_json
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
            return LedgerRun(run_id, topic, RUN_CREATED, 0, dict(config or {}))

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

    def record_rag_evidence_disabled(
        self,
        run_id: str,
        *,
        generation: str,
        reason: str = "retention_unavailable",
    ) -> LedgerEvent:
        """Append a fail-closed RAG downgrade without rewriting the run specification."""
        with self.transaction() as conn:
            row = conn.execute("SELECT 1 FROM research_runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown research run: {run_id}")
            return self.append_event(
                conn,
                run_id,
                actor="director",
                event_type="rag_evidence_disabled",
                payload={"generation": generation, "reason": reason},
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

    def record_tool_intent(
        self,
        *,
        tool_call_id: str,
        run_id: str,
        step_id: str,
        attempt_id: str,
        worker_id: str,
        tool_name: str,
        source: str,
        side_effect: str,
        node_name: str,
        proposal: Mapping[str, Any],
        idempotency_key: str | None,
        lease_seconds: float | None = None,
    ) -> LedgerToolCall:
        """Durably record an authorized intent before an external handler runs."""
        with self.transaction() as conn:
            self._require_active_tool_attempt(
                conn,
                run_id=run_id,
                step_id=step_id,
                attempt_id=attempt_id,
                worker_id=worker_id,
            )
            now = time.time()
            call = LedgerToolCall(
                tool_call_id=tool_call_id,
                run_id=run_id,
                step_id=step_id,
                attempt_id=attempt_id,
                node_name=node_name,
                tool_name=tool_name,
                source=source,
                side_effect=side_effect,
                status="intent",
                idempotency_key=idempotency_key,
                proposal=dict(proposal),
                observation={},
                created_at=now,
                updated_at=now,
            )
            self._insert_tool_call(conn, call)
            if lease_seconds is not None:
                conn.execute(
                    """
                    UPDATE research_steps
                    SET lease_expires_at = ?, updated_at = ?
                    WHERE step_id = ?
                    """,
                    (now + max(1.0, float(lease_seconds)), now, step_id),
                )
            self.append_event(
                conn,
                run_id,
                actor="tool_broker",
                event_type="tool_intended",
                payload={
                    "tool_call_id": tool_call_id,
                    "step_id": step_id,
                    "attempt_id": attempt_id,
                    "node_name": node_name,
                    "tool_name": tool_name,
                    "source": source,
                    "side_effect": side_effect,
                    "idempotency_key": idempotency_key,
                    "proposal": dict(proposal),
                },
            )
            return call

    def record_tool_decision(
        self,
        *,
        tool_call_id: str,
        run_id: str,
        step_id: str,
        attempt_id: str,
        worker_id: str,
        tool_name: str,
        source: str,
        side_effect: str,
        node_name: str,
        proposal: Mapping[str, Any],
        idempotency_key: str | None,
        status: str,
        reason: str,
    ) -> LedgerToolCall:
        """Persist a denied or approval-waiting proposal that never reached a handler."""
        if status not in {"denied", "waiting_approval"}:
            raise ValueError(f"invalid non-execution tool status {status!r}")
        with self.transaction() as conn:
            self._require_active_tool_attempt(
                conn,
                run_id=run_id,
                step_id=step_id,
                attempt_id=attempt_id,
                worker_id=worker_id,
            )
            now = time.time()
            call = LedgerToolCall(
                tool_call_id=tool_call_id,
                run_id=run_id,
                step_id=step_id,
                attempt_id=attempt_id,
                node_name=node_name,
                tool_name=tool_name,
                source=source,
                side_effect=side_effect,
                status=status,
                idempotency_key=idempotency_key,
                proposal=dict(proposal),
                observation={"reason": reason},
                created_at=now,
                updated_at=now,
            )
            self._insert_tool_call(conn, call)
            self.append_event(
                conn,
                run_id,
                actor="tool_broker",
                event_type="tool_denied" if status == "denied" else "tool_waiting_approval",
                payload={
                    "tool_call_id": tool_call_id,
                    "step_id": step_id,
                    "attempt_id": attempt_id,
                    "node_name": node_name,
                    "tool_name": tool_name,
                    "reason": reason,
                    "proposal": dict(proposal),
                },
            )
            return call

    def record_tool_observation(
        self,
        *,
        tool_call_id: str,
        run_id: str,
        step_id: str,
        attempt_id: str,
        worker_id: str,
        status: str,
        observation: Mapping[str, Any],
        lease_seconds: float | None = None,
    ) -> LedgerToolCall:
        """Settle a prior intent with a durable observation under the same lease."""
        if status not in {"succeeded", "failed", "timed_out", "unknown"}:
            raise ValueError(f"invalid tool observation status {status!r}")
        with self.transaction() as conn:
            self._require_active_tool_attempt(
                conn,
                run_id=run_id,
                step_id=step_id,
                attempt_id=attempt_id,
                worker_id=worker_id,
            )
            row = conn.execute(
                """
                SELECT * FROM research_tool_calls
                WHERE tool_call_id = ? AND run_id = ? AND step_id = ? AND attempt_id = ?
                """,
                (tool_call_id, run_id, step_id, attempt_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown tool call {tool_call_id!r}")
            if str(row["status"]) != "intent":
                raise RuntimeError(f"tool call {tool_call_id!r} is not awaiting an observation")
            now = time.time()
            conn.execute(
                """
                UPDATE research_tool_calls
                SET status = ?, observation_json = ?, updated_at = ?
                WHERE tool_call_id = ?
                """,
                (status, _as_json(dict(observation)), now, tool_call_id),
            )
            if lease_seconds is not None:
                conn.execute(
                    """
                    UPDATE research_steps
                    SET lease_expires_at = ?, updated_at = ?
                    WHERE step_id = ?
                    """,
                    (now + max(1.0, float(lease_seconds)), now, step_id),
                )
            self.append_event(
                conn,
                run_id,
                actor="tool_broker",
                event_type="tool_observed",
                payload={
                    "tool_call_id": tool_call_id,
                    "step_id": step_id,
                    "attempt_id": attempt_id,
                    "status": status,
                    "observation": dict(observation),
                },
            )
            updated = conn.execute(
                "SELECT * FROM research_tool_calls WHERE tool_call_id = ?", (tool_call_id,)
            ).fetchone()
            assert updated is not None  # protected by the update above
            return self._tool_call_from_row(updated)

    def record_evidence_bundle(
        self,
        *,
        run_id: str,
        step_id: str,
        attempt_id: str,
        worker_id: str,
        bundle: Mapping[str, Any],
    ) -> None:
        """Append generation-pinned retrieval evidence to the durable event trail."""
        with self.transaction() as conn:
            self._require_active_tool_attempt(
                conn,
                run_id=run_id,
                step_id=step_id,
                attempt_id=attempt_id,
                worker_id=worker_id,
            )
            self.append_event(
                conn,
                run_id,
                actor="rag_evidence",
                event_type="rag_evidence_recorded",
                payload={
                    "step_id": step_id,
                    "attempt_id": attempt_id,
                    "bundle": dict(bundle),
                },
            )

    def renew_lease(
        self,
        *,
        run_id: str,
        step_id: str,
        attempt_id: str,
        worker_id: str,
        lease_seconds: float,
    ) -> None:
        """Extend an owned running attempt lease without changing its progress."""
        with self.transaction() as conn:
            self._require_active_tool_attempt(
                conn,
                run_id=run_id,
                step_id=step_id,
                attempt_id=attempt_id,
                worker_id=worker_id,
            )
            now = time.time()
            conn.execute(
                """
                UPDATE research_steps
                SET lease_expires_at = ?, updated_at = ?
                WHERE step_id = ?
                """,
                (now + max(1.0, float(lease_seconds)), now, step_id),
            )

    def tool_calls(self, run_id: str) -> list[LedgerToolCall]:
        """Read the durable tool trail in issue order for one run."""
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM research_tool_calls WHERE run_id = ? ORDER BY created_at, tool_call_id",
                (run_id,),
            ).fetchall()
            return [self._tool_call_from_row(row) for row in rows]

    def latest_tool_call_for_idempotency(
        self,
        run_id: str,
        *,
        step_id: str,
        idempotency_key: str,
    ) -> LedgerToolCall | None:
        """Return the latest durable result for one deterministic idempotency key."""
        with self.transaction() as conn:
            row = conn.execute(
                """
                SELECT * FROM research_tool_calls
                WHERE run_id = ? AND step_id = ? AND idempotency_key = ?
                ORDER BY created_at DESC, tool_call_id DESC
                LIMIT 1
                """,
                (run_id, step_id, idempotency_key),
            ).fetchone()
            return self._tool_call_from_row(row) if row is not None else None

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

    def active_attempt_id(self, step_id: str) -> str | None:
        """Return the current running attempt for ``step_id``, if any."""
        with self.transaction() as conn:
            row = conn.execute(
                """
                SELECT attempt_id FROM research_attempts
                WHERE step_id = ? AND status = 'running'
                ORDER BY attempt_no DESC
                LIMIT 1
                """,
                (step_id,),
            ).fetchone()
            return str(row["attempt_id"]) if row is not None else None

    def active_leased_steps(self, run_id: str) -> list[str]:
        """Return in-flight steps still protected by an unexpired worker lease."""
        with self.transaction() as conn:
            rows = conn.execute(
                """
                SELECT step_id FROM research_steps
                WHERE run_id = ?
                  AND status IN ('running', 'reconciling')
                  AND lease_owner IS NOT NULL
                  AND lease_expires_at > ?
                ORDER BY created_at
                """,
                (run_id, time.time()),
            ).fetchall()
            return [str(row["step_id"]) for row in rows]

    def recoverable_step_ids(self, run_id: str) -> list[str]:
        """Return interrupted steps that require a checkpoint or manual review."""
        with self.transaction() as conn:
            rows = conn.execute(
                """
                SELECT step_id FROM research_steps
                WHERE run_id = ? AND status IN ('unknown', 'reconciling')
                ORDER BY created_at
                """,
                (run_id,),
            ).fetchall()
            return [str(row["step_id"]) for row in rows]

    def record_checkpoint(
        self,
        *,
        run_id: str,
        step_id: str,
        attempt_id: str,
        worker_id: str,
        lease_seconds: float,
        step_name: str,
        context_payload: Mapping[str, Any],
        workflow_state: Mapping[str, Any],
        manifest: Mapping[str, Any],
    ) -> LedgerCheckpoint:
        """Atomically persist a JSON checkpoint and renew its worker lease."""
        with self.transaction() as conn:
            now = time.time()
            step = conn.execute(
                """
                SELECT run_id, lease_owner, lease_expires_at
                FROM research_steps WHERE step_id = ?
                """,
                (step_id,),
            ).fetchone()
            if step is None or str(step["run_id"]) != run_id:
                raise KeyError(f"unknown step {step_id!r} for run {run_id!r}")
            if (
                step["lease_owner"] != worker_id
                or step["lease_expires_at"] is None
                or float(step["lease_expires_at"]) <= now
            ):
                raise LeaseOwnershipError(f"worker {worker_id!r} does not own active lease")
            attempt = conn.execute(
                "SELECT status FROM research_attempts WHERE attempt_id = ? AND step_id = ?",
                (attempt_id, step_id),
            ).fetchone()
            if attempt is None or str(attempt["status"]) != "running":
                raise LeaseOwnershipError(f"attempt {attempt_id!r} is not running")

            row = conn.execute(
                """
                SELECT COALESCE(MAX(checkpoint_seq), 0) + 1 AS next_seq
                FROM research_checkpoints WHERE step_id = ? AND attempt_id = ?
                """,
                (step_id, attempt_id),
            ).fetchone()
            checkpoint_seq = int(row["next_seq"])
            checkpoint_id = uuid.uuid4().hex
            checkpoint = LedgerCheckpoint(
                checkpoint_id=checkpoint_id,
                run_id=run_id,
                step_id=step_id,
                attempt_id=attempt_id,
                checkpoint_seq=checkpoint_seq,
                step_name=step_name,
                context_payload=dict(context_payload),
                workflow_state=dict(workflow_state),
                manifest=dict(manifest),
                created_at=now,
            )
            conn.execute(
                """
                INSERT INTO research_checkpoints(
                    checkpoint_id, run_id, step_id, attempt_id, checkpoint_seq,
                    step_name, context_json, workflow_state_json, manifest_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint.checkpoint_id,
                    checkpoint.run_id,
                    checkpoint.step_id,
                    checkpoint.attempt_id,
                    checkpoint.checkpoint_seq,
                    checkpoint.step_name,
                    _as_json(checkpoint.context_payload),
                    _as_json(checkpoint.workflow_state),
                    _as_json(checkpoint.manifest),
                    checkpoint.created_at,
                ),
            )
            conn.execute(
                "UPDATE research_attempts SET checkpoint_ref = ? WHERE attempt_id = ?",
                (checkpoint_id, attempt_id),
            )
            conn.execute(
                """
                INSERT INTO research_attempt_progress(
                    step_id, attempt_id, node_name, boundary_kind, checkpoint_id, updated_at
                ) VALUES (?, ?, ?, 'checkpointed', ?, ?)
                ON CONFLICT(step_id) DO UPDATE SET
                    attempt_id = excluded.attempt_id,
                    node_name = excluded.node_name,
                    boundary_kind = excluded.boundary_kind,
                    checkpoint_id = excluded.checkpoint_id,
                    updated_at = excluded.updated_at
                """,
                (step_id, attempt_id, step_name, checkpoint_id, now),
            )
            conn.execute(
                """
                UPDATE research_steps
                SET lease_expires_at = ?, updated_at = ?
                WHERE step_id = ?
                """,
                (now + max(1.0, float(lease_seconds)), now, step_id),
            )
            self.append_event(
                conn,
                run_id,
                actor="checkpoint",
                event_type="workflow_checkpointed",
                payload={
                    "checkpoint_id": checkpoint_id,
                    "step_id": step_id,
                    "attempt_id": attempt_id,
                    "checkpoint_seq": checkpoint_seq,
                    "step_name": step_name,
                    "manifest": dict(manifest),
                },
            )
            return checkpoint

    def record_workflow_step_started(
        self,
        *,
        run_id: str,
        step_id: str,
        attempt_id: str,
        worker_id: str,
        lease_seconds: float,
        node_name: str,
    ) -> None:
        """Record an internal node's start before it can issue an external tool."""
        with self.transaction() as conn:
            now = time.time()
            step = conn.execute(
                """
                SELECT run_id, lease_owner, lease_expires_at
                FROM research_steps WHERE step_id = ?
                """,
                (step_id,),
            ).fetchone()
            if step is None or str(step["run_id"]) != run_id:
                raise KeyError(f"unknown step {step_id!r} for run {run_id!r}")
            if (
                step["lease_owner"] != worker_id
                or step["lease_expires_at"] is None
                or float(step["lease_expires_at"]) <= now
            ):
                raise LeaseOwnershipError(f"worker {worker_id!r} does not own active lease")
            attempt = conn.execute(
                "SELECT status FROM research_attempts WHERE attempt_id = ? AND step_id = ?",
                (attempt_id, step_id),
            ).fetchone()
            if attempt is None or str(attempt["status"]) != "running":
                raise LeaseOwnershipError(f"attempt {attempt_id!r} is not running")

            conn.execute(
                """
                INSERT INTO research_attempt_progress(
                    step_id, attempt_id, node_name, boundary_kind, checkpoint_id, updated_at
                ) VALUES (?, ?, ?, 'started', NULL, ?)
                ON CONFLICT(step_id) DO UPDATE SET
                    attempt_id = excluded.attempt_id,
                    node_name = excluded.node_name,
                    boundary_kind = excluded.boundary_kind,
                    checkpoint_id = NULL,
                    updated_at = excluded.updated_at
                """,
                (step_id, attempt_id, node_name, now),
            )
            conn.execute(
                """
                UPDATE research_steps
                SET lease_expires_at = ?, updated_at = ?
                WHERE step_id = ?
                """,
                (now + max(1.0, float(lease_seconds)), now, step_id),
            )
            self.append_event(
                conn,
                run_id,
                actor="checkpoint",
                event_type="workflow_step_started",
                payload={
                    "step_id": step_id,
                    "attempt_id": attempt_id,
                    "node_name": node_name,
                },
            )

    def inflight_workflow_step(self, step_id: str) -> str | None:
        """Return the node that started after the last safe checkpoint, if any."""
        with self.transaction() as conn:
            row = conn.execute(
                """
                SELECT node_name FROM research_attempt_progress
                WHERE step_id = ? AND boundary_kind = 'started'
                """,
                (step_id,),
            ).fetchone()
            return str(row["node_name"]) if row is not None else None

    def latest_checkpoint_for_step(self, step_id: str) -> LedgerCheckpoint | None:
        """Return the newest checkpoint for one logical cycle step."""
        with self.transaction() as conn:
            row = conn.execute(
                """
                SELECT checkpoint_id, run_id, step_id, attempt_id, checkpoint_seq,
                       step_name, context_json, workflow_state_json, manifest_json, created_at
                FROM research_checkpoints WHERE step_id = ?
                ORDER BY created_at DESC, checkpoint_seq DESC
                LIMIT 1
                """,
                (step_id,),
            ).fetchone()
            return self._checkpoint_from_row(row)

    @staticmethod
    def _require_active_tool_attempt(
        conn: sqlite3.Connection,
        *,
        run_id: str,
        step_id: str,
        attempt_id: str,
        worker_id: str,
    ) -> None:
        now = time.time()
        step = conn.execute(
            """
            SELECT run_id, lease_owner, lease_expires_at
            FROM research_steps WHERE step_id = ?
            """,
            (step_id,),
        ).fetchone()
        if step is None or str(step["run_id"]) != run_id:
            raise KeyError(f"unknown step {step_id!r} for run {run_id!r}")
        if (
            step["lease_owner"] != worker_id
            or step["lease_expires_at"] is None
            or float(step["lease_expires_at"]) <= now
        ):
            raise LeaseOwnershipError(f"worker {worker_id!r} does not own active lease")
        attempt = conn.execute(
            "SELECT status FROM research_attempts WHERE attempt_id = ? AND step_id = ?",
            (attempt_id, step_id),
        ).fetchone()
        if attempt is None or str(attempt["status"]) != "running":
            raise LeaseOwnershipError(f"attempt {attempt_id!r} is not running")

    @staticmethod
    def _insert_tool_call(conn: sqlite3.Connection, call: LedgerToolCall) -> None:
        conn.execute(
            """
            INSERT INTO research_tool_calls(
                tool_call_id, run_id, step_id, attempt_id, node_name, tool_name,
                source, side_effect, status, idempotency_key, proposal_json,
                observation_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                call.tool_call_id,
                call.run_id,
                call.step_id,
                call.attempt_id,
                call.node_name,
                call.tool_name,
                call.source,
                call.side_effect,
                call.status,
                call.idempotency_key,
                _as_json(call.proposal),
                _as_json(call.observation),
                call.created_at,
                call.updated_at,
            ),
        )

    @staticmethod
    def _run_from_row(row: sqlite3.Row | None) -> LedgerRun | None:
        if row is None:
            return None
        config = _from_json(row["config_json"], {})
        return LedgerRun(
            run_id=str(row["run_id"]),
            topic=str(row["topic"]),
            status=str(row["status"]),
            last_projected_event=int(row["last_projected_event"]),
            config=dict(config) if isinstance(config, Mapping) else {},
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

    @staticmethod
    def _checkpoint_from_row(row: sqlite3.Row | None) -> LedgerCheckpoint | None:
        if row is None:
            return None
        return LedgerCheckpoint(
            checkpoint_id=str(row["checkpoint_id"]),
            run_id=str(row["run_id"]),
            step_id=str(row["step_id"]),
            attempt_id=str(row["attempt_id"]),
            checkpoint_seq=int(row["checkpoint_seq"]),
            step_name=str(row["step_name"]),
            context_payload=_from_json(row["context_json"], {}),
            workflow_state=_from_json(row["workflow_state_json"], {}),
            manifest=_from_json(row["manifest_json"], {}),
            created_at=float(row["created_at"]),
        )

    @staticmethod
    def _tool_call_from_row(row: sqlite3.Row) -> LedgerToolCall:
        return LedgerToolCall(
            tool_call_id=str(row["tool_call_id"]),
            run_id=str(row["run_id"]),
            step_id=str(row["step_id"]),
            attempt_id=str(row["attempt_id"]),
            node_name=str(row["node_name"]),
            tool_name=str(row["tool_name"]),
            source=str(row["source"]),
            side_effect=str(row["side_effect"]),
            status=str(row["status"]),
            idempotency_key=(str(row["idempotency_key"]) if row["idempotency_key"] else None),
            proposal=dict(_from_json(row["proposal_json"], {})),
            observation=dict(_from_json(row["observation_json"], {})),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )
