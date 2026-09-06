"""Tests for first-class Claim/Evidence objects (schema v14+).

Covers the ``evidence`` and ``claims`` tables, the ``record_evidence`` /
``record_claim`` write helpers, the answer-recording path that materializes
sparse evidence + claim rows, and migration idempotency.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from drbrain.storage.database import Database

EVIDENCE_COLS = [
    "evidence_id",
    "paper_id",
    "node_id",
    "page",
    "snippet",
    "value",
    "unit",
    "conditions",
    "provenance",
    "authority",
    "created_at",
]

CLAIM_COLS = [
    "claim_id",
    "label",
    "claim_text",
    "claim_type",
    "authority",
    "provenance",
    "confidence",
    "valid_from",
    "valid_to",
    "created_at",
]


def _table_names(db: Database) -> list[str]:
    return [
        r[0]
        for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    ]


def _cols(db: Database, table: str) -> list[str]:
    return [r[1] for r in db.conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _versions(db: Database) -> list[int]:
    return [
        r[0]
        for r in db.conn.execute("SELECT version FROM schema_versions ORDER BY version").fetchall()
    ]


# ── schema (fresh database) ───────────────────────────────────────


def test_fresh_db_has_claims_and_evidence_tables(tmp_db):
    tables = _table_names(tmp_db)
    assert "evidence" in tables
    assert "claims" in tables


def test_fresh_db_evidence_columns(tmp_db):
    cols = _cols(tmp_db, "evidence")
    for c in EVIDENCE_COLS:
        assert c in cols, f"evidence missing column {c}"


def test_fresh_db_claims_columns(tmp_db):
    cols = _cols(tmp_db, "claims")
    for c in CLAIM_COLS:
        assert c in cols, f"claims missing column {c}"


# ── record_evidence ────────────────────────────────────────────────


def test_record_evidence_inserts_and_roundtrips(tmp_db):
    eid = tmp_db.record_evidence(
        "paper-1",
        "node-7",
        page="12",
        snippet="the yield is 42%",
        value="42",
        unit="%",
        conditions="room temperature",
        provenance="TOOL_RESULT",
        authority="peer_reviewed",
    )
    assert eid == "paper-1:node-7"

    row = tmp_db.conn.execute(
        "SELECT evidence_id, paper_id, node_id, page, snippet, value, unit, "
        "conditions, provenance, authority FROM evidence"
    ).fetchone()
    assert row == (
        "paper-1:node-7",
        "paper-1",
        "node-7",
        "12",
        "the yield is 42%",
        "42",
        "%",
        "room temperature",
        "TOOL_RESULT",
        "peer_reviewed",
    )


def test_record_evidence_defaults(tmp_db):
    eid = tmp_db.record_evidence("p1")
    assert eid == "p1"  # paper-only grounding
    row = tmp_db.conn.execute(
        "SELECT node_id, page, snippet, value, unit, conditions, provenance, authority "
        "FROM evidence"
    ).fetchone()
    assert row == ("", "", "", "", "", "", "", "")


def test_record_evidence_idempotent(tmp_db):
    first = tmp_db.record_evidence("p1", "n1")
    second = tmp_db.record_evidence("p1", "n1")
    assert first == second == "p1:n1"
    count = tmp_db.conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
    assert count == 1


# ── record_claim ───────────────────────────────────────────────────


def test_record_claim_inserts_and_roundtrips(tmp_db):
    cid = tmp_db.record_claim(
        "Band gap of perovskite X",
        "The band gap is 1.5 eV.",
        claim_type="Conclusion",
        authority="peer_reviewed",
        provenance="MODEL_INFERRED",
        confidence=0.9,
        valid_from=2020,
    )
    assert cid.startswith("claim_")

    row = tmp_db.conn.execute(
        "SELECT label, claim_text, claim_type, authority, provenance, confidence, "
        "valid_from, valid_to FROM claims"
    ).fetchone()
    assert row == (
        "Band gap of perovskite X",
        "The band gap is 1.5 eV.",
        "Conclusion",
        "peer_reviewed",
        "MODEL_INFERRED",
        0.9,
        2020,
        None,
    )


def test_record_claim_idempotent(tmp_db):
    first = tmp_db.record_claim("q?", "a", claim_type="Conclusion")
    second = tmp_db.record_claim("q?", "a", claim_type="Conclusion")
    assert first == second
    count = tmp_db.conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    assert count == 1


# ── answer path materializes evidence + claim ──────────────────────


def test_record_answer_materializes_evidence_and_claim(tmp_db):
    answer_id = tmp_db.record_answer(
        "What is the band gap?",
        "1.5 eV.",
        evidence_ids=[
            "10.1002_adma.202308655:0000",
            "10.1002_adma.202308655:0001",
            "arxiv_1234.5678",
        ],
        provenance="TOOL_RESULT",
    )
    assert answer_id != 0

    # Evidence rows: each ``paper:node`` (or bare ``paper``) became a row.
    rows = tmp_db.conn.execute(
        "SELECT evidence_id, paper_id, node_id, provenance FROM evidence ORDER BY evidence_id"
    ).fetchall()
    assert rows == [
        ("10.1002_adma.202308655:0000", "10.1002_adma.202308655", "0000", "TOOL_RESULT"),
        ("10.1002_adma.202308655:0001", "10.1002_adma.202308655", "0001", "TOOL_RESULT"),
        ("arxiv_1234.5678", "arxiv_1234.5678", "", "TOOL_RESULT"),
    ]

    # A single claim row was materialized for the answer.
    claims = tmp_db.conn.execute(
        "SELECT label, claim_text, claim_type, provenance FROM claims"
    ).fetchall()
    assert claims == [("What is the band gap?", "1.5 eV.", "", "TOOL_RESULT")]


def test_record_answer_without_evidence_records_only_claim(tmp_db):
    tmp_db.record_answer("q?", "a")
    assert tmp_db.conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 0
    assert tmp_db.conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 1


def test_record_answer_evidence_idempotent(tmp_db):
    """Re-recording the same evidence across answers dedups to one row."""
    tmp_db.record_answer("q1?", "a1", evidence_ids=["p1:n1"])
    tmp_db.record_answer("q2?", "a2", evidence_ids=["p1:n1", "p2:n2"])
    assert tmp_db.conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 2
    assert tmp_db.conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 2


# ── migration (pre-existing database) ──────────────────────────────


def _make_v13_db(path: Path) -> None:
    """Build a minimal pre-v14 database (schema_versions 1-13 only)."""
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE schema_versions (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.executemany(
        "INSERT INTO schema_versions (version) VALUES (?)",
        [(i,) for i in range(1, 14)],
    )
    conn.commit()
    conn.close()


def test_migrate_v13_adds_claims_and_evidence(tmp_path):
    """Opening a v13 database applies every later migration in place."""
    db_path = tmp_path / "old.db"
    _make_v13_db(db_path)

    db = Database(db_path)

    assert _versions(db) == list(range(1, 20))
    assert "evidence" in _table_names(db)
    assert "claims" in _table_names(db)
    assert "claim_evidence" in _table_names(db)
    assert "owner_principal" in _cols(db, "agent_sessions")

    # Both tables are immediately writable.
    db.record_evidence("p1", "n1")
    db.record_claim("q?", "a")
    assert db.conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 1
    assert db.conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 1

    db.close()


def test_migration_is_idempotent(tmp_path):
    """Re-opening an already-migrated database applies no new versions."""
    db_path = tmp_path / "mig.db"
    _make_v13_db(db_path)

    db = Database(db_path)
    db.close()

    db2 = Database(db_path)
    assert _versions(db2) == list(range(1, 20))
    assert "evidence" in _table_names(db2)
    assert "claims" in _table_names(db2)
    assert "claim_evidence" in _table_names(db2)
    db2.close()


def test_record_claim_evidence_preserves_many_to_many_links(tmp_db):
    claim_id = tmp_db.record_claim("research", "Verified claim", claim_type="Conclusion")
    evidence_id = tmp_db.record_evidence(
        "paper-1",
        "section-2",
        evidence_id="ev-real",
        snippet="The supporting passage.",
    )

    linked = tmp_db.record_claim_evidence(claim_id, [evidence_id, evidence_id])

    assert linked == ["ev-real"]
    assert tmp_db.conn.execute("SELECT claim_id, evidence_id FROM claim_evidence").fetchall() == [
        (claim_id, "ev-real")
    ]

    # Re-recording an idempotent claim must not cascade-delete its evidence link.
    tmp_db.record_claim("research", "Verified claim", claim_type="Conclusion")
    assert tmp_db.conn.execute("SELECT claim_id, evidence_id FROM claim_evidence").fetchall() == [
        (claim_id, "ev-real")
    ]

    with pytest.raises(sqlite3.IntegrityError):
        tmp_db.record_claim_evidence(claim_id, ["unknown-evidence"])

    # Updating the evidence must not delete an existing claim relationship.
    tmp_db.record_evidence(
        "paper-1",
        "section-2",
        evidence_id="ev-real",
        snippet="A richer supporting passage.",
    )
    assert tmp_db.conn.execute("SELECT claim_id, evidence_id FROM claim_evidence").fetchall() == [
        (claim_id, "ev-real")
    ]
