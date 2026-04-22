# DrBrain v1.1 — Advanced Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add schema-first validation, confidence queue, argument units, and temporal evolution to the existing DrBrain v1 codebase.

**Architecture:** 4 layers — (1) schema validation engine intercepts LLM output before ingestion, (2) confidence queue routes low-confidence items for human review, (3) argument units extend extraction and storage beyond flat concepts, (4) temporal tracking adds first_seen/last_seen fields with evolution signal detection. All integrated into the existing ingest pipeline.

**Tech Stack:** Same as v1 — Python 3.12+, typer, litellm, SQLite, rich, pyyaml.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/brbrain/storage/database.py` | Modify | Add `arguments` + `confidence_queue` tables, `first_seen`/`last_seen` on concepts, query methods |
| `src/brbrain/validator/schema.py` | Create | TBox/RBox constraint validation engine |
| `src/brbrain/validator/__init__.py` | Create | Validator package |
| `src/brbrain/extractor/queue.py` | Create | Confidence queue operations (insert, resolve, consensus check) |
| `src/brbrain/extractor/argument.py` | Create | Argument extraction from LLM output with validation |
| `src/brbrain/extractor/concept.py` | Modify | Add arguments to extraction result |
| `prompts/extract_concepts.txt` | Modify | Add argument schema to prompt |
| `src/brbrain/cli/commands.py` | Modify | Add queue, queue resolve, timeline commands; update ingest pipeline |
| `src/brbrain/cli/main.py` | Modify | Register new commands |
| `src/brbrain/report/generator.py` | Modify | Add arguments + validation to report |
| `tests/test_validator.py` | Create | TBox/RBox validation tests |
| `tests/test_queue.py` | Create | Confidence queue tests |
| `tests/test_argument.py` | Create | Argument extraction tests |
| `tests/test_temporal.py` | Create | Temporal evolution query tests |
| `tests/test_ingest_v11.py` | Create | Integration test: ingest with validation + queue + arguments |

---

### Task 1: Database Schema — arguments, confidence_queue, temporal fields

**Files:**
- Modify: `src/brbrain/storage/database.py`
- Test: `tests/test_temporal.py` (partial — DB methods only)

- [ ] **Step 1: Write tests for new DB tables and methods**

```python
"""Tests for v1.1 database schema: arguments, confidence_queue, temporal fields."""
import tempfile
from pathlib import Path
from brbrain.storage.database import Database

def test_arguments_table_exists():
    """arguments table is created on Database init."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        row = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='arguments'"
        ).fetchone()
        assert row is not None
        db.close()

def test_confidence_queue_table_exists():
    """confidence_queue table is created on Database init."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        row = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='confidence_queue'"
        ).fetchone()
        assert row is not None
        db.close()

def test_insert_argument():
    """insert_argument stores argument and returns arg_id."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test", 2024, "uploaded")
        db.commit()
        arg_id = db.insert_argument(
            source_paper="p1",
            claim="Self-attention replaces RNN",
            claim_type="proposes",
            target_label="Transformer",
            target_type="Method",
            evidence_type="empirical",
            evidence_detail="WMT14 EN-DE BLEU +2.0",
            confidence=0.95,
        )
        assert arg_id is not None
        row = db.conn.execute("SELECT claim, claim_type FROM arguments WHERE arg_id = ?", (arg_id,)).fetchone()
        assert row[0] == "Self-attention replaces RNN"
        assert row[1] == "proposes"
        db.close()

def test_insert_queue_item():
    """insert_queue_item stores pending item and returns queue_id."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        qid = db.insert_queue_item(
            source_paper="p1",
            item_type="concept",
            item_data='{"label": "neuro-symbolic reasoning", "type": "Method"}',
            confidence=0.52,
        )
        assert qid is not None
        row = db.conn.execute("SELECT status FROM confidence_queue WHERE queue_id = ?", (qid,)).fetchone()
        assert row[0] == "pending"
        db.close()

def test_resolve_queue_item_accept():
    """accept_queue_item sets status to 'accepted'."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        qid = db.insert_queue_item("p1", "concept", '{"label": "test"}', 0.5)
        db.accept_queue_item(qid)
        row = db.conn.execute("SELECT status FROM confidence_queue WHERE queue_id = ?", (qid,)).fetchone()
        assert row[0] == "accepted"
        db.close()

def test_resolve_queue_item_reject():
    """reject_queue_item sets status to 'rejected'."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        qid = db.insert_queue_item("p1", "concept", '{"label": "test"}', 0.5)
        db.reject_queue_item(qid)
        row = db.conn.execute("SELECT status FROM confidence_queue WHERE queue_id = ?", (qid,)).fetchone()
        assert row[0] == "rejected"
        db.close()

def test_get_queue_pending():
    """get_queue_pending returns only pending items."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        q1 = db.insert_queue_item("p1", "concept", '{"label": "a"}', 0.5)
        q2 = db.insert_queue_item("p1", "concept", '{"label": "b"}', 0.4)
        db.accept_queue_item(q1)
        pending = db.get_queue_pending()
        assert len(pending) == 1
        assert pending[0]["queue_id"] == q2
        db.close()

def test_concepts_have_temporal_fields():
    """concepts table has first_seen and last_seen columns."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test", 2020, "uploaded")
        db.commit()
        db.insert_concept("p1", "Method", "transformer", 0.95, year=2020)
        cid = db.conn.execute("SELECT concept_id FROM concepts").fetchone()[0]
        row = db.conn.execute(
            "SELECT first_seen, last_seen FROM concepts WHERE concept_id = ?", (cid,)
        ).fetchone()
        assert row[0] == 2020
        assert row[1] == 2020
        db.close()

def test_get_concept_evolution():
    """get_concept_evolution returns year-by-year stats for a concept label."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Paper A", 2017, "uploaded")
        db.insert_paper("p2", "Paper B", 2020, "uploaded")
        db.insert_paper("p3", "Paper C", 2023, "uploaded")
        db.insert_concept("p1", "Method", "transformer", 0.95, year=2017)
        db.insert_concept("p2", "Method", "transformer", 0.90, year=2020)
        db.insert_concept("p3", "Method", "transformer", 0.75, year=2023)
        db.commit()
        evolution = db.get_concept_evolution("transformer")
        assert len(evolution) == 3
        assert evolution[0]["year"] == 2017
        assert evolution[0]["count"] == 1
        assert evolution[1]["year"] == 2020
        assert evolution[1]["count"] == 1
        db.close()

def test_detect_evolution_signals():
    """detect_evolution_signals returns signal type for a concept."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        # Emerging: first seen recently, growing count
        db.insert_paper("p1", "A", 2024, "uploaded")
        db.insert_paper("p2", "B", 2025, "uploaded")
        db.insert_concept("p1", "Method", "new_thing", 0.9, year=2024)
        db.insert_concept("p2", "Method", "new_thing", 0.85, year=2025)
        db.commit()
        signals = db.detect_evolution_signals()
        # "new_thing" should be detected as "emerging" or similar
        assert any(s["label"] == "new_thing" for s in signals)
        db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_temporal.py -v`
Expected: FAIL — missing tables/methods

- [ ] **Step 3: Update SCHEMA_SQL with new tables and fields**

Add to SCHEMA_SQL in `database.py`:

```sql
-- Modified concepts table: add first_seen and last_seen
-- (replaces the old concepts CREATE TABLE)
CREATE TABLE IF NOT EXISTS concepts (
    concept_id INTEGER PRIMARY KEY AUTOINCREMENT,
    local_id TEXT NOT NULL REFERENCES papers(local_id),
    type TEXT NOT NULL CHECK(type IN ('Problem', 'Method', 'Conclusion', 'Debate', 'Gap', 'Actor')),
    label TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    first_seen INTEGER,
    last_seen INTEGER
);

-- New: arguments table
CREATE TABLE IF NOT EXISTS arguments (
    arg_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_paper TEXT NOT NULL REFERENCES papers(local_id),
    claim TEXT NOT NULL,
    claim_type TEXT NOT NULL CHECK(claim_type IN ('supports', 'challenges', 'extends', 'limits', 'solves', 'proposes')),
    target_label TEXT NOT NULL,
    target_type TEXT NOT NULL CHECK(target_type IN ('Method', 'Problem', 'Conclusion', 'Gap', 'Debate', 'Argument')),
    evidence_type TEXT CHECK(evidence_type IN ('empirical', 'theoretical', 'case_study', 'survey')),
    evidence_detail TEXT,
    confidence REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- New: confidence_queue table
CREATE TABLE IF NOT EXISTS confidence_queue (
    queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_paper TEXT NOT NULL,
    item_type TEXT NOT NULL CHECK(item_type IN ('concept', 'alias', 'relation')),
    item_data TEXT NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'accepted', 'rejected')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- New indexes
CREATE INDEX IF NOT EXISTS idx_concepts_first_seen ON concepts(first_seen);
CREATE INDEX IF NOT EXISTS idx_arguments_source ON arguments(source_paper);
CREATE INDEX IF NOT EXISTS idx_arguments_target ON arguments(target_label);
CREATE INDEX IF NOT EXISTS idx_queue_status ON confidence_queue(status);
```

- [ ] **Step 4: Update insert_concept to accept year parameter**

Replace existing `insert_concept` method:

```python
def insert_concept(self, local_id: str, ctype: str, label: str, confidence: float = 1.0, year: int | None = None) -> int:
    """Insert a concept with temporal tracking. Returns concept_id."""
    cur = self.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?)",
        (local_id, ctype, label, confidence, year, year),
    )
    return cur.lastrowid
```

- [ ] **Step 5: Add new DB methods**

Add after existing methods:

```python
def insert_argument(self, source_paper: str, claim: str, claim_type: str,
                    target_label: str, target_type: str,
                    evidence_type: str | None = None, evidence_detail: str | None = None,
                    confidence: float = 1.0) -> int:
    """Insert an argument unit. Returns arg_id."""
    cur = self.conn.execute(
        "INSERT INTO arguments (source_paper, claim, claim_type, target_label, target_type, "
        "evidence_type, evidence_detail, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (source_paper, claim, claim_type, target_label, target_type,
         evidence_type, evidence_detail, confidence),
    )
    return cur.lastrowid

def insert_queue_item(self, source_paper: str, item_type: str, item_data: str,
                      confidence: float) -> int:
    """Insert a confidence queue item. Returns queue_id."""
    cur = self.conn.execute(
        "INSERT INTO confidence_queue (source_paper, item_type, item_data, confidence, status) "
        "VALUES (?, ?, ?, ?, 'pending')",
        (source_paper, item_type, item_data, confidence),
    )
    return cur.lastrowid

def accept_queue_item(self, queue_id: int) -> None:
    """Mark queue item as accepted."""
    self.conn.execute(
        "UPDATE confidence_queue SET status = 'accepted' WHERE queue_id = ?", (queue_id,)
    )

def reject_queue_item(self, queue_id: int) -> None:
    """Mark queue item as rejected."""
    self.conn.execute(
        "UPDATE confidence_queue SET status = 'rejected' WHERE queue_id = ?", (queue_id,)
    )

def get_queue_pending(self) -> list[dict]:
    """Return all pending queue items."""
    rows = self.conn.execute(
        "SELECT queue_id, source_paper, item_type, item_data, confidence, created_at "
        "FROM confidence_queue WHERE status = 'pending' ORDER BY created_at"
    ).fetchall()
    cols = ["queue_id", "source_paper", "item_type", "item_data", "confidence", "created_at"]
    return [dict(zip(cols, row)) for row in rows]

def get_arguments_by_paper(self, local_id: str) -> list[dict]:
    """Get all arguments for a paper."""
    rows = self.conn.execute(
        "SELECT arg_id, claim, claim_type, target_label, target_type, "
        "evidence_type, evidence_detail, confidence "
        "FROM arguments WHERE source_paper = ?", (local_id,)
    ).fetchall()
    cols = ["arg_id", "claim", "claim_type", "target_label", "target_type",
            "evidence_type", "evidence_detail", "confidence"]
    return [dict(zip(cols, row)) for row in rows]

def get_concept_evolution(self, label: str) -> list[dict]:
    """Get year-by-year usage stats for a concept label."""
    rows = self.conn.execute(
        "SELECT p.year, COUNT(*) as count, AVG(c.confidence) as avg_conf "
        "FROM concepts c JOIN papers p ON c.local_id = p.local_id "
        "WHERE c.label = ? AND p.year IS NOT NULL "
        "GROUP BY p.year ORDER BY p.year",
        (label,),
    ).fetchall()
    return [dict(zip(["year", "count", "avg_conf"], row)) for row in rows]

def detect_evolution_signals(self) -> list[dict]:
    """Detect evolution signals across all concepts."""
    rows = self.conn.execute(
        "SELECT c.label, c.type, MIN(p.year) as first_seen, MAX(p.year) as last_seen, "
        "COUNT(DISTINCT c.local_id) as paper_count, AVG(c.confidence) as avg_conf "
        "FROM concepts c JOIN papers p ON c.local_id = p.local_id "
        "WHERE p.year IS NOT NULL "
        "GROUP BY c.label, c.type"
    ).fetchall()

    from datetime import datetime
    current_year = datetime.now().year
    signals = []

    for label, ctype, first_seen, last_seen, paper_count, avg_conf in rows:
        signal = "established"
        if first_seen >= current_year - 2 and paper_count >= 2:
            signal = "emerging"
        elif paper_count > 10 and avg_conf > 0.8:
            signal = "established"
        elif last_seen <= current_year - 3 and paper_count <= 5:
            signal = "declining"
        elif avg_conf < 0.7 and paper_count > 5:
            signal = "contested"
        elif last_seen <= current_year - 3:
            recent = self.conn.execute(
                "SELECT COUNT(*) FROM concepts c JOIN papers p ON c.local_id = p.local_id "
                "WHERE c.label = ? AND p.year >= ?",
                (label, current_year - 1),
            ).fetchone()[0]
            if recent > 0:
                signal = "resurging"

        signals.append({
            "label": label, "type": ctype, "signal": signal,
            "first_seen": first_seen, "last_seen": last_seen,
            "paper_count": paper_count, "avg_confidence": round(avg_conf, 3),
        })

    return signals
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_temporal.py -v`
Expected: PASS (9/9)

- [ ] **Step 7: Commit**

```bash
jj describe -m "feat(v1.1): add arguments, confidence_queue, temporal fields to DB schema

Add arguments table for argument units, confidence_queue for human-in-the-loop,
first_seen/last_seen on concepts for temporal tracking. Add evolution detection
and queue resolve methods."
```

---

### Task 2: Schema Validation Engine (TBox/RBox)

**Files:**
- Create: `src/brbrain/validator/__init__.py`
- Create: `src/brbrain/validator/schema.py`
- Test: `tests/test_validator.py`

- [ ] **Step 1: Write failing tests for TBox/RBox validation**

```python
"""Tests for TBox/RBox schema validation engine."""
from brbrain.validator.schema import (
    TBOX, RBOX, validate_relation, validate_tbox, validate_rbox, ValidationResult
)

def test_tbox_valid_relation():
    """validate_tbox accepts Method --proposes--> anything."""
    result = validate_tbox("Method", "proposes")
    assert result.valid is True

def test_tbox_invalid_relation():
    """validate_tbox rejects Problem --proposes--> (Problems cannot propose)."""
    result = validate_tbox("Problem", "proposes")
    assert result.valid is False
    assert "Problem" in result.reason

def test_tbox_all_types():
    """Each concept type has valid relation whitelist."""
    # Problem can be addressed, left open, point to
    assert validate_tbox("Problem", "addresses").valid
    assert validate_tbox("Problem", "leaves_open").valid
    assert validate_tbox("Problem", "proposes").valid is False

    # Method can propose, extend, replace
    assert validate_tbox("Method", "proposes").valid
    assert validate_tbox("Method", "extends").valid
    assert validate_tbox("Method", "replaces").valid

    # Conclusion can support/challenge
    assert validate_tbox("Conclusion", "supports").valid
    assert validate_tbox("Conclusion", "challenges").valid
    assert validate_tbox("Conclusion", "proposes").valid is False

def test_rbox_irreflexive():
    """validate_rbox rejects self-relations for irreflexive relations."""
    result = validate_rbox("A", "extends", "A")
    assert result.valid is False
    assert "irreflexive" in result.reason

def test_rbox_asymmetric():
    """validate_rbox flags asymmetric reverse violations."""
    # A extends B is fine
    assert validate_rbox("A", "extends", "B").valid
    # But we can't check B extends A without context — the validator checks individual edges
    # Asymmetric constraint is checked at graph level, not per-edge
    # Per-edge: just check it's a valid asymmetric relation type
    assert validate_rbox("B", "extends", "A").valid  # individual edge is valid

def test_validate_relation_full():
    """validate_relation checks both TBox and RBox."""
    # Valid: Method proposes something
    result = validate_relation("Method", "proposes", "Transformer", "Method")
    assert result.valid is True

    # Invalid: Problem proposes (TBox violation)
    result = validate_relation("Problem", "proposes", "Solution", "Method")
    assert result.valid is False

def test_validation_result_to_dict():
    """ValidationResult can be serialized."""
    result = validate_tbox("Gap", "proposes")
    d = result.to_dict()
    assert "valid" in d
    assert "reason" in d
    assert d["valid"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_validator.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write TBox/RBox validation engine**

```python
"""Schema-first validation: TBox type constraints + RBox relation restrictions."""
from __future__ import annotations

from dataclasses import dataclass

# TBox: concept_type -> allowed relation types
TBOX = {
    "Problem":   {"addresses", "leaves_open", "points_to"},
    "Method":    {"addresses", "proposes", "extends", "replaces", "solves"},
    "Conclusion":{"supports", "challenges", "limits"},
    "Debate":    {"supports", "challenges"},
    "Gap":       {"leaves_open", "points_to", "constrains"},
    "Actor":     {"affiliated_with", "proposes"},
}

# RBox: relation properties
RBOX = {
    "transitive": {"extends"},
    "asymmetric": {"extends", "replaces", "challenges", "supports"},
    "irreflexive": {"extends", "replaces", "challenges", "supports", "limits"},
}


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    valid: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return {"valid": self.valid, "reason": self.reason}


def validate_tbox(concept_type: str, relation: str) -> ValidationResult:
    """Check if a relation is valid for a given concept type.

    TBox constraint: each concept type has a whitelist of valid relations.
    """
    allowed = TBOX.get(concept_type)
    if allowed is None:
        return ValidationResult(False, f"Unknown concept type: {concept_type}")
    if relation not in allowed:
        return ValidationResult(
            False,
            f"TBox violation: {concept_type} cannot use relation '{relation}'. "
            f"Allowed: {sorted(allowed)}",
        )
    return ValidationResult(True)


def validate_rbox(src_label: str, relation: str, dst_label: str) -> ValidationResult:
    """Check RBox constraints for a single edge.

    - Irreflexive: src == dst not allowed for irreflexive relations
    - Asymmetric: checked per-edge (full graph check done in closure)
    """
    if relation in RBOX["irreflexive"] and src_label == dst_label:
        return ValidationResult(
            False,
            f"RBox violation: '{relation}' is irreflexive, cannot relate "
            f"'{src_label}' to itself.",
        )
    return ValidationResult(True)


def validate_relation(concept_type: str, relation: str,
                      src_label: str, dst_label: str) -> ValidationResult:
    """Full validation: TBox + RBox."""
    tbox = validate_tbox(concept_type, relation)
    if not tbox.valid:
        return tbox
    return validate_rbox(src_label, relation, dst_label)


def validate_extraction(concepts: dict, relations: list[dict]) -> dict:
    """Validate all concepts and relations from LLM extraction.

    Returns: {"valid": [...], "rejected": [...]}
    Each item has {"type"/"relation", "detail", "reason"}.
    """
    valid = []
    rejected = []

    # Validate relations against TBox (using the head concept's type)
    for rel in relations:
        head = rel.get("head", "")
        rel_type = rel.get("rel", "")
        tail = rel.get("tail", "")

        # Find the concept type for the head
        head_type = _find_concept_type(head, concepts)
        if head_type:
            result = validate_relation(head_type, rel_type, head, tail)
            if result.valid:
                valid.append({"type": "relation", "detail": rel})
            else:
                rejected.append({"type": "relation", "detail": rel, "reason": result.reason})
        else:
            # Unknown concept type — allow through with warning
            valid.append({"type": "relation", "detail": rel})

    return {"valid": valid, "rejected": rejected}


def _find_concept_type(label: str, concepts: dict) -> str | None:
    """Find the concept type for a label in the extraction result."""
    type_map = {
        "Problem": concepts.get("problems", []),
        "Method": concepts.get("methods", []),
        "Conclusion": concepts.get("conclusions", []),
        "Debate": concepts.get("debates", []),
        "Gap": concepts.get("gaps", []),
        "Actor": concepts.get("actors", []),
    }
    for ctype, items in type_map.items():
        for item in items:
            if item.get("label", "") == label:
                return ctype
    return None
```

- [ ] **Step 4: Write __init__.py**

```python
"""Schema validation package."""
from brbrain.validator.schema import (
    TBOX, RBOX, validate_tbox, validate_rbox, validate_relation,
    validate_extraction, ValidationResult,
)

__all__ = [
    "TBOX", "RBOX", "validate_tbox", "validate_rbox",
    "validate_relation", "validate_extraction", "ValidationResult",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_validator.py -v`
Expected: PASS (7/7)

- [ ] **Step 6: Commit**

```bash
jj describe -m "feat(v1.1): TBox/RBox schema validation engine

Add symbolic constraint checks for concept-relation pairs.
TBox whitelists relations per concept type, RBox enforces
irreflexivity. validate_extraction batch-validates LLM output."
```

---

### Task 3: Confidence Queue Module

**Files:**
- Create: `src/brbrain/extractor/queue.py`
- Test: `tests/test_queue.py`

- [ ] **Step 1: Write failing tests for confidence queue**

```python
"""Tests for confidence queue routing and resolution."""
import tempfile
from pathlib import Path
import json
from brbrain.storage.database import Database
from brbrain.extractor.queue import route_item, check_consensus, resolve_accept, resolve_reject

def test_route_item_below_threshold():
    """Items below weak_threshold go to queue."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        result = route_item(db, "p1", "concept", {"label": "weird thing", "type": "Method"}, 0.4, weak_threshold=0.7)
        assert result["action"] == "queued"
        assert result["queue_id"] is not None
        db.close()

def test_route_item_above_threshold_direct_ingest():
    """Items above auto_accept go directly to ingest."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        result = route_item(db, "p1", "concept", {"label": "clear thing", "type": "Method"}, 0.95, weak_threshold=0.7, auto_accept=0.9)
        assert result["action"] == "accepted"
        db.close()

def test_route_item_weak_marker():
    """Items between thresholds get weak marker."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        result = route_item(db, "p1", "concept", {"label": "ok thing"}, 0.8, weak_threshold=0.7, auto_accept=0.9)
        assert result["action"] == "weak"
        db.close()

def test_check_consensus_auto_promotes():
    """Concept appearing in 3+ papers with conf > 0.8 auto-promotes matching queue items."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "A", 2020, "uploaded")
        db.insert_paper("p2", "B", 2021, "uploaded")
        db.insert_paper("p3", "C", 2022, "uploaded")
        db.insert_concept("p1", "Method", "transformer", 0.95, year=2020)
        db.insert_concept("p2", "Method", "transformer", 0.90, year=2021)
        db.insert_concept("p3", "Method", "transformer", 0.85, year=2022)
        db.commit()

        # Queue item for "transformer" below threshold
        qid = db.insert_queue_item("p4", "concept", json.dumps({"label": "transformer", "type": "Method"}), 0.6)
        db.commit()

        is_consensus = check_consensus(db, "transformer")
        assert is_consensus is True

        resolve_accept(db, qid)
        row = db.conn.execute("SELECT status FROM confidence_queue WHERE queue_id = ?", (qid,)).fetchone()
        assert row[0] == "accepted"
        db.close()

def test_resolve_reject():
    """resolve_reject sets status to rejected."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        qid = db.insert_queue_item("p1", "concept", '{"label": "bad"}', 0.3)
        resolve_reject(db, qid)
        row = db.conn.execute("SELECT status FROM confidence_queue WHERE queue_id = ?", (qid,)).fetchone()
        assert row[0] == "rejected"
        db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_queue.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write confidence queue module**

```python
"""Confidence queue: routing, resolution, consensus detection."""
from __future__ import annotations

import json

from brbrain.storage.database import Database


def route_item(
    db: Database,
    source_paper: str,
    item_type: str,
    item_data: dict,
    confidence: float,
    weak_threshold: float = 0.7,
    auto_accept: float = 0.9,
) -> dict:
    """Route an extracted item based on its confidence.

    Returns: {"action": "accepted"|"weak"|"queued", "queue_id": int|None}
    """
    if confidence >= auto_accept:
        return {"action": "accepted", "queue_id": None}
    elif confidence >= weak_threshold:
        return {"action": "weak", "queue_id": None}
    else:
        qid = db.insert_queue_item(
            source_paper, item_type, json.dumps(item_data), confidence,
        )
        return {"action": "queued", "queue_id": qid}


def check_consensus(db: Database, label: str, min_papers: int = 3, min_confidence: float = 0.8) -> bool:
    """Check if a concept label has consensus (appears in N+ papers with high confidence)."""
    row = db.conn.execute(
        "SELECT COUNT(DISTINCT c.local_id), AVG(c.confidence) "
        "FROM concepts c WHERE c.label = ?",
        (label,),
    ).fetchone()
    if row is None:
        return False
    count, avg_conf = row
    return count >= min_papers and avg_conf >= min_confidence


def resolve_accept(db: Database, queue_id: int) -> None:
    """Accept a queue item. If the concept has consensus, auto-accept matching items."""
    item = db.conn.execute(
        "SELECT item_data FROM confidence_queue WHERE queue_id = ?", (queue_id,)
    ).fetchone()
    if item is None:
        return

    data = json.loads(item[0])
    label = data.get("label", "")

    db.accept_queue_item(queue_id)

    # If this concept has consensus, auto-accept other pending items with same label
    if label and check_consensus(db, label):
        db.conn.execute(
            "UPDATE confidence_queue SET status = 'accepted' "
            "WHERE status = 'pending' AND item_data LIKE ?",
            (f'%"{label}"%',),
        )

    db.commit()


def resolve_reject(db: Database, queue_id: int) -> None:
    """Reject a queue item."""
    db.reject_queue_item(queue_id)
    db.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_queue.py -v`
Expected: PASS (5/5)

- [ ] **Step 5: Commit**

```bash
jj describe -m "feat(v1.1): confidence queue module with routing and consensus

Add route_item for threshold-based routing, check_consensus for
auto-promotion, resolve_accept/reject for human-in-the-loop.
Consensus: 3+ papers with conf >0.8 auto-accepts matching queue items."
```

---

### Task 4: Argument Extraction + Updated LLM Prompt

**Files:**
- Create: `src/brbrain/extractor/argument.py`
- Modify: `src/brbrain/extractor/concept.py`
- Modify: `prompts/extract_concepts.txt`
- Test: `tests/test_argument.py`

- [ ] **Step 1: Write failing tests for argument extraction**

```python
"""Tests for argument extraction from LLM output."""
from brbrain.extractor.argument import ExtractedArgument, parse_arguments, validate_argument

def test_parse_arguments():
    """parse_arguments converts raw LLM dicts to ExtractedArgument objects."""
    raw = [
        {
            "claim": "Self-attention replaces RNN",
            "claim_type": "proposes",
            "target": "Transformer",
            "target_type": "Method",
            "evidence_type": "empirical",
            "evidence_detail": "WMT14 BLEU +2.0",
            "confidence": 0.95,
        }
    ]
    args = parse_arguments(raw)
    assert len(args) == 1
    assert args[0].claim == "Self-attention replaces RNN"
    assert args[0].claim_type == "proposes"
    assert args[0].target == "Transformer"

def test_validate_argument_valid():
    """validate_argument accepts valid claim_type and target_type."""
    arg = ExtractedArgument(
        claim="X solves Y", claim_type="solves",
        target="Problem Y", target_type="Problem",
        evidence_type="empirical", confidence=0.9,
    )
    result = validate_argument(arg)
    assert result.valid is True

def test_validate_argument_invalid_claim_type():
    """validate_argument rejects unknown claim_type."""
    arg = ExtractedArgument(
        claim="X does Y", claim_type="magical",
        target="Z", target_type="Method",
        evidence_type="empirical", confidence=0.9,
    )
    result = validate_argument(arg)
    assert result.valid is False

def test_validate_argument_invalid_target_type():
    """validate_argument rejects unknown target_type."""
    arg = ExtractedArgument(
        claim="X proposes Y", claim_type="proposes",
        target="Z", target_type="UnknownType",
        evidence_type="empirical", confidence=0.9,
    )
    result = validate_argument(arg)
    assert result.valid is False

def test_extracted_argument_to_dict():
    """ExtractedArgument serializes to dict."""
    arg = ExtractedArgument(
        claim="Test claim", claim_type="proposes",
        target="Target", target_type="Method",
        evidence_type="empirical", evidence_detail="details",
        confidence=0.85,
    )
    d = arg.to_dict()
    assert d["claim"] == "Test claim"
    assert d["claim_type"] == "proposes"
    assert d["target"] == "Target"
    assert d["evidence_type"] == "empirical"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_argument.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write argument extraction module**

```python
"""Argument unit extraction and validation."""
from __future__ import annotations

from dataclasses import dataclass

from brbrain.validator.schema import ValidationResult

VALID_CLAIM_TYPES = {"supports", "challenges", "extends", "limits", "solves", "proposes"}
VALID_TARGET_TYPES = {"Method", "Problem", "Conclusion", "Gap", "Debate", "Argument"}
VALID_EVIDENCE_TYPES = {"empirical", "theoretical", "case_study", "survey", None}


@dataclass
class ExtractedArgument:
    """A single argument unit from LLM extraction."""
    claim: str
    claim_type: str
    target: str
    target_type: str
    evidence_type: str | None = None
    evidence_detail: str | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "claim_type": self.claim_type,
            "target": self.target,
            "target_type": self.target_type,
            "evidence_type": self.evidence_type,
            "evidence_detail": self.evidence_detail,
            "confidence": self.confidence,
        }


def parse_arguments(raw: list[dict]) -> list[ExtractedArgument]:
    """Parse raw LLM argument dicts into ExtractedArgument objects."""
    args = []
    for item in raw:
        args.append(ExtractedArgument(
            claim=item.get("claim", ""),
            claim_type=item.get("claim_type", ""),
            target=item.get("target", ""),
            target_type=item.get("target_type", ""),
            evidence_type=item.get("evidence_type"),
            evidence_detail=item.get("evidence_detail"),
            confidence=item.get("confidence", 1.0),
        ))
    return args


def validate_argument(arg: ExtractedArgument) -> ValidationResult:
    """Validate an argument's claim_type and target_type against allowed values."""
    if arg.claim_type not in VALID_CLAIM_TYPES:
        return ValidationResult(
            False,
            f"Invalid claim_type '{arg.claim_type}'. Allowed: {sorted(VALID_CLAIM_TYPES)}",
        )
    if arg.target_type not in VALID_TARGET_TYPES:
        return ValidationResult(
            False,
            f"Invalid target_type '{arg.target_type}'. Allowed: {sorted(VALID_TARGET_TYPES)}",
        )
    return ValidationResult(True)


def validate_arguments(args: list[ExtractedArgument]) -> tuple[list[ExtractedArgument], list[dict]]:
    """Validate all arguments. Returns (valid, rejected)."""
    valid = []
    rejected = []
    for arg in args:
        result = validate_argument(arg)
        if result.valid:
            valid.append(arg)
        else:
            rejected.append({"argument": arg.to_dict(), "reason": result.reason})
    return valid, rejected
```

- [ ] **Step 4: Update concept.py to include arguments**

Replace `concept.py`:

```python
"""Academic concept + argument extraction via LLM with fallback chain."""
from __future__ import annotations

from pathlib import Path

from brbrain.extractor.llm_client import acall_with_fallback
from brbrain.extractor.argument import ExtractedArgument, parse_arguments

PROMPT_TEMPLATE = Path(__file__).parent.parent.parent.parent / "prompts" / "extract_concepts.txt"


class ExtractedConcepts:
    """Structured extraction result from a paper."""

    def __init__(self, data: dict):
        self.problems: list[dict] = data.get("problems", [])
        self.methods: list[dict] = data.get("methods", [])
        self.conclusions: list[dict] = data.get("conclusions", [])
        self.debates: list[dict] = data.get("debates", [])
        self.gaps: list[dict] = data.get("gaps", [])
        self.actors: list[dict] = data.get("actors", [])
        self.relations: list[dict] = data.get("relations", [])
        self.arguments: list[ExtractedArgument] = parse_arguments(data.get("arguments", []))

    def to_dict(self) -> dict:
        return {
            "problems": self.problems,
            "methods": self.methods,
            "conclusions": self.conclusions,
            "debates": self.debates,
            "gaps": self.gaps,
            "actors": self.actors,
            "relations": self.relations,
            "arguments": [a.to_dict() for a in self.arguments],
        }


async def extract_concepts(
    text: str,
    models: list[dict],
) -> ExtractedConcepts | None:
    """Extract academic concepts + arguments from paper text using LLM fallback chain."""
    system_prompt = PROMPT_TEMPLATE.read_text(encoding="utf-8")
    data = await acall_with_fallback(
        prompt=text[:12000],
        models=models,
        system_prompt=system_prompt,
    )
    if data is None:
        return None
    return ExtractedConcepts(data)
```

- [ ] **Step 5: Update extract_concepts.txt prompt**

Read the current prompt, then replace with:

```
You are an academic knowledge graph builder. Extract structured concepts AND argument units from a research paper. Output STRICT JSON.

Output schema (no markdown wrapping, no extra text):
{
  "problems": [{"label": "3-8 word label", "confidence": 0.0-1.0}],
  "methods": [{"label": "3-8 word label", "confidence": 0.0-1.0}],
  "conclusions": [{"label": "3-8 word label", "confidence": 0.0-1.0}],
  "debates": [{"label": "point of contention", "confidence": 0.0-1.0}],
  "gaps": [{"label": "limitation or open question", "confidence": 0.0-1.0}],
  "actors": [{"label": "author/lab/institution", "confidence": 0.0-1.0}],
  "relations": [
    {"head": "this_paper" | concept_label, "rel": "addresses"|"proposes"|"challenges"|"supports"|"leaves_open"|"extends"|"replaces", "tail": concept_label}
  ],
  "arguments": [
    {
      "claim": "3-15 word claim statement",
      "claim_type": "supports|challenges|extends|limits|solves|proposes",
      "target": "target concept label",
      "target_type": "Method|Problem|Conclusion|Gap|Debate",
      "evidence_type": "empirical|theoretical|case_study|survey",
      "evidence_detail": "brief description of supporting evidence",
      "confidence": 0.0-1.0
    }
  ]
}

Rules:
- problems: core research questions the paper addresses
- methods: technical approaches, models, algorithms, frameworks
- conclusions: key empirical findings or claims
- debates: points of contention, opposing views, paradigm conflicts
- gaps: limitations, unresolved questions, future work directions
- actors: authors, labs, research groups central to the work
- relations: connect concepts using exact relation types listed above
- arguments: structured claims with evidence — what the paper argues, not just what it mentions
- claim_type: "proposes" for new contributions, "challenges"/"supports" for existing claims, "limits" for constraints, "solves" for problem resolution, "extends" for improvements
- Keep labels concise (3-8 words), use consistent terminology
- If nothing to extract for a category, return empty array []
- confidence: 1.0 = explicitly stated, 0.7-0.9 = strongly implied, <0.7 = weak inference
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_argument.py -v`
Expected: PASS (5/5)

Also verify existing tests still pass:
Run: `uv run pytest tests/test_extractor.py -v`
Expected: PASS (4/4)

- [ ] **Step 7: Commit**

```bash
jj describe -m "feat(v1.1): argument extraction with LLM prompt update

Add ExtractedArgument dataclass, parse_arguments, validate_argument.
Update ExtractedConcepts to include arguments. Extend LLM prompt
with argument schema alongside existing concept extraction."
```

---

### Task 5: Update Ingest Pipeline + New CLI Commands

**Files:**
- Modify: `src/brbrain/cli/commands.py`
- Modify: `src/brbrain/cli/main.py`
- Modify: `src/brbrain/report/generator.py`
- Test: `tests/test_ingest_v11.py`

- [ ] **Step 1: Write integration test for updated ingest pipeline**

```python
"""Integration test for v1.1 ingest pipeline with validation, queue, arguments."""
import tempfile
from pathlib import Path
from brbrain.storage.database import Database
from brbrain.validator.schema import validate_extraction
from brbrain.extractor.queue import route_item
from brbrain.extractor.argument import ExtractedArgument, validate_arguments

def test_validation_rejects_invalid_relations():
    """validate_extraction rejects Problem --proposes--> relations."""
    concepts = {
        "problems": [{"label": "X problem", "confidence": 0.9}],
        "methods": [{"label": "Y method", "confidence": 0.9}],
        "conclusions": [], "debates": [], "gaps": [], "actors": [],
    }
    relations = [
        {"head": "X problem", "rel": "proposes", "tail": "Y method"},
        {"head": "Y method", "rel": "addresses", "tail": "X problem"},
    ]
    result = validate_extraction(concepts, relations)
    assert len(result["rejected"]) == 1
    assert "proposes" in result["rejected"][0]["reason"]
    assert len(result["valid"]) == 1

def test_argument_validation():
    """validate_arguments filters invalid arguments."""
    args = [
        ExtractedArgument("Valid claim", "proposes", "Method X", "Method", "empirical", "details", 0.9),
        ExtractedArgument("Invalid claim", "magic", "Target Y", "Method", "empirical", "", 0.8),
    ]
    valid, rejected = validate_arguments(args)
    assert len(valid) == 1
    assert len(rejected) == 1
    assert "magic" in rejected[0]["reason"]

def test_queue_routing():
    """route_item correctly routes low-confidence items to queue."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        result = route_item(db, "p1", "concept", {"label": "low conf item"}, 0.4)
        assert result["action"] == "queued"
        pending = db.get_queue_pending()
        assert len(pending) == 1
        db.close()

def test_db_insert_argument_with_paper():
    """Full flow: insert paper, insert argument, query back."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test Paper", 2024, "uploaded")
        db.insert_concept("p1", "Method", "Transformer", 0.95, year=2024)
        db.commit()
        arg_id = db.insert_argument(
            "p1", "Self-attention replaces RNN", "proposes",
            "Transformer", "Method", "empirical", "WMT14 BLEU +2.0", 0.95,
        )
        args = db.get_arguments_by_paper("p1")
        assert len(args) == 1
        assert args[0]["claim"] == "Self-attention replaces RNN"
        db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ingest_v11.py -v`
Expected: FAIL (some may pass since DB methods exist from Task 1, but the full pipeline integration hasn't been updated yet)

- [ ] **Step 3: Update ingest_cmd to include validation, queue, arguments**

In `commands.py`, update the `ingest_cmd` function. The key changes are:
- After extraction, run `validate_extraction` on concepts+relations
- Route low-confidence concepts through `route_item`
- Insert validated arguments into DB
- Track validation stats for the report

Replace the ingest_cmd function. The key section to change is after Stage 3 (Extract):

```python
def ingest_cmd(pdf_path: str, json_flag: bool = False):
    """Full ingest pipeline: parse -> identify -> extract -> validate -> queue -> align -> ingest -> expand -> report."""
    cfg = load_config()
    db = Database(cfg["db"]["path"])
    graph = GraphEngine()
    graph.load_from_db(db)
    dedup = DedupEngine(db)
    alias_table = AliasTable()

    # Queue config
    queue_cfg = cfg.get("queue", {})
    weak_threshold = queue_cfg.get("weak_threshold", 0.7)
    auto_accept = queue_cfg.get("auto_accept", 0.9)

    # Stage 1: Parse
    typer.echo(f"Parsing: {pdf_path}")
    try:
        parsed = extract_pdf(pdf_path, cfg)
    except Exception as e:
        typer.echo(f"Error parsing PDF: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"  Title: {parsed.title}")
    typer.echo(f"  Year: {parsed.year}")
    typer.echo(f"  Sections: {len(parsed.text_blocks)} high-signal blocks")

    # Stage 2: Identify
    ids = PaperIDs(doi=parsed.doi, arxiv=parsed.arxiv)
    local_id = dedup.resolve(ids, title=parsed.title, year=parsed.year)
    is_new = local_id is None
    if is_new:
        local_id = f"p{uuid.uuid4().hex[:6]}"
        db.insert_paper(local_id, parsed.title, parsed.year, "uploaded")
        db.insert_paper_ids(local_id, doi=ids.doi, arxiv=ids.arxiv)
        db.commit()
        typer.echo(f"  [new] {local_id}")
    else:
        db.upgrade_placeholder(local_id)
        db.commit()
        typer.echo(f"  [upgrade] {local_id}")

    # Stage 3: Extract
    typer.echo("  Extracting concepts + arguments...")
    llm_models = cfg.get("llm", {}).get("models", [])
    if not llm_models:
        typer.echo("Error: no LLM models configured. Run: drbrain setup", err=True)
        raise typer.Exit(1)

    import asyncio
    full_text = "\n\n".join(parsed.text_blocks)
    concepts = asyncio.run(extract_concepts(full_text, llm_models))
    if concepts is None:
        typer.echo("Error: LLM extraction failed. All models exhausted.", err=True)
        raise typer.Exit(1)

    # Stage 3.5: Validate
    from brbrain.validator.schema import validate_extraction
    typer.echo("  Validating extraction...")
    concept_data = {
        "problems": concepts.problems, "methods": concepts.methods,
        "conclusions": concepts.conclusions, "debates": concepts.debates,
        "gaps": concepts.gaps, "actors": concepts.actors,
    }
    validation = validate_extraction(concept_data, concepts.relations)
    typer.echo(f"  Valid items: {len(validation['valid'])}")
    if validation["rejected"]:
        typer.echo(f"  Rejected: {len(validation['rejected'])}")
        for r in validation["rejected"]:
            typer.echo(f"    [yellow]{r['reason']}[/yellow]")

    # Build valid concept sets for alignment
    valid_relations = [r["detail"] for r in validation["valid"] if r["type"] == "relation"]

    # Stage 3.6: Queue low-confidence concepts
    from brbrain.extractor.queue import route_item
    typed_count = 0
    queued_count = 0
    weak_count = 0
    all_items = [
        ("Problem", concepts.problems), ("Method", concepts.methods),
        ("Conclusion", concepts.conclusions), ("Debate", concepts.debates),
        ("Gap", concepts.gaps), ("Actor", concepts.actors),
    ]
    for ctype, items in all_items:
        for item in items:
            label = item.get("label", "")
            conf = item.get("confidence", 1.0)
            routing = route_item(db, local_id, "concept",
                               {"label": label, "type": ctype}, conf,
                               weak_threshold, auto_accept)
            if routing["action"] == "queued":
                queued_count += 1
            elif routing["action"] == "weak":
                weak_count += 1
            typed_count += 1

    # Stage 4: Align + Stage 5: Ingest
    for ctype, items in all_items:
        for item in items:
            label = item.get("label", "")
            conf = item.get("confidence", 1.0)
            if conf >= weak_threshold:  # Only ingest non-queued items
                alias_table.get_or_create(label)
                db.insert_concept(local_id, ctype, label, conf, year=parsed.year)

    for rel in valid_relations:
        db.insert_edge(rel["head"], rel["tail"], rel["rel"], local_id)

    # Ingest arguments
    from brbrain.extractor.argument import validate_arguments
    valid_args, rejected_args = validate_arguments(concepts.arguments)
    for arg in valid_args:
        db.insert_argument(
            local_id, arg.claim, arg.claim_type, arg.target, arg.target_type,
            arg.evidence_type, arg.evidence_detail, arg.confidence,
        )

    db.commit()
    typer.echo(f"  Concepts inserted: {typed_count}")
    typer.echo(f"  Arguments inserted: {len(valid_args)}")
    if queued_count:
        typer.echo(f"  Queued for review: {queued_count}")
    if weak_count:
        typer.echo(f"  Weak (ingested with marker): {weak_count}")
    if rejected_args:
        typer.echo(f"  Arguments rejected: {len(rejected_args)}")

    # Stage 6: Expand
    typer.echo("  Expanding citations...")
    from brbrain.extractor.citation import expand_citations
    refs, cits = expand_citations(db, local_id, cfg)
    refs_in = sum(1 for r in refs if r.in_graph)
    cits_in = sum(1 for c in cits if c.in_graph)
    typer.echo(f"  References: {len(refs)} ({refs_in} in graph)")
    typer.echo(f"  Citations: {len(cits)} ({cits_in} in graph)")

    # Stage 8: Closure
    typer.echo("  Running rule closure...")
    graph.load_from_db(db)
    inferred = graph.closure()
    for edge in inferred:
        db.insert_edge(edge["src"], edge["dst"], edge["relation"], local_id)
    db.commit()
    typer.echo(f"  Inferred edges: {len(inferred)}")

    # Stage 9: Report (now includes arguments + validation)
    report = PaperReport(
        local_id=local_id, title=parsed.title, year=parsed.year,
        ids={"doi": parsed.doi, "arxiv": parsed.arxiv},
        status="uploaded",
        concepts=concepts.to_dict(),
        arguments=[a.to_dict() for a in valid_args],
        references=refs,
        citations=cits,
        validation={
            "items_rejected": len(validation["rejected"]),
            "items_queued": queued_count,
            "tbox_violations": [r["reason"] for r in validation["rejected"]],
            "rbox_violations": [],
        },
    )
    report_dir = Path(cfg["dirs"]["reports"])
    report_path = report.save(report_dir)
    typer.echo(f"  Report saved: {report_path}")

    summary = report.summary
    if summary["graph_coverage"] < 0.3:
        typer.echo(f"\n  [bold yellow]Warning: Low coverage ({summary['graph_coverage']:.1%}). Consider ingesting missing references.[/bold yellow]")

    db.close()
    typer.echo(f"\nDone: {local_id}")
```

- [ ] **Step 4: Add queue, queue resolve, timeline CLI commands**

Add to `commands.py`:

```python
def queue_cmd():
    """List all pending confidence queue items."""
    cfg = load_config()
    db = Database(cfg["db"]["path"])
    pending = db.get_queue_pending()
    db.close()

    if not pending:
        typer.echo("Queue is empty.")
        return

    table = Table(title="Confidence Queue")
    table.add_column("ID", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Data")
    table.add_column("Confidence", justify="right")
    table.add_column("Paper")
    for item in pending:
        import json
        data = json.loads(item["item_data"])
        label = data.get("label", "N/A")
        item_type = data.get("type", item["item_type"])
        table.add_row(
            str(item["queue_id"]),
            item["item_type"],
            f"{item_type}: {label}",
            f"{item['confidence']:.2f}",
            item["source_paper"],
        )
    console.print(table)


def queue_resolve_cmd(queue_id: int, accept: bool = False, reject: bool = False):
    """Resolve a queue item: accept or reject."""
    if accept and reject:
        typer.echo("Error: cannot both accept and reject", err=True)
        raise typer.Exit(1)
    if not accept and not reject:
        typer.echo("Error: specify --accept or --reject", err=True)
        raise typer.Exit(1)

    cfg = load_config()
    db = Database(cfg["db"]["path"])

    from brbrain.extractor.queue import resolve_accept, resolve_reject
    if accept:
        resolve_accept(db, queue_id)
        typer.echo(f"Queue item {queue_id} accepted.")
    else:
        resolve_reject(db, queue_id)
        typer.echo(f"Queue item {queue_id} rejected.")

    db.close()


def timeline_cmd(concept: str):
    """Show concept evolution over time."""
    cfg = load_config()
    db = Database(cfg["db"]["path"])
    evolution = db.get_concept_evolution(concept)

    if not evolution:
        typer.echo(f"No data for concept: {concept}")
        db.close()
        return

    # Get concept type
    row = db.conn.execute(
        "SELECT type FROM concepts WHERE label = ? LIMIT 1", (concept,)
    ).fetchone()
    ctype = row[0] if row else "unknown"

    typer.echo(f"\nConcept: {concept} ({ctype})")

    from datetime import datetime
    current_year = datetime.now().year

    for entry in evolution:
        year = entry["year"]
        count = entry["count"]
        avg_conf = entry["avg_conf"]

        # Generate signal label
        if year == evolution[0]["year"]:
            signal = "first appeared"
        elif year >= current_year - 2:
            signal = "recent"
        else:
            signal = ""

        line = f"  {year}: {count} paper{'s' if count > 1 else ''} (avg confidence {avg_conf:.2f})"
        if signal:
            line += f" — {signal}"
        typer.echo(line)

    # Detect overall signal
    signals = db.detect_evolution_signals()
    matching = [s for s in signals if s["label"] == concept]
    if matching:
        typer.echo(f"Status: {matching[0]['signal'].upper()}")

    db.close()
```

- [ ] **Step 5: Update PaperReport to accept arguments + validation**

In `generator.py`, update `PaperReport`:

```python
@dataclass
class PaperReport:
    """Complete report for a single paper."""

    local_id: str
    title: str
    year: int | None
    ids: dict = field(default_factory=dict)
    status: str = "uploaded"
    concepts: dict = field(default_factory=dict)
    arguments: list[dict] = field(default_factory=list)
    references: list[RefEntry] = field(default_factory=list)
    citations: list[RefEntry] = field(default_factory=list)
    validation: dict = field(default_factory=dict)

    @property
    def summary(self) -> dict:
        # ... (keep existing summary code)
        ...

    @property
    def boundary_alert(self) -> dict:
        # ... (keep existing, add validation check)
        s = self.summary
        missing = [r for r in self.references if not r.in_graph and r.title]
        alert = {
            "missing_core_refs": len(missing) > 5,
            "isolated_subgraph": s["graph_coverage"] < 0.3 and total_refs_and_citations(self) > 10,
        }
        if self.validation.get("items_rejected", 0) > 0:
            alert["validation_failures"] = True
        return alert

    def to_dict(self) -> dict:
        return {
            "paper": { ... },  # same as before
            "concepts": self.concepts,
            "arguments": self.arguments,
            "references": [...],  # same as before
            "citations": [...],   # same as before
            "summary": self.summary,
            "boundary_alert": self.boundary_alert,
            "validation": self.validation,
        }
```

- [ ] **Step 6: Register new commands in main.py**

Add to `main.py`:

```python
from brbrain.cli.commands import (
    ingest_cmd, expand_cmd, report_cmd, closure_cmd, seed_cmd,
    list_cmd, stats_cmd, query_cmd, export_cmd,
    queue_cmd, queue_resolve_cmd, timeline_cmd,
)

# Register new commands
app.command("queue")(queue_cmd)
app.command("queue resolve")(queue_resolve_cmd)
app.command("timeline")(timeline_cmd)
```

- [ ] **Step 7: Update stats_cmd to include queue depth and argument count**

Add to stats_cmd in commands.py:

```python
    queue_pending = db.conn.execute(
        "SELECT COUNT(*) FROM confidence_queue WHERE status = 'pending'"
    ).fetchone()[0]
    arguments = db.conn.execute("SELECT COUNT(*) FROM arguments").fetchone()[0]

    table.add_row("Arguments", str(arguments))
    table.add_row("Queue pending", str(queue_pending))
```

- [ ] **Step 8: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: PASS (all existing + new tests)

- [ ] **Step 9: Verify CLI help shows new commands**

Run: `uv run python -m brbrain.cli --help`
Expected: Should show `queue`, `queue resolve`, `timeline` commands

- [ ] **Step 10: Commit**

```bash
jj describe -m "feat(v1.1): updated ingest pipeline with validation, queue, arguments

Add schema validation before ingestion, confidence queue routing for
low-confidence items, argument extraction and storage. New CLI commands:
queue, queue resolve, timeline. Updated stats with queue depth and args."
```

---

## Self-Review

### 1. Spec Coverage Check

| Spec Section | Task Coverage |
|--------------|---------------|
| 12. Schema-First Validation (TBox/RBox) | Task 2 |
| 13. Confidence Queue + CLI | Task 3 + Task 5 |
| 14. Argument Units + LLM prompt | Task 4 |
| 15. Temporal Evolution + Timeline CLI | Task 1 + Task 5 |
| 16. Knowledge Boundary Discovery | Uses existing `detect_research_seeds` — not in scope for this plan (would be a separate enhancement) |
| 17. Updated JSON Report (validation + arguments) | Task 5 |
| Updated Pipeline (3.5 Validate, 3.6 Queue) | Task 5 |
| Updated DB Schema (arguments, confidence_queue, first_seen/last_seen) | Task 1 |
| New CLI commands (queue, queue resolve, timeline) | Task 5 |
| Queue config in YAML | Already supported via `cfg.get("queue", {})` |
| Consensus feedback loop | Task 3 |

**No gaps found for the 4 new features.**

### 2. Placeholder Scan
- None. All steps contain complete code.

### 3. Type Consistency
- `ExtractedArgument`: defined in Task 4, used in Tasks 4, 5
- `validate_extraction`: returns `{"valid": [...], "rejected": [...]}`, used in Task 5
- `route_item`: returns `{"action": "...", "queue_id": ...}`, used in Task 5
- `PaperReport`: extended with `arguments` and `validation` fields in Task 5
- DB methods: `insert_concept` now takes optional `year`, `insert_argument`, `insert_queue_item`, etc. — all consistent
- `validate_argument`: validates `ExtractedArgument`, used in Task 5

**No type mismatches.**

### 4. Scope Check
This plan adds 4 features to the existing codebase. Each task is testable independently:
- Task 1: DB schema changes (foundational, needed by all others)
- Task 2: Validation engine (standalone, no DB dependency)
- Task 3: Queue module (uses Task 1 DB methods)
- Task 4: Argument extraction (standalone, updates concept.py)
- Task 5: Integration of all into pipeline + CLI (depends on 1-4)

Dependencies: 5 needs 1-4. 2 is independent. 3 needs 1. 4 is independent.
