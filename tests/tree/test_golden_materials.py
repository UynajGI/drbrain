"""T55: fixed golden materials are verifiable and split up front."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from drbrain.rag.eval_data import validate_golden_materials

ACCEPTANCE = Path(__file__).resolve().parents[2] / "data" / "integration" / "unified-tree"
GOLDEN = ACCEPTANCE / "golden.jsonl"
DB_PATH = ACCEPTANCE / "data" / "drbrain.db"

TEXTS = {
    ("p1", "nl-1"): "say hello world to the reader",
    ("p2", "nl-2"): "second paper text with a distinctive phrase",
}


def _resolve(paper_id, node_id):
    return TEXTS.get((paper_id, node_id))


def _entry(**overrides):
    entry = {
        "id": "q1",
        "query": "what is X?",
        "kind": "term",
        "material": "md",
        "split": "dev",
        "relevant_papers": ["p1"],
        "relevant_nodes": ["nl-1"],
        "evidence": [{"paper_id": "p1", "node_id": "nl-1", "quote": "hello world"}],
    }
    entry.update(overrides)
    return entry


def _multi(**overrides):
    entry = _entry(
        id="q-multi",
        kind="multi",
        split="holdout",
        relevant_papers=["p1", "p2"],
        relevant_nodes=["nl-1", "nl-2"],
        evidence=[
            {"paper_id": "p1", "node_id": "nl-1", "quote": "hello world"},
            {"paper_id": "p2", "node_id": "nl-2", "quote": "distinctive phrase"},
        ],
    )
    entry.update(overrides)
    return entry


class TestValidator:
    def test_clean_set_has_no_problems(self):
        assert validate_golden_materials([_entry(), _multi()], _resolve) == []

    def test_mutations_are_reported(self):
        problems = validate_golden_materials(
            [
                _entry(
                    quote=None,
                    evidence=[{"paper_id": "p1", "node_id": "nl-1", "quote": "not in the text"}],
                ),
                _entry(
                    id="q2",
                    evidence=[{"paper_id": "p9", "node_id": "nr-region", "quote": "hello world"}],
                ),
                _entry(id="q1"),  # duplicate id
                _entry(id="q3", split="test"),
                _entry(id="q4", kind="unknown"),
                _multi(id="q5", evidence=[{"paper_id": "p1", "node_id": "nl-1", "quote": "hello world"}]),
                _entry(id="q6", query=""),
            ],
            _resolve,
        )
        joined = "\n".join(problems)
        assert "quote not found verbatim" in joined
        assert "is not a readable leaf" in joined
        assert "duplicate id: q1" in joined
        assert "split must be dev or holdout" in joined
        assert "unknown kind" in joined
        assert "multi-evidence entries need >= 2 papers" in joined
        assert "empty query" in joined

    def test_missing_holdout_split_is_reported(self):
        problems = validate_golden_materials([_entry(), _entry(id="q2")], _resolve)
        assert "no holdout split entries" in problems


@pytest.mark.skipif(not GOLDEN.is_file(), reason="acceptance golden set absent")
class TestAcceptanceGolden:
    def _entries(self) -> list[dict]:
        return [
            json.loads(line)
            for line in GOLDEN.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _resolver(self):
        conn = sqlite3.connect(DB_PATH)

        def resolve(paper_id: str, node_id: str) -> str | None:
            row = conn.execute(
                "SELECT b.text FROM tree_nodes n JOIN content_blocks b ON b.block_id = n.block_id "
                "WHERE n.local_id = ? AND n.node_id = ? AND n.kind = 'leaf' AND n.state = 'ready'",
                (paper_id, node_id),
            ).fetchone()
            return str(row[0]) if row else None

        return conn, resolve

    def test_real_materials_validate_against_the_pinned_revision(self):
        entries = self._entries()
        assert entries
        conn, resolve = self._resolver()
        try:
            problems = validate_golden_materials(entries, resolve)
        finally:
            conn.close()
        assert problems == []

    def test_coverage_and_split_are_explicit(self):
        entries = self._entries()
        assert {entry["split"] for entry in entries} == {"dev", "holdout"}
        assert {"term", "formula", "structure", "multi"} <= {
            entry["kind"] for entry in entries
        }
        assert {"pdf", "tex", "md"} <= {entry["material"] for entry in entries}
        papers = {paper for entry in entries for paper in entry["relevant_papers"]}
        assert len(papers) == 9  # the full mixed corpus is covered
        assert all(
            evidence["revision"] == 1
            for entry in entries
            for evidence in entry["evidence"]
        )
