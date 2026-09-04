"""Tests for scripts/eval/build_physics_golden_set.py (physics golden set, R-I5)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "build_physics_golden_set_testee",
        REPO / "scripts" / "eval" / "build_physics_golden_set.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gold = _load_script_module()

P1_TITLE = "Topological Superconductivity on the Kagome Lattice"
P2_TITLE = "Anisotropic Spin Transport in Kagome Metals"


def _seed_papers(db) -> None:
    db.insert_paper("p1", P1_TITLE, 2024, "uploaded", paper_type="preprint")
    db.set_paper_abstract(
        "p1",
        "We study topological superconductivity on the kagome lattice with spin orbit "
        "coupling and find a topological phase transition near van Hove filling.",
    )
    db.insert_paper("p2", P2_TITLE, 2023, "uploaded", paper_type="preprint")
    db.set_paper_abstract(
        "p2",
        "Kagome metals show anisotropic spin transport. We compute the spin "
        "conductivity with a boltzmann approach and discuss berry curvature effects.",
    )
    db.commit()


def test_mutual_citations_produce_cases(tmp_db):
    _seed_papers(tmp_db)
    tmp_db.insert_paper_cite_keys("p1", ["p2"])
    tmp_db.insert_paper_cite_keys("p2", ["p1"])
    tmp_db.resolve_paper_cite_keys({"p1": "p1", "p2": "p2"})
    tmp_db.commit()

    cases = gold.build_cases(tmp_db, now="2026-09-04T00:00:00+00:00")
    assert len(cases) == 2
    for case in cases:
        assert set(case) == {"query", "expected_paper_id", "source", "created_at"}
        assert case["source"] == "arxiv-citation"
        assert case["expected_paper_id"] in {"p1", "p2"}
        assert case["query"].strip()
        assert case["created_at"] == "2026-09-04T00:00:00+00:00"

    # the case whose query carries p1's title must expect p2 (and vice versa)
    by_query = {c["query"]: c["expected_paper_id"] for c in cases}
    p1_query = next(q for q in by_query if q.startswith(P1_TITLE))
    p2_query = next(q for q in by_query if q.startswith(P2_TITLE))
    assert by_query[p1_query] == "p2"
    assert by_query[p2_query] == "p1"
    # query = citing title + TF keywords from the citing abstract
    assert "kagome" in p1_query


def test_unresolved_citations_are_ignored(tmp_db):
    _seed_papers(tmp_db)
    tmp_db.insert_paper_cite_keys("p1", ["someone-else-2020"])  # cited_local_id stays NULL
    tmp_db.commit()
    assert gold.build_cases(tmp_db) == []


def test_empty_library_yields_empty_cases(tmp_db):
    assert gold.build_cases(tmp_db) == []


def test_empty_library_main_writes_empty_set(tmp_db, tmp_path, capsys):
    out = tmp_path / "golden.json"
    rc = gold.main(["--db", str(tmp_db.path), "--out", str(out)])
    assert rc == 0
    assert json.loads(out.read_text(encoding="utf-8")) == {"cases": []}
    assert "no resolved in-library citations" in capsys.readouterr().out


def test_limit_and_seed_reproducibility(tmp_db):
    _seed_papers(tmp_db)
    for cid in ("c1", "c2", "c3", "c4", "c5"):
        tmp_db.insert_paper(cid, f"Cited paper {cid}", 2020, "uploaded", paper_type="preprint")
    tmp_db.insert_paper_cite_keys("p1", ["c1", "c2", "c3", "c4", "c5"])
    tmp_db.resolve_paper_cite_keys({cid: cid for cid in ("c1", "c2", "c3", "c4", "c5")})
    tmp_db.commit()

    all_expected = {c["expected_paper_id"] for c in gold.build_cases(tmp_db)}
    assert len(all_expected) == 5  # c1..c5 (this db has no p2 citation)

    limited_a = gold.build_cases(tmp_db, limit=3, seed=42)
    limited_b = gold.build_cases(tmp_db, limit=3, seed=42)
    assert len(limited_a) == 3
    assert limited_a == limited_b  # same seed → identical sample
    assert {c["expected_paper_id"] for c in limited_a} <= all_expected

    unseeded = gold.build_cases(tmp_db, limit=3)
    assert len(unseeded) == 3
    assert {c["expected_paper_id"] for c in unseeded} <= all_expected


def test_limit_zero_returns_all(tmp_db):
    _seed_papers(tmp_db)
    tmp_db.insert_paper_cite_keys("p1", ["p2"])
    tmp_db.resolve_paper_cite_keys({"p2": "p2"})
    tmp_db.commit()
    assert len(gold.build_cases(tmp_db, limit=0)) == 1
