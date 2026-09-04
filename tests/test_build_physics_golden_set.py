"""Tests for scripts/eval/build_physics_golden_set.py (physics golden set, R-I5).

The builder emits JSONL in exactly the schema ``drbrain.rag.eval.load_golden``
consumes, so ``drbrain rag eval`` scores against it without a translation
layer (round-4 cleanup: wire the builder into the eval pipeline).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

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


def test_mutual_citations_produce_load_golden_cases(tmp_db):
    _seed_papers(tmp_db)
    tmp_db.insert_paper_cite_keys("p1", ["p2"])
    tmp_db.insert_paper_cite_keys("p2", ["p1"])
    tmp_db.resolve_paper_cite_keys({"p1": "p1", "p2": "p2"})
    tmp_db.commit()

    cases = gold.build_cases(tmp_db, now="2026-09-04T00:00:00+00:00")
    assert len(cases) == 2
    p1_abs = (
        "We study topological superconductivity on the kagome lattice with spin orbit "
        "coupling and find a topological phase transition near van Hove filling."
    )
    p2_abs = (
        "Kagome metals show anisotropic spin transport. We compute the spin "
        "conductivity with a boltzmann approach and discuss berry curvature effects."
    )
    for case in cases:
        # exact load_golden schema (plus provenance extras)
        assert set(case) == {
            "query",
            "relevant_papers",
            "relevant_nodes",
            "split",
            "reference_answer",
            "source",
            "created_at",
        }
        assert case["source"] == "arxiv-citation"
        assert len(case["relevant_papers"]) == 1
        assert case["relevant_papers"][0] in {"p1", "p2"}
        assert case["relevant_nodes"] == []
        assert case["split"] in {"dev", "val", "test"}
        assert case["reference_answer"].strip()
        assert case["query"].strip()
        assert case["created_at"] == "2026-09-04T00:00:00+00:00"

    # the case whose query carries p1's title must expect p2 (and vice versa),
    # and reference_answer must be the CITED paper's abstract — eval scores
    # answer correctness against it, so the mapping has to be exact
    by_query = {c["query"]: c for c in cases}
    p1_case = next(c for q, c in by_query.items() if q.startswith(P1_TITLE))
    p2_case = next(c for q, c in by_query.items() if q.startswith(P2_TITLE))
    assert p1_case["relevant_papers"] == ["p2"]
    assert p2_case["relevant_papers"] == ["p1"]
    assert p1_case["reference_answer"] == p2_abs
    assert p2_case["reference_answer"] == p1_abs
    # query = citing title + TF keywords from the citing abstract
    assert "kagome" in p1_case["query"]


def test_output_is_load_golden_jsonl(tmp_db, tmp_path):
    """End-to-end: load_golden reads the emitted file unchanged (the wiring)."""
    from drbrain.config import Config, LlamaIndexConfig
    from drbrain.rag.eval import load_golden

    _seed_papers(tmp_db)
    tmp_db.insert_paper_cite_keys("p1", ["p2"])
    tmp_db.insert_paper_cite_keys("p2", ["p1"])
    tmp_db.resolve_paper_cite_keys({"p1": "p1", "p2": "p2"})
    tmp_db.commit()

    out = tmp_path / "golden.jsonl"
    rc = gold.main(["--db", str(tmp_db.path), "--out", str(out)])
    assert rc == 0

    cfg = Config(llamaindex=LlamaIndexConfig(enabled=True))
    cfg.llamaindex.eval.golden_set = str(out)
    loaded = load_golden(cfg)
    assert len(loaded) == 2
    assert {entry["relevant_papers"][0] for entry in loaded} == {"p1", "p2"}
    dev = load_golden(cfg, split="dev")
    assert all(entry["split"] == "dev" for entry in dev)


def test_split_assignment_is_deterministic_and_partitioned(tmp_db):
    _seed_papers(tmp_db)
    cids = tuple(f"c{i}" for i in range(1, 21))  # 20 cited papers
    for cid in cids:
        tmp_db.insert_paper(cid, f"Cited {cid}", 2020, "uploaded", paper_type="preprint")
    tmp_db.insert_paper_cite_keys("p1", list(cids))
    tmp_db.resolve_paper_cite_keys({cid: cid for cid in cids})
    tmp_db.commit()

    cases = gold.build_cases(tmp_db, seed=42)
    assert len(cases) == 20
    counts = {name: sum(1 for c in cases if c["split"] == name) for name in ("dev", "val", "test")}
    assert counts == {"dev": 12, "val": 4, "test": 4}  # 60/20/20 of 20
    # same seed → identical split assignment
    assert gold.build_cases(tmp_db, seed=42) == cases


def test_unresolved_citations_are_ignored(tmp_db):
    _seed_papers(tmp_db)
    tmp_db.insert_paper_cite_keys("p1", ["someone-else-2020"])  # cited_local_id stays NULL
    tmp_db.commit()
    assert gold.build_cases(tmp_db) == []


def test_empty_library_yields_empty_cases(tmp_db):
    assert gold.build_cases(tmp_db) == []


def test_empty_library_main_writes_empty_set(tmp_db, tmp_path, capsys):
    out = tmp_path / "golden.jsonl"
    rc = gold.main(["--db", str(tmp_db.path), "--out", str(out)])
    assert rc == 0
    assert out.read_text(encoding="utf-8") == ""
    assert "no resolved in-library citations" in capsys.readouterr().out


def test_limit_and_seed_reproducibility(tmp_db):
    _seed_papers(tmp_db)
    for cid in ("c1", "c2", "c3", "c4", "c5"):
        tmp_db.insert_paper(cid, f"Cited paper {cid}", 2020, "uploaded", paper_type="preprint")
    tmp_db.insert_paper_cite_keys("p1", ["c1", "c2", "c3", "c4", "c5"])
    tmp_db.resolve_paper_cite_keys({cid: cid for cid in ("c1", "c2", "c3", "c4", "c5")})
    tmp_db.commit()

    all_expected = {c["relevant_papers"][0] for c in gold.build_cases(tmp_db)}
    assert len(all_expected) == 5  # c1..c5 (this db has no p2 citation)

    limited_a = gold.build_cases(tmp_db, limit=3, seed=42)
    limited_b = gold.build_cases(tmp_db, limit=3, seed=42)
    assert len(limited_a) == 3
    assert limited_a == limited_b  # same seed → identical sample
    assert {c["relevant_papers"][0] for c in limited_a} <= all_expected

    unseeded = gold.build_cases(tmp_db, limit=3)
    assert len(unseeded) == 3
    assert {c["relevant_papers"][0] for c in unseeded} <= all_expected


def test_limit_zero_returns_all(tmp_db):
    _seed_papers(tmp_db)
    tmp_db.insert_paper_cite_keys("p1", ["p2"])
    tmp_db.resolve_paper_cite_keys({"p2": "p2"})
    tmp_db.commit()
    assert len(gold.build_cases(tmp_db, limit=0)) == 1
