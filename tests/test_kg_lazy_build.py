"""Tests for scripts/pipeline/kg_lazy_build.py (lazy KG build, L1/L2 layers)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "kg_lazy_build_testee", REPO / "scripts" / "pipeline" / "kg_lazy_build.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kg = _load_script_module()

ABSTRACT = (
    "Topological superconductivity on the kagome lattice has become a central "
    "problem in condensed matter physics, and how topological superconductivity "
    "emerges from the kagome lattice remains an open problem. In this work we "
    "propose a minimal two-orbital model for topological superconductivity on the "
    "kagome lattice, and we develop a self-consistent mean field approach to the "
    "model. Our approach shows that spin orbit coupling drives a topological "
    "phase transition. The phase transition is characterized by a change of the "
    "Chern number. "
)

CONCLUSION_MD = (
    "## Conclusion\n\n"
    "We find that the kagome lattice supports topological superconductivity near "
    "the van Hove filling. Our results show that the van Hove singularity pins "
    "the phase transition and enhances the superconducting dome.\n\n"
    "## Acknowledgements\n\n"
    "We thank the funding agency for support.\n"
)


def _make_paper(db, pid: str, title: str, abstract: str = "", year: int = 2024) -> None:
    db.insert_paper(pid, title, year, "uploaded", paper_type="preprint")
    if abstract:
        db.set_paper_abstract(pid, abstract)
    db.commit()


@pytest.fixture
def papers_root(tmp_path):
    root = tmp_path / "papers"
    (root / "p1").mkdir(parents=True)
    (root / "p1" / "raw.md").write_text(
        "# Title\n\n" + ABSTRACT + "\n\n" + CONCLUSION_MD, encoding="utf-8"
    )
    return root


# ── conclusion-section extraction ────────────────────────────────────────────


def test_extract_conclusion_section_body_only():
    body = kg._extract_conclusion_section(CONCLUSION_MD)
    assert "van Hove filling" in body
    assert "Acknowledg" not in body  # stops at the next heading
    assert body.startswith("We find")


def test_extract_conclusion_section_missing():
    assert kg._extract_conclusion_section("# Title\n\nNo conclusions here.\n") == ""


# ── heuristic extractor ──────────────────────────────────────────────────────


def test_heuristic_concept_count_bounds():
    concepts = kg.heuristic_extract(
        [(ABSTRACT, "abstract"), (kg._extract_conclusion_section(CONCLUSION_MD), "conclusion")]
    )
    assert 3 <= len(concepts) <= 5
    labels = [c["label"] for c in concepts]
    assert len(set(labels)) == len(labels)
    assert all(c["label"] for c in concepts)
    assert all(c["type"] in kg.VALID_TYPES for c in concepts)
    assert all(0.0 <= c["confidence"] <= 1.0 for c in concepts)
    # conclusion-derived concept carries its section
    assert any(c["section"] == "conclusion" for c in concepts)


def test_heuristic_concept_types_match_context():
    concepts = kg.heuristic_extract([(ABSTRACT, "abstract")])
    # "we propose ... model" sentence → Method
    assert any(c["type"] == "Method" for c in concepts)
    # "central problem / open problem" sentence → Problem
    assert any(c["type"] == "Problem" for c in concepts)


# ── L1 pass ──────────────────────────────────────────────────────────────────


def test_l1_inserts_and_sets_sections(tmp_db, papers_root):
    _make_paper(tmp_db, "p1", "Kagome superconductivity", abstract=ABSTRACT)
    _make_paper(tmp_db, "p2", "Another kagome study", abstract=ABSTRACT)
    _make_paper(tmp_db, "p3", "No abstract")  # not selected: no abstract
    _make_paper(tmp_db, "p4", "Already extracted", abstract=ABSTRACT)
    tmp_db.insert_concept("p4", "Method", "pre-existing", 0.9, year=2024)
    tmp_db.commit()

    stats = kg.run_l1(tmp_db, papers_root, extractor="heuristic")
    assert stats["selected"] == 2  # p1, p2 — p4 already filtered out by the SQL
    assert stats["processed"] == 2
    assert stats["inserted"] >= 6  # 3-5 concepts each

    rows_p1 = tmp_db.execute(
        "SELECT type, label, section FROM concepts WHERE local_id = 'p1'"
    ).fetchall()
    assert 3 <= len(rows_p1) <= 5
    assert all(r[0] in kg.VALID_TYPES for r in rows_p1)
    assert any(r[2] == "abstract" for r in rows_p1)

    rows_p3 = tmp_db.execute("SELECT COUNT(*) FROM concepts WHERE local_id = 'p3'").fetchone()
    assert rows_p3[0] == 0
    rows_p4 = tmp_db.execute("SELECT COUNT(*) FROM concepts WHERE local_id = 'p4'").fetchone()
    assert rows_p4[0] == 1  # only the pre-existing concept


def test_l1_idempotent_rerun(tmp_db, papers_root):
    _make_paper(tmp_db, "p1", "Kagome superconductivity", abstract=ABSTRACT)
    first = kg.run_l1(tmp_db, papers_root, extractor="heuristic")
    assert first["inserted"] > 0
    count_before = tmp_db.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]

    second = kg.run_l1(tmp_db, papers_root, extractor="heuristic")
    assert second["selected"] == 0  # paper now has concepts → not selected
    assert second["inserted"] == 0
    count_after = tmp_db.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
    assert count_after == count_before


def test_l1_limit_controls_batch(tmp_db, papers_root):
    for pid in ("a", "b", "c"):
        _make_paper(tmp_db, pid, f"Paper {pid}", abstract=ABSTRACT)
    stats = kg.run_l1(tmp_db, papers_root, extractor="heuristic", limit=1)
    assert stats["selected"] == 1
    assert stats["processed"] == 1
    done = tmp_db.execute("SELECT COUNT(DISTINCT local_id) FROM concepts").fetchone()[0]
    assert done == 1


# ── spark4b extractor: missing model must exit cleanly ───────────────────────


def test_spark4b_missing_model_exits_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(kg, "SPARK_MODEL_DIR", tmp_path / "no-such-model")
    monkeypatch.setattr(kg, "_spark_state", None)
    with pytest.raises(SystemExit) as exc:
        kg.spark4b_extract([(ABSTRACT, "abstract")])
    assert exc.value.code == 2
    assert "spark4b model not found" in capsys.readouterr().err


# ── worklist: mark-retrieved append ──────────────────────────────────────────


def test_mark_retrieved_appends_idempotently(tmp_db, tmp_path):
    wl_path = tmp_path / "kg_worklist.json"
    _make_paper(tmp_db, "2401.01234", "Retrieved paper", abstract=ABSTRACT)

    assert kg.mark_retrieved("2401.01234", wl_path, db=tmp_db) is True
    assert kg.mark_retrieved("2401.01234", wl_path, db=tmp_db) is False  # duplicate

    wl = kg.load_worklist(wl_path)
    assert [e["paper_id"] for e in wl["pending"]] == ["2401.01234"]
    assert wl["pending"][0]["marked_at"]
    assert wl["done"] == []


# ── L2: worklist consumption (extraction stubbed) ────────────────────────────


def test_l2_consumes_worklist_and_marks_done(tmp_db, tmp_path, papers_root, monkeypatch):
    _make_paper(tmp_db, "p1", "To extract fully", abstract=ABSTRACT)
    wl_path = tmp_path / "kg_worklist.json"
    kg.mark_retrieved("p1", wl_path)

    calls: list[str] = []

    def fake_full_extract(db, pid, cfg, root, skip_refine=True):
        calls.append(pid)
        return {"ok": True, "local_id": pid}

    monkeypatch.setattr(kg, "_full_extract", fake_full_extract)
    stats = kg.run_l2(tmp_db, papers_root, worklist_path=wl_path, cfg={"llm": {"models": ["stub"]}})
    assert stats["ok"] == 1 and stats["failed"] == 0
    assert calls == ["p1"]

    wl = kg.load_worklist(wl_path)
    assert wl["pending"] == []
    assert [e["paper_id"] for e in wl["done"]] == ["p1"]
    assert wl["done"][0]["extracted_at"]


def test_l2_failure_keeps_pending(tmp_db, tmp_path, papers_root, monkeypatch):
    _make_paper(tmp_db, "p1", "Failing extraction", abstract=ABSTRACT)
    wl_path = tmp_path / "kg_worklist.json"
    kg.mark_retrieved("p1", wl_path)

    monkeypatch.setattr(
        kg,
        "_full_extract",
        lambda db, pid, cfg, root, skip_refine=True: {"ok": False, "error": "boom"},
    )
    stats = kg.run_l2(tmp_db, papers_root, worklist_path=wl_path, cfg={"llm": {"models": ["stub"]}})
    assert stats["failed"] == 1
    wl = kg.load_worklist(wl_path)
    assert [e["paper_id"] for e in wl["pending"]] == ["p1"]
    assert wl["done"] == []


def test_l2_skips_paper_that_already_has_concepts(tmp_db, tmp_path, papers_root, monkeypatch):
    _make_paper(tmp_db, "p2", "Already built", abstract=ABSTRACT)
    tmp_db.insert_concept("p2", "Method", "existing concept", 0.9, year=2024)
    tmp_db.commit()
    wl_path = tmp_path / "kg_worklist.json"
    kg.mark_retrieved("p2", wl_path)

    calls: list[str] = []

    def fake_full_extract(db, pid, cfg, root, skip_refine=True):
        calls.append(pid)
        return {"ok": True, "local_id": pid}

    monkeypatch.setattr(kg, "_full_extract", fake_full_extract)
    stats = kg.run_l2(tmp_db, papers_root, worklist_path=wl_path, cfg={"llm": {"models": ["stub"]}})
    assert stats["skipped"] == 1 and stats["ok"] == 0
    assert calls == []  # no extraction call for an already-built paper
    wl = kg.load_worklist(wl_path)
    assert wl["pending"] == []
    assert [e["paper_id"] for e in wl["done"]] == ["p2"]


def test_l2_explicit_papers_list(tmp_db, tmp_path, papers_root, monkeypatch):
    _make_paper(tmp_db, "p1", "Explicit target", abstract=ABSTRACT)
    monkeypatch.setattr(
        kg, "_full_extract", lambda db, pid, cfg, root, skip_refine=True: {"ok": True}
    )
    stats = kg.run_l2(
        tmp_db,
        papers_root,
        paper_ids=["p1"],
        worklist_path=tmp_path / "wl.json",
        cfg={"llm": {"models": ["stub"]}},
    )
    assert stats["ok"] == 1


def test_l2_requires_llm_models(tmp_db, tmp_path, papers_root):
    with pytest.raises(SystemExit):
        kg.run_l2(tmp_db, papers_root, paper_ids=["missing"], cfg={})
