"""Tests for the concept co-occurrence graph builder."""

from __future__ import annotations

import tempfile
from pathlib import Path

from drbrain.concept_graph.builder import (
    apply_filter,
    build_cliques,
    concepts_for_paper,
    is_formula,
    is_noise,
    normalize_concept,
)
from drbrain.storage.database import Database


def _tmp_db() -> tuple[Database, tempfile.TemporaryDirectory]:
    td = tempfile.TemporaryDirectory()
    return Database(Path(td.name) / "test.db"), td


# ── normalization (aligns with paper Table 1) ────────────────────────────────


def test_normalize_lowercases_and_strips() -> None:
    assert normalize_concept("  Graphene Oxide ") == "graphene oxide"


def test_normalize_removes_fill_word_of() -> None:
    # "Removal of Metal Impurities" -> "metal impurity removal" (of dropped, singular)
    assert normalize_concept("Removal of Metal Impurities") == "removal metal impurity"


def test_normalize_singularizes() -> None:
    assert normalize_concept("al2o3 coatings") == "al2o3 coating"
    assert normalize_concept("ribbons") == "ribbon"
    assert normalize_concept("batteries") == "battery"
    assert normalize_concept("processes") == "process"
    assert normalize_concept("gases") == "gas"
    assert normalize_concept("catalysts") == "catalyst"


def test_normalize_keeps_uncountable_exceptions() -> None:
    # EXCEPT set: collective/uncountable nouns keep their plural form.
    assert normalize_concept("series") == "series"
    assert normalize_concept("species") == "species"
    assert normalize_concept("status") == "status"
    assert normalize_concept("analyses") == "analyses"
    assert normalize_concept("materials") == "materials"
    assert normalize_concept("properties") == "properties"


def test_normalize_unifies_dimension_notation() -> None:
    # 2d/3d -> 2D/3D writing unification (materials context).
    assert normalize_concept("2d materials") == "2D materials"
    assert normalize_concept("3D printing") == "3D printing"
    assert normalize_concept("1D nanowires") == "1D nanowire"
    assert normalize_concept("al2o3 coatings") == "al2o3 coating"  # digits untouched


def test_normalize_empty() -> None:
    assert normalize_concept("") == ""
    assert normalize_concept("of the") == ""


# ── noise / formula heuristics ───────────────────────────────────────────────


def test_is_noise_flags_biomedical_residue() -> None:
    assert is_noise("cancer")
    assert is_noise("drug delivery")
    assert is_noise("Alzheimer's disease")
    assert not is_noise("graphene oxide")
    assert not is_noise("battery")


def test_is_noise_flags_trivial_labels() -> None:
    assert is_noise("x")
    assert is_noise("12345")
    assert is_noise("---")
    assert is_noise("2019 review")
    assert is_noise("a" * 81)


def test_is_formula_detects_chemical_formulas() -> None:
    assert is_formula("Al2O3")
    assert is_formula("TiO2")
    assert is_formula("SiO2")
    assert is_formula("Fe2O3")
    assert is_formula("H2O")
    assert is_formula("ZnO")
    assert not is_formula("graphene")
    assert not is_formula("graphene oxide")
    assert not is_formula("battery")
    assert not is_formula("2D materials")


# ── concept source ───────────────────────────────────────────────────────────


def test_concepts_for_paper_from_terms() -> None:
    db, td = _tmp_db()
    try:
        db.insert_paper("p1", "T", 2020, "extracted")
        db.insert_paper_term("p1", "graphene oxide", kind="keyword")
        db.insert_paper_term("p1", "Graphene Oxide", kind="topic")  # duplicate after norm
        db.insert_paper_term("p1", "battery", kind="keyword")
        db.conn.commit()
        concepts = concepts_for_paper(db, "p1", source="terms")
        assert concepts == ["graphene oxide", "battery"]
    finally:
        db.close()
        td.cleanup()


def test_concepts_for_paper_from_concepts_table() -> None:
    db, td = _tmp_db()
    try:
        db.insert_paper("p1", "T", 2020, "extracted")
        db.insert_concept("p1", "Method", "Solar Cells", 0.9, year=2020)
        db.insert_concept("p1", "Gap", "stability", 0.8, year=2020)
        db.conn.commit()
        concepts = concepts_for_paper(db, "p1", source="concepts")
        assert "solar cell" in concepts
        assert "stability" in concepts
    finally:
        db.close()
        td.cleanup()


# ── clique building ──────────────────────────────────────────────────────────


def test_build_cliques_edge_count_and_year() -> None:
    db, td = _tmp_db()
    try:
        db.insert_paper("p1", "T", 2021, "extracted")
        for term in ("concept one", "concept two", "concept three"):
            db.insert_paper_term("p1", term, kind="keyword")
        db.conn.commit()

        edges = build_cliques(db, source="terms")
        assert edges == 3  # C(3, 2)
        rows = db.conn.execute(
            "SELECT src_label, dst_label, year, paper_id FROM concept_cooccurrence"
        ).fetchall()
        assert len(rows) == 3
        assert all(r[2] == 2021 for r in rows)
        assert all(r[3] == "p1" for r in rows)
    finally:
        db.close()
        td.cleanup()


def test_build_cliques_skips_single_concept() -> None:
    db, td = _tmp_db()
    try:
        db.insert_paper("p1", "T", 2020, "extracted")
        db.insert_paper_term("p1", "solo concept", kind="keyword")
        db.conn.commit()
        assert build_cliques(db, source="terms") == 0
    finally:
        db.close()
        td.cleanup()


# ── filtering / aggregation ──────────────────────────────────────────────────


def test_apply_filter_thresholds() -> None:
    db, td = _tmp_db()
    try:
        # "common phrase" appears in 3 papers; "rare phrase" in 1; "x" is 1 word.
        for pid in ("p1", "p2", "p3"):
            db.insert_paper(pid, "T", 2020, "extracted")
            db.insert_paper_term(pid, "common phrase", kind="keyword")
            db.insert_paper_term(pid, "other concept", kind="keyword")
            db.conn.commit()
        db.insert_paper("p4", "T", 2020, "extracted")
        db.insert_paper_term("p4", "rare phrase", kind="keyword")
        db.insert_paper_term("p4", "common phrase", kind="keyword")
        db.conn.commit()

        build_cliques(db, source="terms")
        stats = apply_filter(db, min_freq=3, min_words=2)
        kept_labels = {r[0] for r in db.conn.execute("SELECT label FROM concept_nodes").fetchall()}
        # "common phrase" in 4 papers, "other concept" in 3 -> kept; "rare phrase" in 1 -> dropped
        assert "common phrase" in kept_labels
        assert "other concept" in kept_labels
        assert "rare phrase" not in kept_labels
        assert stats["kept"] == len(kept_labels)
    finally:
        db.close()
        td.cleanup()
