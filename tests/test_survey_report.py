"""Tests for the one-shot survey report generator (report/survey.py)."""

from __future__ import annotations

from drbrain.report.survey import (
    H_CROSS,
    H_EVIDENCE,
    H_GAPS,
    generate_survey,
    generate_survey_data,
)


def _seed_graph(db) -> None:
    """Build a minimal knowledge graph: papers, concepts, edges, citations, answer."""
    db.insert_paper("p1", "Gap Source Paper", 2021, "uploaded")
    db.insert_paper("p2", "Supporting Paper", 2022, "uploaded")
    db.insert_paper("p3", "Challenging Paper", 2023, "uploaded")

    db.insert_concept("p1", "Gap", "gap_x", 0.9, year=2021, section="discussion", node_id="n1")
    db.insert_concept("p2", "Debate", "debate_y", 0.8, year=2022, section="results", node_id="n2")
    db.insert_concept("p1", "Method", "method_a", 0.7, year=2021)
    db.insert_concept("p2", "Method", "concept_sup", 0.7, year=2022)
    db.insert_concept("p3", "Method", "concept_cha", 0.7, year=2023)

    # Epistemic columns on the Gap concept (provenance / authority / validity).
    db.conn.execute(
        "UPDATE concepts SET provenance=?, authority=?, valid_from=?, valid_to=? WHERE label=?",
        ("SOURCE", "peer_reviewed", 2021, None, "gap_x"),
    )

    # Gap is left open (no solves/addresses) -> unaddressed_gap seed.
    db.insert_edge("method_a", "gap_x", "leaves_open", "p1")
    # Conflicting views -> debate_zone seed.
    db.insert_edge("concept_sup", "debate_y", "supports", "p2")
    db.insert_edge("concept_cha", "debate_y", "challenges", "p3")

    # Citation graph: p2 and p3 both cite p1 (direct + shared reference).
    db.insert_paper_citation("p2", "p1")
    db.insert_paper_citation("p3", "p1")

    db.record_answer(
        "什么是 gap_x?",
        "gap_x 指当前研究尚未解决的问题。",
        evidence_ids=["c1", "c2"],
        provenance="GRAPH",
    )
    db.commit()


def test_survey_has_three_sections(tmp_db):
    _seed_graph(tmp_db)

    md = generate_survey(tmp_db)

    assert H_GAPS in md
    assert H_CROSS in md
    assert H_EVIDENCE in md


def test_survey_lists_gap_and_debate(tmp_db):
    _seed_graph(tmp_db)

    md = generate_survey(tmp_db)

    # Gap section surfaces the seeded Gap + Debate concepts and their seeds.
    assert "gap_x" in md
    assert "debate_y" in md
    # The unaddressed-gap seed description is rendered.
    assert "no proposed solution" in md
    # The debate-zone seed description is rendered.
    assert "active debate" in md


def test_survey_cross_references(tmp_db):
    _seed_graph(tmp_db)

    md = generate_survey(tmp_db)

    # Direct citation table (who-cites-whom).
    assert "Supporting Paper" in md
    assert "Challenging Paper" in md
    assert "Gap Source Paper" in md


def test_survey_evidence_chain_traces_source(tmp_db):
    _seed_graph(tmp_db)

    md = generate_survey(tmp_db)

    # Evidence chain must trace gap_x back to authority + paper + section.
    assert "gap_x" in md
    assert "peer_reviewed" in md
    assert "Gap Source Paper" in md
    assert "discussion" in md
    # Answer record is surfaced with its evidence ids.
    assert "什么是 gap_x?" in md
    assert "c1" in md


def test_survey_topic_filters(tmp_db):
    _seed_graph(tmp_db)

    md = generate_survey(tmp_db, topic="gap_x")

    assert "gap_x" in md
    # debate_y does not match the topic, so it should be filtered out.
    assert "debate_y" not in md


def test_survey_unmatched_topic_does_not_crash(tmp_db):
    _seed_graph(tmp_db)

    md = generate_survey(tmp_db, topic="no_such_topic_anywhere")

    assert H_GAPS in md
    assert H_CROSS in md
    assert H_EVIDENCE in md


def test_survey_empty_library_does_not_crash(tmp_db):
    md = generate_survey(tmp_db)

    assert H_GAPS in md
    assert H_CROSS in md
    assert H_EVIDENCE in md


def test_survey_data_is_structured(tmp_db):
    _seed_graph(tmp_db)

    data = generate_survey_data(tmp_db)

    assert data["summary"]["paper_count"] == 3
    assert data["summary"]["gap_concepts"] >= 1
    assert data["summary"]["debate_concepts"] >= 1
    assert data["cross_references"]["counts"]["direct_citations"] == 2
    assert data["cross_references"]["counts"]["shared_references"] >= 1
    assert data["evidence_chain"]["counts"]["gap_evidence"] >= 1
