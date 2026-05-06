"""Tests for report/generator.py — PaperReport, RefEntry, and helpers."""

from drbrain.report.generator import PaperReport, RefEntry, total_refs_and_citations


def test_ref_entry_creation():
    """RefEntry constructs with expected defaults."""
    ref = RefEntry(title="Test Paper", year=2026)
    assert ref.title == "Test Paper"
    assert ref.year == 2026
    assert ref.ids == {}
    assert ref.in_graph is False
    assert ref.local_id is None


def test_ref_entry_with_all_fields():
    """RefEntry accepts all fields."""
    ref = RefEntry(
        title="Full Entry",
        year=2024,
        ids={"doi": "10.1234/test"},
        in_graph=True,
        local_id="p1",
    )
    assert ref.local_id == "p1"
    assert ref.in_graph is True
    assert ref.ids == {"doi": "10.1234/test"}


def test_paper_report_creation():
    """PaperReport constructs and to_dict works."""
    report = PaperReport(local_id="p1", title="Test", year=2026, concepts={}, arguments=[])
    assert report.local_id == "p1"
    assert report.title == "Test"
    d = report.to_dict()
    assert "paper" in d
    assert d["paper"]["local_id"] == "p1"
    assert d["paper"]["title"] == "Test"


def test_paper_report_with_references():
    """PaperReport summary counts references and citations."""
    report = PaperReport(
        local_id="p1",
        title="Test",
        year=2026,
        references=[
            RefEntry(title="Ref1", year=2020, in_graph=True),
            RefEntry(title="Ref2", year=2021, in_graph=False),
        ],
        citations=[
            RefEntry(title="Cit1", year=2022, in_graph=True),
        ],
    )
    s = report.summary
    assert s["total_refs"] == 2
    assert s["total_cits"] == 1
    assert s["refs_in_graph"] == 1
    assert s["cits_in_graph"] == 1
    assert s["graph_coverage"] == round(2 / 3, 3)


def test_paper_report_empty_coverage():
    """PaperReport with no refs/citations has 0.0 coverage."""
    report = PaperReport(local_id="p1", title="Empty", year=2026)
    s = report.summary
    assert s["graph_coverage"] == 0.0
    assert s["total_refs"] == 0
    assert s["total_cits"] == 0


def test_total_refs_and_citations():
    """total_refs_and_citations sums references + citations."""
    report = PaperReport(
        local_id="p1",
        title="Test",
        year=2026,
        references=[RefEntry(title="r1", year=2020), RefEntry(title="r2", year=2021)],
        citations=[RefEntry(title="c1", year=2022)],
    )
    assert total_refs_and_citations(report) == 3


def test_total_refs_and_citations_empty():
    """total_refs_and_citations returns 0 for empty report."""
    report = PaperReport(local_id="p1", title="Empty", year=2026)
    assert total_refs_and_citations(report) == 0


def test_boundary_alert():
    """boundary_alert flags when many core refs are missing."""
    report = PaperReport(
        local_id="p1",
        title="Test",
        year=2026,
        references=[RefEntry(title=f"Ref{i}", year=2020, in_graph=False) for i in range(7)],
    )
    alert = report.boundary_alert
    assert alert["missing_core_refs"] is True
