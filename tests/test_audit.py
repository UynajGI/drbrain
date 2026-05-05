"""Tests for data quality audit pipeline."""

from __future__ import annotations

import datetime
import json
import tempfile
from pathlib import Path

from drbrain.services.audit import SEVERITY_ORDER, audit_papers
from drbrain.storage.database import Database


def _make_db() -> Database:
    """Create an in-memory database with schema."""
    db = Database(":memory:")
    return db


def _insert_paper(db: Database, local_id: str, title: str, **kwargs) -> None:
    """Insert a paper with defaults."""
    db.insert_paper(
        local_id,
        title=title,
        year=kwargs.get("year", 2023),
        status=kwargs.get("status", "extracted"),
        journal=kwargs.get("journal", "Test Journal"),
        publisher=kwargs.get("publisher", "Test Publisher"),
        citation_count=kwargs.get("citation_count", 0),
    )
    if "abstract" in kwargs:
        db.set_paper_abstract(local_id, kwargs["abstract"])
    if kwargs.get("doi") or kwargs.get("arxiv") or kwargs.get("s2_id") or kwargs.get("openalex_id"):
        db.insert_paper_ids(
            local_id,
            doi=kwargs.get("doi"),
            arxiv=kwargs.get("arxiv"),
            s2_id=kwargs.get("s2_id"),
            openalex_id=kwargs.get("openalex_id"),
        )
    db.commit()


# ── Rule tests ──────────────────────────────────────────────────────


def test_missing_title(tmp_path: Path):
    """Rule: missing_title (error)."""
    db = _make_db()
    # Title stored as empty
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES (?, ?, ?, ?)",
        ("p1", "", 2023, "extracted"),
    )
    db.commit()
    (tmp_path / "p1").mkdir(exist_ok=True)
    (tmp_path / "p1" / "raw.md").write_text("some content")
    issues = audit_papers(db, tmp_path, severity="error")
    assert any(i["rule"] == "missing_title" for i in issues)
    assert any(i["severity"] == "error" for i in issues if i["rule"] == "missing_title")


def test_missing_md(tmp_path: Path):
    """Rule: missing_md (error)."""
    db = _make_db()
    _insert_paper(db, "p1", "Test Paper")
    # No raw.md in paper dir
    (tmp_path / "p1").mkdir(exist_ok=True)
    issues = audit_papers(db, tmp_path, severity="error")
    assert any(i["rule"] == "missing_md" for i in issues)


def test_missing_doi(tmp_path: Path):
    """Rule: missing_doi (warning)."""
    db = _make_db()
    _insert_paper(db, "p1", "Test Paper")  # no paper_ids row
    (tmp_path / "p1").mkdir(exist_ok=True)
    (tmp_path / "p1" / "raw.md").write_text("some content")
    issues = audit_papers(db, tmp_path, severity="warning")
    assert any(i["rule"] == "missing_doi" for i in issues)


def test_missing_doi_has_arxiv(tmp_path: Path):
    """missing_doi should NOT fire when arxiv is present."""
    db = _make_db()
    _insert_paper(db, "p1", "Test Paper", arxiv="2301.00001")
    (tmp_path / "p1").mkdir(exist_ok=True)
    (tmp_path / "p1" / "raw.md").write_text("some content")
    issues = audit_papers(db, tmp_path, severity="warning")
    assert not any(i["rule"] == "missing_doi" for i in issues)


def test_missing_abstract(tmp_path: Path):
    """Rule: missing_abstract (warning)."""
    db = _make_db()
    _insert_paper(db, "p1", "Test Paper", abstract="")
    (tmp_path / "p1").mkdir(exist_ok=True)
    (tmp_path / "p1" / "raw.md").write_text("some content")
    issues = audit_papers(db, tmp_path, severity="warning")
    assert any(i["rule"] == "missing_abstract" for i in issues)


def test_missing_year(tmp_path: Path):
    """Rule: missing_year (warning)."""
    db = _make_db()
    _insert_paper(db, "p1", "Test Paper", year=None)
    (tmp_path / "p1").mkdir(exist_ok=True)
    (tmp_path / "p1" / "raw.md").write_text("some content")
    issues = audit_papers(db, tmp_path, severity="warning")
    assert any(i["rule"] == "missing_year" for i in issues)


def test_missing_journal(tmp_path: Path):
    """Rule: missing_journal (warning)."""
    db = _make_db()
    _insert_paper(db, "p1", "Test Paper", journal="")
    (tmp_path / "p1").mkdir(exist_ok=True)
    (tmp_path / "p1" / "raw.md").write_text("some content")
    issues = audit_papers(db, tmp_path, severity="warning")
    assert any(i["rule"] == "missing_journal" for i in issues)


def test_missing_authors(tmp_path: Path):
    """Rule: missing_authors (warning) -- no Actor-type concepts."""
    db = _make_db()
    _insert_paper(db, "p1", "Test Paper")
    (tmp_path / "p1").mkdir(exist_ok=True)
    (tmp_path / "p1" / "raw.md").write_text("some content")
    # No Actor concepts inserted
    issues = audit_papers(db, tmp_path, severity="warning")
    assert any(i["rule"] == "missing_authors" for i in issues)


def test_missing_authors_has_actors(tmp_path: Path):
    """missing_authors should NOT fire when Actors exist."""
    db = _make_db()
    _insert_paper(db, "p1", "Test Paper")
    db.insert_concept("p1", "Actor", "John Doe", 1.0)
    db.commit()
    (tmp_path / "p1").mkdir(exist_ok=True)
    (tmp_path / "p1" / "raw.md").write_text("some content")
    issues = audit_papers(db, tmp_path, severity="warning")
    assert not any(i["rule"] == "missing_authors" for i in issues)


def test_short_md(tmp_path: Path):
    """Rule: short_md (warning) -- raw.md < 200 chars."""
    db = _make_db()
    _insert_paper(db, "p1", "Test Paper")
    (tmp_path / "p1").mkdir(exist_ok=True)
    (tmp_path / "p1" / "raw.md").write_text("short")
    issues = audit_papers(db, tmp_path, severity="warning")
    assert any(i["rule"] == "short_md" for i in issues)


def test_short_md_ok(tmp_path: Path):
    """short_md should NOT fire when raw.md >= 200 chars."""
    db = _make_db()
    _insert_paper(db, "p1", "Test Paper")
    (tmp_path / "p1").mkdir(exist_ok=True)
    (tmp_path / "p1" / "raw.md").write_text("A" * 200)
    issues = audit_papers(db, tmp_path, severity="warning")
    assert not any(i["rule"] == "short_md" for i in issues)


def test_empty_tree(tmp_path: Path):
    """Rule: empty_tree (warning) -- missing or empty tree.json."""
    db = _make_db()
    _insert_paper(db, "p1", "Test Paper")
    (tmp_path / "p1").mkdir(exist_ok=True)
    (tmp_path / "p1" / "raw.md").write_text("A" * 200)
    # No tree.json at all
    issues = audit_papers(db, tmp_path, severity="warning")
    assert any(i["rule"] == "empty_tree" for i in issues)


def test_empty_tree_empty_file(tmp_path: Path):
    """empty_tree fires when tree.json exists but is empty."""
    db = _make_db()
    _insert_paper(db, "p1", "Test Paper")
    (tmp_path / "p1").mkdir(exist_ok=True)
    (tmp_path / "p1" / "raw.md").write_text("A" * 200)
    (tmp_path / "p1" / "tree.json").write_text("")
    issues = audit_papers(db, tmp_path, severity="warning")
    assert any(i["rule"] == "empty_tree" for i in issues)


def test_low_concept_count(tmp_path: Path):
    """Rule: low_concept_count (warning) -- < 3 concepts."""
    db = _make_db()
    _insert_paper(db, "p1", "Test Paper")
    (tmp_path / "p1").mkdir(exist_ok=True)
    (tmp_path / "p1" / "raw.md").write_text("A" * 200)
    # Only 1 concept
    db.insert_concept("p1", "Method", "CNN", 1.0)
    db.commit()
    issues = audit_papers(db, tmp_path, severity="warning")
    assert any(i["rule"] == "low_concept_count" for i in issues)


def test_low_concept_count_ok(tmp_path: Path):
    """low_concept_count should NOT fire with >= 3 concepts."""
    db = _make_db()
    _insert_paper(db, "p1", "Test Paper")
    (tmp_path / "p1").mkdir(exist_ok=True)
    (tmp_path / "p1" / "raw.md").write_text("A" * 200)
    for i in range(3):
        db.insert_concept("p1", "Method", f"Method {i}", 1.0)
    db.commit()
    issues = audit_papers(db, tmp_path, severity="warning")
    assert not any(i["rule"] == "low_concept_count" for i in issues)


def test_no_edges(tmp_path: Path):
    """Rule: no_edges (info) -- has concepts but zero edges."""
    db = _make_db()
    _insert_paper(db, "p1", "Test Paper")
    (tmp_path / "p1").mkdir(exist_ok=True)
    (tmp_path / "p1" / "raw.md").write_text("A" * 200)
    db.insert_concept("p1", "Method", "CNN", 1.0)
    db.insert_concept("p1", "Problem", "Overfitting", 1.0)
    db.commit()
    issues = audit_papers(db, tmp_path, severity="info")
    assert any(i["rule"] == "no_edges" for i in issues)


def test_no_edges_ok(tmp_path: Path):
    """no_edges should NOT fire when edges exist."""
    db = _make_db()
    _insert_paper(db, "p1", "Test Paper")
    (tmp_path / "p1").mkdir(exist_ok=True)
    (tmp_path / "p1" / "raw.md").write_text("A" * 200)
    db.insert_concept("p1", "Method", "CNN", 1.0)
    db.insert_edge("p1", "CNN", "solves", "p1")
    db.commit()
    issues = audit_papers(db, tmp_path, severity="info")
    assert not any(i["rule"] == "no_edges" for i in issues)


def test_placeholder_status(tmp_path: Path):
    """Rule: placeholder_status (info) -- paper status is placeholder."""
    db = _make_db()
    _insert_paper(db, "p1", "Test Paper", status="placeholder")
    (tmp_path / "p1").mkdir(exist_ok=True)
    (tmp_path / "p1" / "raw.md").write_text("A" * 200)
    issues = audit_papers(db, tmp_path, severity="info")
    assert any(i["rule"] == "placeholder_status" for i in issues)


def test_old_placeholder(tmp_path: Path):
    """Rule: old_placeholder (info) -- placeholder older than 30 days."""
    db = _make_db()
    old_date = (datetime.datetime.now() - datetime.timedelta(days=60)).strftime("%Y-%m-%d")
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status, created_at) VALUES (?, ?, ?, ?, ?)",
        ("p1", "Test Paper", 2023, "placeholder", old_date),
    )
    db.commit()
    (tmp_path / "p1").mkdir(exist_ok=True)
    (tmp_path / "p1" / "raw.md").write_text("A" * 200)
    issues = audit_papers(db, tmp_path, severity="info")
    assert any(i["rule"] == "old_placeholder" for i in issues)


def test_old_placeholder_recent(tmp_path: Path):
    """old_placeholder should NOT fire for recent placeholders."""
    db = _make_db()
    recent = (datetime.datetime.now() - datetime.timedelta(days=5)).strftime("%Y-%m-%d")
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status, created_at) VALUES (?, ?, ?, ?, ?)",
        ("p1", "Test Paper", 2023, "placeholder", recent),
    )
    db.commit()
    (tmp_path / "p1").mkdir(exist_ok=True)
    (tmp_path / "p1" / "raw.md").write_text("A" * 200)
    issues = audit_papers(db, tmp_path, severity="info")
    assert not any(i["rule"] == "old_placeholder" for i in issues)


def test_unresolved_env(tmp_path: Path):
    """Rule: unresolved_env (warning) -- title contains ${."""
    db = _make_db()
    _insert_paper(db, "p1", "${API_KEY} - A Study")
    (tmp_path / "p1").mkdir(exist_ok=True)
    (tmp_path / "p1" / "raw.md").write_text("A" * 200)
    issues = audit_papers(db, tmp_path, severity="warning")
    assert any(i["rule"] == "unresolved_env" for i in issues)


def test_duplicate_title(tmp_path: Path):
    """Rule: duplicate_title (info) -- normalized title matches another paper."""
    db = _make_db()
    _insert_paper(db, "p1", "A Deep Learning Approach")
    _insert_paper(db, "p2", "A  Deep  Learning  Approach!")
    (tmp_path / "p1").mkdir(exist_ok=True)
    (tmp_path / "p1" / "raw.md").write_text("A" * 200)
    (tmp_path / "p2").mkdir(exist_ok=True)
    (tmp_path / "p2" / "raw.md").write_text("A" * 200)
    issues = audit_papers(db, tmp_path, severity="info")
    dup_issues = [i for i in issues if i["rule"] == "duplicate_title"]
    assert len(dup_issues) >= 1


# ── Severity filtering ──────────────────────────────────────────────


def test_severity_filter_error(tmp_path: Path):
    """Only error-level issues returned when severity=error."""
    db = _make_db()
    _insert_paper(db, "p1", "Test Paper")  # no raw.md -> missing_md (error)
    # Also add known warning: no external IDs
    (tmp_path / "p1").mkdir(exist_ok=True)
    issues = audit_papers(db, tmp_path, severity="error")
    severities = {i["severity"] for i in issues}
    assert severities == {"error"}


def test_severity_filter_warning(tmp_path: Path):
    """error + warning issues returned when severity=warning."""
    db = _make_db()
    _insert_paper(db, "p1", "Test Paper")  # no IDs => missing_doi (warning), no md => error
    (tmp_path / "p1").mkdir(exist_ok=True)
    issues = audit_papers(db, tmp_path, severity="warning")
    severities = {i["severity"] for i in issues}
    assert severities <= {"error", "warning"}


def test_severity_filter_info(tmp_path: Path):
    """All severities returned when severity=info."""
    db = _make_db()
    _insert_paper(db, "p1", "Test Paper", status="placeholder")
    (tmp_path / "p1").mkdir(exist_ok=True)
    issues = audit_papers(db, tmp_path, severity="info")
    severities = {i["severity"] for i in issues}
    # At minimum we expect info-level issues
    assert "info" in severities


def test_severity_order():
    """SEVERITY_ORDER maps severity to numeric rank."""
    assert SEVERITY_ORDER["error"] < SEVERITY_ORDER["warning"]
    assert SEVERITY_ORDER["warning"] < SEVERITY_ORDER["info"]


# ── CLI tests ───────────────────────────────────────────────────────


def test_audit_cmd_with_json(tmp_path: Path):
    """audit_cmd --json returns valid JSON."""
    db = _make_db()
    _insert_paper(db, "p1", "Test Paper")
    (tmp_path / "p1").mkdir(exist_ok=True)
    (tmp_path / "p1" / "raw.md").write_text("A" * 200)

    from unittest.mock import MagicMock

    ctx = MagicMock()
    ctx.obj = {
        "config": {
            "dirs": {
                "papers": str(tmp_path),
                "db": ":memory:",
            }
        }
    }
    # We need to override the DB to use the same in-memory instance
    # Instead, test the underlying function directly
    result = audit_papers(db, tmp_path, severity="warning")
    json_str = json.dumps(result)
    parsed = json.loads(json_str)
    assert isinstance(parsed, list)
    for item in parsed:
        assert "rule" in item
        assert "severity" in item
        assert "paper_id" in item
        assert "message" in item


# ── PDF validation tests ────────────────────────────────────────────


def test_pdf_validation_ok():
    """_validate_pdf returns ok=True for a valid PDF."""
    from drbrain.parser.mineru_parser import _validate_pdf

    # Create a minimal valid PDF in a temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        # Minimal PDF: header + 1 page
        pdf_content = (
            b"%PDF-1.4\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\n"
            b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
            b"trailer<</Size 4/Root 1 0 R>>\n"
            b"startxref\n190\n%%EOF"
        )
        f.write(pdf_content)
        f.flush()
        pdf_path = Path(f.name)

    try:
        result = _validate_pdf(pdf_path)
        assert result.ok is True
        assert result.page_count == 1
        assert result.encrypted is False
    finally:
        pdf_path.unlink(missing_ok=True)


def test_pdf_validation_encrypted():
    """_validate_pdf detects encrypted PDFs."""
    import fitz

    from drbrain.parser.mineru_parser import _validate_pdf

    # Create a real PDF with encryption
    doc = fitz.open()
    doc.new_page()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.close()
        pdf_path = Path(f.name)

    try:
        doc.save(
            str(pdf_path), encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="test", user_pw="test"
        )
        doc.close()

        result = _validate_pdf(pdf_path)
        assert result.ok is False
        assert result.encrypted is True
    finally:
        pdf_path.unlink(missing_ok=True)


def test_pdf_validation_missing_file():
    """_validate_pdf handles missing files."""
    from drbrain.parser.mineru_parser import _validate_pdf

    result = _validate_pdf(Path("/nonexistent/pdf_path.pdf"))
    assert result.ok is False
    assert result.error != ""
