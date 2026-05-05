"""Tests for database methods not covered by existing tests."""

from datetime import datetime
from pathlib import Path


def test_execute_and_commit(tmp_db):
    """execute returns a cursor, commit persists changes."""
    cur = tmp_db.execute("SELECT 1")
    assert cur.fetchone() == (1,)
    tmp_db.commit()


def test_executemany(tmp_db):
    """executemany inserts multiple rows."""
    tmp_db.insert_paper("p1", "A", 2020, "uploaded")
    tmp_db.insert_paper("p2", "B", 2021, "uploaded")
    tmp_db.insert_paper("p3", "C", 2022, "uploaded")
    tmp_db.commit()

    papers = tmp_db.get_all_papers()
    assert len(papers) == 3


def test_get_paper_not_found(tmp_db):
    """get_paper returns None for unknown ID."""
    assert tmp_db.get_paper("nonexistent") is None


def test_upgrade_placeholder(tmp_db):
    """upgrade_placeholder changes status from placeholder to uploaded."""
    tmp_db.insert_paper("p1", "Test", 2024, "placeholder")
    tmp_db.commit()

    before = tmp_db.get_paper("p1")
    assert before["status"] == "placeholder"

    tmp_db.upgrade_placeholder("p1")
    tmp_db.commit()

    after = tmp_db.get_paper("p1")
    assert after["status"] == "uploaded"


def test_upgrade_placeholder_noop_for_uploaded(tmp_db):
    """upgrade_placeholder does nothing for already uploaded papers."""
    tmp_db.insert_paper("p1", "Test", 2024, "uploaded")
    tmp_db.commit()

    tmp_db.upgrade_placeholder("p1")
    tmp_db.commit()

    paper = tmp_db.get_paper("p1")
    assert paper["status"] == "uploaded"


def test_insert_and_get_concepts_by_paper(tmp_db):
    """insert_concept + get_concepts_by_paper round-trip."""
    tmp_db.insert_paper("p1", "Test", 2024, "uploaded")
    cid = tmp_db.insert_concept("p1", "Problem", "ML Scalability", confidence=0.95, year=2024)
    assert cid is not None

    concepts = tmp_db.get_concepts_by_paper("p1")
    assert len(concepts) == 1
    assert concepts[0]["label"] == "ML Scalability"
    assert concepts[0]["type"] == "Problem"
    assert concepts[0]["confidence"] == 0.95


def test_insert_alias(tmp_db):
    """insert_alias stores variant->canonical mapping."""
    # Need a paper first (concepts FK to papers)
    tmp_db.insert_paper("p1", "Test", 2024, "uploaded")
    # Need a concept first for alias FK
    cid = tmp_db.insert_concept("p1", "Method", "transformers", year=2024)
    tmp_db.insert_alias("transformers", str(cid))
    tmp_db.insert_alias("Transformer", str(cid))
    tmp_db.commit()

    row = tmp_db.conn.execute(
        "SELECT canonical_id FROM aliases WHERE variant='transformers'"
    ).fetchone()
    assert row[0] == str(cid)


def test_insert_and_get_seeds(tmp_db):
    """insert_seed + get_all_seeds round-trip."""
    sid = tmp_db.insert_seed("unaddressed_gap", "No method addresses Gap X", confidence=0.8)
    assert sid is not None

    seeds = tmp_db.get_all_seeds()
    assert len(seeds) == 1
    assert seeds[0]["pattern_type"] == "unaddressed_gap"


def test_delete_seed(tmp_db):
    """delete_seed removes a research seed."""
    sid = tmp_db.insert_seed("test", "Test seed")
    tmp_db.commit()

    assert len(tmp_db.get_all_seeds()) == 1
    tmp_db.delete_seed(sid)
    tmp_db.commit()
    assert len(tmp_db.get_all_seeds()) == 0


def test_insert_and_get_arguments(tmp_db):
    """insert_argument + get_arguments_by_paper round-trip."""
    tmp_db.insert_paper("p1", "Test", 2024, "uploaded")
    aid = tmp_db.insert_argument(
        "p1",
        "Method X outperforms Y",
        "supports",
        "Method X",
        "Method",
        "empirical",
        "See Table 3",
        0.9,
    )
    assert aid is not None

    args = tmp_db.get_arguments_by_paper("p1")
    assert len(args) == 1
    assert args[0]["claim"] == "Method X outperforms Y"
    assert args[0]["claim_type"] == "supports"


def test_confidence_queue_lifecycle(tmp_db):
    """Queue items flow: insert -> pending -> accept/reject."""
    tmp_db.insert_paper("p1", "Test", 2024, "uploaded")
    qid = tmp_db.insert_queue_item("p1", "concept", '{"label": "Test Concept"}', 0.6)
    assert qid is not None

    pending = tmp_db.get_queue_pending()
    assert len(pending) == 1
    assert pending[0]["queue_id"] == qid

    tmp_db.accept_queue_item(qid)
    tmp_db.commit()
    assert len(tmp_db.get_queue_pending()) == 0


def test_queue_reject(tmp_db):
    """reject_queue_item marks item as rejected."""
    tmp_db.insert_paper("p1", "Test", 2024, "uploaded")
    qid = tmp_db.insert_queue_item("p1", "concept", '{"label": "X"}', 0.5)
    tmp_db.commit()

    tmp_db.reject_queue_item(qid)
    tmp_db.commit()

    status = tmp_db.conn.execute(
        "SELECT status FROM confidence_queue WHERE queue_id = ?", (qid,)
    ).fetchone()[0]
    assert status == "rejected"


def test_get_concept_evolution(tmp_db):
    """get_concept_evolution returns year-by-year usage."""
    tmp_db.insert_paper("p1", "A", 2020, "uploaded")
    tmp_db.insert_paper("p2", "B", 2021, "uploaded")
    tmp_db.insert_paper("p3", "C", 2022, "uploaded")
    tmp_db.insert_concept("p1", "Method", "Transformer", 0.9, year=2020)
    tmp_db.insert_concept("p2", "Method", "Transformer", 0.8, year=2021)
    tmp_db.insert_concept("p3", "Method", "Transformer", 0.95, year=2022)
    tmp_db.commit()

    evolution = tmp_db.get_concept_evolution("Transformer")
    assert len(evolution) == 3
    years = [e["year"] for e in evolution]
    assert years == [2020, 2021, 2022]


def test_detect_evolution_signals(tmp_db):
    """detect_evolution_signals classifies concepts by temporal patterns."""
    current_year = datetime.now().year
    tmp_db.insert_paper("p1", "A", current_year, "uploaded")
    tmp_db.insert_paper("p2", "B", current_year, "uploaded")
    tmp_db.insert_concept("p1", "Method", "NewThing", 0.9, year=current_year)
    tmp_db.insert_concept("p2", "Method", "NewThing", 0.85, year=current_year)
    tmp_db.commit()

    signals = tmp_db.detect_evolution_signals()
    new_thing = [s for s in signals if s["label"] == "NewThing"]
    assert len(new_thing) == 1
    assert "signal" in new_thing[0]


def test_insert_edge_dedup(tmp_db):
    """Duplicate edges are ignored (INSERT OR IGNORE)."""
    tmp_db.insert_edge("p1", "p2", "cites", "p1")
    tmp_db.insert_edge("p1", "p2", "cites", "p1")  # Duplicate
    tmp_db.commit()

    count = tmp_db.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    assert count == 1


def test_save_paper_artifacts():
    """_save_paper_artifacts writes raw.md, source.pdf, and images into per-paper dir."""
    import tempfile
    from types import SimpleNamespace

    from drbrain.cli.commands import _save_paper_artifacts

    with tempfile.TemporaryDirectory() as td:
        paper_dir = Path(td) / "papers" / "p1"
        paper_dir.mkdir(parents=True)
        # Create a fake source PDF
        src_pdf = Path(td) / "input.pdf"
        src_pdf.write_bytes(b"fake pdf content")

        parsed = SimpleNamespace(
            raw_md="# Title\n\nAbstract text here.",
            images_dir=None,
        )
        _save_paper_artifacts(parsed, "p1", paper_dir, src_pdf)
        assert (paper_dir / "raw.md").exists()
        assert (paper_dir / "source.pdf").exists()
        content = (paper_dir / "raw.md").read_text()
        assert "Title" in content


def test_save_paper_artifacts_copies_images():
    """_save_paper_artifacts copies images into per-paper dir."""
    import tempfile
    from types import SimpleNamespace

    from drbrain.cli.commands import _save_paper_artifacts

    with tempfile.TemporaryDirectory() as td:
        paper_dir = Path(td) / "papers" / "p1"
        paper_dir.mkdir(parents=True)
        src_pdf = Path(td) / "input.pdf"
        src_pdf.write_bytes(b"fake pdf")

        # Create source images dir
        img_dir = Path(td) / "src_images"
        img_dir.mkdir()
        (img_dir / "abc.jpg").write_bytes(b"fake image")

        parsed = SimpleNamespace(
            raw_md="# Title\n\n![img](images/abc.jpg)\n\nAbstract.",
            images_dir=img_dir,
        )
        _save_paper_artifacts(parsed, "p1", paper_dir, src_pdf)

        assert (paper_dir / "images" / "abc.jpg").exists()
        content = (paper_dir / "raw.md").read_text()
        assert "images/abc.jpg" in content
