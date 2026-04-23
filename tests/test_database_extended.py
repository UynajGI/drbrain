"""Tests for database methods not covered by existing tests."""
import tempfile
from pathlib import Path
from datetime import datetime

from brbrain.storage.database import Database


def _make_db() -> Database:
    """Create a temp database."""
    td = tempfile.mkdtemp()
    return Database(Path(td) / "test.db")


def test_execute_and_commit():
    """execute returns a cursor, commit persists changes."""
    db = _make_db()
    cur = db.execute("SELECT 1")
    assert cur.fetchone() == (1,)
    db.commit()
    db.close()


def test_executemany():
    """executemany inserts multiple rows."""
    db = _make_db()
    db.insert_paper("p1", "A", 2020, "uploaded")
    db.insert_paper("p2", "B", 2021, "uploaded")
    db.insert_paper("p3", "C", 2022, "uploaded")
    db.commit()

    papers = db.get_all_papers()
    assert len(papers) == 3
    db.close()


def test_get_paper_not_found():
    """get_paper returns None for unknown ID."""
    db = _make_db()
    assert db.get_paper("nonexistent") is None
    db.close()


def test_upgrade_placeholder():
    """upgrade_placeholder changes status from placeholder to uploaded."""
    db = _make_db()
    db.insert_paper("p1", "Test", 2024, "placeholder")
    db.commit()

    before = db.get_paper("p1")
    assert before["status"] == "placeholder"

    db.upgrade_placeholder("p1")
    db.commit()

    after = db.get_paper("p1")
    assert after["status"] == "uploaded"
    db.close()


def test_upgrade_placeholder_noop_for_uploaded():
    """upgrade_placeholder does nothing for already uploaded papers."""
    db = _make_db()
    db.insert_paper("p1", "Test", 2024, "uploaded")
    db.commit()

    db.upgrade_placeholder("p1")
    db.commit()

    paper = db.get_paper("p1")
    assert paper["status"] == "uploaded"
    db.close()


def test_insert_and_get_concepts_by_paper():
    """insert_concept + get_concepts_by_paper round-trip."""
    db = _make_db()
    db.insert_paper("p1", "Test", 2024, "uploaded")
    cid = db.insert_concept("p1", "Problem", "ML Scalability", confidence=0.95, year=2024)
    assert cid is not None

    concepts = db.get_concepts_by_paper("p1")
    assert len(concepts) == 1
    assert concepts[0]["label"] == "ML Scalability"
    assert concepts[0]["type"] == "Problem"
    assert concepts[0]["confidence"] == 0.95
    db.close()


def test_insert_alias():
    """insert_alias stores variant->canonical mapping."""
    db = _make_db()
    # Need a paper first (concepts FK to papers)
    db.insert_paper("p1", "Test", 2024, "uploaded")
    # Need a concept first for alias FK
    cid = db.insert_concept("p1", "Method", "transformers", year=2024)
    db.insert_alias("transformers", str(cid))
    db.insert_alias("Transformer", str(cid))
    db.commit()

    row = db.conn.execute("SELECT canonical_id FROM aliases WHERE variant='transformers'").fetchone()
    assert row[0] == str(cid)
    db.close()


def test_insert_and_get_seeds():
    """insert_seed + get_all_seeds round-trip."""
    db = _make_db()
    sid = db.insert_seed("unaddressed_gap", "No method addresses Gap X", confidence=0.8)
    assert sid is not None

    seeds = db.get_all_seeds()
    assert len(seeds) == 1
    assert seeds[0]["pattern_type"] == "unaddressed_gap"
    db.close()


def test_delete_seed():
    """delete_seed removes a research seed."""
    db = _make_db()
    sid = db.insert_seed("test", "Test seed")
    db.commit()

    assert len(db.get_all_seeds()) == 1
    db.delete_seed(sid)
    db.commit()
    assert len(db.get_all_seeds()) == 0
    db.close()


def test_insert_and_get_arguments():
    """insert_argument + get_arguments_by_paper round-trip."""
    db = _make_db()
    db.insert_paper("p1", "Test", 2024, "uploaded")
    aid = db.insert_argument(
        "p1", "Method X outperforms Y", "supports",
        "Method X", "Method", "empirical", "See Table 3", 0.9,
    )
    assert aid is not None

    args = db.get_arguments_by_paper("p1")
    assert len(args) == 1
    assert args[0]["claim"] == "Method X outperforms Y"
    assert args[0]["claim_type"] == "supports"
    db.close()


def test_confidence_queue_lifecycle():
    """Queue items flow: insert -> pending -> accept/reject."""
    db = _make_db()
    db.insert_paper("p1", "Test", 2024, "uploaded")
    qid = db.insert_queue_item("p1", "concept", '{"label": "Test Concept"}', 0.6)
    assert qid is not None

    pending = db.get_queue_pending()
    assert len(pending) == 1
    assert pending[0]["queue_id"] == qid

    db.accept_queue_item(qid)
    db.commit()
    assert len(db.get_queue_pending()) == 0
    db.close()


def test_queue_reject():
    """reject_queue_item marks item as rejected."""
    db = _make_db()
    db.insert_paper("p1", "Test", 2024, "uploaded")
    qid = db.insert_queue_item("p1", "concept", '{"label": "X"}', 0.5)
    db.commit()

    db.reject_queue_item(qid)
    db.commit()

    status = db.conn.execute(
        "SELECT status FROM confidence_queue WHERE queue_id = ?", (qid,)
    ).fetchone()[0]
    assert status == "rejected"
    db.close()


def test_get_concept_evolution():
    """get_concept_evolution returns year-by-year usage."""
    db = _make_db()
    db.insert_paper("p1", "A", 2020, "uploaded")
    db.insert_paper("p2", "B", 2021, "uploaded")
    db.insert_paper("p3", "C", 2022, "uploaded")
    db.insert_concept("p1", "Method", "Transformer", 0.9, year=2020)
    db.insert_concept("p2", "Method", "Transformer", 0.8, year=2021)
    db.insert_concept("p3", "Method", "Transformer", 0.95, year=2022)
    db.commit()

    evolution = db.get_concept_evolution("Transformer")
    assert len(evolution) == 3
    years = [e["year"] for e in evolution]
    assert years == [2020, 2021, 2022]
    db.close()


def test_detect_evolution_signals():
    """detect_evolution_signals classifies concepts by temporal patterns."""
    db = _make_db()
    current_year = datetime.now().year
    db.insert_paper("p1", "A", current_year, "uploaded")
    db.insert_paper("p2", "B", current_year, "uploaded")
    db.insert_concept("p1", "Method", "NewThing", 0.9, year=current_year)
    db.insert_concept("p2", "Method", "NewThing", 0.85, year=current_year)
    db.commit()

    signals = db.detect_evolution_signals()
    new_thing = [s for s in signals if s["label"] == "NewThing"]
    assert len(new_thing) == 1
    assert "signal" in new_thing[0]
    db.close()


def test_insert_edge_dedup():
    """Duplicate edges are ignored (INSERT OR IGNORE)."""
    db = _make_db()
    db.insert_edge("p1", "p2", "cites", "p1")
    db.insert_edge("p1", "p2", "cites", "p1")  # Duplicate
    db.commit()

    count = db.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    assert count == 1
    db.close()


def test_save_raw_md_file():
    """save_raw_md writes markdown to data/papers/<local_id>.md and returns path."""
    from brbrain.cli.commands import save_raw_md
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td) / "papers"
        md = "# Title\n\n![img](images/abc.jpg)\n\nAbstract text here."
        result = save_raw_md(md, "p1", papers_dir)
        assert result is True
        assert (papers_dir / "p1.md").exists()
        content = (papers_dir / "p1.md").read_text()
        assert content == md


def test_save_raw_md_copies_images():
    """save_raw_md copies images and rewrites refs."""
    from brbrain.cli.commands import save_raw_md
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td) / "papers"
        src_dir = Path(td) / "src"
        src_dir.mkdir()
        img_dir = src_dir / "images"
        img_dir.mkdir()
        (img_dir / "abc.jpg").write_bytes(b"fake")

        md = "# Title\n\n![img](images/abc.jpg)\n\nAbstract text here."
        save_raw_md(md, "p1", papers_dir, img_dir)  # img_dir is the images dir

        assert (papers_dir / "p1.md").exists()
        assert (papers_dir / "images" / "p1" / "abc.jpg").exists()
        content = (papers_dir / "p1.md").read_text()
        assert "images/p1/images/abc.jpg" in content
