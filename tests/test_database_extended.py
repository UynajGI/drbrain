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


# -- Volume/pages interface tests --


def test_insert_paper_accepts_volume_pages(tmp_db):
    """insert_paper must accept and store volume/pages."""
    db = tmp_db
    db.insert_paper("ptest", "Test Paper", 2024, "uploaded", volume="42", pages="100-120")
    p = db.get_paper("ptest")
    assert p["volume"] == "42"
    assert p["pages"] == "100-120"


def test_get_paper_returns_volume_pages(tmp_db):
    """get_paper must include volume and pages in result dict."""
    db = tmp_db
    db.insert_paper("ptest2", "Test", 2023, "uploaded", volume="10", pages="50-55")
    p = db.get_paper("ptest2")
    assert "volume" in p
    assert "pages" in p
    assert p["volume"] == "10"
    assert p["pages"] == "50-55"


def test_insert_paper_volume_pages_default_empty(tmp_db):
    """insert_paper defaults volume/pages to empty strings."""
    db = tmp_db
    db.insert_paper("ptest3", "Test", 2022, "uploaded")
    p = db.get_paper("ptest3")
    assert p["volume"] == ""
    assert p["pages"] == ""


# ── Temporal evolution signals ──────────────────────────────────


def _seed_papers_and_concepts(db, label, ctype, year_confidence_pairs):
    for i, (year, conf) in enumerate(year_confidence_pairs):
        pid = f"p{i:03d}_{label.replace(' ', '_')}"
        db.insert_paper(pid, f"Paper about {label} ({year})", year, "uploaded")
        db.insert_concept(pid, ctype, label, conf, year=year)
    db.commit()


def test_signal_emerging(tmp_db):
    current = datetime.now().year
    _seed_papers_and_concepts(
        tmp_db,
        "quantum transformer",
        "Method",
        [
            (current - 2, 0.9),
            (current - 1, 0.88),
            (current - 1, 0.91),
            (current, 0.85),
            (current, 0.90),
            (current, 0.87),
            (current, 0.92),
        ],
    )
    signals = tmp_db.detect_evolution_signals()
    matching = [s for s in signals if s["label"] == "quantum transformer"]
    assert len(matching) == 1
    assert matching[0]["signal"] == "emerging"


def test_signal_established(tmp_db):
    current = datetime.now().year
    pairs = [(current - 5 + (i % 6), 0.85 + (i % 10) * 0.01) for i in range(12)]
    _seed_papers_and_concepts(tmp_db, "attention mechanism", "Method", pairs)
    signals = tmp_db.detect_evolution_signals()
    matching = [s for s in signals if s["label"] == "attention mechanism"]
    assert len(matching) == 1
    assert matching[0]["signal"] == "established"


def test_signal_declining(tmp_db):
    current = datetime.now().year
    _seed_papers_and_concepts(
        tmp_db,
        "rnn language model",
        "Method",
        [
            (current - 8, 0.9),
            (current - 7, 0.88),
            (current - 5, 0.85),
            (current - 4, 0.82),
        ],
    )
    signals = tmp_db.detect_evolution_signals()
    matching = [s for s in signals if s["label"] == "rnn language model"]
    assert len(matching) == 1
    assert matching[0]["signal"] == "declining"


def test_signal_contested(tmp_db):
    _seed_papers_and_concepts(
        tmp_db,
        "consciousness in llm",
        "Debate",
        [
            (2023, 0.5),
            (2023, 0.6),
            (2024, 0.55),
            (2024, 0.65),
            (2024, 0.45),
            (2025, 0.6),
            (2025, 0.5),
            (2025, 0.7),
        ],
    )
    signals = tmp_db.detect_evolution_signals()
    matching = [s for s in signals if s["label"] == "consciousness in llm"]
    assert len(matching) == 1
    assert matching[0]["signal"] == "contested"


def test_signal_resurging(tmp_db):
    current = datetime.now().year
    _seed_papers_and_concepts(
        tmp_db,
        "symbolic ai",
        "Method",
        [
            (current - 10, 0.9),
            (current - 9, 0.88),
            (current - 8, 0.85),
            (current - 1, 0.75),
            (current, 0.80),
        ],
    )
    signals = tmp_db.detect_evolution_signals()
    matching = [s for s in signals if s["label"] == "symbolic ai"]
    assert len(matching) == 1
    assert matching[0]["signal"] == "resurging"


def test_signal_unknown(tmp_db):
    current = datetime.now().year
    _seed_papers_and_concepts(
        tmp_db,
        "obscure method",
        "Method",
        [
            (current - 2, 0.9),
        ],
    )
    signals = tmp_db.detect_evolution_signals()
    matching = [s for s in signals if s["label"] == "obscure method"]
    assert len(matching) == 1
    assert matching[0]["signal"] in ("unknown", "established")


def test_get_concept_signal(tmp_db):
    current = datetime.now().year
    _seed_papers_and_concepts(
        tmp_db,
        "transformer",
        "Method",
        [
            (current - 5, 0.95),
            (current - 4, 0.93),
            (current - 4, 0.91),
            (current - 3, 0.90),
            (current - 3, 0.88),
        ],
    )
    signal = tmp_db.get_concept_signal("transformer")
    assert signal is not None
    assert "label" in signal
    assert "signal" in signal
    assert tmp_db.get_concept_signal("nonexistent") is None


def test_get_concept_evolution(tmp_db):
    current = datetime.now().year
    _seed_papers_and_concepts(
        tmp_db,
        "diffusion model",
        "Method",
        [
            (current - 3, 0.9),
            (current - 2, 0.88),
            (current - 2, 0.91),
            (current - 1, 0.85),
            (current - 1, 0.90),
            (current - 1, 0.87),
        ],
    )
    evolution = tmp_db.get_concept_evolution("diffusion model")
    assert len(evolution) == 3
    assert evolution[0]["year"] == current - 3
    assert "trend" in evolution[0]
    last = evolution[-1]
    assert last["year"] == current - 1
    assert last["count"] == 3


# ── get_stats ─────────────────────────────────────────────────────


def test_get_stats_returns_counts(tmp_db):
    """get_stats returns zero counts for an empty database."""
    stats = tmp_db.get_stats()
    assert stats["papers"] == 0
    assert stats["concepts"] == 0
    assert stats["edges"] == 0
    assert stats["arguments"] == 0
    assert stats["aliases"] == 0
    assert stats["research_seeds"] == 0
    assert stats["queue_pending"] == 0
    assert stats["uploaded"] == 0
    assert stats["placeholders"] == 0


def test_get_stats_with_data(tmp_db):
    """get_stats returns correct counts after inserting data."""
    tmp_db.insert_paper("p1", "A", 2024, "extracted")
    tmp_db.insert_paper("p2", "B", 2024, "uploaded")
    tmp_db.insert_paper("p3", "C", 2024, "placeholder")
    tmp_db.insert_concept("p1", "Method", "X", 0.9, year=2024)
    tmp_db.insert_concept("p2", "Problem", "Y", 0.8, year=2024)
    tmp_db.insert_edge("p1", "p2", "cites", "p1")
    tmp_db.insert_argument("p1", "claim", "supports", "Y", "Method")
    tmp_db.insert_queue_item("p1", "concept", '{"label": "Z"}', 0.6)
    tmp_db.commit()

    stats = tmp_db.get_stats()
    assert stats["papers"] == 3
    assert stats["uploaded"] == 1
    assert stats["placeholders"] == 1
    assert stats["concepts"] == 2
    assert stats["edges"] == 1
    assert stats["arguments"] == 1
    assert stats["queue_pending"] == 1


def test_get_stats_with_paper_ids_filter(tmp_db):
    """get_stats filters counts when paper_ids is provided."""
    tmp_db.insert_paper("p1", "A", 2024, "uploaded")
    tmp_db.insert_paper("p2", "B", 2024, "placeholder")
    tmp_db.insert_paper("p3", "C", 2024, "extracted")
    tmp_db.insert_concept("p1", "Method", "X", 0.9, year=2024)
    tmp_db.insert_concept("p2", "Problem", "Y", 0.8, year=2024)
    tmp_db.insert_edge("p1", "p2", "cites", "p1")
    tmp_db.insert_argument("p1", "claim", "supports", "Y", "Method")
    tmp_db.commit()

    stats = tmp_db.get_stats(paper_ids=["p1"])
    assert stats["papers"] == 1
    assert stats["uploaded"] == 1
    assert stats["placeholders"] == 0
    assert stats["concepts"] == 1
    assert stats["edges"] == 1
    assert stats["arguments"] == 1

    stats_all = tmp_db.get_stats(paper_ids=["p1", "p2"])
    assert stats_all["papers"] == 2
    assert stats_all["concepts"] == 2
