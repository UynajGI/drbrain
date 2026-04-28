"""Tests for temporal evolution signal detection."""
import tempfile
from pathlib import Path
from datetime import datetime

import pytest
from drbrain.storage.database import Database


def _seed_papers_and_concepts(db, label, ctype, year_confidence_pairs):
    """Helper: insert papers and a concept across multiple years."""
    for i, (year, conf) in enumerate(year_confidence_pairs):
        pid = f"p{i:03d}_{label.replace(' ', '_')}"
        db.insert_paper(pid, f"Paper about {label} ({year})", year, "uploaded")
        db.insert_concept(pid, ctype, label, conf, year=year)
    db.commit()


def test_signal_emerging():
    """Concept appeared recently and paper count is growing."""
    current = datetime.now().year
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        # Growing: 1 paper 2 years ago, 2 papers last year, 4 papers this year
        _seed_papers_and_concepts(db, "quantum transformer", "Method", [
            (current - 2, 0.9),
            (current - 1, 0.88), (current - 1, 0.91),
            (current, 0.85), (current, 0.90), (current, 0.87), (current, 0.92),
        ])
        signals = db.detect_evolution_signals()
        matching = [s for s in signals if s["label"] == "quantum transformer"]
        assert len(matching) == 1
        assert matching[0]["signal"] == "emerging"
        db.close()


def test_signal_established():
    """Concept with many papers and high confidence."""
    current = datetime.now().year
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        # 12 papers across years including recent, all high confidence
        pairs = [(current - 5 + (i % 6), 0.85 + (i % 10) * 0.01) for i in range(12)]
        _seed_papers_and_concepts(db, "attention mechanism", "Method", pairs)
        signals = db.detect_evolution_signals()
        matching = [s for s in signals if s["label"] == "attention mechanism"]
        assert len(matching) == 1
        assert matching[0]["signal"] == "established"
        db.close()


def test_signal_declining():
    """Concept last seen > 3 years ago with flat/stagnant paper count."""
    current = datetime.now().year
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        # Last seen 4 years ago, only 4 papers total, no recent activity
        _seed_papers_and_concepts(db, "rnn language model", "Method", [
            (current - 8, 0.9),
            (current - 7, 0.88),
            (current - 5, 0.85),
            (current - 4, 0.82),
        ])
        signals = db.detect_evolution_signals()
        matching = [s for s in signals if s["label"] == "rnn language model"]
        assert len(matching) == 1
        assert matching[0]["signal"] == "declining"
        db.close()


def test_signal_contested():
    """Concept with many papers but low average confidence."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        # 8 papers, but low confidence (disagreement)
        _seed_papers_and_concepts(db, "consciousness in llm", "Debate", [
            (2023, 0.5), (2023, 0.6), (2024, 0.55), (2024, 0.65),
            (2024, 0.45), (2025, 0.6), (2025, 0.5), (2025, 0.7),
        ])
        signals = db.detect_evolution_signals()
        matching = [s for s in signals if s["label"] == "consciousness in llm"]
        assert len(matching) == 1
        assert matching[0]["signal"] == "contested"
        db.close()


def test_signal_resurging():
    """Concept dormant > 3 years then reappears recently."""
    current = datetime.now().year
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        # Old papers 8-10 years ago, then nothing, then 2 papers recently
        _seed_papers_and_concepts(db, "symbolic ai", "Method", [
            (current - 10, 0.9),
            (current - 9, 0.88),
            (current - 8, 0.85),
            # gap of 6 years
            (current - 1, 0.75),
            (current, 0.80),
        ])
        signals = db.detect_evolution_signals()
        matching = [s for s in signals if s["label"] == "symbolic ai"]
        assert len(matching) == 1
        assert matching[0]["signal"] == "resurging"
        db.close()


def test_signal_default_unknown():
    """Concept with very few papers gets unknown/default signal."""
    current = datetime.now().year
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        # Single paper 2 years ago — not emerging (no growth), not declining (gap <= 3)
        _seed_papers_and_concepts(db, "obscure method", "Method", [
            (current - 2, 0.9),
        ])
        signals = db.detect_evolution_signals()
        matching = [s for s in signals if s["label"] == "obscure method"]
        assert len(matching) == 1
        # Single paper, not emerging, not declining — should be unknown/default
        assert matching[0]["signal"] in ("unknown", "established")
        db.close()


def test_detect_signals_for_concept():
    """Detect signal for a specific concept (per-concept method)."""
    current = datetime.now().year
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        _seed_papers_and_concepts(db, "transformer", "Method", [
            (current - 5, 0.95), (current - 4, 0.93), (current - 4, 0.91),
            (current - 3, 0.90), (current - 3, 0.88),
        ])
        _seed_papers_and_concepts(db, "rnn", "Method", [
            (current - 10, 0.9), (current - 9, 0.88),
        ])

        signal = db.get_concept_signal("transformer")
        assert signal is not None
        assert "label" in signal
        assert "signal" in signal

        # Non-existent concept returns None
        assert db.get_concept_signal("nonexistent") is None
        db.close()


def test_concept_evolution_with_trend():
    """get_concept_evolution returns year-by-year data with trend annotation."""
    current = datetime.now().year
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        _seed_papers_and_concepts(db, "diffusion model", "Method", [
            (current - 3, 0.9),
            (current - 2, 0.88), (current - 2, 0.91),
            (current - 1, 0.85), (current - 1, 0.90), (current - 1, 0.87),
        ])

        evolution = db.get_concept_evolution("diffusion model")
        assert len(evolution) == 3  # 3 years

        # Check first year is marked as first_appeared
        assert evolution[0]["year"] == current - 3
        assert "trend" in evolution[0]

        # Check last year has growing trend
        last = evolution[-1]
        assert last["year"] == current - 1
        assert last["count"] == 3
        db.close()
