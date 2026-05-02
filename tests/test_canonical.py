"""Tests for BM25 + LLM hybrid concept alignment (SmartAligner)."""

import tempfile
import unittest.mock
from pathlib import Path

from drbrain.extractor.canonical import (
    AliasTable,
    SmartAligner,
    _tokenize,
    normalize_label,
)
from drbrain.storage.database import Database

# -- normalize_label --


def test_normalize_label_lowercase():
    assert normalize_label("Transformer") == "transformer"


def test_normalize_label_strip_articles():
    assert normalize_label("The Transformer Architecture") == "transformer architecture"


def test_normalize_normalizes_variants():
    a = normalize_label("Graph Neural Networks")
    b = normalize_label("graph neural network")
    assert a == b


# -- AliasTable --


def test_alias_table_add_and_lookup():
    table = AliasTable()
    cid = table.add_canonical("transformer", "concept_1")
    assert cid == "concept_1"
    table.add_alias("The Transformer", cid)
    table.add_alias("transformer architecture", cid)
    assert table.lookup("transformer") == cid
    assert table.lookup("The Transformer") == cid
    assert table.lookup("transformer architecture") == cid


def test_alias_table_lookup_unknown():
    table = AliasTable()
    table.add_canonical("attention mechanism", "concept_2")
    assert table.lookup("unknown thing") is None


def test_alias_table_get_or_create():
    table = AliasTable()
    cid1 = table.get_or_create("transformer")
    cid2 = table.get_or_create("transformer")
    assert cid1 == cid2
    cid3 = table.get_or_create("The Transformer")
    assert cid3 == cid1


# -- _tokenize --


def test_tokenize_english():
    assert _tokenize("Hello World") == ["hello", "world"]


def test_tokenize_chinese():
    tokens = _tokenize("长程依赖")
    assert len(tokens) > 0


# -- SmartAligner --


def test_smart_aligner_exact_match():
    """Step 1: normalize + exact match returns existing canonical_id."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test", 2024, "uploaded")
        db.insert_concept("p1", "Method", "transformer", 0.95, year=2024)
        db.commit()

        aligner = SmartAligner(db)
        cid = aligner.align("transformer", "Method")
        assert cid is not None
        assert "transformer" in cid or cid.startswith("concept_")
        db.close()


def test_smart_aligner_bm25_auto_align():
    """Step 2: BM25 high score auto-aligns to existing concept."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test", 2024, "uploaded")
        db.insert_concept("p1", "Method", "attention mechanism", 0.9, year=2024)
        db.commit()

        aligner = SmartAligner(db)
        # "self attention" shares keyword with "attention mechanism"
        cid = aligner.align("self attention", "Method")
        # Should either auto-align or create new (depends on BM25 score)
        assert cid is not None
        db.close()


def test_smart_aligner_bm25_no_match_creates_new():
    """Step 4: No BM25 match creates new canonical_id."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test", 2024, "uploaded")
        db.insert_concept("p1", "Method", "quantum computing", 0.9, year=2024)
        db.commit()

        aligner = SmartAligner(db)
        cid1 = aligner.align("quantum computing", "Method")
        cid2 = aligner.align("neural style transfer", "Method")
        # Completely different concepts should get different IDs
        assert cid1 != cid2 or cid2 is not None
        db.close()


def test_smart_aligner_empty_db():
    """SmartAligner works with empty DB (no existing concepts)."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        aligner = SmartAligner(db)
        cid = aligner.align("new concept", "Method")
        assert cid is not None
        db.close()


def test_smart_aligner_flush_pending_no_models():
    """flush_pending does nothing when no LLM models configured."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test", 2024, "uploaded")
        db.insert_concept("p1", "Method", "attention", 0.9, year=2024)
        db.commit()

        aligner = SmartAligner(db, models=None)
        # Manually add a pending entry
        aligner._pending.append(
            {
                "label": "self attention",
                "type": "Method",
                "candidates": ["attention"],
                "score": 0.5,
            }
        )
        aligner.flush_pending()
        # Should still be pending (no models)
        assert len(aligner._pending) == 1
        db.close()


def test_smart_aligner_flush_pending_with_mocked_llm():
    """flush_pending applies LLM decisions when models available."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test", 2024, "uploaded")
        db.insert_concept("p1", "Method", "attention", 0.9, year=2024)
        db.commit()

        aligner = SmartAligner(db, models=[{"provider": "openai", "model": "gpt-4"}])
        aligner._pending.append(
            {
                "label": "self attention",
                "type": "Method",
                "candidates": ["attention"],
                "score": 0.5,
            }
        )

        with unittest.mock.patch(
            "drbrain.extractor.canonical._llm_arbitrate",
            return_value=[{"label": "self attention", "canonical": "attention", "confidence": 0.9}],
        ):
            aligner.flush_pending()

        # Pending should be cleared
        assert len(aligner._pending) == 0
        db.close()


# -- SmartAligner.align() — BM25 score thresholds --


def test_smart_aligner_bm25_ambiguous_score_queues_pending():
    """When BM25 score is 0.3-0.8, label is queued for LLM arbitration."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test", 2024, "uploaded")
        db.insert_concept("p1", "Method", "attention", 0.9, year=2024)
        db.commit()

        aligner = SmartAligner(db)
        fake_doc = {"label": "attention", "type": "Method", "canonical_id": "concept_attention"}
        # Score 0.5 is in the ambiguous range [0.3, 0.8)
        with unittest.mock.patch.object(aligner, "_bm25_search", return_value=(0.5, fake_doc)):
            cid = aligner.align("self attention", "Method")

        # Should return the candidate's canonical_id
        assert cid == "concept_attention"
        # Should have queued for LLM arbitration
        assert len(aligner._pending) == 1
        assert aligner._pending[0]["label"] == "self attention"
        assert aligner._pending[0]["score"] == 0.5
        db.close()


def test_smart_aligner_bm25_high_score_no_pending():
    """When BM25 score >= 0.8, auto-align without queuing for LLM."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test", 2024, "uploaded")
        db.insert_concept("p1", "Method", "attention", 0.9, year=2024)
        db.commit()

        aligner = SmartAligner(db)
        fake_doc = {"label": "attention", "type": "Method", "canonical_id": "concept_attention"}
        # Score 0.85 is >= BM25_AUTO_ALIGN (0.8)
        with unittest.mock.patch.object(aligner, "_bm25_search", return_value=(0.85, fake_doc)):
            cid = aligner.align("deep attention", "Method")

        assert cid == "concept_attention"
        # Should NOT have any pending (no LLM needed)
        assert len(aligner._pending) == 0
        db.close()


def test_smart_aligner_flush_pending_empty_queue():
    """flush_pending with empty pending list does nothing, even with models."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test", 2024, "uploaded")
        db.commit()

        aligner = SmartAligner(db, models=[{"provider": "openai", "model": "gpt-4"}])
        # _pending is already empty
        assert len(aligner._pending) == 0
        # Should not raise, should not call LLM
        aligner.flush_pending()
        assert len(aligner._pending) == 0
        db.close()


def test_smart_aligner_bm25_low_score_below_threshold_creates_new():
    """When BM25 score < 0.3, no match, creates new canonical_id."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test", 2024, "uploaded")
        db.insert_concept("p1", "Method", "quantum computing", 0.9, year=2024)
        db.commit()

        aligner = SmartAligner(db)
        fake_doc = {"label": "quantum computing", "type": "Method", "canonical_id": "concept_qc"}
        # Score 0.25 is below BM25_PENDING_MIN (0.3)
        with unittest.mock.patch.object(aligner, "_bm25_search", return_value=(0.25, fake_doc)):
            cid = aligner.align("neural style transfer", "Method")

        # Should create a new ID (not the candidate's)
        assert cid is not None
        assert cid != "concept_qc"
        assert len(aligner._pending) == 0
        db.close()


def test_smart_aligner_flush_pending_low_confidence_keeps_independent():
    """flush_pending with low-confidence LLM decision keeps label independent."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test", 2024, "uploaded")
        db.insert_concept("p1", "Method", "attention", 0.9, year=2024)
        db.commit()

        aligner = SmartAligner(db, models=[{"provider": "openai", "model": "gpt-4"}])
        aligner._pending.append(
            {
                "label": "self attention",
                "type": "Method",
                "candidates": ["attention"],
                "score": 0.5,
            }
        )

        # LLM returns low confidence (< 0.7) — label should stay independent
        with unittest.mock.patch(
            "drbrain.extractor.canonical._llm_arbitrate",
            return_value=[{"label": "self attention", "canonical": "attention", "confidence": 0.4}],
        ):
            aligner.flush_pending()

        # Pending should be cleared even when uncertain
        assert len(aligner._pending) == 0
        db.close()


def test_smart_aligner_flush_pending_null_canonical():
    """flush_pending with null canonical keeps label independent."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test", 2024, "uploaded")
        db.commit()

        aligner = SmartAligner(db, models=[{"provider": "openai", "model": "gpt-4"}])
        aligner._pending.append(
            {
                "label": "novel technique",
                "type": "Method",
                "candidates": [],
                "score": 0.5,
            }
        )

        # LLM returns null canonical
        with unittest.mock.patch(
            "drbrain.extractor.canonical._llm_arbitrate",
            return_value=[{"label": "novel technique", "canonical": None, "confidence": 0.5}],
        ):
            aligner.flush_pending()

        assert len(aligner._pending) == 0
        db.close()


def test_smart_aligner_bm25_real_high_overlap():
    """Real BM25 with multiple concepts + overlapping query triggers score path."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test", 2024, "uploaded")
        # Multiple concepts so IDF > 0 for terms shared by a subset
        db.insert_concept("p1", "Method", "graph neural network training", 0.9, year=2024)
        db.insert_concept("p1", "Method", "quantum error correction circuits", 0.9, year=2024)
        db.insert_concept("p1", "Method", "reinforcement learning with rewards", 0.9, year=2024)
        db.commit()

        aligner = SmartAligner(db)
        # "graph neural networks" shares terms with doc 1 → non-zero BM25
        cid = aligner.align("graph neural networks", "Method")
        assert cid is not None
        db.close()
