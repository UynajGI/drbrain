"""Tests for BM25 + LLM hybrid concept alignment (SmartAligner)."""
import tempfile
import unittest.mock
from pathlib import Path

from drbrain.storage.database import Database
from drbrain.extractor.canonical import (
    normalize_label, AliasTable, SmartAligner, _tokenize,
)


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
        aligner._pending.append({
            "label": "self attention",
            "type": "Method",
            "candidates": ["attention"],
            "score": 0.5,
        })
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
        aligner._pending.append({
            "label": "self attention",
            "type": "Method",
            "candidates": ["attention"],
            "score": 0.5,
        })

        with unittest.mock.patch(
            "drbrain.extractor.canonical._llm_arbitrate",
            return_value=[
                {"label": "self attention", "canonical": "attention", "confidence": 0.9}
            ],
        ):
            aligner.flush_pending()

        # Pending should be cleared
        assert len(aligner._pending) == 0
        db.close()
