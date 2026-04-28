"""Tests for concept label canonicalization and alias matching."""
from drbrain.extractor.canonical import normalize_label, AliasTable

def test_normalize_label_lowercase():
    assert normalize_label("Transformer") == "transformer"

def test_normalize_label_strip_articles():
    assert normalize_label("The Transformer Architecture") == "transformer architecture"

def test_normalize_normalizes_variants():
    a = normalize_label("Graph Neural Networks")
    b = normalize_label("graph neural network")
    assert a == b

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
