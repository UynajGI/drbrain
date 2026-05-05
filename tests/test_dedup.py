"""Tests for cross-paper concept deduplication."""


def _setup_db_with_duplicates(db):
    """Insert papers + concepts with known duplicates."""
    db.insert_paper("p1", "Paper A", 2023, "uploaded")
    db.insert_paper("p2", "Paper B", 2023, "uploaded")
    # Same concept, slightly different labels
    db.insert_concept("p1", "Method", "transformer architecture", 0.9, year=2023)
    db.insert_concept("p2", "Method", "Transformer model architecture", 0.85, year=2023)
    # Different concept
    db.insert_concept("p1", "Method", "gradient descent", 0.9, year=2023)
    db.insert_concept("p2", "Problem", "vanishing gradients", 0.8, year=2023)
    # Same concept, identical label (exact match)
    db.insert_concept("p1", "Method", "attention mechanism", 0.9, year=2023)
    db.insert_concept("p2", "Method", "attention mechanism", 0.85, year=2023)
    db.commit()


def test_exact_label_dedup(tmp_db):
    """Concepts with identical labels across papers are deduplicated."""
    _setup_db_with_duplicates(tmp_db)

    from drbrain.extractor.concept import dedup_concepts_by_label

    merged = dedup_concepts_by_label(tmp_db)
    # Exact match: attention mechanism should be merged
    assert merged >= 1  # at least the exact match pair
    count = tmp_db.conn.execute(
        "SELECT COUNT(*) FROM concepts WHERE label = 'attention mechanism'"
    ).fetchone()[0]
    assert count == 1  # deduped to single entry


def test_fuzzy_label_dedup(tmp_db):
    """Similar labels across papers are identified for LLM review."""
    _setup_db_with_duplicates(tmp_db)

    from drbrain.extractor.concept import find_similar_labels

    pairs = find_similar_labels(tmp_db, threshold=0.6)
    # "transformer architecture" vs "Transformer model architecture" should match
    assert len(pairs) > 0
    labels = {(a, b) for a, b, _ in pairs}
    assert any("transformer" in a.lower() or "transformer" in b.lower() for a, b in labels)


def test_dedup_only_same_type(tmp_db):
    """Dedup only merges concepts of the same type."""
    _setup_db_with_duplicates(tmp_db)

    from drbrain.extractor.concept import find_similar_labels

    pairs = find_similar_labels(tmp_db, threshold=0.6)
    # "gradient descent" (Method) should NOT match "vanishing gradients" (Problem)
    for a, b, score in pairs:
        assert not ("gradient" in a.lower() and "vanishing" in b.lower()), (
            f"Should not match across types: {a} <-> {b}"
        )
