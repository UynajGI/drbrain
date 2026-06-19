"""Concept deduplication and label similarity."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def dedup_concepts_by_label(db) -> int:
    """Merge concepts with identical labels (case-insensitive) across papers.
    Keeps the highest confidence entry, updates edges to point to it.
    Returns number of merged pairs.
    """
    # Find exact label matches
    rows = db.conn.execute("""
        SELECT LOWER(label) as norm_label, type, COUNT(*) as cnt
        FROM concepts
        GROUP BY LOWER(label), type
        HAVING cnt > 1
    """).fetchall()

    merged = 0
    for norm_label, ctype, count in rows:
        # Get all entries for this label
        entries = db.conn.execute(
            "SELECT concept_id, label, confidence, local_id "
            "FROM concepts WHERE LOWER(label) = ? AND type = ? "
            "ORDER BY confidence DESC",
            (norm_label, ctype),
        ).fetchall()

        if len(entries) < 2:
            continue

        canonical = entries[0]  # highest confidence
        canonical_label = canonical[1]

        for dup in entries[1:]:
            dup_id = dup[0]
            dup_label = dup[1]
            # Redirect edges from the duplicate label to the canonical one,
            # then delete the duplicate concept. Both go through database.py.
            db.redirect_edge_endpoint(dup_label, canonical_label)
            db.delete_concept(dup_id)
            merged += 1

    db.commit()
    return merged


def find_similar_labels(db, threshold: float = 0.6) -> list[tuple[str, str, float]]:
    """Find pairs of concept labels that are similar but not identical.
    Uses word overlap ratio. Returns list of (label_a, label_b, similarity).
    """
    rows = db.conn.execute(
        "SELECT DISTINCT label, type FROM concepts ORDER BY type, label"
    ).fetchall()

    pairs = []
    # Group by type
    by_type: dict[str, list[str]] = {}
    for label, ctype in rows:
        by_type.setdefault(ctype, []).append(label)

    for ctype, labels in by_type.items():
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                sim = _label_similarity(labels[i], labels[j])
                if sim >= threshold and sim < 1.0:
                    pairs.append((labels[i], labels[j], round(sim, 3)))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def _label_similarity(a: str, b: str) -> float:
    """Jaccard similarity between two label word sets."""
    import re

    a_words = set(re.split(r"[\s\-_]+", a.strip().lower()))
    b_words = set(re.split(r"[\s\-_]+", b.strip().lower()))
    if not a_words or not b_words:
        return 0.0
    union = a_words | b_words
    overlap = len(a_words & b_words)
    return overlap / len(union)
