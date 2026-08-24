"""Tests for resumable lean concept extraction."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from drbrain.concept_graph.concept_extract import (
    cached_concepts_for_paper,
    extract_paper_concepts_batch,
)
from drbrain.storage.database import Database


def test_empty_extraction_is_cached_as_a_terminal_outcome():
    """An empty response is recorded so an unlimited run cannot select it forever."""
    db = Database(":memory:")
    db.conn.execute(
        "INSERT INTO papers (local_id, title, abstract, year, status) VALUES (?, ?, ?, ?, ?)",
        ("p1", "A paper", "x" * 101, 2024, "uploaded"),
    )
    db.commit()

    cfg = SimpleNamespace(llm=SimpleNamespace(models=[]))
    with (
        patch("drbrain.config.load_config", return_value=cfg),
        patch(
            "drbrain.concept_graph.concept_extract._extract_one",
            new=AsyncMock(return_value=([], "")),
        ),
    ):
        result = extract_paper_concepts_batch(db, limit=1, rpm=0)

    assert result == {"processed": 0, "failed": 1, "cached_total": 1}
    assert cached_concepts_for_paper(db, "p1") == []
