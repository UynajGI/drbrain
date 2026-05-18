"""Tests for user behavior metrics.

TDD: tests written before implementation.
"""

from __future__ import annotations


class TestMetricsStore:
    """Test the metrics recording and querying layer."""

    def test_metrics_db_created(self, tmp_path):
        from drbrain.services.metrics_panel import _ensure_metrics_db

        db_path = tmp_path / "metrics.db"
        _ensure_metrics_db(db_path)
        assert db_path.exists()

    def test_record_search_event(self, tmp_path):
        from drbrain.services.metrics_panel import (
            _ensure_metrics_db,
            get_top_keywords,
            record_search,
        )

        db_path = tmp_path / "metrics.db"
        _ensure_metrics_db(db_path)
        record_search(db_path, "machine learning")
        record_search(db_path, "deep learning")
        record_search(db_path, "machine learning")

        keywords = get_top_keywords(db_path, limit=5)
        assert len(keywords) >= 1
        # "machine learning" should be top (count=2)
        top = keywords[0]
        assert top["keyword"] == "machine learning"
        assert top["count"] == 2

    def test_record_read_event(self, tmp_path):
        from drbrain.services.metrics_panel import (
            _ensure_metrics_db,
            get_most_read_papers,
            record_read,
        )

        db_path = tmp_path / "metrics.db"
        _ensure_metrics_db(db_path)
        record_read(db_path, "paper-001", "Attention Is All You Need")
        record_read(db_path, "paper-001", "Attention Is All You Need")
        record_read(db_path, "paper-002", "BERT")

        papers = get_most_read_papers(db_path, limit=5)
        assert papers[0]["local_id"] == "paper-001"
        assert papers[0]["count"] == 2

    def test_keyword_normalization(self, tmp_path):
        from drbrain.services.metrics_panel import (
            _ensure_metrics_db,
            get_top_keywords,
            record_search,
        )

        db_path = tmp_path / "metrics.db"
        _ensure_metrics_db(db_path)
        record_search(db_path, "  Machine LEARNING  ")
        record_search(db_path, "machine learning")

        keywords = get_top_keywords(db_path, limit=5)
        assert keywords[0]["count"] == 2

    def test_empty_metrics(self, tmp_path):
        from drbrain.services.metrics_panel import (
            _ensure_metrics_db,
            get_top_keywords,
        )

        db_path = tmp_path / "metrics.db"
        _ensure_metrics_db(db_path)

        keywords = get_top_keywords(db_path)
        assert keywords == []

    def test_weekly_trend(self, tmp_path):
        from drbrain.services.metrics_panel import (
            _ensure_metrics_db,
            get_weekly_trend,
            record_search,
        )

        db_path = tmp_path / "metrics.db"
        _ensure_metrics_db(db_path)
        record_search(db_path, "transformer")
        record_search(db_path, "GNN")

        trend = get_weekly_trend(db_path)
        assert isinstance(trend, dict)
        assert "total_searches" in trend
        assert trend["total_searches"] >= 2
