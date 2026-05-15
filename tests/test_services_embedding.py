"""Tests for embedding service: GPU batch sizing, post_filter, model resolution."""

from __future__ import annotations

import pytest

from drbrain.services.embedding import (
    _compute_batch_size,
    _estimate_mem_per_sample,
    _post_filter,
    _resolve_model_path,
)

# ── GPU availability check ─────────────────────────────────────────────────


def _no_gpu() -> bool:
    """Return True if CUDA is unavailable (for test skipping)."""
    try:
        import torch

        return not torch.cuda.is_available()
    except ImportError:
        return True


# ── _post_filter ───────────────────────────────────────────────────────────


class TestPostFilter:
    def test_removes_low_score_results(self):
        results = [
            {"node_id": "n1", "paper_id": "p1", "score": 0.5},
            {"node_id": "n2", "paper_id": "p2", "score": 0.1},
            {"node_id": "n3", "paper_id": "p3", "score": -0.05},
            {"node_id": "n4", "paper_id": "p4", "score": -1.0},
        ]
        filtered = _post_filter(results, min_score=0.0)
        assert len(filtered) == 2
        assert all(r["score"] >= 0.0 for r in filtered)

    def test_removes_empty_node_id_when_require_text(self):
        results = [
            {"node_id": "n1", "paper_id": "p1", "score": 0.5},
            {"node_id": None, "paper_id": "p2", "score": 0.5},
            {"node_id": "", "paper_id": "p3", "score": 0.5},
        ]
        filtered = _post_filter(results, min_score=0.0, require_text=True)
        assert len(filtered) == 1
        assert filtered[0]["node_id"] == "n1"

    def test_require_text_false_keeps_empty(self):
        results = [
            {"node_id": "n1", "paper_id": "p1", "score": 0.5},
            {"node_id": "", "paper_id": "p3", "score": 0.5},
        ]
        filtered = _post_filter(results, min_score=0.0, require_text=False)
        assert len(filtered) == 2

    def test_default_min_score_zero(self):
        results = [
            {"node_id": "n1", "paper_id": "p1", "score": 0.0},
            {"node_id": "n2", "paper_id": "p2", "score": -0.01},
        ]
        filtered = _post_filter(results)
        assert len(filtered) == 1
        assert filtered[0]["node_id"] == "n1"

    def test_higher_min_score_threshold(self):
        results = [
            {"node_id": "n1", "paper_id": "p1", "score": 0.5},
            {"node_id": "n2", "paper_id": "p2", "score": 0.3},
            {"node_id": "n3", "paper_id": "p3", "score": 0.7},
        ]
        filtered = _post_filter(results, min_score=0.4)
        assert len(filtered) == 2
        assert all(r["score"] >= 0.4 for r in filtered)

    def test_empty_input(self):
        assert _post_filter([]) == []
        assert _post_filter([], min_score=0.5) == []

    def test_missing_score_field(self):
        """Results without a score field default to 0.0 for threshold comparison."""
        results = [
            {"node_id": "n1", "paper_id": "p1"},
            {"node_id": "n2", "paper_id": "p2", "score": 0.5},
        ]
        # Missing score treated as 0.0 by get() default
        filtered = _post_filter(results, min_score=0.0)
        assert len(filtered) == 2
        # With higher threshold, missing score (0.0) is filtered out
        filtered = _post_filter(results, min_score=0.1)
        assert len(filtered) == 1


# ── _estimate_mem_per_sample ───────────────────────────────────────────────


class TestEstimateMemPerSample:
    def test_linear_interpolation(self):
        profile = {"per_sample": {"64": 100, "128": 200}}
        # tokens=96: 100 + (96-64)/(128-64) * (200-100) = 100 + 0.5 * 100 = 150
        mem = _estimate_mem_per_sample(96, profile)
        assert mem == 150

    def test_below_range_returns_first_point(self):
        profile = {"per_sample": {"64": 100, "128": 200}}
        mem = _estimate_mem_per_sample(32, profile)
        assert mem == 100

    def test_exact_match_returns_correct_value(self):
        profile = {"per_sample": {"64": 100, "128": 200, "256": 400}}
        mem = _estimate_mem_per_sample(128, profile)
        assert mem == 200

    def test_quadratic_extrapolation_above_range(self):
        profile = {"per_sample": {"64": 100, "128": 200}}
        # tokens=256: ratio=2, 200 * 2 * 2 = 800
        mem = _estimate_mem_per_sample(256, profile)
        assert mem == 800

    def test_empty_profile_returns_zero(self):
        assert _estimate_mem_per_sample(128, {}) == 0
        assert _estimate_mem_per_sample(128, {"per_sample": {}}) == 0

    def test_single_point_profile(self):
        profile = {"per_sample": {"128": 500}}
        # Below -> first point
        assert _estimate_mem_per_sample(64, profile) == 500
        # Match
        assert _estimate_mem_per_sample(128, profile) == 500
        # Above -> quadratic extrapolation: ratio=2, 500*4=2000
        assert _estimate_mem_per_sample(256, profile) == 2000


# ── _compute_batch_size ────────────────────────────────────────────────────


class TestComputeBatchSize:
    def test_basic_computation(self):
        profile = {
            "gpu_total_bytes": 8 * 1024**3,  # 8 GB
            "baseline_bytes": 2 * 1024**3,  # 2 GB model weights
            "per_sample": {"128": 100 * 1024**2},  # 100 MB per sample
        }
        bs = _compute_batch_size(128, profile, safety_factor=0.85)
        # available = 8*0.85 - 2 = 4.8 GB
        # bs = 4.8 GB / 100 MB = 49.152... -> 49
        expected = int((8 * 1024**3 * 0.85 - 2 * 1024**3) / (100 * 1024**2))
        assert bs == expected
        assert bs > 0

    def test_empty_profile_returns_default_8(self):
        assert _compute_batch_size(128, {}) == 8
        assert _compute_batch_size(128, {"per_sample": {}}) == 8

    def test_zero_mem_per_sample_returns_default_8(self):
        profile = {
            "gpu_total_bytes": 8 * 1024**3,
            "baseline_bytes": 0,
            "per_sample": {"128": 0},
        }
        assert _compute_batch_size(128, profile) == 8

    def test_no_available_memory_returns_1(self):
        profile = {
            "gpu_total_bytes": 1024,
            "baseline_bytes": 2000,
            "per_sample": {"128": 100},
        }
        assert _compute_batch_size(128, profile) == 1

    def test_capped_at_128(self):
        profile = {
            "gpu_total_bytes": 1000 * 1024**3,  # huge
            "baseline_bytes": 0,
            "per_sample": {"128": 1},  # tiny
        }
        bs = _compute_batch_size(128, profile)
        assert bs == 128

    def test_minimum_batch_size_is_1(self):
        profile = {
            "gpu_total_bytes": 10 * 1024,
            "baseline_bytes": 0,
            "per_sample": {"128": 10 * 1024},
        }
        bs = _compute_batch_size(128, profile, safety_factor=1.0)
        assert bs == 1


# ── _resolve_model_path ────────────────────────────────────────────────────


class TestResolveModelPath:
    def test_returns_none_for_huggingface_source(self):
        path = _resolve_model_path(
            "Qwen/Qwen3-Embedding-0.6B",
            cache_dir="/tmp/fake_models",
            source="huggingface",
        )
        assert path is None

    def test_returns_none_for_nonexistent_model(self):
        path = _resolve_model_path(
            "nonexistent_org/nonexistent_model_99999",
            cache_dir="/tmp/fake_models_nonexistent",
            source="modelscope",
        )
        # When modelscope is not installed, returns None on ImportError.
        # When modelscope IS installed but model doesn't exist, raises or returns None.
        assert path is None


# ── GPU profiling (skip if no GPU) ─────────────────────────────────────────


@pytest.mark.skipif(_no_gpu(), reason="No GPU available")
class TestGpuProfiling:
    def test_estimate_mem_per_sample_from_real_like_profile(self):
        """Verify _estimate_mem_per_sample works with realistic profile dict."""
        profile = {
            "gpu_total_bytes": 8 * 1024**3,
            "baseline_bytes": 2 * 1024**3,
            "gpu_name": "Test GPU",
            "model_name": "test-model",
            "per_sample": {
                "64": 50 * 1024**2,
                "128": 100 * 1024**2,
                "256": 200 * 1024**2,
            },
            "profiled_at": "2025-01-01T00:00:00",
        }
        mem = _estimate_mem_per_sample(96, profile)
        assert mem > 0
        # Between 50 MB (64 tok) and 100 MB (128 tok), closer to 75 MB
        assert 60 * 1024**2 < mem < 90 * 1024**2
