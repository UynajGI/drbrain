"""Tests for confidence propagation along graph edges."""

import pytest

from drbrain.extractor.confidence_propagation import (
    multi_path_confidence,
    propagate_confidence,
)


def test_propagate_direct():
    """Direct propagation: source_conf * decay."""
    result = propagate_confidence(0.9, 0.85)
    assert result == pytest.approx(0.765)


def test_propagate_chain_two():
    """Two-hop chain: 0.9 * 0.85^2."""
    conf = 0.9
    for _ in range(2):
        conf = propagate_confidence(conf, 0.85)
    assert conf == pytest.approx(0.9 * 0.85 * 0.85)


def test_propagate_no_decay():
    """Decay=1.0 preserves confidence."""
    result = propagate_confidence(0.8, 1.0)
    assert result == pytest.approx(0.8)


def test_propagate_zero():
    """Zero source stays zero."""
    result = propagate_confidence(0.0, 0.85)
    assert result == pytest.approx(0.0)


def test_multi_path_single():
    """Single path: returns its confidence."""
    paths = [0.7]
    result = multi_path_confidence(paths)
    assert result == pytest.approx(0.7)


def test_multi_path_independent():
    """Two independent paths: 1 - (1-a)*(1-b)."""
    paths = [0.5, 0.5]
    result = multi_path_confidence(paths)
    # 1 - (1-0.5)*(1-0.5) = 1 - 0.25 = 0.75
    assert result == pytest.approx(0.75)


def test_multi_path_empty():
    """Empty paths returns 0.0."""
    result = multi_path_confidence([])
    assert result == pytest.approx(0.0)


def test_multi_path_redundant():
    """Three redundant paths converge higher."""
    paths = [0.6, 0.6, 0.6]
    result = multi_path_confidence(paths)
    # 1 - (0.4)^3 = 1 - 0.064 = 0.936
    assert result == pytest.approx(0.936)
