"""Tests for confidence propagation along graph edges."""

import pytest

from drbrain.extractor.confidence_propagation import (
    multi_path_confidence,
    propagate_confidence,
    propagate_confidence_with_section,
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


# -- Section-aware propagation --


def test_section_aware_methods_preserves_more():
    """Methods section has higher decay (preserves more confidence)."""
    result_methods = propagate_confidence_with_section(0.9, "Methods")
    result_discussion = propagate_confidence_with_section(0.9, "Discussion")
    assert result_methods > result_discussion


def test_section_aware_results_high_decay():
    """Results section has grounded evidence (decay=0.90)."""
    result = propagate_confidence_with_section(0.9, "Results")
    assert result == pytest.approx(0.9 * 0.90)


def test_section_aware_related_work_low_decay():
    """Related Work section has speculative content (decay=0.80)."""
    result = propagate_confidence_with_section(0.9, "Related Work")
    assert result == pytest.approx(0.9 * 0.80)


def test_section_aware_unknown_uses_base():
    """Unknown section falls back to base_decay."""
    result = propagate_confidence_with_section(0.9, "Appendix", base_decay=0.85)
    assert result == pytest.approx(0.9 * 0.85)


def test_section_aware_case_insensitive():
    """Section matching is case-insensitive."""
    assert propagate_confidence_with_section(0.9, "methods") == pytest.approx(
        propagate_confidence_with_section(0.9, "Methods")
    )
