"""Tests for paper type detection."""

import pytest

from drbrain.extractor.detection import PAPER_TYPES, detect_paper_type


def test_detect_paper_type_returns_valid_type():
    """Detection returns one of the valid paper_type values."""
    result = detect_paper_type(
        title="A Novel Approach to Graph Neural Networks",
        abstract="We propose a new method for training GNNs on large-scale graphs.",
    )
    assert result in PAPER_TYPES


def test_paper_types_enum():
    """PAPER_TYPES contains expected values."""
    assert PAPER_TYPES == {"paper", "review", "thesis", "preprint", "book", "document"}


@pytest.mark.asyncio
async def test_detect_paper_type_async():
    """Async version returns valid type when models are provided."""
    from drbrain.extractor.detection import detect_paper_type_async

    result = await detect_paper_type_async(
        title="Survey of Deep Learning Methods",
        abstract="This survey covers recent advances in deep learning.",
    )
    assert result in PAPER_TYPES


def test_detect_from_first_page():
    """Detection from first page content works."""
    text = "Submitted to arXiv. We present a comprehensive review of NLP methods."
    result = detect_paper_type(title="NLP Review", abstract=None, first_page=text)
    assert result in PAPER_TYPES
