"""Confidence propagation along graph paths.

Implements uncertainty decay for multi-hop inferences:
- Each hop multiplies confidence by decay factor (default 0.85).
- Multiple independent paths merge via probabilistic OR:
  P = 1 - prod(1 - p_i) for each path confidence p_i.
"""

from __future__ import annotations

DEFAULT_DECAY = 0.85


def propagate_confidence(confidence: float, decay: float = DEFAULT_DECAY) -> float:
    """Apply one hop of confidence decay.

    Args:
        confidence: Confidence before this hop (0.0 to 1.0).
        decay: Multiplicative decay factor per hop.

    Returns:
        Confidence after one hop.
    """
    return confidence * decay


def multi_path_confidence(path_confidences: list[float]) -> float:
    """Merge multiple independent path confidences.

    Uses probabilistic OR: P = 1 - prod(1 - p_i).
    This models that if multiple independent paths support a conclusion,
    the combined confidence is higher than any single path.

    Args:
        path_confidences: List of confidence values from different paths.

    Returns:
        Combined confidence (0.0 to 1.0).
    """
    if not path_confidences:
        return 0.0

    product = 1.0
    for p in path_confidences:
        product *= 1.0 - p
    return 1.0 - product
