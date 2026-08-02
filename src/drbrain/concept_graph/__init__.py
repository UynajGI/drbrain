"""Concept co-occurrence graph layer.

A paper-level / concept-level knowledge graph built from structured metadata,
abstracts and external academic APIs (Sciverse, OpenAlex, ...), replicating the
concept-graph methodology of Marwitz et al. (2026, NMI). This layer is additive
and does not modify the existing typed KG, full-text parser or RAG retrieval.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
