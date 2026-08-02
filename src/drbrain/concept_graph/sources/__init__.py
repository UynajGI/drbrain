"""Multi-source corpus adapters for the concept graph layer.

Each adapter implements the :class:`~drbrain.concept_graph.sources.base.CorpusSource`
protocol so corpus ingestion is source-agnostic. Sciverse is implemented first;
OpenAlex wraps the existing client; CrossRef / Semantic Scholar / arXiv are reserved.
"""

from __future__ import annotations

from drbrain.concept_graph.sources.base import (
    Author,
    CorpusSource,
    PaperRecord,
    PaperRelations,
    is_success_envelope,
)
from drbrain.concept_graph.sources.registry import get_source, register_source

__all__ = [
    "Author",
    "CorpusSource",
    "PaperRecord",
    "PaperRelations",
    "is_success_envelope",
    "get_source",
    "register_source",
]
