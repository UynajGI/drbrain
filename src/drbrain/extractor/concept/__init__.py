"""Concept extraction package — split from the original concept.py monolith.

All public and private names are re-exported so existing imports
``from drbrain.extractor.concept import X`` continue to work unchanged.
"""

from drbrain.extractor.argument import ExtractedArgument, parse_arguments
from drbrain.extractor.concept.dedup import (
    _label_similarity,
    dedup_concepts_by_label,
    find_similar_labels,
)
from drbrain.extractor.concept.merge import (
    _merge_concepts,
    extract_concepts_from_tree,
    extract_section_concepts,
)
from drbrain.extractor.concept.pipeline import (
    _build_ontology,
    _build_tree_hierarchy_text,
    _extract_entities,
    _extract_relations,
    _is_plateau_reached,
    _refine_extraction,
    _resolve_coreferences,
    build_graph_from_tree,
)
from drbrain.extractor.concept.tree_helpers import (
    _apply_tree_weights,
    _build_tree_edges,
    _collect_leaf_nodes,
    _is_quality_content,
    _section_type_hints,
    _tree_position_weight,
)
from drbrain.extractor.concept.types import (
    ExtractedConcepts,
    extract_concepts,
    validate_extraction,
)
from drbrain.extractor.llm_client import acall_with_fallback

__all__ = [
    "ExtractedArgument",
    "ExtractedConcepts",
    "acall_with_fallback",
    "build_graph_from_tree",
    "dedup_concepts_by_label",
    "extract_concepts",
    "extract_concepts_from_tree",
    "extract_section_concepts",
    "find_similar_labels",
    "parse_arguments",
    "validate_extraction",
    "_build_ontology",
    "_build_tree_edges",
    "_build_tree_hierarchy_text",
    "_collect_leaf_nodes",
    "_extract_entities",
    "_extract_relations",
    "_is_plateau_reached",
    "_is_quality_content",
    "_label_similarity",
    "_merge_concepts",
    "_refine_extraction",
    "_resolve_coreferences",
    "_section_type_hints",
    "_tree_position_weight",
    "_apply_tree_weights",
]
