"""Genealogy package — split from the original genealogy.py monolith.

All public names re-exported so existing imports work unchanged.
"""

from drbrain.graph.genealogy.display import (
    _mermaid_nodes,
    _to_mermaid,
    _to_text_tree,
    format_tree,
)
from drbrain.graph.genealogy.landscape import (
    _get_concepts_by_type,
    landscape_workspace,
)
from drbrain.graph.genealogy.lineage import (
    _bfs_ancestors,
    _bfs_descendants,
    _collect_reachable_labels,
    _preload_concept_info,
    _reroot_with_ancestors,
    evolve_concept,
    trace_descendants,
)
from drbrain.graph.genealogy.paradigm import (
    _format_provenance,
    _get_concept_provenance,
    analyze_difficulty,
    analyze_frontier,
    detect_paradigm_shifts,
)
from drbrain.graph.genealogy.transfer import (
    _cluster_by_similarity,
    _score_transfer_pairs,
    find_transfer_history,
    find_transfer_opportunities,
    find_transfer_opportunities_auto,
)

__all__ = [
    "analyze_difficulty",
    "analyze_frontier",
    "detect_paradigm_shifts",
    "evolve_concept",
    "find_transfer_history",
    "find_transfer_opportunities",
    "find_transfer_opportunities_auto",
    "format_tree",
    "landscape_workspace",
    "trace_descendants",
    "_bfs_ancestors",
    "_bfs_descendants",
    "_cluster_by_similarity",
    "_collect_reachable_labels",
    "_format_provenance",
    "_get_concept_provenance",
    "_get_concepts_by_type",
    "_mermaid_nodes",
    "_preload_concept_info",
    "_reroot_with_ancestors",
    "_score_transfer_pairs",
    "_to_mermaid",
    "_to_text_tree",
]
