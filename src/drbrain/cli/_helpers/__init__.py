"""CLI helper package — split from the original _common.py monolith.

All names re-exported so existing imports
``from drbrain.cli._common import X`` continue to work.
"""

from drbrain.cli._helpers.db_ingest import (
    _check_and_merge_duplicates,
    _fetch_citations_interested,
    _ingest_single_paper,
    _log_error,
    _merge_papers,
    _move_to_pending,
    _resolve_node_type,
    _resolve_workspace_papers,
    _save_paper_artifacts,
    open_db,
)
from drbrain.cli._helpers.display import (
    _apply_mined_rules,
    _build_closure_context,
    _enrich_tree_with_sections,
    _export_paper_to_meta,
    _extend_chain,
    _match_pattern,
    _print_analyze_report,
    _render_landscape,
    _show_actor,
)
from drbrain.cli._helpers.enrich import (
    _enrich_doi_from_crossref,
    _enrich_doi_from_crossref_arxiv,
    _enrich_doi_from_crossref_doi,
    _enrich_doi_from_openalex,
)

__all__ = [
    "_apply_mined_rules",
    "_build_closure_context",
    "_check_and_merge_duplicates",
    "_enrich_doi_from_crossref",
    "_enrich_doi_from_crossref_arxiv",
    "_enrich_doi_from_crossref_doi",
    "_enrich_doi_from_openalex",
    "_enrich_tree_with_sections",
    "_export_paper_to_meta",
    "_extend_chain",
    "_fetch_citations_interested",
    "_ingest_single_paper",
    "_log_error",
    "_match_pattern",
    "_merge_papers",
    "_move_to_pending",
    "_print_analyze_report",
    "_render_landscape",
    "_resolve_node_type",
    "_resolve_workspace_papers",
    "_save_paper_artifacts",
    "_show_actor",
    "open_db",
]
