"""Backward-compatibility proxy for the original _common.py.

The implementation has been split into the ``_helpers`` package:
- _helpers/db_ingest.py: DB, ingestion, merge helpers
- _helpers/enrich.py: DOI/citation enrichment
- _helpers/display.py: display, analysis, export helpers

All names are re-exported here so existing imports
``from drbrain.cli._common import X`` continue to work unchanged.
"""

from drbrain.cli._helpers import *  # noqa: F401, F403
from drbrain.cli._helpers import (  # noqa: F401
    _apply_mined_rules,
    _build_closure_context,
    _check_and_merge_duplicates,
    _enrich_doi_from_crossref,
    _enrich_doi_from_crossref_arxiv,
    _enrich_doi_from_crossref_doi,
    _enrich_doi_from_openalex,
    _enrich_tree_with_sections,
    _export_paper_to_meta,
    _extend_chain,
    _fetch_citations_interested,
    _ingest_single_paper,
    _log_error,
    _match_pattern,
    _merge_papers,
    _move_to_pending,
    _print_analyze_report,
    _render_landscape,
    _resolve_node_type,
    _resolve_workspace_papers,
    _save_paper_artifacts,
    _show_actor,
    open_db,
)
