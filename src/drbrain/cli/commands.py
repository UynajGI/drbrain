"""Backward-compatible re-exports from split command modules.

All command functions and private helpers are now in separate files.
Import from the specific module in new code:

    from drbrain.cli.ingest_commands import ingest_cmd
    from drbrain.cli._common import _resolve_workspace_papers
"""

# Re-export common utilities that were module-level names in old commands.py
from drbrain.cli._common import (  # noqa: F401
    _apply_mined_rules,
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
)
from drbrain.cli.analysis_commands import (  # noqa: F401
    ask_cmd,
    descendants_cmd,
    difficulty_cmd,
    evolve_cmd,
    frontier_cmd,
    isomorphism_cmd,
    landscape_cmd,
    paradigm_cmd,
    reason_cmd,
    transfers_cmd,
)
from drbrain.cli.build_commands import (  # noqa: F401
    build_cmd,
    embed_cmd,
    translate_cmd,
)
from drbrain.cli.check_commands import (  # noqa: F401
    analyze_cmd,
    check_cmd,
    clean_cmd,
)
from drbrain.cli.export_commands import (  # noqa: F401
    backup_cmd,
    delete_cmd,
    export_cmd,
    lineage_cmd,
    queue_cmd,
    queue_resolve_all_cmd,
    queue_resolve_cmd,
)
from drbrain.cli.ingest_commands import (  # noqa: F401
    check_citations_cmd,
    citations_cmd,
    closure_cmd,
    fetch_cmd,
    ingest_cmd,
    report_cmd,
)
from drbrain.cli.query_commands import (  # noqa: F401
    index_cmd,
    list_cmd,
    query_cmd,
    seed_cmd,
    show_cmd,
    stats_cmd,
)
from drbrain.cli.repair_commands import (  # noqa: F401
    import_cmd,
    repair_cmd,
)
from drbrain.config import load_config  # noqa: F401
from drbrain.parser.mineru_parser import extract_pdf  # noqa: F401
from drbrain.query.tree_retrieval import query_by_structure_hybrid  # noqa: F401
