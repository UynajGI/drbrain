"""PageIndex parser package — tree extraction, summarization, validation, and retrieval."""

from drbrain.parser.pageindex.builder import (  # noqa: F401
    DocumentTree,
    TreeConfig,
    _build_tree_from_nodes,
    _clean_tree_for_output,
    _extract_node_text_content,
    _extract_nodes_from_markdown,
    _find_all_children,
    _recursive_split_large_nodes,
    _split_large_text,
    _structure_to_list,
    _tree_thinning_for_index,
    _update_node_list_with_text_token_count,
    md_to_tree,
)
from drbrain.parser.pageindex.retrieval import (  # noqa: F401
    _collect_line_ranges,
    _create_clean_structure_for_description,
    _find_node_by_id,
    _format_structure,
    _write_node_id,
    get_document_structure_json,
    get_node_content,
    get_node_content_by_title,
)
from drbrain.parser.pageindex.summary import (  # noqa: F401
    _generate_doc_description,
    _generate_node_summary,
    _generate_summaries_for_structure_md,
)
from drbrain.parser.pageindex.validation import (  # noqa: F401
    _build_tree_from_outline,
    _cap_depth,
    _count_leaves,
    _extract_pdf_outline,
    _flatten_single_chains,
    _llm_segment_document,
    _split_single_leaf,
    _verify_and_correct_tree,
    _verify_tree_sample,
    md_to_tree_with_fallback,
    validate_and_fix_tree,
)
