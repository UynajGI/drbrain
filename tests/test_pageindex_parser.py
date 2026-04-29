"""Tests for PageIndex markdown tree extraction."""

from drbrain.parser.pageindex_parser import (
    DocumentTree,
    TreeConfig,
    _build_tree_from_nodes,
    _extract_node_text_content,
    _extract_nodes_from_markdown,
    _recursive_split_large_nodes,
    _split_large_text,
    _tree_thinning_for_index,
    _update_node_list_with_text_token_count,
    _write_node_id,
    get_document_structure_json,
    get_node_content,
    get_node_content_by_title,
)

# -- Node extraction --


def test_extract_nodes_simple():
    """Extract headers from simple markdown."""
    md = "# Title\n\nSome text\n\n## Section A\n\nContent\n\n## Section B\n"
    nodes, lines = _extract_nodes_from_markdown(md)
    assert len(nodes) == 3
    assert nodes[0]["node_title"] == "Title"
    assert nodes[0]["line_num"] == 1
    assert nodes[1]["node_title"] == "Section A"


def test_extract_nodes_nested():
    """Extract nested headers (h1-h3)."""
    md = "# H1\n\n## H2\n\n### H3\n"
    nodes, _ = _extract_nodes_from_markdown(md)
    assert len(nodes) == 3
    assert nodes[2]["node_title"] == "H3"


def test_extract_nodes_skip_code_block():
    """Headers inside code blocks are ignored."""
    md = "# Real Header\n\n```\n# Not a header\n```\n\n## Another Header\n"
    nodes, _ = _extract_nodes_from_markdown(md)
    assert len(nodes) == 2


# -- Node text content --


def test_extract_text_content():
    """Node text spans from its header to the next header."""
    md = "## A\nText A\n\n## B\nText B\n"
    nodes, lines = _extract_nodes_from_markdown(md)
    content = _extract_node_text_content(nodes, lines)
    assert "Text A" in content[0]["text"]
    assert "Text B" in content[1]["text"]


def test_extract_text_content_last_node():
    """Last node spans to end of document."""
    md = "## A\nText\n\n## B\nLast text\n"
    nodes, lines = _extract_nodes_from_markdown(md)
    content = _extract_node_text_content(nodes, lines)
    assert "Last text" in content[-1]["text"]


# -- Tree building --


def test_build_tree_flat():
    """Flat headers become flat tree."""
    md = "## A\n\n## B\n\n## C\n"
    nodes, lines = _extract_nodes_from_markdown(md)
    content = _extract_node_text_content(nodes, lines)
    tree = _build_tree_from_nodes(content)
    assert len(tree) == 3


def test_build_tree_nested():
    """Nested headers become hierarchical tree."""
    md = "# H1\n\n## H1.A\n\n## H1.B\n\n# H2\n"
    nodes, lines = _extract_nodes_from_markdown(md)
    content = _extract_node_text_content(nodes, lines)
    tree = _build_tree_from_nodes(content)
    assert len(tree) == 2  # Two h1 roots
    # First root should have two children
    assert len(tree[0]["nodes"]) == 2


def test_write_node_id():
    """Node IDs are sequential zero-padded."""
    tree = [{"title": "A", "nodes": [{"title": "B", "nodes": []}]}, {"title": "C", "nodes": []}]
    _write_node_id(tree)
    assert tree[0]["node_id"] == "0000"
    assert tree[0]["nodes"][0]["node_id"] == "0001"
    assert tree[1]["node_id"] == "0002"


# -- Tree thinning --


def test_tree_thinning_merges_small_nodes():
    """Small nodes are merged into parents."""
    nodes = [
        {"title": "A", "level": 1, "text": "big " * 2000, "line_num": 1},
        {"title": "B", "level": 2, "text": "small", "line_num": 2},
    ]
    nodes = _update_node_list_with_text_token_count(nodes)
    result = _tree_thinning_for_index(nodes, min_node_token=5000)
    # Small node B should be merged into A
    assert len(result) == 1


# -- Recursive splitting --


def test_split_large_text_paragraphs():
    """Large text is split on paragraph boundaries."""
    # Each paragraph ~10 tokens, 20 paragraphs = ~200 tokens
    text = "\n\n".join([f"Paragraph {i} with some content here." for i in range(20)])
    chunks = _split_large_text(text, max_tokens=50)
    assert len(chunks) > 1
    for chunk in chunks:
        # Each chunk should be under the limit (with some margin)
        assert len(chunk) > 0


def test_split_large_text_single_paragraph_too_large():
    """A single oversized paragraph is split by lines."""
    # One paragraph with many lines
    text = "\n".join([f"Line {i} with some extra content to fill tokens." for i in range(50)])
    chunks = _split_large_text(text, max_tokens=50)
    assert len(chunks) > 1


def test_recursive_split_large_nodes():
    """Leaf nodes exceeding max_node_tokens are split into sub-nodes."""
    big_text = "\n\n".join([f"Section paragraph {i} with content." for i in range(100)])
    nodes = [
        {"title": "Small", "level": 1, "text": "tiny", "line_num": 1},
        {"title": "Big", "level": 1, "text": big_text, "line_num": 2},
    ]
    result = _recursive_split_large_nodes(nodes, max_node_tokens=50)
    # Small node stays, Big node gets split
    assert len(result) > 2
    # First chunk keeps original title
    big_node = [n for n in result if n["title"] == "Big"]
    assert len(big_node) == 1
    # Sub-nodes have "(part N)" suffix
    part_nodes = [n for n in result if "(part " in n["title"]]
    assert len(part_nodes) >= 1
    # Sub-nodes are level+1
    for pn in part_nodes:
        assert pn["level"] == 2


def test_recursive_split_respects_hierarchy():
    """Nodes with children are not split (children carry the content)."""
    nodes = [
        {"title": "Parent", "level": 1, "text": "short", "line_num": 1},
        {"title": "Child", "level": 2, "text": "short", "line_num": 2},
    ]
    result = _recursive_split_large_nodes(nodes, max_node_tokens=50)
    assert len(result) == 2  # No splitting


# -- DocumentTree --


def test_document_tree_to_dict():
    """DocumentTree serializes to dict."""
    dt = DocumentTree(
        doc_name="test_doc",
        line_count=100,
        structure=[{"title": "A"}],
        doc_description="A test doc",
    )
    d = dt.to_dict()
    assert d["doc_name"] == "test_doc"
    assert d["doc_description"] == "A test doc"


def test_document_tree_to_json():
    """DocumentTree serializes to JSON."""
    dt = DocumentTree(doc_name="test", line_count=10, structure=[])
    j = dt.to_json()
    assert '"doc_name": "test"' in j


# -- Integration --


def test_md_to_tree_sync(tmp_path):
    """md_to_tree produces a valid DocumentTree (without LLM summaries)."""
    import asyncio

    from drbrain.parser.pageindex_parser import md_to_tree

    md = tmp_path / "test.md"
    md.write_text(
        "# Paper Title\n\nAbstract text.\n\n## Introduction\n\nIntro content.\n\n## Methods\n\nMethod details.\n\n## Conclusion\n\nFinal thoughts.\n"
    )

    config = TreeConfig(
        if_thinning=False,
        if_add_node_summary=False,  # No LLM
        if_add_doc_description=False,
        if_add_node_id=True,
    )
    result = asyncio.run(md_to_tree(md, config=config))
    assert result.doc_name == "test"
    assert result.line_count > 0
    assert len(result.structure) >= 1


# -- Content retrieval --


def test_get_node_content(tmp_path):
    """get_node_content extracts text for a specific node by node_id."""
    md = tmp_path / "test.md"
    md.write_text(
        "# Title\n\nSome text.\n\n## Section A\n\nContent A.\n\n## Section B\n\nContent B.\n"
    )

    import asyncio

    from drbrain.parser.pageindex_parser import md_to_tree

    config = TreeConfig(
        if_thinning=False,
        if_add_node_summary=False,
        if_add_doc_description=False,
        if_add_node_id=True,
    )
    result = asyncio.run(md_to_tree(md, config=config))

    # Get content for first section
    content_a = get_node_content(md, result.structure, "0001")
    assert content_a is not None
    assert "Section A" in content_a
    assert "Content A" in content_a

    # Content B should not include Content A
    content_b = get_node_content(md, result.structure, "0002")
    assert content_b is not None
    assert "Section B" in content_b
    assert "Content A" not in content_b


def test_get_node_content_not_found(tmp_path):
    """get_node_content returns None for unknown node_id."""
    md = tmp_path / "test.md"
    md.write_text("# Title\n\nText.\n")
    assert get_node_content(md, [{"title": "X", "node_id": "0000", "nodes": []}], "9999") is None


def test_get_node_content_by_title(tmp_path):
    """get_node_content_by_title finds node by title match."""
    md = tmp_path / "test.md"
    md.write_text(
        "# Title\n\nText.\n\n## Methods\n\nMethod details here.\n\n## Results\n\nResults here.\n"
    )

    import asyncio

    from drbrain.parser.pageindex_parser import md_to_tree

    config = TreeConfig(
        if_thinning=False,
        if_add_node_summary=False,
        if_add_doc_description=False,
        if_add_node_id=True,
    )
    result = asyncio.run(md_to_tree(md, config=config))

    content = get_node_content_by_title(md, result.structure, "Methods")
    assert content is not None
    assert "Method details" in content


def test_get_document_structure_json():
    """get_document_structure_json strips text fields from output."""
    structure = [
        {
            "title": "A",
            "node_id": "0000",
            "line_num": 1,
            "text": "should be removed",
            "summary": "keep this",
            "nodes": [
                {
                    "title": "B",
                    "node_id": "0001",
                    "line_num": 3,
                    "text": "also removed",
                    "nodes": [],
                }
            ],
        }
    ]
    import json

    result = json.loads(get_document_structure_json(structure))
    assert "text" not in result[0]
    assert result[0]["summary"] == "keep this"
    assert "text" not in result[0]["nodes"][0]
