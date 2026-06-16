"""Tree validation, repair, and fallback parsing strategies."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from drbrain.extractor.llm_client import acall_text_with_fallback


def _count_leaves(nodes: list[dict]) -> int:
    """Count leaf nodes in a tree structure."""
    total = 0
    for n in nodes:
        if not n.get("nodes"):
            total += 1
        else:
            total += _count_leaves(n["nodes"])
    return total


def _flatten_single_chains(nodes: list[dict]) -> list[dict]:
    """Collapse parent→single-child chains where parent has no text."""
    result = []
    for node in nodes:
        # First, flatten children recursively
        children = node.get("nodes", [])
        if children:
            node["nodes"] = _flatten_single_chains(children)

        # Then check if this node can be collapsed (iteratively for cascading)
        while True:
            children = node.get("nodes", [])
            text = node.get("text", "").strip()
            summary = node.get("summary", "").strip()
            prefix_summary = node.get("prefix_summary", "").strip()
            if not text and not summary and not prefix_summary and len(children) == 1:
                node = children[0]
            else:
                break

        result.append(node)
    return result


def _cap_depth(nodes: list[dict], max_depth: int = 5, current_depth: int = 1) -> list[dict]:
    """Merge nodes deeper than max_depth into their parent at max_depth."""
    result = []
    for node in nodes:
        children = node.get("nodes", [])
        if current_depth >= max_depth and children:
            # Merge all children's text into this node
            child_texts = []
            for child in children:
                ct = child.get("text", "").strip()
                if ct:
                    child_texts.append(ct)

                # Also collect grandchildren text recursively
                def _collect_descendant_text(n):
                    texts = []
                    t = n.get("text", "").strip()
                    if t:
                        texts.append(t)
                    for gc in n.get("nodes", []):
                        texts.extend(_collect_descendant_text(gc))
                    return texts

                child_texts.extend(_collect_descendant_text(child))
            if child_texts:
                existing = node.get("text", "").strip()
                all_texts = ([existing] if existing else []) + child_texts
                node["text"] = "\n\n".join(all_texts)
            node["nodes"] = []
            result.append(node)
        else:
            if children:
                node["nodes"] = _cap_depth(children, max_depth, current_depth + 1)
            result.append(node)
    return result


def _split_single_leaf(structure: list[dict]) -> list[dict]:
    """If tree has only one leaf, split it into multiple nodes by paragraphs."""
    if _count_leaves(structure) > 1:
        return structure

    # Find the single leaf
    def _find_leaf(nodes):
        for n in nodes:
            if not n.get("nodes"):
                return n
            found = _find_leaf(n["nodes"])
            if found:
                return found
        return None

    leaf = _find_leaf(structure)
    if not leaf:
        return structure

    text = leaf.get("text", "").strip()
    if not text:
        return structure

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) < 2:
        return structure

    # Split: first paragraph stays in original leaf, rest become siblings
    leaf["text"] = paragraphs[0]
    new_nodes = []
    for i, para in enumerate(paragraphs[1:], 1):
        new_nodes.append(
            {
                "title": f"{leaf['title']} (part {i + 1})",
                "node_id": "",  # will be reassigned by caller
                "line_num": leaf.get("line_num", 0),
                "text": para,
                "nodes": [],
            }
        )

    # Insert new nodes after the leaf's parent in the structure
    def _insert_after_leaf(nodes):
        result = []
        for n in nodes:
            result.append(n)
            if n is leaf:
                result.extend(new_nodes)
            elif n.get("nodes"):
                n["nodes"] = _insert_after_leaf(n["nodes"])
        return result

    return _insert_after_leaf(structure)


def validate_and_fix_tree(structure: list[dict]) -> list[dict]:
    """Validate tree structure and fix common issues.

    1. Remove empty leaf nodes (no text, no summary, no children)
    2. Flatten single-child chains (parent with no text + one child → child)
    3. Cap depth at 5 levels
    4. Split single leaf into multiple nodes by paragraphs
    """

    # 1. Remove empty leaves (bottom-up: process children first)
    def _remove_empty_leaves(nodes):
        result = []
        for n in nodes:
            if n.get("nodes"):
                n["nodes"] = _remove_empty_leaves(n["nodes"])
            text = n.get("text", "").strip()
            summary = n.get("summary", "").strip()
            prefix_summary = n.get("prefix_summary", "").strip()
            children = n.get("nodes", [])
            # Keep if has content, summary, or children
            if text or summary or prefix_summary or children:
                result.append(n)
        return result

    structure = _remove_empty_leaves(structure)

    # 2. Flatten single-child chains
    structure = _flatten_single_chains(structure)

    # 3. Cap depth
    structure = _cap_depth(structure, max_depth=5)

    # 4. Split single leaf
    structure = _split_single_leaf(structure)

    return structure


# -- TOC-based fallback tree construction --


def _extract_pdf_outline(pdf_path: str | Path) -> list[tuple[int, str, int]]:
    """Extract TOC bookmarks from a PDF file.

    Returns list of (level, title, page_index) tuples.
    Returns empty list if PDF has no outline.
    """
    import fitz

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return []

    try:
        doc = fitz.open(str(pdf_path))
        toc = doc.get_toc(simple=True)
        doc.close()
    except Exception:
        return []

    items = []
    for item in toc:
        # simple=True returns (level, title, page) tuples
        items.append((item[0], item[1], item[2] - 1))  # page_index is 0-based
    return items


def _build_tree_from_outline(
    outline: list[tuple[int, str, int]],
    md_path: str | Path,
):
    """Build a DocumentTree from PDF outline items.

    Each outline item becomes a tree node. The text content is assigned
    by splitting the markdown into equal chunks based on page count.
    """
    from drbrain.parser.pageindex.builder import DocumentTree, _build_tree_from_nodes
    from drbrain.parser.pageindex.retrieval import _write_node_id

    md_path = Path(md_path)
    with open(md_path, encoding="utf-8") as f:
        lines = f.readlines()
    total_lines = len(lines)

    if not outline:
        return DocumentTree(doc_name=md_path.stem, line_count=total_lines, structure=[])

    # Assign line ranges: split total_lines evenly across outline items
    n = len(outline)
    chunk_size = max(1, total_lines // n)

    flat_nodes = []
    for i, (level, title, _page_idx) in enumerate(outline):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n - 1 else total_lines
        text = "".join(lines[start:end]).strip()
        flat_nodes.append(
            {
                "title": title,
                "level": level,
                "line_num": start + 1,  # 1-indexed
                "text": text,
            }
        )

    tree_structure = _build_tree_from_nodes(flat_nodes)
    _write_node_id(tree_structure)

    return DocumentTree(
        doc_name=md_path.stem,
        line_count=total_lines,
        structure=tree_structure,
    )


async def _llm_segment_document(
    md_path: str | Path,
    config=None,
    models: list[dict] | None = None,
):
    """LLM-based document segmentation as last-resort fallback.

    Asks the LLM to identify section boundaries in plain text,
    then builds a tree from the identified sections.
    """
    from drbrain.parser.pageindex.builder import DocumentTree, TreeConfig, _build_tree_from_nodes
    from drbrain.parser.pageindex.retrieval import _write_node_id

    config = config or TreeConfig()
    models = models or []

    md_path = Path(md_path)
    with open(md_path, encoding="utf-8") as f:
        content = f.read()
    lines = content.split("\n")
    total_lines = len(lines)

    if not models:
        # No LLM available — return single-node tree
        return DocumentTree(
            doc_name=md_path.stem,
            line_count=total_lines,
            structure=[
                {
                    "title": "Document",
                    "node_id": "0000",
                    "text": content,
                    "line_num": 1,
                    "nodes": [],
                }
            ],
        )

    prompt = (
        "You are given a plain text document without clear section headers. "
        "Identify the major sections and return a JSON array of section objects.\n\n"
        f"Document text (first 4000 chars):\n{content[:4000]}\n\n"
        "Return JSON array (no markdown):\n"
        '[{"title": "Section Name", "start_line": 1, "end_line": 50}, ...]\n'
        "Line numbers are 1-indexed. Cover the entire document."
    )

    response = await acall_text_with_fallback(prompt, models, max_tokens=1024)
    if not response:
        return DocumentTree(doc_name=md_path.stem, line_count=total_lines, structure=[])

    try:
        sections = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return DocumentTree(doc_name=md_path.stem, line_count=total_lines, structure=[])

    if not isinstance(sections, list) or not sections:
        return DocumentTree(doc_name=md_path.stem, line_count=total_lines, structure=[])

    flat_nodes = []
    for sec in sections:
        title = sec.get("title", "Untitled")
        start = max(0, (sec.get("start_line", 1) - 1))
        end = min(total_lines, sec.get("end_line", total_lines))
        text = "\n".join(lines[start:end]).strip()
        flat_nodes.append(
            {
                "title": title,
                "level": 1,
                "line_num": start + 1,
                "text": text,
            }
        )

    tree_structure = _build_tree_from_nodes(flat_nodes)
    if config.if_add_node_id:
        _write_node_id(tree_structure)

    return DocumentTree(
        doc_name=md_path.stem,
        line_count=total_lines,
        structure=tree_structure,
    )


async def _verify_tree_sample(
    tree_items: list[dict], md_lines: list[str], models: list[dict]
) -> float:
    """LLM checks that sampled sections actually start at their claimed lines. Returns accuracy."""
    if not tree_items or not models:
        return 1.0

    import random

    sample_size = min(10, len(tree_items))
    sampled = random.sample(tree_items, sample_size)

    line_chunks = []
    for item in sampled:
        start = max(0, item["line_num"] - 2)
        end = min(len(md_lines), item["line_num"] + 5)
        chunk = "\n".join(md_lines[start:end])
        line_chunks.append((item["title"], item["line_num"], chunk))

    prompt = (
        "Check if each section title starts at the claimed line in the text. "
        "Respond ONLY with a JSON array of objects with keys 'title' and 'correct' (true/false).\n\n"
    )
    for title, line_num, chunk in line_chunks:
        prompt += f"Title: {title} (claimed line {line_num})\nText:\n{chunk}\n\n"

    result = await acall_text_with_fallback(prompt, models, max_tokens=500)
    try:
        checks = json.loads(result) if result else []
        if not checks:
            return 0.0
        correct = sum(1 for c in checks if c.get("correct"))
        return correct / len(checks)
    except Exception:
        return 0.0


async def _verify_and_correct_tree(tree, md_path: str, models: list[dict], max_retries: int = 2):
    """Verify tree structure, re-extract with adjusted settings if accuracy is low."""
    if not models or not tree.structure:
        return tree

    from drbrain.parser.pageindex.builder import TreeConfig, md_to_tree
    from drbrain.parser.pageindex.retrieval import _collect_line_ranges

    md_lines = Path(md_path).read_text(encoding="utf-8").splitlines()
    items = _collect_line_ranges(tree.structure)
    items = [i for i in items if i["line_num"] > 0]

    if len(items) < 3:
        return tree

    for attempt in range(max_retries + 1):
        accuracy = await _verify_tree_sample(items, md_lines, models)
        logger.info(f"Tree verification attempt {attempt}: accuracy={accuracy:.2f}")

        if accuracy >= 0.7:
            return tree

        if attempt < max_retries:
            logger.info("Re-extracting with adjusted thresholds...")
            config = TreeConfig(
                if_thinning=True,
                min_token_threshold=max(2000, 5000 - attempt * 1500),
                max_node_tokens=10000,
                if_add_node_summary=False,
                if_add_doc_description=False,
                if_add_node_id=True,
                if_add_node_text=False,
            )
            new_tree = await md_to_tree(md_path, config, models)
            if new_tree.structure:
                tree = new_tree
                items = _collect_line_ranges(tree.structure)
                items = [i for i in items if i["line_num"] > 0]

    return tree


async def md_to_tree_with_fallback(
    md_path: str | Path,
    config=None,
    models: list[dict] | None = None,
    pdf_path: str | Path | None = None,
):
    """Convert markdown to tree with three-level fallback.

    Level 1: Header-based extraction (existing md_to_tree)
    Level 2: PDF TOC outline (if pdf_path provided)
    Level 3: LLM-based segmentation (if models provided)
    """
    from drbrain.parser.pageindex.builder import md_to_tree

    # Level 1: header-based
    logger.info("[tree] Level 1: header-based extraction for %s", Path(md_path).name)
    tree = await md_to_tree(md_path, config, models)
    if tree.structure:
        logger.info("[tree] Level 1 succeeded — %d sections", len(tree.structure))
        if models:
            tree = await _verify_and_correct_tree(tree, md_path, models)
        return tree

    # Level 2: PDF outline
    logger.info("[tree] Level 1 failed (no structure), trying Level 2: PDF outline")
    if pdf_path:
        outline = _extract_pdf_outline(pdf_path)
        if outline:
            logger.info("[tree] Level 2 succeeded via PDF outline")
            return _build_tree_from_outline(outline, md_path)

    # Level 3: LLM segmentation
    logger.info("[tree] Level 2 failed, trying Level 3: LLM segmentation")
    return await _llm_segment_document(md_path, config, models)
