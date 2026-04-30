"""Markdown document structuring via PageIndex tree extraction.

Adapted from PageIndex (https://github.com/vectify-ai/pageindex).
Original code Copyright (c) 2025 Vectify AI, MIT License.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path

import litellm

from drbrain.extractor.llm_client import acall_text_with_fallback


@dataclass
class TreeConfig:
    """Configuration for markdown tree extraction."""

    if_thinning: bool = False
    min_token_threshold: int = 5000
    if_add_node_summary: bool = True
    summary_token_threshold: int = 200
    if_add_doc_description: bool = True
    if_add_node_text: bool = False
    if_add_node_id: bool = True
    max_node_tokens: int = 10000


@dataclass
class DocumentTree:
    """Structured document tree output."""

    doc_name: str
    line_count: int
    structure: list[dict]
    doc_description: str | None = None

    def to_dict(self) -> dict:
        d = {
            "doc_name": self.doc_name,
            "line_count": self.line_count,
            "structure": self.structure,
        }
        if self.doc_description:
            d["doc_description"] = self.doc_description
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# -- Core tree-building logic (adapted from page_index_md.py) --


def _extract_nodes_from_markdown(markdown_content: str) -> tuple[list[dict], list[str]]:
    """Extract header nodes from markdown content."""
    header_pattern = r"^(#{1,6})\s+(.+)$"
    code_block_pattern = r"^```"
    node_list = []
    in_code_block = False

    for line_num, line in enumerate(markdown_content.split("\n"), 1):
        stripped = line.strip()
        if re.match(code_block_pattern, stripped):
            in_code_block = not in_code_block
            continue
        if not stripped:
            continue
        if not in_code_block:
            match = re.match(header_pattern, stripped)
            if match:
                node_list.append({"node_title": match.group(2).strip(), "line_num": line_num})

    return node_list, markdown_content.split("\n")


def _extract_node_text_content(node_list: list[dict], markdown_lines: list[str]) -> list[dict]:
    """Attach text content to each node (from its header to the next header)."""
    all_nodes = []
    for node in node_list:
        line_content = markdown_lines[node["line_num"] - 1]
        header_match = re.match(r"^(#{1,6})", line_content)
        if header_match is None:
            continue
        all_nodes.append(
            {
                "title": node["node_title"],
                "line_num": node["line_num"],
                "level": len(header_match.group(1)),
            }
        )

    for i, node in enumerate(all_nodes):
        start_line = node["line_num"] - 1
        end_line = (
            all_nodes[i + 1]["line_num"] - 1 if i + 1 < len(all_nodes) else len(markdown_lines)
        )
        node["text"] = "\n".join(markdown_lines[start_line:end_line]).strip()
    return all_nodes


def _find_all_children(parent_index: int, parent_level: int, node_list: list[dict]) -> list[int]:
    """Find all descendants of a parent node."""
    children = []
    for i in range(parent_index + 1, len(node_list)):
        if node_list[i]["level"] <= parent_level:
            break
        children.append(i)
    return children


def _update_node_list_with_text_token_count(
    node_list: list[dict], model: str | None = None
) -> list[dict]:
    """Compute token count for each node including its children's text."""
    result = node_list.copy()
    for i in range(len(result) - 1, -1, -1):
        children_indices = _find_all_children(i, result[i]["level"], result)
        total_text = result[i].get("text", "")
        for ci in children_indices:
            child_text = result[ci].get("text", "")
            if child_text:
                total_text += "\n" + child_text
        result[i]["text_token_count"] = litellm.token_counter(model=model, text=total_text)
    return result


def _tree_thinning_for_index(
    node_list: list[dict], min_node_token: int, model: str | None = None
) -> list[dict]:
    """Merge small nodes into their parents to reduce tree depth."""
    result = node_list.copy()
    to_remove: set[int] = set()

    for i in range(len(result) - 1, -1, -1):
        if i in to_remove:
            continue
        if result[i].get("text_token_count", 0) < min_node_token:
            children_indices = _find_all_children(i, result[i]["level"], result)
            children_texts = []
            for ci in sorted(children_indices):
                if ci not in to_remove:
                    ct = result[ci].get("text", "")
                    if ct.strip():
                        children_texts.append(ct)
                    to_remove.add(ci)
            if children_texts:
                parent_text = result[i].get("text", "")
                merged = parent_text
                for ct in children_texts:
                    if merged and not merged.endswith("\n"):
                        merged += "\n\n"
                    merged += ct
                result[i]["text"] = merged
                result[i]["text_token_count"] = litellm.token_counter(model=model, text=merged)

    for idx in sorted(to_remove, reverse=True):
        result.pop(idx)
    return result


def _split_large_text(text: str, max_tokens: int, model: str | None = None) -> list[str]:
    """Split text into chunks that each stay under max_tokens.

    Splits on paragraph boundaries (double newlines), falling back to
    single newlines if a single paragraph exceeds the limit.
    """
    paragraphs = re.split(r"\n\n+", text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if litellm.token_counter(model=model, text=candidate) <= max_tokens:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If a single paragraph is too large, split by single newlines
            if litellm.token_counter(model=model, text=para) > max_tokens:
                lines = para.split("\n")
                line_chunk = ""
                for line in lines:
                    line_candidate = f"{line_chunk}\n{line}".strip() if line_chunk else line
                    if litellm.token_counter(model=model, text=line_candidate) <= max_tokens:
                        line_chunk = line_candidate
                    else:
                        if line_chunk:
                            chunks.append(line_chunk)
                        line_chunk = line
                if line_chunk:
                    current = line_chunk
                else:
                    current = ""
            else:
                current = para

    if current:
        chunks.append(current)
    return chunks


def _recursive_split_large_nodes(
    nodes: list[dict], max_node_tokens: int, model: str | None = None
) -> list[dict]:
    """Walk the flat node list and split any node whose text exceeds max_node_tokens.

    Large leaf nodes (no children in the flat list) are split into paragraph-based
    chunks. Each chunk becomes a new synthetic child node at level+1.
    """
    result: list[dict] = []
    for i, node in enumerate(nodes):
        text = node.get("text", "")
        token_count = litellm.token_counter(model=model, text=text)

        # Check if this node has children (next node has higher level)
        has_children = i + 1 < len(nodes) and nodes[i + 1]["level"] > node["level"]

        if token_count > max_node_tokens and not has_children:
            # Split into chunks
            chunks = _split_large_text(text, max_node_tokens, model)
            if len(chunks) > 1:
                # Keep the node with first chunk as its text
                node["text"] = chunks[0]
                result.append(node)
                # Add synthetic sub-nodes for remaining chunks
                for ci, chunk in enumerate(chunks[1:], 1):
                    result.append(
                        {
                            "title": f"{node['title']} (part {ci + 1})",
                            "line_num": node["line_num"],
                            "level": node["level"] + 1,
                            "text": chunk,
                        }
                    )
                continue
        result.append(node)
    return result


def _build_tree_from_nodes(node_list: list[dict]) -> list[dict]:
    """Convert flat node list to hierarchical tree."""
    if not node_list:
        return []
    stack = []
    root_nodes = []
    node_counter = 1

    for node in node_list:
        tree_node = {
            "title": node["title"],
            "node_id": str(node_counter).zfill(4),
            "text": node.get("text", ""),
            "line_num": node["line_num"],
            "nodes": [],
        }
        node_counter += 1

        while stack and stack[-1][1] >= node["level"]:
            stack.pop()

        if not stack:
            root_nodes.append(tree_node)
        else:
            stack[-1][0]["nodes"].append(tree_node)

        stack.append((tree_node, node["level"]))

    return root_nodes


def _clean_tree_for_output(tree_nodes: list[dict]) -> list[dict]:
    """Clean tree for JSON output (remove internal fields recursively)."""
    cleaned = []
    for node in tree_nodes:
        c = {
            "title": node["title"],
            "node_id": node["node_id"],
            "text": node["text"],
            "line_num": node["line_num"],
        }
        if node["nodes"]:
            c["nodes"] = _clean_tree_for_output(node["nodes"])
        cleaned.append(c)
    return cleaned


def _write_node_id(data, node_id: int = 0) -> int:
    """Assign sequential node_ids to tree nodes."""
    if isinstance(data, dict):
        data["node_id"] = str(node_id).zfill(4)
        node_id += 1
        for key in list(data.keys()):
            if "nodes" in key:
                node_id = _write_node_id(data[key], node_id)
    elif isinstance(data, list):
        for item in data:
            node_id = _write_node_id(item, node_id)
    return node_id


def _structure_to_list(structure) -> list[dict]:
    """Flatten tree to list of nodes."""
    if isinstance(structure, dict):
        nodes = [structure]
        if "nodes" in structure:
            nodes.extend(_structure_to_list(structure["nodes"]))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(_structure_to_list(item))
        return nodes
    return []


def _format_structure(structure, order: list[str] | None = None):
    """Reorder dict keys and remove empty nodes arrays."""
    if not order:
        return structure
    if isinstance(structure, dict):
        if "nodes" in structure:
            structure["nodes"] = _format_structure(structure["nodes"], order)
        if not structure.get("nodes"):
            structure.pop("nodes", None)
        return {k: structure[k] for k in order if k in structure}
    elif isinstance(structure, list):
        return [_format_structure(item, order) for item in structure]
    return structure


def _create_clean_structure_for_description(structure) -> dict | list:
    """Create structure without text fields for doc description generation."""
    if isinstance(structure, dict):
        clean = {}
        for key in ("title", "node_id", "summary", "prefix_summary"):
            if key in structure:
                clean[key] = structure[key]
        if "nodes" in structure and structure["nodes"]:
            clean["nodes"] = _create_clean_structure_for_description(structure["nodes"])
        return clean
    elif isinstance(structure, list):
        return [_create_clean_structure_for_description(item) for item in structure]
    return structure


# -- LLM-based summary generation --


async def _generate_node_summary(node: dict, models: list[dict]) -> str:
    """Generate summary for a single node via LLM."""
    prompt = (
        "You are given a part of a document. Generate a concise description "
        "of what the main points are covered.\n\n"
        f"Partial Document Text: {node.get('text', '')}\n\n"
        "Directly return the description, do not include any other text."
    )
    response = await acall_text_with_fallback(prompt, models, max_tokens=256)
    return response or ""


async def _generate_doc_description(structure: dict, models: list[dict]) -> str:
    """Generate one-sentence document description via LLM."""
    prompt = (
        "You are an expert in generating descriptions for documents.\n"
        "Given a document structure, generate a one-sentence description "
        "that distinguishes this document from others.\n\n"
        f"Document Structure: {structure}\n\n"
        "Directly return the description, do not include any other text."
    )
    response = await acall_text_with_fallback(prompt, models, max_tokens=128)
    return response or ""


async def _generate_summaries_for_structure_md(
    structure: list[dict],
    summary_token_threshold: int,
    model: str,
    models: list[dict],
) -> list[dict]:
    """Generate summaries for all nodes in the structure."""
    nodes = _structure_to_list(structure)

    async def _get_summary(node: dict) -> str:
        node_text = node.get("text", "")
        num_tokens = litellm.token_counter(model=model, text=node_text)
        if num_tokens < summary_token_threshold:
            return node_text
        return await _generate_node_summary(node, models)

    tasks = [_get_summary(node) for node in nodes]
    summaries = await asyncio.gather(*tasks)

    for node, summary in zip(nodes, summaries):
        if not node.get("nodes"):
            node["summary"] = summary
        else:
            node["prefix_summary"] = summary

    return structure


# -- Main entry point --


async def md_to_tree(
    md_path: str | Path,
    config: TreeConfig | None = None,
    models: list[dict] | None = None,
) -> DocumentTree:
    """Convert a Markdown file to a structured document tree.

    Args:
        md_path: Path to the markdown file.
        config: Tree extraction options.
        models: LLM model configs for summary generation.

    Returns:
        DocumentTree with structured hierarchy and optional summaries.
    """
    config = config or TreeConfig()
    models = models or []

    md_path = Path(md_path)
    with open(md_path, encoding="utf-8") as f:
        markdown_content = f.read()

    line_count = markdown_content.count("\n") + 1
    model = models[0].get("model", None) if models else None

    # Extract nodes
    node_list, markdown_lines = _extract_nodes_from_markdown(markdown_content)
    nodes_with_content = _extract_node_text_content(node_list, markdown_lines)

    # Optional thinning
    if config.if_thinning and config.min_token_threshold:
        nodes_with_content = _update_node_list_with_text_token_count(nodes_with_content, model)
        nodes_with_content = _tree_thinning_for_index(
            nodes_with_content, config.min_token_threshold, model
        )

    # Split large nodes into sub-chunks
    if config.max_node_tokens:
        nodes_with_content = _recursive_split_large_nodes(
            nodes_with_content, config.max_node_tokens, model
        )

    # Build tree
    tree_structure = _build_tree_from_nodes(nodes_with_content)
    if config.if_add_node_id:
        _write_node_id(tree_structure)

    # Generate summaries
    if config.if_add_node_summary and models:
        tree_structure = _format_structure(
            tree_structure,
            order=["title", "node_id", "line_num", "summary", "prefix_summary", "text", "nodes"],
        )
        tree_structure = await _generate_summaries_for_structure_md(
            tree_structure, config.summary_token_threshold, model, models
        )

        if not config.if_add_node_text:
            tree_structure = _format_structure(
                tree_structure,
                order=["title", "node_id", "line_num", "summary", "prefix_summary", "nodes"],
            )

    # Generate doc description
    doc_description = None
    if config.if_add_doc_description and models:
        clean_structure = _create_clean_structure_for_description(tree_structure)
        doc_description = await _generate_doc_description(clean_structure, models)

    return DocumentTree(
        doc_name=md_path.stem,
        line_count=line_count,
        structure=tree_structure,
        doc_description=doc_description,
    )


# -- Tree validation and repair --


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
    import pypdfium2 as pdfium

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return []

    try:
        doc = pdfium.PdfDocument(str(pdf_path))
        toc = list(doc.get_toc())
        doc.close()
    except Exception:
        return []

    items = []
    for item in toc:
        # PdfOutlineItem(level, title, is_closed, n_kids, page_index, view_mode, view_pos)
        items.append((item.level, item.title, item.page_index))
    return items


def _build_tree_from_outline(
    outline: list[tuple[int, str, int]],
    md_path: str | Path,
) -> DocumentTree:
    """Build a DocumentTree from PDF outline items.

    Each outline item becomes a tree node. The text content is assigned
    by splitting the markdown into equal chunks based on page count.
    """
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
    config: TreeConfig | None = None,
    models: list[dict] | None = None,
) -> DocumentTree:
    """LLM-based document segmentation as last-resort fallback.

    Asks the LLM to identify section boundaries in plain text,
    then builds a tree from the identified sections.
    """
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


async def md_to_tree_with_fallback(
    md_path: str | Path,
    config: TreeConfig | None = None,
    models: list[dict] | None = None,
    pdf_path: str | Path | None = None,
) -> DocumentTree:
    """Convert markdown to tree with three-level fallback.

    Level 1: Header-based extraction (existing md_to_tree)
    Level 2: PDF TOC outline (if pdf_path provided)
    Level 3: LLM-based segmentation (if models provided)
    """
    # Level 1: header-based
    tree = await md_to_tree(md_path, config, models)
    if tree.structure:
        return tree

    # Level 2: PDF outline
    if pdf_path:
        outline = _extract_pdf_outline(pdf_path)
        if outline:
            return _build_tree_from_outline(outline, md_path)

    # Level 3: LLM segmentation
    return await _llm_segment_document(md_path, config, models)


# -- Content retrieval (on-demand) --


def _find_node_by_id(structure: list[dict], node_id: str) -> dict | None:
    """Find a node in the tree by its node_id."""
    for node in structure:
        if node.get("node_id") == node_id:
            return node
        found = _find_node_by_id(node.get("nodes", []), node_id)
        if found:
            return found
    return None


def _collect_line_ranges(structure: list[dict]) -> list[dict]:
    """Flatten tree into a sorted list of {node_id, title, line_num} for range calculation."""
    items = []
    for node in structure:
        items.append(
            {
                "node_id": node.get("node_id", ""),
                "title": node.get("title", ""),
                "line_num": node.get("line_num", 0),
            }
        )
        items.extend(_collect_line_ranges(node.get("nodes", [])))
    return sorted(items, key=lambda x: x["line_num"])


def get_node_content(
    md_path: str | Path,
    structure: list[dict],
    node_id: str,
) -> str | None:
    """Load the markdown text content for a specific node by its node_id.

    Reads the MD file and extracts lines from the node's header to the
    next sibling/parent header. Returns None if node_id not found.
    """
    target = _find_node_by_id(structure, node_id)
    if not target:
        return None

    md_path = Path(md_path)
    if not md_path.exists():
        return None

    with open(md_path, encoding="utf-8") as f:
        lines = f.readlines()

    start_line = target["line_num"] - 1  # 0-indexed

    # Find end: next node at same or higher level
    all_nodes = _collect_line_ranges(structure)
    end_line = len(lines)
    found_target = False
    for n in all_nodes:
        if n["line_num"] == target["line_num"]:
            found_target = True
            continue
        if found_target and n["line_num"] > target["line_num"]:
            end_line = n["line_num"] - 1  # exclusive, 1-indexed → 0-indexed
            break

    return "".join(lines[start_line:end_line]).strip()


def get_node_content_by_title(
    md_path: str | Path,
    structure: list[dict],
    title: str,
) -> str | None:
    """Load content for the first node matching the given title."""

    def _find(nodes: list[dict]) -> str | None:
        for node in nodes:
            if node.get("title", "").strip().lower() == title.strip().lower():
                return node.get("node_id")
            found = _find(node.get("nodes", []))
            if found:
                return found
        return None

    node_id = _find(structure)
    if not node_id:
        return None
    return get_node_content(md_path, structure, node_id)


def get_document_structure_json(structure: list[dict]) -> str:
    """Return tree structure as JSON string with text fields removed (saves tokens)."""

    def _remove_text(nodes: list[dict]) -> list[dict]:
        cleaned = []
        for node in nodes:
            c = {k: v for k, v in node.items() if k != "text"}
            if "nodes" in c:
                c["nodes"] = _remove_text(c["nodes"])
            cleaned.append(c)
        return cleaned

    return json.dumps(_remove_text(structure), ensure_ascii=False, indent=2)
