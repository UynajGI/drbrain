"""Tree construction from markdown — node extraction, tree building, and md_to_tree entry point."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import litellm

from drbrain.parser.pageindex.retrieval import (
    _create_clean_structure_for_description,
    _format_structure,
    _write_node_id,
)


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
        import json

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
    stack: list = []
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
    from drbrain.parser.pageindex.summary import (
        _generate_doc_description,
        _generate_summaries_for_structure_md,
    )

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
            tree_structure, config.summary_token_threshold, str(model) if model else "", models
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
        doc_description = await _generate_doc_description(
            clean_structure if isinstance(clean_structure, dict) else {}, models
        )

    return DocumentTree(
        doc_name=md_path.stem,
        line_count=line_count,
        structure=tree_structure,
        doc_description=doc_description,
    )
