"""Tree display: text and Mermaid rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def format_tree(nodes: list[dict], indent: str = "", mermaid: bool = False) -> str:
    """Format lineage tree as text or Mermaid diagram."""
    if mermaid:
        return _to_mermaid(nodes)
    return _to_text_tree(nodes)


def _to_text_tree(nodes: list[dict], prefix: str = "") -> str:
    """Render as indented text tree with box-drawing characters."""
    lines: list[str] = []
    for i, node in enumerate(nodes):
        is_last = i == len(nodes) - 1
        connector = "└─ " if is_last else "├─ "

        year_str = f" ({node['year']})" if node.get("year") else ""
        rel_str = f" — {node['relation']}" if node.get("relation") else ""
        type_str = f" [{node.get('type', '')}]" if node.get("type") else ""

        lines.append(f"{prefix}{connector}{node['label']}{type_str}{year_str}{rel_str}")

        # Show provenance for evolve nodes
        section = node.get("section", "")
        if section:
            lines.append(f"{prefix}    source: {section}")

        # Show bridge provenance for descendant nodes
        via_prov = node.get("via_provenance", "")
        if via_prov:
            lines.append(f"{prefix}    {via_prov}")

        children = node.get("children", [])
        if children:
            child_prefix = prefix + ("   " if is_last else "│  ")
            lines.append(_to_text_tree(children, child_prefix))

    return "\n".join(lines)


def _to_mermaid(nodes: list[dict]) -> str:
    """Render as Mermaid flowchart."""
    lines = ["graph TD"]
    _mermaid_nodes(lines, nodes, None)
    return "\n".join(lines)


def _mermaid_nodes(lines: list[str], nodes: list[dict], parent_id: str | None):
    """Recursively add Mermaid nodes and edges."""
    for node in nodes:
        nid = node["label"].replace(" ", "_")[:50]
        year_str = f" ({node['year']})" if node.get("year") else ""
        section = node.get("section", "")
        via_section = node.get("via_section", "")
        tooltip = section or via_section or ""
        tooltip_str = f"<br/>{tooltip}" if tooltip else ""
        lines.append(f'    {nid}["{node["label"]}{year_str}{tooltip_str}"]')
        if parent_id:
            rel = node.get("relation", "")
            via_prov = node.get("via_provenance", "")
            edge_label = f"{rel}: {via_prov}" if via_prov else rel
            lines.append(f"    {parent_id} -->|{edge_label}| {nid}")
        children = node.get("children", [])
        if children:
            _mermaid_nodes(lines, children, nid)
