"""Knowledge graph export in GraphML, JSON-LD, and Cypher formats.

These formats enable interoperability with Neo4j (Cypher), Gephi/Cytoscape
(GraphML), and RDF/semantic-web tooling (JSON-LD).
"""

from __future__ import annotations

import json
from typing import Any

import networkx as nx
from loguru import logger


def export_graphml(graph: Any, db: Any, path: str) -> None:
    """Export KG as GraphML (Gephi/Cytoscape compatible).

    Nodes: concepts with type/label attributes.
    Edges: relations with relation type + confidence.
    Uses networkx.write_graphml().
    """
    g = graph.graph

    # Enrich nodes with concept metadata from DB
    node_attrs: dict[str, dict[str, str]] = {}
    try:
        rows = db.conn.execute("SELECT label, type FROM concepts").fetchall()
        label_to_type: dict[str, str] = {r[0]: r[1] for r in rows}
    except Exception:
        label_to_type = {}

    for node in g.nodes():
        attrs = {"label": str(node)}
        node_type = label_to_type.get(node, "unknown")
        attrs["type"] = node_type
        node_attrs[node] = attrs

    nx.set_node_attributes(g, node_attrs)

    logger.info(
        "[export:graphml] writing %d nodes, %d edges → %s",
        g.number_of_nodes(),
        g.number_of_edges(),
        path,
    )
    nx.write_graphml(g, path)


def export_jsonld(graph: Any, db: Any, path: str) -> None:
    """Export KG as JSON-LD (RDF-compatible).

    Each concept → {"@id": label, "@type": type}
    Each edge → {"@id": edge_id, "subject": src, "predicate": rel, "object": dst}
    """
    g = graph.graph

    # Build label→type mapping from DB
    label_to_type: dict[str, str] = {}
    try:
        rows = db.conn.execute("SELECT label, type FROM concepts").fetchall()
        label_to_type = {r[0]: r[1] for r in rows}
    except Exception:
        pass

    nodes = []
    seen_labels: set[str] = set()
    for node in g.nodes():
        node_str = str(node)
        if node_str in seen_labels:
            continue
        seen_labels.add(node_str)
        entry: dict[str, Any] = {
            "@id": node_str,
            "@type": label_to_type.get(node_str, "Concept"),
        }
        nodes.append(entry)

    edges = []
    for idx, (u, v, data) in enumerate(g.edges(data=True)):
        edge_entry: dict[str, Any] = {
            "@id": f"edge_{idx}",
            "subject": str(u),
            "predicate": data.get("relation", "related_to"),
            "object": str(v),
        }
        if "weight" in data:
            edge_entry["weight"] = data["weight"]
        if "source_paper" in data:
            edge_entry["source_paper"] = data["source_paper"]
        edges.append(edge_entry)

    doc: dict[str, Any] = {
        "@context": {
            "@vocab": "https://drbrain.org/kg/",
            "subject": {"@type": "@id"},
            "object": {"@type": "@id"},
        },
        "@graph": nodes + edges,
    }

    logger.info("[export:jsonld] writing %d nodes, %d edges → %s", len(nodes), len(edges), path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)


def export_cypher(graph: Any, db: Any, path: str) -> None:
    """Export KG as Cypher script (Neo4j import).

    Generates CREATE/MERGE statements for nodes and relationships.
    """
    g = graph.graph

    # Build label→type mapping from DB
    label_to_type: dict[str, str] = {}
    try:
        rows = db.conn.execute("SELECT label, type FROM concepts").fetchall()
        label_to_type = {r[0]: r[1] for r in rows}
    except Exception:
        pass

    lines: list[str] = []
    lines.append("// DrBrain Knowledge Graph — Cypher export")
    lines.append(f"// Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}")
    lines.append("")

    # Collect unique nodes and sanitize labels for Cypher
    seen_nodes: dict[str, str] = {}  # label -> sanitized
    for node in g.nodes():
        node_str = str(node)
        sanitized = node_str.replace("'", "\\'")
        seen_nodes[node_str] = sanitized

    # Node CREATE statements
    lines.append("// ── Nodes ──")
    for node_str, sanitized in seen_nodes.items():
        node_type = label_to_type.get(node_str, "Concept")
        cypher_type = node_type.replace(" ", "_")
        lines.append(f"MERGE (n:`{cypher_type}` {{label: '{sanitized}'}});")

    lines.append("")

    # Relationship CREATE statements
    lines.append("// ── Relationships ──")
    for u, v, data in g.edges(data=True):
        u_san = str(u).replace("'", "\\'")
        v_san = str(v).replace("'", "\\'")
        rel_type = data.get("relation", "RELATED_TO").upper().replace(" ", "_")
        weight = data.get("weight", 1.0)
        source = str(data.get("source_paper", "")).replace("'", "\\'")

        u_type = label_to_type.get(str(u), "Concept").replace(" ", "_")
        v_type = label_to_type.get(str(v), "Concept").replace(" ", "_")

        lines.append(
            f"MERGE (a:`{u_type}` {{label: '{u_san}'}}) "
            f"MERGE (b:`{v_type}` {{label: '{v_san}'}}) "
            f"CREATE (a)-[r:`{rel_type}` {{weight: {weight}, source: '{source}'}}]->(b);"
        )

    output = "\n".join(lines) + "\n"

    logger.info(
        "[export:cypher] writing %d node stmts, %d edge stmts → %s",
        len(seen_nodes),
        g.number_of_edges(),
        path,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(output)
