"""Tests for graph export formats: GraphML, JSON-LD, Cypher."""

import json
import xml.etree.ElementTree as ET

import pytest

from drbrain.graph.engine import GraphEngine
from drbrain.storage.database import Database
from drbrain.storage.graph_export import export_cypher, export_graphml, export_jsonld


@pytest.fixture
def populated_db(tmp_path):
    """Create an in-memory DB with papers, concepts, and edges."""
    db = Database(":memory:")
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Paper A', 2023, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p1', 'Method', 'Transformer', 0.9, 'method')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p1', 'Problem', 'Attention Bottleneck', 0.8, 'introduction')"
    )
    db.commit()
    yield db
    db.close()


@pytest.fixture
def populated_graph(populated_db):
    """Create a GraphEngine with test edges."""
    g = GraphEngine()
    g.add_edge(
        "Transformer", "Attention Bottleneck", relation="addresses", source_paper="p1", weight=1.0
    )
    g.add_edge("Transformer", "Self-Attention", relation="extends", source_paper="p1", weight=0.9)
    return g


class TestExportGraphML:
    def test_graphml_creates_valid_xml(self, populated_graph, populated_db, tmp_path):
        path = str(tmp_path / "kg.graphml")
        export_graphml(populated_graph, populated_db, path)

        # File should exist and be valid XML
        tree = ET.parse(path)
        root = tree.getroot()
        assert root.tag == "graphml" or root.tag.endswith("graphml")


class TestExportJSONLD:
    def test_jsonld_creates_valid_json(self, populated_graph, populated_db, tmp_path):
        path = str(tmp_path / "kg.jsonld")
        export_jsonld(populated_graph, populated_db, path)

        with open(path) as f:
            data = json.load(f)

        # Should have @context and @graph
        assert "@context" in data
        assert "@graph" in data

        # Split nodes and edges
        nodes = [item for item in data["@graph"] if "@id" in item and "subject" not in item]
        edges = [item for item in data["@graph"] if "subject" in item]

        # Should have at least our test nodes and edges
        assert len(nodes) >= 3  # Transformer, Attention Bottleneck, Self-Attention
        assert len(edges) >= 2  # addresses, extends

        # Check structure of a node
        node_ids = {n["@id"] for n in nodes}
        assert "Transformer" in node_ids
        assert "Attention Bottleneck" in node_ids

        # Check structure of an edge
        edge = edges[0]
        assert "subject" in edge
        assert "predicate" in edge
        assert "object" in edge


class TestExportCypher:
    def test_cypher_contains_create_merge(self, populated_graph, populated_db, tmp_path):
        path = str(tmp_path / "kg.cypher")
        export_cypher(populated_graph, populated_db, path)

        with open(path) as f:
            content = f.read()

        # Should contain MERGE statements for nodes
        assert "MERGE" in content

        # Should contain CREATE for relationships
        assert "CREATE" in content

        # Should reference our test nodes
        assert "Transformer" in content
        assert "Attention Bottleneck" in content

        # Should have relationship types derived from relation names
        assert "ADDRESSES" in content or "addresses" in content

    def test_cypher_has_node_and_edge_sections(self, populated_graph, populated_db, tmp_path):
        path = str(tmp_path / "kg.cypher")
        export_cypher(populated_graph, populated_db, path)

        with open(path) as f:
            content = f.read()

        # Should have section comments
        assert "// ── Nodes ──" in content
        assert "// ── Relationships ──" in content
