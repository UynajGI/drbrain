"""Tests for build agents (idempotency, validation, provenance)."""

import pytest

from drbrain.extractor.agent import (
    CorefAgent,
    EntityAgent,
    OntologyAgent,
    RefineAgent,
    RelationAgent,
    get_agent,
)
from drbrain.extractor.concept import _build_tree_hierarchy_text

# -- Tree hierarchy text rendering --


def test_tree_hierarchy_flat():
    """Flat structure renders all nodes at depth 0."""
    structure = [
        {"title": "Introduction", "node_id": "1", "nodes": []},
        {"title": "Methods", "node_id": "2", "nodes": []},
        {"title": "Results", "node_id": "3", "nodes": []},
    ]
    text = _build_tree_hierarchy_text(structure)
    assert "Introduction [depth=0]" in text
    assert "Methods [depth=0]" in text
    assert "Results [depth=0]" in text
    # Last node uses └──
    assert "└── Results" in text
    # First nodes use ├──
    assert "├── Introduction" in text
    assert "├── Methods" in text


def test_tree_hierarchy_nested():
    """Nested structure shows indentation and depth."""
    structure = [
        {
            "title": "3. Methodology",
            "node_id": "1",
            "nodes": [
                {"title": "3.1 Dataset", "node_id": "1.1", "nodes": []},
                {"title": "3.2 Metrics", "node_id": "1.2", "nodes": []},
            ],
        },
    ]
    text = _build_tree_hierarchy_text(structure)
    assert "3. Methodology [depth=0]" in text
    assert "3.1 Dataset [depth=1]" in text
    assert "3.2 Metrics [depth=1]" in text
    # Depth=1 nodes should be indented
    assert "    ├── 3.1" in text or "    └── 3.2" in text


def test_tree_hierarchy_with_children_key():
    """Support nodes that use 'children' instead of 'nodes'."""
    structure = [
        {
            "title": "Parent",
            "node_id": "1",
            "children": [
                {"title": "Child", "node_id": "1.1", "children": []},
            ],
        },
    ]
    text = _build_tree_hierarchy_text(structure)
    assert "Parent [depth=0]" in text
    assert "Child [depth=1]" in text


# -- Agent output validation --


def test_ontology_agent_validation():
    agent = OntologyAgent()
    raw = {"Method": ["GNN", "Attention"], "Problem": ["Scalability"]}
    result = agent._validate_output(raw)
    assert "Method" in result
    assert result["Method"] == ["GNN", "Attention"]
    assert result["Problem"] == ["Scalability"]


def test_ontology_agent_filters_invalid_types():
    agent = OntologyAgent()
    raw = {"Method": ["GNN"], "InvalidType": ["X"], "AnotherBad": ["Y"]}
    result = agent._validate_output(raw)
    assert "Method" in result
    assert "InvalidType" not in result
    assert "AnotherBad" not in result


def test_entity_agent_requires_label_and_type():
    agent = EntityAgent()
    raw = {
        "concepts": [
            {
                "label": "GNN",
                "type": "Method",
                "confidence": 0.9,
                "section": "3.1",
                "node_id": "1.1",
            },
            {"label": "", "type": "Method"},  # missing label
            {"label": "Bad", "type": ""},  # missing type
            {"label": "Good", "type": "Method"},  # minimal valid
        ]
    }
    result = agent._validate_output(raw)
    concepts = result["concepts"]
    assert len(concepts) == 2
    assert concepts[0]["label"] == "GNN"
    assert concepts[0]["node_id"] == "1.1"
    assert concepts[1]["label"] == "Good"


def test_entity_agent_preserves_provenance():
    agent = EntityAgent()
    raw = {
        "concepts": [
            {"label": "GNN", "type": "Method", "section": "3.1 Dataset", "node_id": "0003.001"},
        ]
    }
    result = agent._validate_output(raw)
    c = result["concepts"][0]
    assert c["section"] == "3.1 Dataset"
    assert c["node_id"] == "0003.001"


def test_relation_agent_requires_head_rel_tail():
    agent = RelationAgent()
    raw = {
        "relations": [
            {"head": "GNN", "rel": "uses", "tail": "Dataset", "node_id": "1.1", "section": "3.1"},
            {"head": "GNN", "tail": "Dataset"},  # missing rel
            {"head": "", "rel": "uses", "tail": "X"},  # missing head
        ]
    }
    result = agent._validate_output(raw)
    relations = result["relations"]
    assert len(relations) == 1
    assert relations[0]["head"] == "GNN"
    assert relations[0]["node_id"] == "1.1"


def test_relation_agent_no_provenance_still_valid():
    """Relations without provenance are still accepted (backward compat)."""
    agent = RelationAgent()
    raw = {
        "relations": [
            {"head": "GNN", "rel": "uses", "tail": "Dataset"},
        ]
    }
    result = agent._validate_output(raw)
    assert len(result["relations"]) == 1
    assert result["relations"][0]["node_id"] == ""
    assert result["relations"][0]["section"] == ""


def test_coref_agent_validation():
    agent = CorefAgent()
    raw = {
        "merges": [
            {"canonical": "Graph Neural Network", "variants": ["GNN", "Graph NN"]},
            {"canonical": "", "variants": ["X"]},  # empty canonical
        ]
    }
    result = agent._validate_output(raw)
    merges = result["merges"]
    assert len(merges) == 1
    assert merges[0]["canonical"] == "Graph Neural Network"


def test_refine_agent_with_snapshot():
    agent = RefineAgent()
    agent.set_snapshot(
        [{"label": "GNN"}, {"label": "CNN"}],
        [{"head": "GNN", "rel": "uses", "tail": "Dataset"}],
    )
    raw = {"corrections": [{"type": "relabel", "old": "GNN", "new": "Graph Neural Net"}]}
    result = agent._validate_output(raw)
    assert len(result["corrections"]) == 1
    assert result["diff"] is not None
    assert result["diff"]["before"]["concept_count"] == 2
    assert result["diff"]["before"]["relation_count"] == 1


# -- Agent factory --


def test_get_agent_returns_correct_type():
    assert isinstance(get_agent("ontology"), OntologyAgent)
    assert isinstance(get_agent("entities"), EntityAgent)
    assert isinstance(get_agent("relations"), RelationAgent)
    assert isinstance(get_agent("coreference"), CorefAgent)
    assert isinstance(get_agent("refine"), RefineAgent)


def test_get_agent_unknown_raises():
    with pytest.raises(ValueError, match="Unknown agent"):
        get_agent("nonexistent")


# -- Idempotency guard --


def test_agent_is_complete_true(tmp_db):
    """Agent skips when DB has COMPLETE status."""
    db = tmp_db
    db.conn.execute(
        "INSERT OR REPLACE INTO build_stages (paper_id, stage, status) VALUES (?, ?, ?)",
        ("test-paper", "ontology", "complete"),
    )
    db.commit()

    agent = OntologyAgent()
    assert agent._is_complete(db, "test-paper") is True


def test_agent_is_complete_false(tmp_db):
    """Agent runs when no COMPLETE status exists."""
    db = tmp_db
    agent = OntologyAgent()
    assert agent._is_complete(db, "test-paper") is False


def test_agent_is_complete_in_progress_not_complete(tmp_db):
    """In-progress stage should NOT skip."""
    db = tmp_db
    db.conn.execute(
        "INSERT OR REPLACE INTO build_stages (paper_id, stage, status) VALUES (?, ?, ?)",
        ("test-paper", "ontology", "in_progress"),
    )
    db.commit()

    agent = OntologyAgent()
    assert agent._is_complete(db, "test-paper") is False


def test_agent_save_and_load_result(tmp_db):
    """Results are persisted and recoverable."""
    db = tmp_db
    agent = OntologyAgent()
    agent._save_result(db, "test-paper", {"Method": ["GNN"]})

    cached = agent._load_cached(db, "test-paper")
    assert cached == {"Method": ["GNN"]}


# -- DB integration: edge provenance columns --


def _ensure_test_paper(db, local_id: str = "test") -> None:
    """Insert a minimal paper row if not present (FK prerequisite)."""
    db.conn.execute(
        "INSERT OR IGNORE INTO papers (local_id, title, year, status) VALUES (?, ?, ?, ?)",
        (local_id, "Test Paper", 2024, "uploaded"),
    )
    db.commit()


def test_insert_edge_with_provenance(tmp_db):
    """insert_edge accepts and stores node_id and section."""
    db = tmp_db
    _ensure_test_paper(db)
    db.insert_concept("test", "Method", "GNN", 0.9, section="3.1", node_id="1.1")
    db.insert_concept("test", "Method", "Dataset", 0.8, section="3.1", node_id="1.1")

    db.insert_edge("GNN", "Dataset", "uses", "test", node_id="1.1", section="3.1 Dataset")

    row = db.conn.execute(
        "SELECT node_id, section FROM edges WHERE src_id = ? AND dst_id = ?",
        ("GNN", "Dataset"),
    ).fetchone()
    assert row is not None
    assert row[0] == "1.1"
    assert row[1] == "3.1 Dataset"


def test_insert_edge_without_provenance(tmp_db):
    """Backward compat: edges without provenance get empty defaults."""
    db = tmp_db
    _ensure_test_paper(db)
    db.insert_concept("test", "Method", "A", 0.9)
    db.insert_concept("test", "Method", "B", 0.8)

    db.insert_edge("A", "B", "relates_to", "test")

    row = db.conn.execute(
        "SELECT node_id, section FROM edges WHERE src_id = ? AND dst_id = ?",
        ("A", "B"),
    ).fetchone()
    assert row[0] == ""
    assert row[1] == ""
