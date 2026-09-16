"""T16: extraction consumers read canonical content when no files are present."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from drbrain.extractor.agent_tools import get_document_structure, get_section_content
from drbrain.extractor.concept.tree_helpers import _collect_leaf_nodes
from drbrain.extractor.context import canonical_extraction_inputs
from drbrain.services.canonical_content import write_canonical_content
from drbrain.storage.database import Database
from drbrain.storage.node_projection import collect_canonical_node_records

TEXT = (
    "# Title\n\nIntro paragraph.\n\n## Methods\n\n"
    "Step one of the method.\n\n## Results\n\nThe outcome text is here.\n"
)

LEGACY_RAW = "# Old\n\nlegacy body\n"


def _db(tmp_path) -> Database:
    return Database(tmp_path / "db.sqlite")


def _canonical(db: Database, local_id: str = "p1", text: str = TEXT) -> None:
    if db.get_paper(local_id) is None:
        db.insert_paper(local_id, "T", 2024, "uploaded")
        db.commit()
    result = write_canonical_content(db, local_id, text, media_type="md", parser="test")
    assert result["ok"]


def _legacy_paper(root: Path, local_id: str = "p1") -> Path:
    directory = root / local_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "raw.md").write_text(LEGACY_RAW, encoding="utf-8")
    tree = {"structure": [{"title": "Root", "node_id": "0000", "line_num": 1, "nodes": []}]}
    (directory / "tree.json").write_text(json.dumps(tree), encoding="utf-8")
    return directory


def _add_region(db: Database, local_id: str = "p1", summary: str = "A short summary.") -> str:
    from drbrain.tree.contracts import ChildRef, NodeRecord, region_node_id

    leaves = db.list_tree_nodes(kind="leaf", state="ready", local_id=local_id, limit=100)
    refs = tuple(
        ChildRef(child_id=leaf["node_id"], child_revision=1, ordinal=index)
        for index, leaf in enumerate(leaves[:2])
    )
    contract = {"purpose": "test"}
    region = NodeRecord(
        node_id=region_node_id(refs, contract),
        revision=1,
        kind="region",
        state="ready",
        layer=1,
        content_hash=hashlib.sha256(summary.encode("utf-8")).hexdigest(),
        summary=summary,
        children=refs,
        contract=contract,
    )
    db.insert_tree_node(region, publish=True)
    return region.node_id


class TestExtractionInputs:
    def test_canonical_inputs_reconstruct_the_document_text(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        canonical = canonical_extraction_inputs(db, "p1")
        assert canonical is not None
        structure, texts = canonical
        leaves = _collect_leaf_nodes(structure)
        assert leaves and all(leaf["node_id"] in texts for leaf in leaves)
        # The extraction input is the same document the old MD slice produced.
        assert "".join(texts[leaf["node_id"]] for leaf in leaves) == TEXT
        # No files were needed (and none were created).
        assert not (tmp_path / "papers").exists()

    def test_region_becomes_a_parent_and_supplies_its_summary(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        region_id = _add_region(db, summary="Summary only, no child text.")
        structure, texts = canonical_extraction_inputs(db, "p1")
        assert region_id in texts and texts[region_id] == "Summary only, no child text."
        roots = {node["node_id"] for node in structure}
        assert region_id in roots
        # Leaves stay childless under their region parent.
        leaves = _collect_leaf_nodes(structure)
        assert leaves and all(leaf["node_id"] != region_id for leaf in leaves)

    def test_legacy_only_paper_returns_none(self, tmp_path):
        db = _db(tmp_path)
        db.insert_paper("p1", "T", 2024, "uploaded")
        db.commit()
        _legacy_paper(tmp_path / "papers")
        assert canonical_extraction_inputs(db, "p1") is None


class TestAgentTools:
    def test_tools_serve_db_only_papers(self, tmp_path):
        db = _db(tmp_path)
        _canonical(db)
        nodes = get_document_structure(None, "p1", db=db)
        assert nodes and all(node["node_id"].startswith(("nl-", "nr-")) for node in nodes)
        leaf_id = _collect_leaf_nodes(nodes)[0]["node_id"]
        expected = {
            str(record["node_id"]): str(record["text"])
            for record in collect_canonical_node_records(db.conn, "p1")
        }
        assert get_section_content(None, "p1", leaf_id, db=db) == expected[leaf_id]
        assert not (tmp_path / "papers").exists()

    def test_tools_legacy_fallback_is_unchanged(self, tmp_path):
        db = _db(tmp_path)
        db.insert_paper("p1", "T", 2024, "uploaded")
        db.commit()
        root = tmp_path / "papers"
        _legacy_paper(root)
        nodes = get_document_structure(root, "p1", db=db)
        assert nodes == [{"node_id": "0000", "title": "Root"}]
        assert get_section_content(root, "p1", "0000", db=db) == LEGACY_RAW.strip()

    def test_pipeline_accepts_precomputed_section_texts(self):
        import inspect

        from drbrain.extractor.concept.pipeline import build_graph_from_tree

        parameters = inspect.signature(build_graph_from_tree).parameters
        assert "section_texts" in parameters
