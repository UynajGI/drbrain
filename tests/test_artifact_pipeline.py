"""Offline contracts for the resumable material-to-RAG pipeline."""

from __future__ import annotations

import base64
import json

import pytest

from drbrain.parser.material import extract_material
from drbrain.rag.preparation import prepare_sql_rag
from drbrain.storage.database import Database
from drbrain.storage.inbox import scan_materials
from drbrain.storage.node_projection import collect_tree_node_records


def test_paper_artifacts_are_idempotent_and_queryable(tmp_path):
    db = Database(tmp_path / "drbrain.db")
    try:
        db.insert_paper("p1", "Paper", 2024, "uploaded")
        db.upsert_paper_artifact("p1", "raw", "ready", fingerprint="a")
        db.upsert_paper_artifact("p1", "raw", "ready", fingerprint="b")
        db.commit()
        artifact = db.get_paper_artifact("p1", "raw")
        assert artifact is not None
        assert artifact["status"] == "ready"
        assert artifact["fingerprint"] == "b"
        assert artifact["attempts"] == 2
    finally:
        db.close()


def test_shared_projection_handles_line_num_and_inline_fallback(tmp_path):
    paper_dir = tmp_path / "p1"
    paper_dir.mkdir()
    (paper_dir / "raw.md").write_text("# Intro\nBody\n# Method\nDetails\n", encoding="utf-8")
    (paper_dir / "tree.json").write_text(
        json.dumps(
            {
                "structure": [
                    {"node_id": "1", "title": "Intro", "line_num": 1},
                    {"node_id": "2", "title": "Method", "line_num": 3},
                    {"node_id": "3", "title": "Inline", "line_num": 0, "text": "Fallback"},
                ]
            }
        ),
        encoding="utf-8",
    )
    records = collect_tree_node_records(paper_dir, paper_id="p1")
    assert [row["node_id"] for row in records] == ["1", "2", "3"]
    assert "Body" in records[0]["text"]
    assert "Details" in records[1]["text"]
    assert records[2]["text"].endswith("Fallback")


def test_prepare_sql_rag_builds_and_publishes_atomic_copy(tmp_path):
    db_path = tmp_path / "data" / "drbrain.db"
    papers = tmp_path / "data" / "papers"
    paper_dir = papers / "p1"
    paper_dir.mkdir(parents=True)
    (paper_dir / "raw.md").write_text("# Intro\nBody", encoding="utf-8")
    (paper_dir / "tree.json").write_text(
        json.dumps({"structure": [{"node_id": "1", "title": "Intro", "line_num": 1}]}),
        encoding="utf-8",
    )
    db = Database(db_path)
    db.insert_paper("p1", "Paper", 2024, "uploaded")
    db.commit()
    db.close()

    cfg = {
        "db": {"path": str(db_path)},
        "dirs": {"papers": str(papers)},
        "llamaindex": {"rag_engine": "sql", "storage_dir": str(tmp_path / "data" / "rag")},
    }
    stats = prepare_sql_rag(cfg)
    assert stats["nodes"] == 1
    assert stats["generation"]
    assert (tmp_path / "data" / "drbrain_rag.db").is_file()


def test_text_materials_use_native_adapter_and_inbox_filter(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    source = inbox / "note.md"
    source.write_text("# A useful note\nDOI: 10.1234/example\n", encoding="utf-8")
    (inbox / "ignore.bin").write_bytes(b"binary")

    assert scan_materials(inbox) == [source]
    parsed = extract_material(source, {})
    assert parsed.title == "A useful note"
    assert parsed.doi == "10.1234/example"
    assert parsed.raw_md.startswith("# A useful note")

    source.write_text(
        "# Note\nA date-like value 2024.12345\narXiv:2401.12345v2\n", encoding="utf-8"
    )
    arxiv = extract_material(source, {})
    assert arxiv.arxiv == "2401.12345"


def test_sql_prepare_rejects_partial_snapshot(tmp_path):
    with pytest.raises(ValueError, match="corpus-wide"):
        prepare_sql_rag(
            {"db": {"path": str(tmp_path / "db.sqlite")}},
            paper_ids=["p1"],
            publish=False,
        )


def test_sql_prepare_does_not_publish_empty_snapshot(tmp_path):
    db_path = tmp_path / "db.sqlite"
    papers = tmp_path / "papers"
    db = Database(db_path)
    db.insert_paper("p1", "Waiting for tree", 2024, "uploaded")
    db.commit()
    db.close()
    cfg = {
        "db": {"path": str(db_path)},
        "dirs": {"papers": str(papers)},
        "llamaindex": {"rag_engine": "sql", "storage_dir": str(tmp_path / "rag")},
    }
    stats = prepare_sql_rag(cfg)
    assert stats["status"] == "degraded"
    assert not (tmp_path / "drbrain_rag.db").exists()


def test_raptor_artifact_cleanup_prevents_duplicate_layers(tmp_path):
    db = Database(tmp_path / "db.sqlite")
    try:
        db.insert_paper("p1", "Paper", 2024, "uploaded")
        db.conn.execute(
            "INSERT INTO tree_vectors(node_id, paper_id, embedding, tree_layer) "
            "VALUES (?, ?, ?, ?)",
            ("raptor_p1_L1_old", "p1", b"\x00\x00\x80?", "raptor_L1"),
        )
        db.conn.execute(
            "INSERT INTO tree_summaries(node_id, paper_id, summary_text, tree_layer) "
            "VALUES (?, ?, ?, ?)",
            ("raptor_p1_L1_old", "p1", "old summary", 1),
        )
        removed = db.clear_raptor_artifacts("p1")
        db.commit()
        assert removed == 1
        assert db.conn.execute("SELECT COUNT(*) FROM tree_vectors").fetchone()[0] == 0
        assert db.conn.execute("SELECT COUNT(*) FROM tree_summaries").fetchone()[0] == 0
    finally:
        db.close()


def test_raptor_replacement_accepts_summary_only_stage(tmp_path):
    db = Database(tmp_path / "db.sqlite")
    try:
        db.insert_paper("p1", "Paper", 2024, "uploaded")
        db.conn.execute(
            "INSERT INTO tree_vectors(node_id, paper_id, embedding, tree_layer) "
            "VALUES (?, ?, ?, ?)",
            ("raptor_p1_L1_old", "p1", b"old", "raptor_L1"),
        )
        db.conn.execute(
            "INSERT INTO tree_summaries(node_id, paper_id, summary_text, tree_layer) "
            "VALUES (?, ?, ?, ?)",
            ("raptor_p1_L1_old", "p1", "old summary", 1),
        )
        assert db.replace_raptor_artifacts(
            "p1",
            [
                {
                    "type": "summary",
                    "node_id": "raptor_p1_L1_new",
                    "paper_id": "p1",
                    "summary_text": "new summary",
                    "source_node_ids": ["p1:1"],
                    "tree_layer": 1,
                }
            ],
        ) == {"summaries": 1, "vectors": 0}
        assert db.conn.execute("SELECT node_id FROM tree_summaries").fetchone()[0] == (
            "raptor_p1_L1_new"
        )

        result = db.replace_raptor_artifacts(
            "p1",
            [
                {
                    "type": "summary",
                    "node_id": "raptor_p1_L1_new",
                    "paper_id": "p1",
                    "summary_text": "new summary",
                    "source_node_ids": ["p1:1"],
                    "tree_layer": 1,
                },
                {
                    "type": "vector",
                    "node_id": "raptor_p1_L1_new",
                    "paper_id": "p1",
                    "embedding_blob_b64": base64.b64encode(b"new").decode("ascii"),
                    "content_hash": "hash",
                    "tree_layer": "raptor_L1",
                },
            ],
        )
        db.commit()
        assert result == {"summaries": 1, "vectors": 1}
        assert db.conn.execute("SELECT node_id FROM tree_summaries").fetchone()[0] == (
            "raptor_p1_L1_new"
        )
    finally:
        db.close()
