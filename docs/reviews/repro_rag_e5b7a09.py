"""Additional review contracts for the merged e5b7a09 handoff.

Synthetic data only. Reuses the earlier review's isolated CLI fixture.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "earlier_rag_review", Path(__file__).with_name("repro_rag_48a3c6f.py")
)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
indexed_root = base.indexed_root


def test_status_accepts_ready_unified_routes_when_tree_is_disabled(indexed_root):
    base.configure(indexed_root, retrievers=["bm25", "vector"])
    found = base.helpers._invoke(indexed_root, "search", "p1body0", "--json")
    legs = json.loads(found.stdout)["legs"]
    assert all(leg["status"] in {"ok", "empty"} for leg in legs), legs
    checked = base.helpers._invoke(indexed_root, "index", "status", "--json")
    report = json.loads(checked.stdout)
    assert report["states"]["retrievable"]["ready"] is True, {
        "query_legs": legs,
        "readiness": report["states"]["retrievable"],
    }


def test_fused_tree_evidence_preserves_the_actual_read_range(indexed_root, monkeypatch):
    from drbrain.rag.sql_retrie import retrieve_documents_sql
    from drbrain.storage.database import Database
    from drbrain.tree.navigator import ScriptedPlanner

    cfg = base.tree_config(indexed_root)
    db = Database(cfg.db.path)
    try:
        node_id, text = db.conn.execute(
            "SELECT n.node_id, b.text FROM tree_nodes n "
            "JOIN content_blocks b ON b.block_id=n.block_id "
            "WHERE n.kind='leaf' AND LENGTH(b.text)>40 LIMIT 1"
        ).fetchone()
        planner = ScriptedPlanner(
            [
                {"action": "read", "node_id": node_id, "char_start": 7, "char_end": 27},
                {"action": "finish", "reason": "done"},
            ]
        )
        monkeypatch.setattr(
            "drbrain.tree.leg.resolve_navigation_planner", lambda _cfg: (planner, "scripted")
        )
        rows = retrieve_documents_sql(cfg, db, "p1body0", top_k=5)
        assert rows, "The planner must have read a real leaf before checking fusion."
        row = next(row for row in rows if row["node_id"] == node_id)
        assert (row["char_start"], row["char_end"], row["text"]) == (7, 27, text[7:27]), {
            "requested_range": [7, 27],
            "returned_range": [row["char_start"], row["char_end"]],
            "returned_text_length": len(row["text"]),
        }
    finally:
        db.close()


def test_bm25_paper_scope_is_applied_before_candidate_limit(tmp_path):
    from drbrain.rag.sql_retrie import _unified_bm25_entries
    from drbrain.services.canonical_content import write_canonical_content
    from drbrain.storage.database import Database

    db = Database(tmp_path / "scope.sqlite")
    try:
        for paper_id in ("outside", "inside"):
            db.insert_paper(paper_id, paper_id, 2024, "uploaded")
            result = write_canonical_content(
                db,
                paper_id,
                "scopedneedle has an identical passage in both documents.",
                media_type="md",
                parser="review",
            )
            assert result["ok"]
        ranked = db.search_content('"scopedneedle"', limit=10)
        assert {hit["local_id"] for hit in ranked} == {"inside", "outside"}
        selected_paper = next(
            hit["local_id"] for hit in ranked if hit["local_id"] != ranked[0]["local_id"]
        )
        entries = _unified_bm25_entries(db, "scopedneedle", 1, {selected_paper})
        assert entries, {"scope": selected_paper, "global_first": ranked[0]["local_id"]}
    finally:
        db.close()


def test_unpublished_content_is_not_labelled_as_the_published_generation(indexed_root):
    from drbrain.rag.sql_retrie import retrieve_documents_sql
    from drbrain.services.canonical_content import write_canonical_content
    from drbrain.storage.database import Database
    from drbrain.tree.leg import active_tree_generation
    from drbrain.tree.publish import resolve_tree_generation
    from drbrain.tree.reading import ReadOnlyTreeStore

    cfg = base.tree_config(indexed_root)
    cfg.llamaindex.retrievers = ["bm25"]
    generation = active_tree_generation(cfg)
    resolved = resolve_tree_generation(indexed_root / "data/tree", generation)
    db = Database(cfg.db.path)
    try:
        db.insert_paper("latepaper", "Added after publication", 2024, "uploaded")
        written = write_canonical_content(
            db,
            "latepaper",
            "postpublishuniqueterm was added after the active publication.",
            media_type="md",
            parser="review",
        )
        assert written["ok"]
        with ReadOnlyTreeStore(resolved["snapshot"]) as snapshot:
            assert (
                snapshot.conn.execute(
                    "SELECT COUNT(*) FROM tree_nodes WHERE local_id='latepaper'"
                ).fetchone()[0]
                == 0
            )
        rows = retrieve_documents_sql(cfg, db, "postpublishuniqueterm", top_k=5)
        assert not any(row["paper_id"] == "latepaper" for row in rows), {
            "active_generation": generation,
            "reported_generation": rows.result.generation,
            "papers": [row["paper_id"] for row in rows],
            "snapshot": rows.result.capabilities.get("snapshot"),
        }
    finally:
        db.close()
