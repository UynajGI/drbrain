"""Tests for scripts/pipeline/merge_shards.py — unified-store tables (2026-09).

The shard-pipeline migration prerequisite: canonical content and published
leaves travel with the shard merge, shard-local regions stay behind, the FTS
index follows via the ``content_blocks`` triggers, and a re-run stays
idempotent (INSERT OR REPLACE on content-addressed keys).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from drbrain.services.canonical_content import write_canonical_content
from drbrain.storage.database import Database

REPO = Path(__file__).resolve().parents[1]


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "merge_shards_testee", REPO / "scripts" / "pipeline" / "merge_shards.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


merge_shards = _load_script_module()

TEXT = "# Methods\n\n" + " ".join(f"token{index}" for index in range(40)) + "\n"


def _shard_db(path: Path, local_ids: list[str]) -> Database:
    db = Database(path)
    for local_id in local_ids:
        db.insert_paper(local_id, "T", 2024, "uploaded")
        written = write_canonical_content(db, local_id, TEXT, media_type="md", parser="test")
        assert written["ok"], written
    db.commit()
    return db


def _leaf_count(db: Database) -> int:
    return int(db.conn.execute("SELECT COUNT(*) FROM tree_nodes WHERE kind = 'leaf'").fetchone()[0])


def test_unified_tables_travel_with_the_merge_and_regions_stay_behind(tmp_path):
    shard_a = _shard_db(tmp_path / "shard0.db", ["s0a", "s0b"])
    shard_b = _shard_db(tmp_path / "shard1.db", ["s1a"])
    leaves_a, leaves_b = _leaf_count(shard_a), _leaf_count(shard_b)
    assert leaves_a > 0 and leaves_b > 0
    # A shard-local region must never mix into the main tree.
    shard_a.conn.execute(
        "INSERT INTO tree_nodes (node_id, kind, state, local_id, content_hash) "
        "VALUES ('r-shard0', 'region', 'ready', 's0a', 'deadbeef')"
    )
    shard_a.commit()
    shard_a.close()
    shard_b.close()

    main = Database(tmp_path / "main.db")
    try:
        counts_a = merge_shards.merge_one(tmp_path / "shard0.db", main.conn)
        counts_b = merge_shards.merge_one(tmp_path / "shard1.db", main.conn)
        main.commit()

        assert counts_a["document_revisions"] == 2
        assert counts_a["content_blocks"] > 0
        assert counts_a["tree_nodes"] == leaves_a
        assert counts_b["tree_nodes"] == leaves_b

        papers = {row[0] for row in main.conn.execute("SELECT local_id FROM papers")}
        assert papers == {"s0a", "s0b", "s1a"}
        assert (
            main.conn.execute("SELECT COUNT(*) FROM tree_nodes WHERE kind = 'leaf'").fetchone()[0]
            == leaves_a + leaves_b
        )
        assert (
            main.conn.execute("SELECT COUNT(*) FROM tree_nodes WHERE kind = 'region'").fetchone()[0]
            == 0
        )

        # The FTS index follows the merged blocks via the schema triggers, so
        # merged content is immediately searchable on the main database.
        assert main.content_fts_status()["consistent"] is True
        hits = main.search_content('"token7"', limit=5)
        assert hits, "merged canonical blocks must be searchable"
        assert {hit["local_id"] for hit in hits} <= {"s0a", "s0b", "s1a"}

        # Re-running the merge is idempotent (content-addressed keys).
        blocks_before = main.conn.execute("SELECT COUNT(*) FROM content_blocks").fetchone()[0]
        merge_shards.merge_one(tmp_path / "shard0.db", main.conn)
        main.commit()
        assert (
            main.conn.execute("SELECT COUNT(*) FROM tree_nodes WHERE kind = 'leaf'").fetchone()[0]
            == leaves_a + leaves_b
        )
        assert (
            main.conn.execute("SELECT COUNT(*) FROM content_blocks").fetchone()[0] == blocks_before
        )
    finally:
        main.close()
