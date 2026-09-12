"""Materialize and publish the SQL RAG working copy in one operation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from drbrain.rag.sql_retrie import _default_rag_db
from drbrain.rag.sql_snapshot import publish_sql_snapshot
from drbrain.storage.database import Database
from drbrain.storage.node_projection import collect_tree_node_records
from drbrain.storage.paths import paper_dir
from drbrain.storage.rag_database import rebuild_rag_database


def _cfg_value(cfg: Any, section: str, key: str, default: Any = None) -> Any:
    section_value = getattr(cfg, section, None)
    if section_value is not None:
        value = getattr(section_value, key, None)
        if value is not None:
            return value
    if isinstance(cfg, dict):
        raw = cfg.get(section, {})
        if isinstance(raw, dict):
            return raw.get(key, default)
    return default


def prepare_sql_rag(
    cfg: Any,
    *,
    paper_ids: list[str] | None = None,
    publish: bool = True,
) -> dict[str, Any]:
    """Build ``drbrain_rag.db`` and optionally publish an immutable generation.

    A SQL snapshot is a complete corpus view.  Refusing a paper subset avoids
    silently publishing a generation that drops papers still present in the
    primary database; use the LlamaIndex incremental indexer for per-paper
    rebuilds and run this command again for a new SQL generation.
    """
    if paper_ids:
        raise ValueError("SQL RAG preparation is corpus-wide; omit --paper")
    main_path = _cfg_value(cfg, "db", "path", "data/drbrain.db")
    papers_root = Path(_cfg_value(cfg, "dirs", "papers", "data/papers"))
    db = Database(main_path)
    all_paper_ids: set[str] = set()
    try:
        # Pin one consistent primary-database view while copying vectors and
        # summaries; a concurrent embed/build commit must wait for the next
        # preparation run instead of producing a mixed generation.
        if not db.conn.in_transaction:
            db.conn.execute("BEGIN")
        papers = db.get_all_papers()
        node_rows: list[tuple[str, str, str, str, str]] = []
        eligible: set[str] = set()
        for paper in papers:
            pid = str(paper["local_id"])
            all_paper_ids.add(pid)
            pdir = paper_dir(papers_root, pid)
            records = collect_tree_node_records(pdir, paper_id=pid)
            if not records:
                db.upsert_paper_artifact(pid, "rag_text", "skipped", error="tree unavailable")
                continue
            eligible.add(pid)
            for row in records:
                text = str(row["text"])
                node_rows.append(
                    (
                        str(row["node_key"]),
                        pid,
                        str(row["node_id"]),
                        text,
                        hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                    )
                )

        # Derived rows must use the same eligible-paper scope as node_texts;
        # copying every historical vector when no tree is available would
        # create orphan retrieval candidates.
        allowed = eligible

        def scoped_rows(sql: str) -> list:
            if not allowed:
                return []
            placeholders = ",".join("?" * len(allowed))
            return db.conn.execute(
                f"{sql} WHERE paper_id IN ({placeholders})", tuple(sorted(allowed))
            ).fetchall()

        vector_rows = scoped_rows(
            "SELECT node_id, paper_id, embedding, content_hash, tree_layer FROM tree_vectors"
        )
        summary_rows = scoped_rows(
            "SELECT node_id, paper_id, summary_text, source_node_ids, tree_layer FROM tree_summaries"
        )
        category_rows = db.conn.execute(
            "SELECT local_id, categories FROM papers WHERE categories IS NOT NULL"
            + (
                " AND local_id IN (" + ",".join("?" * len(eligible)) + ")" if eligible else " AND 0"
            ),
            tuple(sorted(eligible)),
        ).fetchall()
        node_counts = Counter(row[1] for row in node_rows)
        if not eligible:
            # Never replace a previously usable snapshot with an empty one
            # merely because every current paper is waiting for PageIndex.
            db.commit()
            return {
                "nodes": 0,
                "vectors": 0,
                "summaries": 0,
                "categories": 0,
                "backend": "sql",
                "status": "degraded",
                "reason": "no eligible PageIndex trees",
            }
        metadata = {
            "source_db": str(main_path),
            "node_projection": "storage.node_projection.v1",
            "paper_count": len(eligible),
        }
        stats = rebuild_rag_database(
            _default_rag_db(cfg),
            node_rows=node_rows,
            vector_rows=vector_rows,
            summary_rows=summary_rows,
            category_rows=category_rows,
            metadata=metadata,
        )
        for pid in eligible:
            db.upsert_paper_artifact(
                pid,
                "rag_text",
                "ready",
                metadata_json=json.dumps({"nodes": node_counts.get(pid, 0)}),
            )
        db.commit()
    finally:
        db.close()

    if publish:
        if all_paper_ids:
            db = Database(main_path)
            try:
                for pid in all_paper_ids:
                    db.upsert_paper_artifact(pid, "rag_snapshot", "running")
                db.commit()
            finally:
                db.close()
        try:
            generation = publish_sql_snapshot(cfg)
        except Exception as exc:
            if all_paper_ids:
                db = Database(main_path)
                try:
                    for pid in all_paper_ids:
                        db.upsert_paper_artifact(pid, "rag_snapshot", "failed", error=str(exc))
                    db.commit()
                finally:
                    db.close()
            raise
        stats["generation"] = generation.get("generation", "")
        if all_paper_ids:
            db = Database(main_path)
            try:
                for pid in all_paper_ids:
                    if pid not in eligible:
                        db.upsert_paper_artifact(
                            pid,
                            "rag_snapshot",
                            "skipped",
                            error="tree unavailable for published snapshot",
                        )
                        continue
                    db.upsert_paper_artifact(
                        pid,
                        "rag_snapshot",
                        "ready",
                        metadata_json=json.dumps({"generation": stats["generation"]}),
                    )
                db.commit()
            finally:
                db.close()
    return stats


__all__ = ["prepare_sql_rag"]
