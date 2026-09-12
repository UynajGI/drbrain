"""Write surface for the derived SQL RAG database.

The SQL retriever uses a deliberately small read-only database.  Keeping its
DDL and writes here makes the preparation step atomic and prevents pipeline
modules from each inventing a slightly different schema.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from typing import Any


def rebuild_rag_database(
    target: str | Path,
    *,
    node_rows: Iterable[tuple[str, str, str, str, str]],
    vector_rows: Iterable[tuple[str, str, bytes, str, str]],
    summary_rows: Iterable[tuple[str, str, str, str, int]],
    category_rows: Iterable[tuple[str, str]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, int | str]:
    """Atomically replace a derived RAG database from prepared rows."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temp = Path(temp_name)
    counts = {"nodes": 0, "vectors": 0, "summaries": 0, "categories": 0}
    try:
        with closing(sqlite3.connect(str(temp))) as conn:
            # The file is private until publication, so use a durable journal
            # rather than trading crash safety for a marginal build speedup.
            conn.execute("PRAGMA synchronous=FULL")
            conn.executescript(
                """
            CREATE TABLE node_texts (
                node_key TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                text TEXT NOT NULL,
                content_hash TEXT NOT NULL
            );
            CREATE INDEX idx_node_texts_paper ON node_texts(paper_id);
            CREATE TABLE tree_vectors (
                node_id TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                content_hash TEXT NOT NULL DEFAULT '',
                tree_layer TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX idx_tree_vectors_layer_paper ON tree_vectors(tree_layer, paper_id);
            CREATE TABLE tree_summaries (
                node_id TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL,
                summary_text TEXT NOT NULL DEFAULT '',
                source_node_ids TEXT NOT NULL DEFAULT '',
                tree_layer INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX idx_tree_summaries_paper ON tree_summaries(paper_id);
            CREATE TABLE paper_categories (
                paper_id TEXT PRIMARY KEY,
                categories TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE rag_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
                """
            )
            node_batch = list(node_rows)
            if node_batch:
                conn.executemany("INSERT INTO node_texts VALUES (?,?,?,?,?)", node_batch)
                counts["nodes"] = len(node_batch)
            vector_batch = list(vector_rows)
            if vector_batch:
                conn.executemany("INSERT INTO tree_vectors VALUES (?,?,?,?,?)", vector_batch)
                counts["vectors"] = len(vector_batch)
            summary_batch = list(summary_rows)
            if summary_batch:
                conn.executemany("INSERT INTO tree_summaries VALUES (?,?,?,?,?)", summary_batch)
                counts["summaries"] = len(summary_batch)
            category_batch = list(category_rows)
            if category_batch:
                conn.executemany("INSERT INTO paper_categories VALUES (?,?)", category_batch)
                counts["categories"] = len(category_batch)
            conn.execute(
                "CREATE VIRTUAL TABLE node_texts_fts USING fts5("
                "text, content='node_texts', content_rowid='rowid', tokenize='porter unicode61')"
            )
            conn.execute("INSERT INTO node_texts_fts(node_texts_fts) VALUES('rebuild')")
            for key, value in (metadata or {}).items():
                conn.execute(
                    "INSERT INTO rag_metadata(key, value) VALUES (?, ?)", (key, str(value))
                )
            conn.commit()
        with temp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp, target)
        try:
            directory_fd = os.open(str(target.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Directory fsync is not available on every supported platform;
            # the atomic replace above still protects readers from a partial
            # database.
            pass
    finally:
        if temp.exists():
            temp.unlink()
    return {**counts, "path": str(target)}


__all__ = ["rebuild_rag_database"]
