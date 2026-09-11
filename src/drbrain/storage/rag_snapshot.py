"""SQLite snapshot copying and complete logical fingerprints for RAG publication."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


def content_fingerprint(conn: sqlite3.Connection) -> str:
    """Hash all ordinary tables, including vector bytes, under one read transaction.

    Called during publication, not on every indexed query. FTS/vec virtual
    tables and their shadows are derived indexes; the ordinary source tables
    and their schemas are the versioned semantic data.
    """
    digest = hashlib.sha256()
    tables = [
        row[1]
        for row in conn.execute("PRAGMA table_list")
        if row[2] == "table" and not row[1].startswith("sqlite_")
    ]
    for name in sorted(tables):
        quoted = '"' + name.replace('"', '""') + '"'
        columns = conn.execute(f"PRAGMA table_info({quoted})").fetchall()
        digest.update(json.dumps([name, columns], default=str).encode())
        # Primary keys give stable order across VACUUM/backup. Rowid covers
        # older schemas without a declared key.
        keys = sorted((column[5], column[1]) for column in columns if column[5])
        order = ",".join('"' + key.replace('"', '""') + '"' for _, key in keys) or "rowid"
        for row in conn.execute(f"SELECT * FROM {quoted} ORDER BY {order}"):
            for value in row:
                data = value if isinstance(value, bytes) else json.dumps(value).encode()
                digest.update(b"b" if isinstance(value, bytes) else b"j")
                digest.update(len(data).to_bytes(8, "big"))
                digest.update(data)
    return "sql-" + digest.hexdigest()


def copy_snapshot(source: Path, destination: Path) -> dict[str, Any]:
    """SQLite backup includes committed WAL data and yields a standalone database."""
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as reader:
        with sqlite3.connect(destination) as writer:
            reader.backup(writer)
    with sqlite3.connect(destination.as_uri() + "?mode=ro", uri=True) as snapshot:
        return {
            "content_fingerprint": content_fingerprint(snapshot),
            "node_count": snapshot.execute("SELECT COUNT(*) FROM node_texts").fetchone()[0],
        }
