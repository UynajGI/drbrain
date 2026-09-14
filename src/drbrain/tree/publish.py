"""Immutable generation publication for the unified store (plan T37).

The publish order is fixed and recoverable:

1. write everything into ``generations/.staging-<gen>`` (SQLite backup, a copy
   of the shared Zvec directory, the manifest);
2. verify the staged copy is internally consistent (ready node-vector
   metadata rows must match what the copied ANN index actually holds, and the
   watermarks must describe the copied files, not live state);
3. rename the staging directory into place, then write the active pointer
   **last**.

A crash therefore leaves either the previous generation active (and an
ignored ``.staging-*`` directory) or a complete new one; readers only ever
resolve a published generation, and ``staging``/``failed`` nodes are never
part of the manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from loguru import logger

MANIFEST_NAME = "manifest.json"
ACTIVE_POINTER_NAME = "active.json"
GENERATIONS_DIR_NAME = "generations"
SNAPSHOT_NAME = "tree.sqlite3"
VECTORS_DIR_NAME = "zvec"
STAGING_PREFIX = ".staging-"


class PublicationError(RuntimeError):
    """The generation could not be published or verified."""


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def new_generation_id() -> str:
    return f"gen-{int(time.time() * 1000):013d}-{os.getpid()}"


def content_watermark(db) -> dict[str, Any]:
    """One entry per document revision: identity plus canonical hash."""
    rows = db.conn.execute(
        "SELECT local_id, revision, canonical_hash FROM document_revisions "
        "WHERE state = 'ready' ORDER BY local_id, revision"
    ).fetchall()
    payload = [f"{row[0]}@{row[1]}:{row[2]}" for row in rows]
    return {
        "documents": len(rows),
        "digest": _sha256("|".join(payload)),
    }


def node_watermark(db, *, states: Iterable[str] = ("ready",)) -> dict[str, Any]:
    states = tuple(states)
    placeholders = ",".join("?" * len(states))
    rows = db.conn.execute(
        f"SELECT node_id, revision, fingerprint FROM tree_nodes "
        f"WHERE state IN ({placeholders}) ORDER BY node_id",
        states,
    ).fetchall()
    payload = [f"{row[0]}@{row[1]}:{row[2]}" for row in rows]
    return {"nodes": len(rows), "digest": _sha256("|".join(payload))}


def vector_watermark(db, *, profile_id: str | None = None) -> dict[str, Any]:
    sql = (
        "SELECT node_id, node_revision, profile_id, content_hash FROM node_vectors "
        "WHERE state = 'ready'"
    )
    params: tuple = ()
    if profile_id is not None:
        sql += " AND profile_id = ?"
        params = (profile_id,)
    sql += " ORDER BY node_id"
    rows = db.conn.execute(sql, params).fetchall()
    payload = [f"{row[0]}@{row[1]}:{row[2]}:{row[3]}" for row in rows]
    dimensions = {
        int(value)
        for value in (
            row[0]
            for row in db.conn.execute(
                "SELECT DISTINCT dimension FROM node_vectors WHERE state = 'ready'"
                + (" AND profile_id = ?" if profile_id else ""),
                params,
            ).fetchall()
        )
    }
    return {
        "vectors": len(rows),
        "dimensions": sorted(dimensions),
        "digest": _sha256("|".join(payload)),
    }


def compute_watermarks(db, *, profile_id: str | None = None) -> dict[str, Any]:
    return {
        "schema": 1,
        "content": content_watermark(db),
        "nodes": node_watermark(db),
        "vectors": vector_watermark(db, profile_id=profile_id),
        "profile_id": profile_id or "",
    }


def _stage_database(source: sqlite3.Connection, target: Path) -> None:
    """Write a standalone, consistent copy of the live database.

    ``VACUUM INTO`` is used instead of ``Connection.backup``: it never blocks
    behind an open reader/writer transaction (backup waits for the source
    connection to become idle, which can deadlock a long-lived pipeline
    connection) and it compacts the snapshot in one step.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    if source.in_transaction:
        source.commit()
    try:
        source.execute("VACUUM INTO ?", (str(target),))
    except sqlite3.Error as exc:
        raise PublicationError(f"cannot stage the database snapshot: {exc}") from exc


def _copy_vectors(source_dir: Path | None, target: Path) -> dict[str, Any]:
    if source_dir is None or not source_dir.is_dir():
        return {"copied": False, "count": -1}
    shutil.copytree(source_dir, target)
    return {"copied": True, "count": -1}


def _count_vectors(path: Path) -> int:
    try:
        import zvec

        collection = zvec.open(str(path))
    except Exception:  # noqa: BLE001 - a missing/older index is a state, not a crash
        return -1
    try:
        stats = getattr(collection, "stats", None)
        if callable(stats):
            try:
                stats = stats()
            except TypeError:
                pass
        for attribute in ("doc_count", "count", "num_docs"):
            if stats is not None and hasattr(stats, attribute):
                return int(getattr(stats, attribute))
        return -1
    finally:
        collection.close()


def publish_tree_generation(
    db,
    storage_dir: str | Path,
    *,
    profile_id: str | None = None,
    vector_dir: str | Path | None = None,
    generation: str | None = None,
) -> dict[str, Any]:
    """Publish a new generation and atomically move the active pointer."""
    root = Path(storage_dir).resolve()
    generations = root / GENERATIONS_DIR_NAME
    generations.mkdir(parents=True, exist_ok=True)
    generation = generation or new_generation_id()
    if not str(generation).startswith("gen-"):
        raise PublicationError("generation ids must start with 'gen-'")
    staging = generations / f"{STAGING_PREFIX}{generation}"
    target = generations / generation
    if target.exists():
        raise PublicationError(f"generation {generation} already exists")
    if staging.exists():
        raise PublicationError(f"staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    published = False
    try:
        _stage_database(db.conn, staging / SNAPSHOT_NAME)
        vectors_source = Path(vector_dir).resolve() if vector_dir else None
        _copy_vectors(vectors_source, staging / VECTORS_DIR_NAME)
        snapshot = sqlite3.connect(str(staging / SNAPSHOT_NAME))
        try:
            watermarks = compute_watermarks_offline(snapshot, profile_id=profile_id)
        finally:
            snapshot.close()
        vector_count = -1
        if (staging / VECTORS_DIR_NAME).exists():
            vector_count = _count_vectors(staging / VECTORS_DIR_NAME)
            expected = int(watermarks["vectors"]["vectors"])
            if vector_count >= 0 and vector_count != expected:
                raise PublicationError(
                    f"staged ANN holds {vector_count} vectors but metadata lists {expected}; "
                    "refusing to publish an inconsistent generation"
                )
        manifest = {
            "schema": 1,
            "generation": generation,
            "created_at": time.time(),
            "profile_id": profile_id or "",
            "watermarks": watermarks,
            "vector_count": vector_count,
            "fingerprint": _sha256(
                json.dumps(
                    {
                        "watermarks": watermarks,
                        "vector_count": vector_count,
                        "profile_id": profile_id or "",
                    },
                    sort_keys=True,
                )
            ),
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(staging, target)
        _write_pointer(root, generation)
        published = True
        return {
            "generation": generation,
            "fingerprint": manifest["fingerprint"],
            "watermarks": watermarks,
            "vector_count": vector_count,
            "published": True,
        }
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def compute_watermarks_offline(conn: sqlite3.Connection, *, profile_id: str | None) -> dict[str, Any]:
    """Watermarks computed from a snapshot file rather than a live Database."""

    def rows(sql: str, params: tuple = ()) -> list:
        return conn.execute(sql, params).fetchall()

    revision_rows = rows(
        "SELECT local_id, revision, canonical_hash FROM document_revisions "
        "WHERE state = 'ready' ORDER BY local_id, revision"
    )
    node_rows = rows(
        "SELECT node_id, revision, fingerprint FROM tree_nodes WHERE state = 'ready' ORDER BY node_id"
    )
    vector_sql = (
        "SELECT node_id, node_revision, profile_id, content_hash, dimension FROM node_vectors "
        "WHERE state = 'ready'"
    )
    params: tuple = ()
    if profile_id:
        vector_sql += " AND profile_id = ?"
        params = (profile_id,)
    vector_sql += " ORDER BY node_id"
    vector_rows = rows(vector_sql, params)
    return {
        "schema": 1,
        "content": {
            "documents": len(revision_rows),
            "digest": _sha256("|".join(f"{r[0]}@{r[1]}:{r[2]}" for r in revision_rows)),
        },
        "nodes": {
            "nodes": len(node_rows),
            "digest": _sha256("|".join(f"{r[0]}@{r[1]}:{r[2]}" for r in node_rows)),
        },
        "vectors": {
            "vectors": len(vector_rows),
            "dimensions": sorted({int(r[4]) for r in vector_rows if r[4]}),
            "digest": _sha256("|".join(f"{r[0]}@{r[1]}:{r[2]}:{r[3]}" for r in vector_rows)),
        },
        "profile_id": profile_id or "",
    }


def _write_pointer(root: Path, generation: str) -> None:
    payload = {"generation": generation, "updated_at": time.time()}
    target = root / ACTIVE_POINTER_NAME
    handle = tempfile.NamedTemporaryFile(
        "w", dir=root, prefix=".active-", suffix=".json", delete=False, encoding="utf-8"
    )
    try:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    os.replace(handle.name, target)


def get_active_tree_generation(storage_dir: str | Path) -> str | None:
    pointer = Path(storage_dir).resolve() / ACTIVE_POINTER_NAME
    if not pointer.is_file():
        return None
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("[tree] unreadable active pointer at {}", pointer)
        return None
    generation = str(payload.get("generation") or "")
    return generation or None


def resolve_tree_generation(storage_dir: str | Path, generation: str) -> dict[str, Any]:
    """Resolve a *published* generation; staging directories are invisible."""
    if generation.startswith(STAGING_PREFIX):
        raise PublicationError(f"{generation!r} is a staging directory, not a publication")
    root = Path(storage_dir).resolve() / GENERATIONS_DIR_NAME / generation
    if not root.is_dir():
        raise PublicationError(f"unknown generation {generation!r}")
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise PublicationError(f"generation {generation!r} has no manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PublicationError(f"unreadable manifest for {generation!r}") from exc
    if not isinstance(manifest, dict) or manifest.get("generation") != generation:
        raise PublicationError(f"manifest does not describe generation {generation!r}")
    snapshot = root / SNAPSHOT_NAME
    if not snapshot.is_file():
        raise PublicationError(f"generation {generation!r} is missing its snapshot")
    return {
        "generation": generation,
        "path": str(root),
        "snapshot": str(snapshot),
        "vectors": str(root / VECTORS_DIR_NAME),
        "manifest": manifest,
    }


def verify_tree_generation(storage_dir: str | Path, generation: str) -> dict[str, Any]:
    """Re-check a published generation against the live files it contains."""
    resolved = resolve_tree_generation(storage_dir, generation)
    manifest = resolved["manifest"]
    conn = sqlite3.connect(resolved["snapshot"])
    try:
        watermarks = compute_watermarks_offline(conn, profile_id=manifest.get("profile_id") or None)
    finally:
        conn.close()
    expected = manifest.get("watermarks") or {}
    mismatches: list[str] = []
    for section in ("content", "nodes", "vectors"):
        if watermarks.get(section, {}).get("digest") != (expected.get(section) or {}).get("digest"):
            mismatches.append(section)
    vector_count = -1
    vectors_path = Path(resolved["vectors"])
    if vectors_path.exists():
        vector_count = _count_vectors(vectors_path)
        if vector_count >= 0 and vector_count != int(watermarks["vectors"]["vectors"]):
            mismatches.append("vector_count")
    fingerprint = _sha256(
        json.dumps(
            {
                "watermarks": watermarks,
                "vector_count": vector_count,
                "profile_id": manifest.get("profile_id") or "",
            },
            sort_keys=True,
        )
    )
    return {
        "generation": generation,
        "ok": not mismatches,
        "mismatches": mismatches,
        "fingerprint": fingerprint,
        "manifest_fingerprint": manifest.get("fingerprint", ""),
    }
