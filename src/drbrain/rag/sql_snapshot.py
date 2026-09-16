"""Publish SQL corpora through the shared immutable index-generation lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from loguru import logger

from drbrain.rag.config import get_llamaindex_config
from drbrain.rag.status import RetrievalUnavailableError
from drbrain.storage.rag_snapshot import copy_snapshot

SNAPSHOT_FILE = "corpus.sqlite3"


def embedding_identity(cfg: Any) -> dict[str, Any]:
    embed = cfg.get("embed", {}) if isinstance(cfg, dict) else cfg.embed
    fields = ("provider", "model", "dim", "max_seq_length")
    return {
        key: embed.get(key) if isinstance(embed, dict) else getattr(embed, key, None)
        for key in fields
    }


def publish_sql_snapshot(cfg: Any) -> dict[str, Any]:
    from drbrain.rag import index_generations as indexer
    from drbrain.rag.sql_retrie import _default_rag_db
    from drbrain.rag.zvec_index import (
        INDEX_DIR_NAME,
        build_zvec_index,
        configured_vector_backend,
    )

    source = _default_rag_db(cfg).resolve()
    if not source.is_file():
        raise RetrievalUnavailableError("SQL corpus is missing; build the corpus before indexing")
    root = Path(get_llamaindex_config(cfg).storage_dir).resolve()
    previous = indexer.get_active_index_generation(cfg)
    generation = indexer._new_generation_id()
    staging = root / indexer.GENERATIONS_DIR_NAME / (".staging-" + generation)
    target = staging.with_name(generation)
    staging.mkdir(parents=True)
    published = False
    try:
        stats = copy_snapshot(source, staging / SNAPSHOT_FILE)
        vector_backend = configured_vector_backend(cfg)
        vector_stats: dict[str, Any] = {"backend": vector_backend, "count": 0, "dimension": 0}
        if vector_backend == "zvec":
            vector_stats = build_zvec_index(staging / SNAPSHOT_FILE, staging / INDEX_DIR_NAME)
        identity = embedding_identity(cfg)
        signature = hashlib.sha256(
            json.dumps(
                [stats["content_fingerprint"], identity, vector_stats], sort_keys=True
            ).encode()
        ).hexdigest()
        manifest = {
            "backend": "sql",
            "generation": generation,
            "embedding": identity,
            "vector_backend": vector_backend,
            "vector_index": INDEX_DIR_NAME
            if vector_backend == "zvec" and vector_stats["count"]
            else None,
            "vector_count": int(vector_stats["count"]),
            "vector_dimension": int(vector_stats["dimension"]),
            "signature": signature,
            **stats,
        }
        indexer._write_manifest(staging, manifest)
        os.chmod(staging / SNAPSHOT_FILE, 0o444)
        os.replace(staging, target)
        indexer._write_json_atomically(
            root / indexer.ACTIVE_POINTER_NAME, {"generation": generation}
        )
        published = True
        try:
            indexer._prune_inactive_generations(
                root, generation, protected_generations=indexer._referenced_generations(root)
            )
        except OSError:
            logger.warning("SQL snapshot published; old generation cleanup could not complete")
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if not published and target.exists():
            shutil.rmtree(target)
    return {
        "generation": generation,
        "previous_generation": previous,
        "nodes": stats["node_count"],
        "backend": "sql",
        "vector_backend": vector_backend,
        "vector_count": int(vector_stats["count"]),
        "vector_dimension": int(vector_stats["dimension"]),
        "signature": signature,
        "storage_dir": str(root),
    }


def resolve_sql_snapshot(cfg: Any, generation: str) -> Path:
    from drbrain.rag import index_generations as indexer

    if generation == indexer.LEGACY_INDEX_GENERATION:
        raise RetrievalUnavailableError("SQL working copies are not immutable legacy snapshots")
    root = indexer._storage_root_for_generation(get_llamaindex_config(cfg).storage_dir, generation)
    if root is None:
        raise RetrievalUnavailableError(f"unavailable SQL generation {generation!r}")
    try:
        manifest = json.loads((root / indexer.MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RetrievalUnavailableError("unreadable SQL snapshot manifest") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("backend") != "sql"
        or manifest.get("generation") != generation
    ):
        raise RetrievalUnavailableError("SQL snapshot identity mismatch")
    if manifest.get("embedding") != embedding_identity(cfg):
        raise RetrievalUnavailableError(
            "embedding configuration differs from the pinned SQL snapshot"
        )
    path = root / SNAPSHOT_FILE
    if not path.is_file() or path.is_symlink():
        raise RetrievalUnavailableError("SQL snapshot database is missing or unsafe")
    return path


def resolve_sql_vector_index(cfg: Any, generation: str) -> Path:
    """Resolve the ANN directory belonging to one immutable SQL generation."""

    from drbrain.rag import index_generations as indexer
    from drbrain.rag.zvec_index import configured_vector_backend

    if configured_vector_backend(cfg) != "zvec":
        raise RetrievalUnavailableError("Zvec vector index is not selected")
    if generation == indexer.LEGACY_INDEX_GENERATION:
        raise RetrievalUnavailableError("SQL working copies do not have an immutable Zvec index")
    root = indexer._storage_root_for_generation(get_llamaindex_config(cfg).storage_dir, generation)
    if root is None:
        raise RetrievalUnavailableError(f"unavailable SQL generation {generation!r}")
    try:
        manifest = json.loads((root / indexer.MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RetrievalUnavailableError("unreadable SQL snapshot manifest") from exc
    if manifest.get("vector_backend") != "zvec":
        raise RetrievalUnavailableError("SQL snapshot was published without a Zvec index")
    relative = manifest.get("vector_index")
    if not isinstance(relative, str) or not relative:
        raise RetrievalUnavailableError("SQL snapshot has no Zvec vector index")
    path = (root / relative).resolve()
    if path.parent != root.resolve() or not path.is_dir() or path.is_symlink():
        raise RetrievalUnavailableError("SQL snapshot Zvec index is missing or unsafe")
    return path


def sql_index_health(cfg: Any) -> dict[str, Any]:
    """Inspect snapshot identity/schema without loading models or changing data."""
    import sqlite3
    from contextlib import closing

    from drbrain.rag.index_generations import get_active_index_generation
    from drbrain.rag.zvec_index import configured_vector_backend

    li = get_llamaindex_config(cfg)
    report: dict[str, Any] = {
        "ready": False,
        "status": "unavailable",
        "backend": "sql",
        "storage_dir": str(li.storage_dir),
        "reasons": [],
        "checks": {},
    }
    vector_backend = configured_vector_backend(cfg)
    report["vector_backend"] = vector_backend
    if not li.enabled:
        report.update(status="disabled", reasons=["disabled"])
        return report
    generation = get_active_index_generation(cfg)
    report["generation"] = generation
    if generation is None:
        report["reasons"] = ["no_published_sql_snapshot"]
        return report
    try:
        path = resolve_sql_snapshot(cfg, generation)
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            conn.execute("SELECT node_key, text FROM node_texts LIMIT 1").fetchall()
            conn.execute("SELECT rowid FROM node_texts_fts LIMIT 1").fetchall()
        checks: dict[str, Any] = {"snapshot": {"loadable": True}}
        if vector_backend == "zvec":
            vector_path = resolve_sql_vector_index(cfg, generation)
            try:
                import zvec

                collection = zvec.open(str(vector_path))
                collection.close()
            except Exception as exc:  # noqa: BLE001 - health reports native errors
                raise RetrievalUnavailableError("Zvec vector index is not loadable") from exc
            try:
                vector_meta = json.loads(
                    (vector_path / "metadata.json").read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as exc:
                raise RetrievalUnavailableError("Zvec vector index metadata is unreadable") from exc
            checks["vector"] = {
                "loadable": True,
                "count": vector_meta.get("count", 0),
                "dimension": vector_meta.get("dimension", 0),
                "backend": "zvec",
            }
    except (OSError, ValueError, sqlite3.Error, RetrievalUnavailableError):
        report["reasons"] = ["sql_snapshot_unavailable_or_incompatible"]
        if vector_backend == "zvec":
            report["reasons"].append("zvec_vector_index_unavailable")
        return report
    report.update(ready=True, status="ready", checks=checks)
    return report
