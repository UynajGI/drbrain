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

    source = _default_rag_db(cfg).resolve()
    if not source.is_file():
        raise RetrievalUnavailableError("SQL corpus is missing; build the corpus before rag index")
    root = Path(get_llamaindex_config(cfg).storage_dir).resolve()
    previous = indexer.get_active_index_generation(cfg)
    generation = indexer._new_generation_id()
    staging = root / indexer.GENERATIONS_DIR_NAME / (".staging-" + generation)
    target = staging.with_name(generation)
    staging.mkdir(parents=True)
    published = False
    try:
        stats = copy_snapshot(source, staging / SNAPSHOT_FILE)
        identity = embedding_identity(cfg)
        signature = hashlib.sha256(
            json.dumps([stats["content_fingerprint"], identity], sort_keys=True).encode()
        ).hexdigest()
        manifest = {
            "backend": "sql",
            "generation": generation,
            "embedding": identity,
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
    if not isinstance(manifest, dict) or manifest.get("backend") != "sql" or manifest.get("generation") != generation:
        raise RetrievalUnavailableError("SQL snapshot identity mismatch")
    if manifest.get("embedding") != embedding_identity(cfg):
        raise RetrievalUnavailableError(
            "embedding configuration differs from the pinned SQL snapshot"
        )
    path = root / SNAPSHOT_FILE
    if not path.is_file() or path.is_symlink():
        raise RetrievalUnavailableError("SQL snapshot database is missing or unsafe")
    return path


def sql_index_health(cfg: Any) -> dict[str, Any]:
    """Inspect snapshot identity/schema without loading models or changing data."""
    import sqlite3
    from drbrain.rag.index_generations import get_active_index_generation

    li = get_llamaindex_config(cfg)
    report = {"ready": False, "status": "unavailable", "backend": "sql",
              "storage_dir": str(li.storage_dir), "reasons": [], "checks": {}}
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
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
            conn.execute("SELECT node_key, text FROM node_texts LIMIT 1").fetchall()
            conn.execute("SELECT rowid FROM node_texts_fts LIMIT 1").fetchall()
    except (OSError, ValueError, sqlite3.Error, RetrievalUnavailableError):
        report["reasons"] = ["sql_snapshot_unavailable_or_incompatible"]
        return report
    report.update(ready=True, status="ready", checks={"snapshot": {"loadable": True}})
    return report
