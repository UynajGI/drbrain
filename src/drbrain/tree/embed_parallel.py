"""Parallel vectors stage for the unified tree: spool first, then one load (T24/T45).

The serial embed→write loop owns both the model and the store in one process,
so compute and the single-writer load serialize.  This module splits the stage:

* one worker process per device (``cuda:N`` and/or ``cpu``) enumerates the same
  pending predicate the serial loop uses, embeds its slice and writes npz
  shards — workers never touch the shared store or the database;
* a single loader writes the shards back through the serial path's contract
  (staging → ``store.upsert`` → ready) in large batches.

The spool is resumable (each worker continues from its first missing part) and
the load is idempotent (a part is deleted only after its batch commits ready).
A spool directory is bound to one pending set: a manifest records the worker
count, part size and pending digest, and a mismatch fails closed instead of
loading a mix of slices.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

SPOOL_DIR_NAME = "embed-spool"
PART_SIZE = 4096
EMBED_CHUNK = 1024
LOAD_BATCH = 20000

__all__ = [
    "SPOOL_DIR_NAME",
    "parallel_vectors_stage",
    "parse_embed_devices",
    "pending_rows",
]


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def parse_embed_devices(embed_cfg: Any) -> list[str]:
    """Device list for the parallel stage; ``[]`` keeps the serial path.

    Only the local provider can be sharded.  ``device: cuda:N`` (when it names
    one GPU) plus ``extra_gpus`` give the GPU workers; ``cpu_workers`` appends
    that many CPU workers.  Fewer than two devices never leaves the serial path.
    """
    if str(_cfg_get(embed_cfg, "provider", "") or "").strip().lower() != "local":
        return []
    devices: list[str] = []
    device = str(_cfg_get(embed_cfg, "device", "") or "").strip()
    if device.startswith("cuda:"):
        devices.append(device)
    for gpu in _cfg_get(embed_cfg, "extra_gpus", None) or []:
        candidate = f"cuda:{int(gpu)}"
        if candidate not in devices:
            devices.append(candidate)
    devices.extend(["cpu"] * max(0, int(_cfg_get(embed_cfg, "cpu_workers", 0) or 0)))
    return devices if len(devices) > 1 else []


def pending_rows(db, profile_id: str, *, force: bool = False) -> list[dict]:
    """Ready nodes whose shared-store entry is missing or stale (stage predicate).

    ``force=True`` mirrors the serial loop: every ready node is pending again.
    """
    from drbrain.tree.prepare import ready_node_rows
    from drbrain.tree.vector_store import needs_write_meta

    rows = ready_node_rows(db)
    if force:
        return rows
    stored: dict[str, dict] = {}
    for nid, state, revision, content_hash, profile in db.conn.execute(
        "SELECT node_id, state, node_revision, content_hash, profile_id FROM node_vectors"
    ):
        stored[str(nid)] = {
            "state": state,
            "node_revision": revision,
            "content_hash": content_hash,
            "profile_id": profile,
        }
    return [
        row
        for row in rows
        if needs_write_meta(
            stored.get(row["node_id"]),
            node_revision=row["revision"],
            content_hash=row["content_hash"],
            profile_id=profile_id,
        )
    ]


def _pending_digest(rows: Sequence[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row['node_id']}:{row['revision']}:{row['content_hash']}\n".encode())
    return digest.hexdigest()[:24]


def _serializable_embed_cfg(embed_cfg: Any) -> dict[str, Any]:
    payload = dataclasses.asdict(embed_cfg)
    payload.pop("api_key", None)
    return payload


def _spawn_workers(
    *,
    devices: Sequence[str],
    db_path: str | Path,
    cfg_path: Path,
    shard_root: Path,
    part_size: int,
    embed_chunk: int,
    force: bool = False,
) -> list[tuple[str, int]]:
    env = dict(os.environ)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    procs: list[tuple[str, subprocess.Popen]] = []
    for index, device in enumerate(devices):
        child_env = dict(env)
        if device == "cpu":
            # One BLAS thread per process keeps a many-worker CPU fleet
            # (``cpu_workers: 64`` on a 72-core host) from oversubscribing
            # the cores; torch reads these at model load.
            child_env["OMP_NUM_THREADS"] = "1"
            child_env["MKL_NUM_THREADS"] = "1"
            child_env["OPENBLAS_NUM_THREADS"] = "1"
        cmd = [
            sys.executable,
            "-m",
            "drbrain.tree.embed_parallel",
            "_worker",
            "--db-path",
            str(db_path),
            "--embed-cfg-json",
            str(cfg_path),
            "--shard-dir",
            str(shard_root),
            "--worker-index",
            str(index),
            "--worker-count",
            str(len(devices)),
            "--device",
            device,
            "--part-size",
            str(part_size),
            "--embed-chunk",
            str(embed_chunk),
        ]
        if force:
            cmd.append("--force")
        procs.append((device, subprocess.Popen(cmd, env=child_env)))
    return [(device, proc.wait()) for device, proc in procs]


def _load_spool(
    db,
    store,
    profile_id: str,
    shard_root: Path,
    *,
    batch_size: int,
    log: Callable[[str], None],
) -> tuple[int, int]:
    """Write every shard part through the serial write contract.

    Returns ``(loaded, parts)``.  A part is deleted only after its batch has
    committed both the vector and the ``ready`` metadata.
    """
    from drbrain.tree.vector_store import VectorEntry

    parts = sorted(shard_root.glob("shard-*/*.npz"))
    batch: list[VectorEntry] = []
    batch_parts: list[Path] = []
    loaded = 0
    consumed_parts = 0

    def upsert_meta(entries: Sequence[VectorEntry], state: str) -> None:
        with db.transaction():
            for entry in entries:
                db.upsert_node_vector(
                    entry.node_id,
                    node_revision=entry.node_revision,
                    kind=entry.kind,
                    profile_id=entry.profile_id,
                    content_hash=entry.content_hash,
                    dimension=len(entry.vector),
                    local_id=entry.local_id,
                    layer=entry.layer,
                    state=state,
                )

    def flush() -> None:
        nonlocal batch, batch_parts, loaded, consumed_parts
        if not batch:
            return
        entries = batch
        upsert_meta(entries, "staging")
        store.upsert(entries)
        upsert_meta(entries, "ready")
        for part in batch_parts:
            part.unlink()
        loaded += len(entries)
        consumed_parts += len(batch_parts)
        log(f"[tree] vector loader: {loaded} spooled vectors written")
        batch = []
        batch_parts = []

    for part in parts:
        with np.load(part) as data:
            profiles = {str(item) for item in data["profile_ids"]}
            if profiles and profiles != {profile_id}:
                raise RuntimeError(
                    f"spool part {part} was written for profile {sorted(profiles)}, "
                    f"expected {profile_id}"
                )
            for index in range(len(data["node_ids"])):
                batch.append(
                    VectorEntry(
                        node_id=str(data["node_ids"][index]),
                        node_revision=int(data["revisions"][index]),
                        kind=str(data["kinds"][index]),
                        local_id=str(data["local_ids"][index]),
                        layer=int(data["layers"][index]),
                        content_hash=str(data["content_hashes"][index]),
                        profile_id=str(data["profile_ids"][index]),
                        vector=tuple(float(value) for value in data["vectors"][index]),
                    )
                )
        batch_parts.append(part)
        if len(batch) >= int(batch_size):
            flush()
    flush()
    return loaded, consumed_parts


def parallel_vectors_stage(
    db,
    store,
    profile: Any,
    embed_cfg: Any,
    *,
    devices: Sequence[str],
    spool_dir: str | Path | None = None,
    part_size: int = PART_SIZE,
    embed_chunk: int = EMBED_CHUNK,
    load_batch: int = LOAD_BATCH,
    force: bool = False,
    spawn: Callable[..., list[tuple[str, int]]] | None = None,
    log: Callable[[str], None] = logger.info,
) -> dict[str, Any]:
    """Spool the pending nodes on ``devices``, then load them in one writer."""
    profile_id = profile.profile_id()
    rows = pending_rows(db, profile_id, force=force)
    if not rows:
        return {"status": "ok", "nodes": 0, "embedded": 0, "pending": 0, "empty": 0}

    shard_root = Path(spool_dir) if spool_dir else Path(store.path).parent / SPOOL_DIR_NAME
    shard_root.mkdir(parents=True, exist_ok=True)
    manifest_path = shard_root / "manifest.json"
    manifest = {
        "worker_count": len(devices),
        "part_size": int(part_size),
        "force": bool(force),
        "profile_id": profile_id,
        "pending_digest": _pending_digest(rows),
        "pending_count": len(rows),
    }
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = None
        if previous != manifest:
            return {
                "status": "failed",
                "error": (
                    f"spool directory {shard_root} belongs to a different pending set; "
                    "remove it and rerun"
                ),
                "nodes": len(rows),
                "embedded": 0,
                "pending": len(rows),
            }
    else:
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=1), encoding="utf-8")
    cfg_path = shard_root / "embed-cfg.json"
    cfg_path.write_text(
        json.dumps(_serializable_embed_cfg(embed_cfg), ensure_ascii=False), encoding="utf-8"
    )
    for stats in shard_root.glob("shard-*/worker-stats.json"):
        stats.unlink()

    spawner = spawn or _spawn_workers
    started = time.monotonic()
    log(
        f"[tree] parallel vectors: {len(rows)} pending, devices={list(devices)}, spool={shard_root}"
    )
    results = spawner(
        devices=list(devices),
        db_path=db.path,
        cfg_path=cfg_path,
        shard_root=shard_root,
        part_size=int(part_size),
        embed_chunk=int(embed_chunk),
        force=bool(force),
    )
    failed = [(device, code) for device, code in results if code]
    if failed:
        return {
            "status": "failed",
            "error": f"embed workers failed: {failed}",
            "nodes": len(rows),
            "embedded": 0,
            "pending": len(rows),
        }

    loaded, parts = _load_spool(db, store, profile_id, shard_root, batch_size=load_batch, log=log)
    empty = 0
    for stats_path in shard_root.glob("shard-*/worker-stats.json"):
        try:
            empty += int(json.loads(stats_path.read_text(encoding="utf-8")).get("empty", 0))
        except (OSError, ValueError):
            continue
    # The spool is fully consumed only when every part loaded; keep the
    # manifest for a resumable remainder, otherwise clear the directory.
    remaining_parts = sorted(shard_root.glob("shard-*/*.npz"))
    if not remaining_parts:
        cfg_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        for shard in shard_root.glob("shard-*"):
            for leftover in shard.iterdir():
                leftover.unlink(missing_ok=True)
            shard.rmdir()
    duration = time.monotonic() - started
    log(
        f"[tree] parallel vectors: {loaded} loaded from {parts} parts "
        f"({', '.join(f'{device}={code}' for device, code in results)}), {duration:.0f}s"
    )
    return {
        "status": "ok",
        "nodes": len(rows),
        "embedded": loaded,
        "pending": 0,
        "empty": empty,
        "parallel": {
            "devices": list(devices),
            "spool_dir": str(shard_root),
            "parts": parts,
            "duration_s": round(duration, 1),
        },
    }


# ── worker process ──────────────────────────────────────────────────────────


def _run_worker(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="drbrain.tree.embed_parallel _worker")
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--embed-cfg-json", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--part-size", type=int, default=PART_SIZE)
    parser.add_argument("--embed-chunk", type=int, default=EMBED_CHUNK)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(list(argv))

    from drbrain.config import EmbedConfig
    from drbrain.services.embedding import _embed_batch
    from drbrain.storage.database import Database
    from drbrain.storage.node_projection import read_node_text
    from drbrain.tree.embedding_identity import profile_from_config

    payload = json.loads(Path(args.embed_cfg_json).read_text(encoding="utf-8"))
    embed_cfg = dataclasses.replace(EmbedConfig(**payload), device=args.device)
    profile = profile_from_config(embed_cfg)
    shard_dir = Path(args.shard_dir) / f"shard-{args.worker_index}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    db = Database(args.db_path)
    try:
        rows = pending_rows(db, profile.profile_id(), force=bool(args.force))
        mine = rows[args.worker_index :: args.worker_count]
        total = len(mine)
        seq = 0
        while (shard_dir / f"part-{seq:05d}.npz").exists():
            seq += 1
        start = seq * int(args.part_size)
        logger.info(
            "[tree] embed worker {} on {}: {} nodes, resuming at {}",
            args.worker_index,
            args.device,
            total,
            start,
        )
        empty = 0
        embedded = 0
        index = start
        while index < total:
            part_rows = mine[index : index + int(args.part_size)]
            texts: list[str] = []
            keep: list[dict] = []
            for row in part_rows:
                text = read_node_text(db.conn, row["node_id"])
                if not text:
                    empty += 1
                    continue
                texts.append(text)
                keep.append(row)
            vectors: list[list[float]] = []
            for offset in range(0, len(texts), int(args.embed_chunk)):
                vectors.extend(
                    _embed_batch(texts[offset : offset + int(args.embed_chunk)], embed_cfg)
                )
            np.savez(
                shard_dir / f"part-{seq:05d}.npz",
                start_index=np.asarray([index], dtype=np.int64),
                node_ids=np.asarray([row["node_id"] for row in keep]),
                revisions=np.asarray([row["revision"] for row in keep], dtype=np.int64),
                kinds=np.asarray([row["kind"] for row in keep]),
                local_ids=np.asarray([row["local_id"] for row in keep]),
                layers=np.asarray([row["layer"] for row in keep], dtype=np.int64),
                content_hashes=np.asarray([row["content_hash"] for row in keep]),
                profile_ids=np.asarray([profile.profile_id()] * len(keep)),
                vectors=np.asarray(vectors, dtype=np.float32),
            )
            embedded += len(keep)
            index += int(args.part_size)
            seq += 1
        (shard_dir / "worker-stats.json").write_text(
            json.dumps({"device": args.device, "embedded": embedded, "empty": empty}),
            encoding="utf-8",
        )
        logger.info(
            "[tree] embed worker {} done: {} embedded, {} empty-text skipped",
            args.worker_index,
            embedded,
            empty,
        )
        return 0
    finally:
        db.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="drbrain.tree.embed_parallel")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("_worker")
    args, rest = parser.parse_known_args(list(argv) if argv is not None else None)
    if args.command == "_worker":
        return _run_worker(rest)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
