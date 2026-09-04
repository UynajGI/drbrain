#!/usr/bin/env python3
"""RAG-ify drbrain.db: fill ``node_texts`` (+ FTS5) from on-disk full texts.

Target architecture (2026-08-27 refactor): full text belongs IN the library
database so RAG is a database feature, not a parallel store. This tool fills
the working copy ``data/drbrain_rag.db``:

  phase extract  -- walk every paper dir with tree.json, build node rows with
                   the EXACT pipeline text construction
                   (``drbrain.services.embedding._collect_tree_nodes``) so the
                   content hashes match ``tree_vectors`` and those vectors are
                   reusable without re-embedding; rows land in shard JSONL.
  phase load     -- stream shards into ``node_texts``, create FTS5 index.
  phase verify   -- hash-match rate against ``tree_vectors``.
  phase categories -- sync papers.categories (main DB) → paper_categories
                   (RAG DB) for category-scoped retrieval.

Usage:
  python scripts/pipeline/ragdb_fill.py extract --workers 24
  python scripts/pipeline/ragdb_fill.py load
  python scripts/pipeline/ragdb_fill.py verify
  python scripts/pipeline/ragdb_fill.py categories
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

PAPERS_ROOT = REPO / "data" / "papers"
RAG_DB = REPO / "data" / "drbrain_rag.db"
MAIN_DB = REPO / "data" / "drbrain.db"
SHARD_DIR = REPO / "data" / "spool" / "ragdb_shards"


# ── extract ──────────────────────────────────────────────────────────────────


def _iter_paper_dirs() -> list[Path]:
    dirs: list[Path] = []
    for entry in PAPERS_ROOT.iterdir():
        if not entry.is_dir():
            continue
        # DOI-keyed papers nest one level (10.1002/xxxx)
        if entry.name.startswith("10."):
            for sub in entry.iterdir():
                if sub.is_dir() and (sub / "tree.json").exists():
                    dirs.append(sub)
        elif (entry / "tree.json").exists():
            dirs.append(entry)
    return dirs


def _extract_chunk(chunk_idx: int, paper_dirs: list[str]) -> str:
    """Worker: build node rows for a chunk of paper dirs → shard JSONL."""
    from drbrain.services.embedding import _collect_tree_nodes, _content_hash

    out_path = SHARD_DIR / f"shard_{chunk_idx:05d}.jsonl"
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for d in paper_dirs:
            pdir = Path(d)
            paper_id = pdir.name
            try:
                nodes = _collect_tree_nodes(pdir)
            except Exception:
                continue
            for node in nodes:
                nid = str(node.get("node_id") or "").strip()
                text = node.get("text") or ""
                if not nid or not text:
                    continue
                f.write(
                    json.dumps(
                        {
                            "k": f"{paper_id}:{nid}",
                            "p": paper_id,
                            "n": nid,
                            "t": text,
                            "h": _content_hash(text),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n += 1
    return f"{chunk_idx}:{n}"


def cmd_extract(workers: int) -> None:
    dirs = [str(d) for d in _iter_paper_dirs()]
    print(f"[extract] {len(dirs)} paper dirs with tree.json", flush=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    chunk_size = max(1, len(dirs) // (workers * 8))
    chunks = [dirs[i : i + chunk_size] for i in range(0, len(dirs), chunk_size)]
    t0 = time.time()
    total = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_extract_chunk, i, c) for i, c in enumerate(chunks)]
        for done in as_completed(futs):
            _, n = done.result().split(":")
            total += int(n)
    print(f"[extract] done: {total} nodes in {time.time() - t0:.0f}s", flush=True)


# ── load ─────────────────────────────────────────────────────────────────────


def cmd_load() -> None:
    conn = sqlite3.connect(str(RAG_DB))
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA cache_size = -2000000")  # 2GB page cache
    conn.execute(
        """CREATE TABLE IF NOT EXISTS node_texts (
               node_key TEXT PRIMARY KEY,
               paper_id TEXT NOT NULL,
               node_id TEXT NOT NULL,
               text TEXT NOT NULL,
               content_hash TEXT NOT NULL
           )"""
    )
    t0 = time.time()
    batch: list[tuple] = []
    n = 0
    shards = sorted(SHARD_DIR.glob("shard_*.jsonl"))
    print(f"[load] {len(shards)} shards → {RAG_DB.name}", flush=True)
    with conn:
        conn.execute("DELETE FROM node_texts")  # idempotent refill
        for shard in shards:
            with open(shard, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    batch.append((r["k"], r["p"], r["n"], r["t"], r["h"]))
                    if len(batch) >= 20000:
                        conn.executemany(
                            "INSERT OR REPLACE INTO node_texts VALUES (?,?,?,?,?)",
                            batch,
                        )
                        n += len(batch)
                        batch.clear()
                        if n % 200000 == 0:
                            print(f"[load] {n} rows ({time.time() - t0:.0f}s)", flush=True)
        if batch:
            conn.executemany("INSERT OR REPLACE INTO node_texts VALUES (?,?,?,?,?)", batch)
            n += len(batch)
    print(f"[load] inserted {n} rows in {time.time() - t0:.0f}s; indexing…", flush=True)
    t0 = time.time()
    with conn:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_node_texts_paper ON node_texts(paper_id)")
    print(f"[load] index done in {time.time() - t0:.0f}s; FTS5 build…", flush=True)
    t0 = time.time()
    with conn:
        conn.execute("DROP TABLE IF EXISTS node_texts_fts")
        conn.execute(
            "CREATE VIRTUAL TABLE node_texts_fts USING fts5("
            "text, content='node_texts', content_rowid='rowid', "
            "tokenize='porter unicode61')"
        )
        conn.execute("INSERT INTO node_texts_fts(node_texts_fts) VALUES('rebuild')")
    print(f"[load] FTS5 done in {time.time() - t0:.0f}s", flush=True)
    conn.close()


# ── verify ───────────────────────────────────────────────────────────────────


def cmd_verify() -> None:
    conn = sqlite3.connect(f"file:{RAG_DB}?mode=ro", uri=True)
    total = conn.execute("SELECT COUNT(*) FROM node_texts").fetchone()[0]
    matched = conn.execute(
        """SELECT COUNT(*) FROM node_texts nt
           JOIN tree_vectors tv ON tv.node_id = nt.node_key
           WHERE tv.tree_layer = 'pageindex' AND tv.content_hash = nt.content_hash"""
    ).fetchone()[0]
    have_vec = conn.execute(
        """SELECT COUNT(*) FROM node_texts nt
           JOIN tree_vectors tv ON tv.node_id = nt.node_key
           WHERE tv.tree_layer = 'pageindex'"""
    ).fetchone()[0]
    print(f"nodes: {total}")
    print(f"nodes with pipeline vector: {have_vec} ({have_vec / total:.1%})")
    print(f"hash-matched (reuse as-is): {matched} ({matched / total:.1%})")
    conn.close()


# ── categories ───────────────────────────────────────────────────────────────


def cmd_categories() -> None:
    """Sync ``papers.categories`` (main DB) → ``paper_categories`` (RAG DB).

    The FTS5 BM25 leg reads this table to restrict recall to a category
    subset (``filters={"categories": [...]}``) — the review §6.2 requirement
    for keeping 0.8M-paper cross-domain homonyms out of physics retrieval.
    """
    if not MAIN_DB.exists():
        print(f"[categories] main DB missing: {MAIN_DB}", flush=True)
        return
    src = sqlite3.connect(f"file:{MAIN_DB}?mode=ro", uri=True)
    try:
        cols = [r[1] for r in src.execute("PRAGMA table_info(papers)").fetchall()]
        if "categories" not in cols:
            print("[categories] papers.categories column missing (run the app once)", flush=True)
            return
        rows = src.execute(
            "SELECT local_id, categories FROM papers WHERE categories IS NOT NULL AND categories != ''"
        ).fetchall()
    finally:
        src.close()
    conn = sqlite3.connect(str(RAG_DB))
    try:
        with conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS paper_categories (
                       paper_id TEXT PRIMARY KEY,
                       categories TEXT NOT NULL DEFAULT ''
                   )"""
            )
            conn.execute("DELETE FROM paper_categories")
        # 分块提交：0.8M 行一次性 executemany 会撑出一个巨型写事务（WAL 暴涨、
        # 中断即全量回滚重来）。每 50k 行独立提交，中断后重跑脚本即可续写。
        chunk_size = 50_000
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i : i + chunk_size]
            with conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO paper_categories (paper_id, categories) VALUES (?, ?)",
                    chunk,
                )
            print(f"[categories] {min(i + chunk_size, len(rows))}/{len(rows)}", flush=True)
        with conn:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_paper_categories_paper ON paper_categories(paper_id)"
            )
    finally:
        conn.close()
    print(f"[categories] synced {len(rows)} papers → {RAG_DB.name}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["extract", "load", "verify", "categories"])
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    if args.phase == "extract":
        cmd_extract(args.workers)
    elif args.phase == "load":
        cmd_load()
    elif args.phase == "categories":
        cmd_categories()
    else:
        cmd_verify()
