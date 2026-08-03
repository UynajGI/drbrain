"""Lean concept extraction for the concept graph layer.

Cost-optimised alternative to the full ``extract_concepts`` pipeline:
one LLM call per paper with a minimal prompt (flat concept list only),
results cached in ``paper_concepts_cache`` so runs are resumable and
re-runs never re-spend tokens.

Validated pilot (glm-4.5-air, 15 abstracts): ~107 output tok/paper,
14.2 concepts/paper, 100% JSON parse rate.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from loguru import logger

from drbrain.storage.database import Database

LEAN_PROMPT_FILE = (
    Path(__file__).parent.parent.parent.parent / "prompts" / "extract_concepts_lean.txt"
)

_CACHE_DDL = """
CREATE TABLE IF NOT EXISTS paper_concepts_cache (
    local_id TEXT PRIMARY KEY,
    concepts_json TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


def ensure_cache_table(db: Database) -> None:
    """Create the extraction cache table if missing."""
    db.conn.execute(_CACHE_DDL)
    db.conn.commit()


def _lean_prompt() -> str:
    return LEAN_PROMPT_FILE.read_text(encoding="utf-8")


class _RateLimiter:
    """Evenly-spaced request gate: at most ``rpm`` LLM calls per minute.

    Serialises request *starts* at a fixed interval (60/rpm seconds) so the
    account never sees bursty traffic. Set ``rpm <= 0`` to disable.
    """

    def __init__(self, rpm: int):
        self._interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def acquire(self) -> None:
        if self._interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            wait = self._next_at - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next_at = max(now, self._next_at) + self._interval


async def _extract_one(
    text: str,
    models: list[dict],
    prompt: str,
    semaphore: asyncio.Semaphore,
    max_tokens: int,
    limiter: _RateLimiter | None = None,
) -> tuple[list[str], str]:
    """Extract concepts for one paper via the LLM fallback chain.

    Returns ``(labels, model_name)``; empty list on failure.
    """
    from drbrain.extractor.llm_client import acall_with_fallback

    async with semaphore:
        if limiter is not None:
            await limiter.acquire()
        data = await acall_with_fallback(
            prompt=text[:6000],
            models=models,
            system_prompt=prompt,
            max_tokens=max_tokens,
        )
    if not isinstance(data, dict):
        return [], ""
    raw = data.get("concepts", [])
    labels: list[str] = []
    for item in raw:
        label = (
            str(item).strip().lower()
            if not isinstance(item, dict)
            else str(item.get("label", "")).strip().lower()
        )
        if label and len(label) <= 80:
            labels.append(label)
    # dedupe, preserve order
    seen: set[str] = set()
    uniq = [x for x in labels if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]
    model_name = models[0].get("model", "") if models else ""
    return uniq, model_name


def extract_paper_concepts_batch(
    db: Database,
    *,
    limit: int | None = None,
    concurrency: int = 2,
    max_tokens: int = 400,
    rpm: int = 10,
    chunk_size: int = 500,
) -> dict:
    """Extract and cache concepts for papers not yet in the cache.

    Streams in chunks of ``chunk_size`` papers: each chunk is awaited and
    committed before the next is fetched, so long runs show progress and
    survive interruptions (cache-based resumability).

    Args:
        db: Database handle.
        limit: Max number of papers to process in this run (None = all).
        concurrency: Max parallel LLM calls.
        max_tokens: Output token budget per paper.
        rpm: Max LLM calls per minute (evenly spaced; 0 = unlimited).
        chunk_size: Papers per await/commit cycle.

    Returns:
        ``{"processed": n, "failed": n, "cached_total": n}``
    """
    from drbrain.config import load_config

    ensure_cache_table(db)
    models = load_config().llm.models
    prompt = _lean_prompt()

    base_query = (
        "SELECT p.local_id, p.title, p.abstract FROM papers p "
        "WHERE p.abstract IS NOT NULL AND length(p.abstract) > 100 "
        "AND p.local_id NOT IN (SELECT local_id FROM paper_concepts_cache) "
        "LIMIT ?"
    )
    insert_sql = (
        "INSERT OR REPLACE INTO paper_concepts_cache (local_id, concepts_json, model) "
        "VALUES (?, ?, ?)"
    )

    semaphore = asyncio.Semaphore(concurrency)
    limiter = _RateLimiter(rpm)
    logger.info("[cg.extract] starting: limit={}, concurrency={}, rpm={}", limit, concurrency, rpm)

    async def _run_chunk(rows: list) -> tuple[int, int]:
        tasks = [
            _extract_one(
                f"{title}\n\n{abstract}".strip(), models, prompt, semaphore, max_tokens, limiter
            )
            for _, title, abstract in rows
        ]
        results = await asyncio.gather(*tasks)
        ok = 0
        chunk_failed = 0
        batch: list[tuple[str, str, str]] = []
        for (local_id, _, _), (labels, model_name) in zip(rows, results):
            if labels:
                batch.append((local_id, json.dumps(labels, ensure_ascii=False), model_name))
                ok += 1
            else:
                chunk_failed += 1
        if batch:
            db.conn.executemany(insert_sql, batch)
            db.conn.commit()
        return ok, chunk_failed

    async def _run_all() -> tuple[int, int]:
        processed = 0
        failed = 0
        remaining = limit
        while True:
            fetch = chunk_size if remaining is None else min(chunk_size, remaining)
            if fetch <= 0:
                break
            rows = db.conn.execute(base_query, (fetch,)).fetchall()
            if not rows:
                break
            ok, chunk_failed = await _run_chunk(rows)
            processed += ok
            failed += chunk_failed
            if remaining is not None:
                remaining -= len(rows)
            total = db.conn.execute("SELECT count(*) FROM paper_concepts_cache").fetchone()[0]
            logger.info(
                "[cg.extract] progress: +{} ok, +{} failed this chunk (cached_total={})",
                ok,
                chunk_failed,
                total,
            )
        return processed, failed

    processed, failed = asyncio.run(_run_all())

    total = db.conn.execute("SELECT count(*) FROM paper_concepts_cache").fetchone()[0]
    logger.info(
        "[cg.extract] done: processed={} failed={} cached_total={}", processed, failed, total
    )
    return {"processed": processed, "failed": failed, "cached_total": total}


def cached_concepts_for_paper(db: Database, local_id: str) -> list[str] | None:
    """Return cached concept labels for a paper, or None if not cached."""
    ensure_cache_table(db)
    row = db.conn.execute(
        "SELECT concepts_json FROM paper_concepts_cache WHERE local_id = ?", (local_id,)
    ).fetchone()
    if row is None:
        return None
    try:
        return list(json.loads(row[0]))
    except (json.JSONDecodeError, TypeError):
        return None
