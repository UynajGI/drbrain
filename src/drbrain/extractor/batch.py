"""Batch LLM orchestration: todo, checkpoint, concurrency, output, done flag.

Pattern source: ``research/scripts/cg_fulltext_concepts_api.py`` — the v2
stability rework (2026-08-08/09) of the 100k+ paper concept-extraction job.
The orchestration moves validated there are generalized here:

- One-time todo list of lightweight IDs (e.g. doi) instead of LIMIT/OFFSET
  rescans over a large table.
- Offset persisted per batch; a restart resumes exactly from it; a stale
  offset (saved against an older, longer todo) auto-resets to 0.
- Output appended to JSONL with a flush per line, so a crash loses at most
  the current batch's unwritten records.
- Completion writes a done flag; external supervisors key off its presence,
  never a volatile offset.

Division of labour: this base class owns orchestration (todo generation,
checkpoint resume, thread pool, JSONL persistence, done flag); subclasses own
single-item processing — extracting a prompt from a todo item and parsing the
raw LLM response into a structured dict.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from loguru import logger

from drbrain.utils.checkpoint import Checkpoint, DoneFlag


@dataclass(frozen=True)
class BatchRunStats:
    """Counters and timing for one :meth:`BatchLLMProcessor.run` execution."""

    ok: int
    fail: int
    total: int
    offset: int
    elapsed_s: float


class BatchLLMProcessor(ABC):
    """Abstract base for batch LLM jobs: orchestration in, item logic out.

    Subclasses implement three per-item pieces: :meth:`build_todo` (which
    lightweight IDs to process), :meth:`build_prompt` (todo item → LLM
    prompt), and :meth:`process_one` (raw LLM response → JSON-serializable
    result dict, typically via ``drbrain.utils.llm_json.parse_llm_json``).
    The base class then runs the whole list with checkpoint resume,
    concurrency, JSONL persistence and a done flag.

    LLM sending goes through the :meth:`_call_llm` hook; its default raises
    ``NotImplementedError``.  A subclass either overrides it (e.g. wrapping
    ``drbrain.extractor.llm_client.call_with_fallback``, optionally with a
    :class:`~drbrain.extractor.llm_client.KeyRotator`) or sends the request
    itself and returns the raw response text.

    Resume semantics mirror the research script: completed items are skipped
    by todo *index* (the checkpoint offset), never by content de-duplication
    — simple and reliable.  A checkpoint offset that is out of bounds of a
    regenerated (shorter) todo list auto-resets to 0.
    """

    def __init__(
        self,
        output_path: str | os.PathLike[str],
        checkpoint_path: str | os.PathLike[str],
        done_flag_path: str | os.PathLike[str],
        *,
        batch_size: int = 64,
    ) -> None:
        """Create a batch processor.

        Args:
            output_path: JSONL file, appended per record.  Each line is one
                ``{"item": ..., "result": ..., "ts": ...}`` record.
            checkpoint_path: Offset file (:class:`Checkpoint`) for resume.
            done_flag_path: Completion marker (:class:`DoneFlag`) written
                once the whole todo list is processed.
            batch_size: Items per batch.  The checkpoint advances one batch
                at a time, so a crash re-runs at most this many items.

        Raises:
            ValueError: when *batch_size* < 1.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._output_path = Path(output_path)
        self._checkpoint = Checkpoint(checkpoint_path)
        self._done_flag = DoneFlag(done_flag_path)
        self._batch_size = batch_size

    @abstractmethod
    def build_todo(self) -> list[str]:
        """Return the todo list: lightweight item IDs to process (e.g. doi).

        Called once per :meth:`run`; may shrink between runs as work is
        completed elsewhere — the checkpoint offset guards against rescanning
        via the out-of-bounds auto-reset.
        """
        ...

    @abstractmethod
    def build_prompt(self, item: str) -> str:
        """Map one todo item to the LLM prompt sent for it."""
        ...

    @abstractmethod
    def process_one(self, item: str, raw_response: str) -> dict:
        """Parse the raw LLM response for *item* into a structured dict.

        The returned dict must be JSON-serializable (it lands under the
        ``"result"`` key of the output record).  Use
        ``drbrain.utils.llm_json.parse_llm_json`` for lenient parsing.
        """
        ...

    def _call_llm(self, prompt: str) -> str:
        """Send *prompt* to the LLM and return the raw response text.

        Default raises ``NotImplementedError``: either override this hook
        (e.g. via ``drbrain.extractor.llm_client``) or send requests from
        your own code and route the raw text through
        :meth:`process_one`.
        """
        raise NotImplementedError(
            "BatchLLMProcessor subclasses must implement _call_llm(prompt) "
            "or send requests themselves"
        )

    def run(self, *, concurrency: int = 1, resume: bool = True) -> BatchRunStats:
        """Process the todo list end-to-end with checkpoint resume.

        Template method: ``build_todo()`` → load checkpoint offset (stale
        offsets auto-reset to 0) → run batches through a thread pool,
        appending successful records to the JSONL output (flush per line) →
        save the offset after each batch → write the done flag once
        everything is processed.

        Args:
            concurrency: Worker threads for the pool (>= 1).  Threads share
                the processor instance, so subclasses must keep
                ``_call_llm``/``process_one`` thread-safe (the pool is a
                ``ThreadPoolExecutor``, matching the research script).
            resume: Continue from the saved checkpoint offset.  When False,
                start from 0 and truncate the output file (fresh run).

        Returns:
            :class:`BatchRunStats` with ok/fail/total counters, the final
            offset and the wall-clock elapsed time.

        Raises:
            ValueError: when *concurrency* < 1.
            NotImplementedError: when :meth:`_call_llm` was left at its
                default.
            BaseException: ``KeyboardInterrupt``/``SystemExit`` from a worker
                abort the run; the checkpoint still reflects the last fully
                completed batch, so a later run resumes from there.
        """
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency}")

        todo = self.build_todo()
        total = len(todo)
        # Checkpoint.load resets stale offsets (>= len(todo)) to 0 — the
        # research script's "offset 超界 → 重置 0" fix.
        offset = self._checkpoint.load(max_value=total) if resume else 0
        if not resume:
            # Fresh run: drop stale output so append mode cannot duplicate.
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            self._output_path.write_text("", encoding="utf-8")

        ok = fail = 0
        t_start = time.monotonic()
        while offset < total:
            batch = todo[offset : offset + self._batch_size]
            batch_ok, batch_fail = self._process_batch(batch, concurrency)
            ok += batch_ok
            fail += batch_fail
            offset += len(batch)
            # Per-batch checkpoint: a crash loses at most the current batch.
            self._checkpoint.save(offset)
            logger.info("[batch] progress offset={}/{} ok={} fail={}", offset, total, ok, fail)

        # Done flag: presence is the supervisor's completion signal, never a
        # volatile offset (research script's concepts_done.flag convention).
        self._done_flag.done(f"ok={ok} fail={fail} offset={offset}")
        logger.info("[batch] done ok={} fail={} total={}", ok, fail, total)
        return BatchRunStats(
            ok=ok,
            fail=fail,
            total=total,
            offset=offset,
            elapsed_s=time.monotonic() - t_start,
        )

    def _process_batch(self, batch: list[str], concurrency: int) -> tuple[int, int]:
        """Run one batch through the pool; returns (ok, fail).

        Opens the output file in append mode for the batch, writes each
        successful record with an immediate flush, then closes before the
        caller saves the checkpoint — a crash cannot leave a half-written
        batch recorded as done.
        """
        ok = fail = 0
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._output_path.open("a", encoding="utf-8") as f:
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futures = {ex.submit(self._process_item, item): item for item in batch}
                for fu in as_completed(futures):
                    succeeded, result = fu.result()
                    if succeeded:
                        assert result is not None  # succeeded implies a record
                        self._write_record(f, futures[fu], result)
                        ok += 1
                    else:
                        fail += 1
        return ok, fail

    def _process_item(self, item: str) -> tuple[bool, dict | None]:
        """One item end-to-end: prompt → LLM call → structured result.

        Runs in a worker thread.  Ordinary exceptions from any step are
        swallowed and reported as a failure (the job keeps going, matching
        the research script's per-item try/except); ``KeyboardInterrupt`` /
        ``SystemExit`` propagate so an operator can stop the job cleanly, and
        ``NotImplementedError`` (e.g. ``_call_llm`` left at its default) is a
        programming error that must abort loudly, not count as item failure.
        """
        try:
            prompt = self.build_prompt(item)
            raw_response = self._call_llm(prompt)
            result = self.process_one(item, raw_response)
            return True, result
        except (KeyboardInterrupt, SystemExit, NotImplementedError):
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("[batch] item {} failed: {}", item, e)
            return False, None

    def _write_record(self, f: TextIO, item: str, result: dict) -> None:
        """Append one JSONL record and flush it to disk immediately."""
        record = {
            "item": item,
            "result": result,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
