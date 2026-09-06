"""Focused process-pool deadline tests for the corpus pipeline workers."""

from __future__ import annotations

import time


def _sleep_worker(task: tuple[str, float]) -> dict:
    """Top-level picklable worker used by the real ProcessPool test."""

    time.sleep(task[1])
    return {"id": task[0], "ok": True}


def _raise_worker(task: tuple[str]) -> dict:
    raise RuntimeError(f"worker exploded for {task[0]}")


def test_process_pool_timeout_terminates_and_aborts_unsubmitted_work() -> None:
    from scripts.pipeline.common import run_process_pool_fail_fast as _run_process_pool_fail_fast

    records: list[dict] = []
    started = time.monotonic()
    complete = _run_process_pool_fail_fast(
        [("slow", 2.0), ("never-submitted", 0.0)],
        _sleep_worker,
        max_workers=1,
        timeout=0.05,
        on_result=records.append,
        timeout_result=lambda task, seconds: {
            "id": task[0],
            "ok": False,
            "error": f"timeout>{seconds:g}s",
        },
        exception_result=lambda task, exc: {
            "id": task[0],
            "ok": False,
            "error": str(exc),
        },
    )

    assert complete is False
    assert records == [{"id": "slow", "ok": False, "error": "timeout>0.05s"}]
    # A terminated child must not keep the command around for the full sleep.
    assert time.monotonic() - started < 2.0


def test_process_pool_unexpected_worker_exception_is_fail_fast() -> None:
    from scripts.pipeline.common import run_process_pool_fail_fast as _run_process_pool_fail_fast

    records: list[dict] = []
    complete = _run_process_pool_fail_fast(
        [("bad",)],
        _raise_worker,
        max_workers=1,
        timeout=1.0,
        on_result=records.append,
        timeout_result=lambda task, seconds: {"id": task[0], "ok": False},
        exception_result=lambda task, exc: {
            "id": task[0],
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        },
    )

    assert complete is False
    assert records and records[0]["id"] == "bad"
    assert "worker exploded" in records[0]["error"]


def test_serial_worker_timeout_interrupts_legacy_in_process_call() -> None:
    from scripts.pipeline.common import (
        run_serial_worker_with_timeout as _run_serial_worker_with_timeout,
    )

    started = time.monotonic()
    result = _run_serial_worker_with_timeout(
        _sleep_worker,
        ("slow", 2.0),
        timeout=0.05,
        timeout_result=lambda task, seconds: {
            "id": task[0],
            "ok": False,
            "error": f"timeout>{seconds:g}s",
        },
        exception_result=lambda task, exc: {
            "id": task[0],
            "ok": False,
            "error": str(exc),
        },
    )

    assert result == {"id": "slow", "ok": False, "error": "timeout>0.05s"}
    assert time.monotonic() - started < 1.0


def test_non_picklable_legacy_worker_is_detected() -> None:
    from scripts.pipeline.common import can_pickle_process_task as _can_pickle_process_task

    marker = object()

    def local_worker(_task: tuple) -> dict:
        return {"marker": marker}

    assert _can_pickle_process_task(local_worker, ("task",)) is False
