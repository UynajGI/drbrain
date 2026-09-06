"""Focused process-pool deadline tests for the corpus pipeline workers."""

from __future__ import annotations

import time

import pytest


def _sleep_worker(task: tuple[str, float]) -> dict:
    """Top-level picklable worker used by the real ProcessPool test."""

    time.sleep(task[1])
    return {"id": task[0], "ok": True}


def _raise_worker(task: tuple[str]) -> dict:
    raise RuntimeError(f"worker exploded for {task[0]}")


def test_process_pool_timeout_terminates_and_aborts_unsubmitted_work() -> None:
    from scripts.pipeline.ingest_scibase import _run_process_pool_fail_fast

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
    from scripts.pipeline.ingest_scibase import _run_process_pool_fail_fast

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
    from scripts.pipeline.ingest_scibase import _run_serial_worker_with_timeout

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
    from scripts.pipeline.ingest_scibase import _can_pickle_process_task

    marker = object()

    def local_worker(_task: tuple) -> dict:
        return {"marker": marker}

    assert _can_pickle_process_task(local_worker, ("task",)) is False


def test_scibase_explicit_empty_source_is_rejected(tmp_path, monkeypatch) -> None:
    from scripts.pipeline import ingest_scibase

    root = tmp_path / "runtime"
    root.mkdir()
    monkeypatch.setenv("DRBRAIN_ROOT", str(root))
    monkeypatch.setattr(
        ingest_scibase,
        "load_cfg",
        lambda *_args, **_kwargs: {"dirs": {}, "llm": {}},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "ingest_scibase.py",
            "--source",
            "",
            "--db",
            str(root / "data" / "shard.db"),
            "--manifest",
            str(root / "data" / "manifest.jsonl"),
        ],
    )

    assert ingest_scibase.main() == 1


def test_rebuild_explicit_empty_list_is_rejected(tmp_path, monkeypatch) -> None:
    from scripts.pipeline import rebuild_trees

    root = tmp_path / "runtime"
    root.mkdir()
    monkeypatch.setenv("DRBRAIN_ROOT", str(root))
    monkeypatch.setattr("sys.argv", ["rebuild_trees.py", "--list", ""])

    assert rebuild_trees.main() == 1


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "not-a-number"])
def test_process_worker_timeout_rejects_non_finite_values(monkeypatch, value: str) -> None:
    from scripts.pipeline.ingest_scibase import _process_worker_timeout

    monkeypatch.setenv("INGEST_PAPER_TIMEOUT", value)
    with pytest.raises(ValueError, match="finite positive"):
        _process_worker_timeout()


def test_rebuild_timeout_uses_its_own_environment_selector(monkeypatch) -> None:
    from scripts.pipeline.rebuild_trees import _process_worker_timeout

    monkeypatch.delenv("REBUILD_TREE_TIMEOUT", raising=False)
    monkeypatch.setenv("REBUILD_WORKER_TIMEOUT", "1.25")
    assert _process_worker_timeout(
        "REBUILD_TREE_TIMEOUT", fallback_env="REBUILD_WORKER_TIMEOUT"
    ) == pytest.approx(1.25)
