#!/usr/bin/env python
"""全量增强管线共享模块：配置加载。

设计要点（吸取旧版教训）：
- load_cfg 始终以 config.local.yaml 为基础（含 llm.models 多引擎/ox-alpha-free 顺序），
  --config 只做增量覆盖（如 config.embed1.yaml 只覆盖 embed 部分）。
"""

from __future__ import annotations

import concurrent.futures
import math
import os
import pickle
import re
import signal
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import yaml

from drbrain.config import _resolve_env_vars, merge_dicts
from drbrain.runtime import RuntimeContext

# ``ROOT`` is retained for callers that imported this module before runtime
# isolation was introduced.  It must not resolve the environment at import
# time: a malformed ``DRBRAIN_ROOT`` should be reported by the command entry
# point, not as an unhandled traceback while importing a helper module.
SOURCE_ROOT = Path(__file__).resolve().parents[2]
ROOT = SOURCE_ROOT
_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


class _SerialWorkerTimeout(BaseException):
    """Internal signal exception used by the compatibility worker path."""


def can_pickle_process_task(worker: Callable[[tuple], dict], task: tuple) -> bool:
    """Return whether a worker/task pair can cross a process boundary."""

    try:
        pickle.dumps((worker, task))
    except Exception:  # noqa: BLE001 - this is a capability probe
        return False
    return True


def run_serial_worker_with_timeout(
    worker: Callable[[tuple], dict],
    task: tuple,
    *,
    timeout: float,
    timeout_result: Callable[[tuple, float], dict],
    exception_result: Callable[[tuple, Exception], dict],
) -> dict:
    """Run an unpicklable legacy worker with a main-thread deadline.

    Signals can interrupt a blocking call only in the main thread.  Embedded
    callers that invoke a pipeline from another thread retain compatibility;
    their result is checked against the elapsed deadline after the call.
    """

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("worker timeout must be a finite positive number")
    started = time.monotonic()
    can_interrupt = (
        os.name == "posix"
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )
    if not can_interrupt:
        try:
            result = worker(task)
        except Exception as exc:  # noqa: BLE001
            return exception_result(task, exc)
        if time.monotonic() - started > timeout:
            return timeout_result(task, timeout)
        return result

    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)

    def _alarm_handler(_signum: int, _frame: object) -> None:
        raise _SerialWorkerTimeout

    try:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            return worker(task)
        except _SerialWorkerTimeout:
            return timeout_result(task, timeout)
        except Exception as exc:  # noqa: BLE001
            return exception_result(task, exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])


def _stop_process_pool(executor: concurrent.futures.ProcessPoolExecutor) -> None:
    """Terminate all child workers without waiting for a stuck provider."""

    processes = list((getattr(executor, "_processes", None) or {}).values())
    own_pgid = os.getpgrp() if hasattr(os, "getpgrp") else None
    for process in processes:
        try:
            if process.is_alive():
                pgid = os.getpgid(process.pid) if own_pgid is not None else None
                if pgid and pgid != own_pgid:
                    os.killpg(pgid, signal.SIGTERM)
                else:
                    process.terminate()
        except (OSError, AttributeError, RuntimeError):
            continue
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except (OSError, RuntimeError):
        pass
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and any(p.is_alive() for p in processes):
        time.sleep(0.02)
    for process in processes:
        try:
            if process.is_alive():
                pgid = os.getpgid(process.pid) if own_pgid is not None else None
                if pgid and pgid != own_pgid:
                    os.killpg(pgid, signal.SIGKILL)
                elif hasattr(process, "kill"):
                    process.kill()
            process.join(timeout=0.1)
        except (OSError, AttributeError, RuntimeError):
            continue


def _worker_process_init() -> None:
    """Put each pool worker in its own session for descendant cleanup."""
    if os.name == "posix":
        try:
            os.setsid()
        except OSError:
            pass
def run_process_pool_fail_fast(
    tasks: Iterable[tuple],
    worker: Callable[[tuple], dict],
    *,
    max_workers: int,
    timeout: float,
    on_result: Callable[[dict], None],
    timeout_result: Callable[[tuple, float], dict],
    exception_result: Callable[[tuple, Exception], dict],
) -> bool:
    """Run bounded process workers and abort promptly on timeout/exception.

    ``tasks`` is consumed lazily, so callers can create per-paper staging
    artifacts only as slots become available.  A deadline starts when a task
    is submitted, not when the whole input list is materialized.
    """

    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("worker timeout must be a finite positive number")
    task_iter = iter(tasks)
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_worker_process_init,
    )
    inflight: dict[object, tuple[tuple, float]] = {}
    aborted = False

    def submit_available() -> None:
        while len(inflight) < max_workers:
            try:
                task = next(task_iter)
            except StopIteration:
                return
            future = executor.submit(worker, task)
            inflight[future] = (task, time.monotonic() + timeout)

    try:
        submit_available()
        while inflight:
            now = time.monotonic()
            next_deadline = min(deadline for _task, deadline in inflight.values())
            wait_for = max(0.0, next_deadline - now)
            finished, _ = concurrent.futures.wait(
                tuple(inflight),
                timeout=wait_for,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            now = time.monotonic()
            # Completed futures win a deadline race; they have already
            # delivered a result and cannot be safely classified as timed out.
            for future in finished:
                task, _deadline = inflight.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    _stop_process_pool(executor)
                    on_result(exception_result(task, exc))
                    aborted = True
                    break
                on_result(result)
                if isinstance(result, dict) and result.get("timeout"):
                    aborted = True
                    break
            if aborted:
                break

            expired = [future for future, (_task, deadline) in inflight.items() if deadline <= now]
            if expired:
                # Stop all workers before invoking potentially slow manifest or
                # DB callbacks, so timed-out children cannot keep mutating
                # artifacts while the parent publishes the failure record.
                _stop_process_pool(executor)
                for future in expired:
                    task, _deadline = inflight.pop(future)
                    future.cancel()
                    on_result(timeout_result(task, timeout))
                aborted = True
                break
            submit_available()
    finally:
        if aborted or inflight:
            for future in inflight:
                future.cancel()
            _stop_process_pool(executor)
        else:
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except (OSError, RuntimeError):
                _stop_process_pool(executor)
                raise
    return not aborted


def _selected_root(root: str | Path | None) -> str | Path:
    """Select a runtime root without treating an explicit empty env as unset."""

    if root is not None:
        return root
    if "DRBRAIN_ROOT" in os.environ:
        return os.environ["DRBRAIN_ROOT"]
    if "DRBRAIN_RUNTIME_ROOT" in os.environ:
        return os.environ["DRBRAIN_RUNTIME_ROOT"]
    # Keep the no-selector behavior aligned with ``RuntimeContext.create`` and
    # ``runtime_root``.  ``SOURCE_ROOT`` is an import anchor only; using it as
    # an implicit data root can split config/DB resolution when a script is
    # launched from another working directory.
    return Path.cwd().resolve()


def runtime_path(
    value: str | Path,
    root: str | Path | None = None,
    *,
    allow_external: bool = False,
) -> Path:
    """Resolve a script argument beneath the selected runtime root.

    Pipeline arguments are data boundaries, so traversal and absolute paths
    outside the selected root fail closed by default.  A caller that
    intentionally reads a shared resource (for example a model cache) must
    opt in explicitly with ``allow_external=True``; mutable pipeline outputs
    should never use that escape hatch.
    """
    selected_root = _selected_root(root)
    # Pass the raw selector through first; converting ``""`` to ``Path('.')``
    # would silently turn an explicitly empty environment value into the
    # process working directory.
    runtime = RuntimeContext.create(selected_root)
    root_path = runtime.root
    try:
        raw_value = os.fspath(value)
    except TypeError as exc:
        raise ValueError(f"pipeline path is not a valid filesystem path: {value!r}") from exc
    if isinstance(raw_value, str) and _URI_SCHEME_RE.match(raw_value):
        raise ValueError(f"pipeline path must be local, not a URI: {raw_value}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root_path / path
    if not allow_external:
        # Reuse the strict lexical + resolved containment check.  In-root
        # symlink aliases are rejected too, since a later mkdir/open could be
        # redirected after the initial containment check.
        return runtime.assert_within_root(path, label="pipeline path")
    return path.resolve()


def get_runtime_context(
    root: str | Path | None = None,
    *,
    run_id: str | None = None,
    overlay_path: str | Path | None = None,
) -> RuntimeContext:
    """Build the shared runtime context used by standalone pipeline scripts."""
    selected_root = _selected_root(root)
    # Leave ``root`` unset only when the caller is intentionally inheriting the
    # launcher namespace.  That lets RuntimeContext pair the launcher's temp
    # root with its selected root; an explicit root gets a fresh default scratch
    # namespace and cannot accidentally reuse another run's temp directory.
    inherited = root is None and (
        "DRBRAIN_ROOT" in os.environ or "DRBRAIN_RUNTIME_ROOT" in os.environ
    )
    context_root = None if inherited else selected_root
    return RuntimeContext.create(context_root, run_id=run_id, overlay_path=overlay_path)


def load_cfg(
    config_path: str | None = None,
    *,
    root: str | Path | None = None,
    context: RuntimeContext | None = None,
) -> dict:
    """Load base/local/explicit config layers within one runtime root."""
    runtime = context or get_runtime_context(root, overlay_path=config_path)

    def read_layer(path: Path, *, required: bool = False) -> dict:
        if path.is_symlink():
            raise ValueError(f"Config file must not be a symlink: {path}")
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Config not found: {path}")
            return {}
        if not path.is_file():
            raise ValueError(f"Config file is not a regular file: {path}")
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            # YAML diagnostics may echo the offending source line.  Config
            # files are allowed to contain credentials, so keep parser detail
            # bounded and scrubbed at this standalone-script boundary too.
            raise ValueError(f"Invalid YAML config: {path.name}: {type(exc).__name__}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Config file must contain a YAML mapping: {path}")
        return value

    base = read_layer(runtime.base_config_path, required=True)
    local = read_layer(runtime.root / "config.local.yaml")
    merged = merge_dicts(base, local)
    if config_path is not None:
        # Validate the caller's lexical path before resolution so an in-root
        # symlink cannot be normalized into an apparently safe real path.
        overlay = runtime.validate_config_file(config_path, label="config overlay", required=True)
        merged = merge_dicts(merged, read_layer(overlay, required=True))
    # Standalone pipeline scripts are write-capable entrypoints too.  Validate
    # the raw layers before normalizing them so an absolute DB/papers/log path
    # from a copied config cannot silently retarget another worktree.
    merged = _resolve_env_vars(merged)
    runtime.validate_config(merged)
    return runtime.apply_config(merged)
