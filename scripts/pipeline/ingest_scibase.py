#!/usr/bin/env python
"""scibase 全文（fulltext-cleaned-20260806/*.json）→ 分片 db 增强入库。

复刻 drbrain ingest 的 identify → tree → paper 阶段（跳过 parse，markdown 现成）：
对每篇写 data/papers/<local_id>/raw.md + tree.json + papers 表记录。

用法:
    uv run python scripts/pipeline/ingest_scibase.py --source data/spool/scibase_shards8/shard0 \
        --db data/shards/shard0.db --manifest data/shards/shard0.ingest.jsonl

断点续传: manifest 里 ok 的 file 跳过。片内并行: INGEST_CONCURRENCY 个 worker 进程。
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import hashlib
import json
import math
import os
import pickle
import signal
import sqlite3
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable
from pathlib import Path

# Keep imports tied to the source checkout.  ``DRBRAIN_ROOT`` is a data-only
# namespace and may not contain a Python package at all.
SOURCE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_SRC = SOURCE_ROOT / "src"
for _import_root in (SOURCE_ROOT, SOURCE_SRC):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from drbrain.dedup.resolver import (  # noqa: E402
    DedupEngine,
    PaperIDs,
    canonical_paper_id,
)
from drbrain.parser.mineru.parser import filter_sections  # noqa: E402
from drbrain.runtime import runtime_root  # noqa: E402
from drbrain.security import configured_secret_values, safe_error  # noqa: E402
from drbrain.storage.database import Database  # noqa: E402
from drbrain.storage.paths import paper_dir, writable_artifact_path  # noqa: E402
from scripts.pipeline.common import load_cfg, runtime_path  # noqa: E402

# Compatibility snapshot only; source and data roots are separate.  Resolving
# the environment here would make ``--help`` fail with an import traceback.
ROOT = SOURCE_ROOT
CLEANED = ROOT / "data/fulltext-cleaned-20260806"
MIN_MD = 500


def _safe_pipeline_error(value: object, cfg: object | None = None) -> str:
    """Bound and redact provider/worker errors before durable publication."""

    try:
        secrets = (
            configured_secret_values(cfg)
            if cfg is not None
            else configured_secret_values(os.environ)
        )
    except Exception:  # noqa: BLE001
        secrets = ()
    if isinstance(value, BaseException):
        value = f"{type(value).__name__}: {value}"
    return safe_error(value, secrets=secrets)


_DEFAULT_PROCESS_WORKER_TIMEOUT = 900.0


class _SerialWorkerTimeout(BaseException):
    """Internal signal exception that must bypass broad worker catches."""


def _can_pickle_process_task(worker: Callable[[tuple], dict], task: tuple) -> bool:
    """Return whether a task can use the isolated ProcessPool path."""

    try:
        pickle.dumps((worker, task))
    except Exception:  # noqa: BLE001
        return False
    return True


def _run_serial_worker_with_timeout(
    worker: Callable[[tuple], dict],
    task: tuple,
    *,
    timeout: float,
    timeout_result: Callable[[tuple, float], dict],
    exception_result: Callable[[tuple, Exception], dict],
) -> dict:
    """Run a legacy in-process worker while enforcing a best-effort deadline.

    Main-thread CLI calls use ``setitimer`` so a blocked provider call is
    interrupted instead of merely being marked late after it returns.  Calls
    from an embedded/non-main thread retain the old behavior and are bounded
    by elapsed-time inspection because Python signals cannot be installed
    there.
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


def _process_worker_timeout(
    env_name: str = "INGEST_PAPER_TIMEOUT",
    *,
    fallback_env: str | None = "INGEST_WORKER_TIMEOUT",
    default: float = _DEFAULT_PROCESS_WORKER_TIMEOUT,
) -> float:
    """Read a finite per-paper deadline for a process-pool worker.

    The deadline is deliberately finite by default.  A zero, NaN, infinity,
    or malformed value would otherwise turn a hung provider call into a
    process that can keep the shard wrapper alive indefinitely.
    """

    raw = os.environ.get(env_name)
    if raw is None and fallback_env:
        raw = os.environ.get(fallback_env)
    if raw is None:
        raw = str(default)
    try:
        timeout = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_name} must be a finite positive number") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"{env_name} must be a finite positive number")
    return timeout


def _stop_process_pool(executor: concurrent.futures.ProcessPoolExecutor) -> None:
    """Best-effort hard stop for workers after a deadline or executor error."""

    # ``Future.cancel`` cannot interrupt a function already running in a
    # child process.  ProcessPoolExecutor exposes its process map privately,
    # so keep this adapter defensive and fall back to normal shutdown on
    # alternate Python implementations.
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
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except (OSError, RuntimeError):
        # Cleanup must not hide the failed pipeline status.
        pass


def _worker_process_init() -> None:
    if os.name == "posix":
        try:
            os.setsid()
        except OSError:
            pass


def _run_process_pool_fail_fast(
    tasks: list[tuple],
    worker: Callable[[tuple], dict],
    *,
    max_workers: int,
    timeout: float,
    on_result: Callable[[dict], None],
    timeout_result: Callable[[tuple, float], dict],
    exception_result: Callable[[tuple, Exception], dict],
) -> bool:
    """Run workers with bounded in-flight work and an aborting deadline.

    Only ``max_workers`` tasks are submitted at a time.  This makes a
    per-task deadline start near the moment a worker can actually begin,
    rather than expiring while thousands of tasks sit in the executor queue.
    On the first timeout or unexpected worker exception all remaining futures
    are cancelled and child processes are terminated before returning.  The
    caller receives the timeout/exception record through ``on_result`` and
    can therefore publish a non-success manifest without waiting for a stuck
    child.
    """

    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("worker timeout must be a finite positive number")
    if not tasks:
        return True
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_worker_process_init,
    )
    task_iter = iter(tasks)
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

            # Handle completed work first when it races the deadline.  A
            # result already delivered by the executor is not a timeout.
            for future in finished:
                task, _deadline = inflight.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    on_result(exception_result(task, exc))
                    aborted = True
                    break
                if not isinstance(result, dict) or result.get("ok") not in (True, False):
                    on_result(exception_result(task, ValueError("worker result protocol invalid")))
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


def _parse_doi_file(fn: str) -> str:
    """cleaned json 文件名是 DOI 的 url 转义（_ → /）。"""
    return fn[:-5].replace("_", "/")


def _paper_ids(cleaned: dict, doi: str | None) -> PaperIDs:
    """Extract and normalize every known external identity from a record."""
    return PaperIDs(
        doi=doi,
        arxiv=cleaned.get("arxiv") or cleaned.get("arxiv_id"),
        s2_id=cleaned.get("s2_id") or cleaned.get("s2"),
        openalex_id=cleaned.get("openalex_id") or cleaned.get("openalex"),
    ).normalized()


def _load_cleaned(fn: str, source_dir: Path | None = None) -> dict | None:
    """Load one cleaned record from the selected runtime source directory."""
    source = source_dir or CLEANED
    try:
        d = json.loads((source / fn).read_text(encoding="utf-8"))
        if d.get("markdown") and len(d["markdown"]) >= MIN_MD:
            return d
    except Exception:  # noqa: BLE001
        pass
    return None


def _preflight_source_file(path: Path) -> tuple[dict | None, str | None]:
    """Read and validate one cleaned record without touching the database."""
    if path.is_symlink():
        return None, "source file is a symlink"
    if not path.is_file():
        return None, "source file is not a regular file"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(value, dict):
        return None, "record must be an object"
    markdown = value.get("markdown")
    if not isinstance(markdown, str) or len(markdown) < MIN_MD:
        return None, f"markdown must be a string of at least {MIN_MD} characters"
    raw_doi = value.get("doi")
    if not isinstance(raw_doi, str) or not raw_doi.strip():
        return None, "record requires an explicit DOI (filename encoding is lossy)"
    for key in ("doi", "title"):
        if key in value and value[key] is not None and not isinstance(value[key], str):
            return None, f"{key} must be a string"
    year = value.get("year")
    if year is not None and not isinstance(year, (int, str)):
        return None, "year must be an integer or string"
    value = dict(value)
    value["_file"] = path.name
    return value, None


def _preflight_manifest(path: Path) -> tuple[set[str], list[str]]:
    """Validate a resume manifest and return successful source filenames."""
    done: set[str] = set()
    errors: list[str] = []
    if not path.exists():
        return done, errors
    if path.is_symlink() or not path.is_file():
        return done, [f"{path}: manifest must be a regular file"]
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return done, [f"{path}: unable to read manifest: {exc}"]
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{line_no}: invalid JSON ({exc})")
            continue
        if not isinstance(record, dict):
            errors.append(f"{path}:{line_no}: record must be an object")
            continue
        if not isinstance(record.get("ok"), bool):
            errors.append(f"{path}:{line_no}: ok must be boolean")
            continue
        filename = record.get("file")
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or "\x00" in filename
            or not filename.endswith(".json")
        ):
            errors.append(f"{path}:{line_no}: file must be a JSON basename")
            continue
        if record["ok"]:
            local_id = record.get("local_id")
            if not isinstance(local_id, str) or not local_id.strip():
                errors.append(f"{path}:{line_no}: successful record missing local_id")
                continue
            raw_doi = record.get("doi")
            if not isinstance(raw_doi, str) or not raw_doi.strip():
                errors.append(f"{path}:{line_no}: successful record requires an explicit DOI")
                continue
            scheme = record.get("id_scheme")
            if scheme is not None and scheme != "canonical-v1":
                errors.append(f"{path}:{line_no}: unsupported id_scheme {scheme!r}")
                continue
            try:
                ids = _paper_ids(record, raw_doi)
                if scheme == "canonical-v1":
                    expected = canonical_paper_id(
                        ids,
                        title=record.get("title") or "",
                        year=record.get("year"),
                        source_key=record.get("file") or "manifest-record",
                    )
                    if local_id != expected:
                        errors.append(
                            f"{path}:{line_no}: local_id {local_id!r} does not match "
                            f"canonical identity {expected!r}"
                        )
                        continue
            except (TypeError, ValueError) as exc:
                errors.append(f"{path}:{line_no}: invalid paper identity: {exc}")
                continue
            done.add(filename)
    return done, errors


def _atomic_write_text(paper_path: Path, filename: str, content: str) -> None:
    """Write a paper artifact without following a pre-existing symlink."""
    target = writable_artifact_path(paper_path, filename)
    fd, temp_name = tempfile.mkstemp(prefix=f".{filename}.", dir=paper_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _publish_generation_marker(paper_path: Path, generation: str) -> None:
    """Publish hashes only after raw/tree are both durable."""
    raw = (paper_path / "raw.md").read_bytes()
    tree = (paper_path / "tree.json").read_bytes()
    marker = json.dumps(
        {
            "generation": generation,
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "tree_sha256": hashlib.sha256(tree).hexdigest(),
        },
        sort_keys=True,
    )
    _atomic_write_text(paper_path, ".generation.json", marker)


def ingest_from_md(cleaned: dict, cfg: dict, db: Database, dedup: DedupEngine) -> dict:
    """Ingest one cached markdown record with an atomic DB/artifact boundary."""
    savepoint = f"ingest_{uuid.uuid4().hex}"
    had_transaction = db.conn.in_transaction
    db.conn.execute(f"SAVEPOINT {savepoint}")
    try:
        result = _ingest_from_md_impl(cleaned, cfg, db, dedup)
        if result.get("ok"):
            db.conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            if not had_transaction:
                db.commit()
        else:
            db.conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            db.conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            if not had_transaction:
                db.conn.rollback()
        return result
    except Exception:
        try:
            db.conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        finally:
            db.conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        if not had_transaction:
            db.conn.rollback()
        raise


def _ingest_from_md_impl(cleaned: dict, cfg: dict, db: Database, dedup: DedupEngine) -> dict:
    """identify → tree → paper（复刻 db_ingest._ingest_single_paper 的 Stage 2-4）。"""
    import time as _time

    from loguru import logger

    t0 = _time.monotonic()
    md = cleaned["markdown"]
    doi = (cleaned.get("doi") or _parse_doi_file(cleaned["_file"])).strip().lower()
    title = cleaned.get("title") or ""
    year = cleaned.get("year")
    if year and isinstance(year, str) and year.isdigit():
        year = int(year)

    # identify
    ids = _paper_ids(cleaned, doi or None)
    local_id = dedup.resolve(ids, title=title, year=year)
    is_new = local_id is None
    if is_new:
        local_id = canonical_paper_id(ids, title=title, year=year, source_key=cleaned["_file"])
        if db.get_paper(local_id) is not None:
            raise ValueError(
                f"canonical paper ID collision for {local_id!r}; refusing to merge records"
            )
        db.insert_paper(local_id, title or doi, year, "uploaded", paper_type="paper", strict=True)
    # A resolved record may contribute identifiers that the existing row did
    # not have yet; always merge them through the strict write boundary.
    db.insert_paper_ids(
        local_id,
        doi=ids.doi,
        arxiv=ids.arxiv,
        s2_id=ids.s2_id,
        openalex_id=ids.openalex_id,
        strict=True,
    )
    # 写 raw.md
    paper_dir_path = paper_dir(Path(cfg.get("dirs", {}).get("papers", "data/papers")), local_id)
    paper_dir_path.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(paper_dir_path, "raw.md", md)

    llm_models = cfg.get("llm", {}).get("models", [])
    if not llm_models:
        return {"ok": False, "local_id": local_id, "error": "no llm models"}

    # detect paper type（LLM，heuristic 兜底）
    try:
        from drbrain.extractor.detection import detect_paper_type_async

        blocks = filter_sections(md)
        first_page = blocks[0] if blocks else None
        paper_type = asyncio.run(
            detect_paper_type_async(
                title=title, abstract=None, first_page=first_page, models=llm_models
            )
        )
        db.set_paper_type(local_id, paper_type or "paper")
    except TimeoutError as e:
        logger.warning(
            "[scibase] paper-type timeout {}: {}", local_id, _safe_pipeline_error(e, cfg)
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[scibase] paper-type failed {}: {}", local_id, _safe_pipeline_error(e, cfg))

    # tree（LLM 摘要；短节点(<2000 token)原文当摘要不调 LLM，省 ~80% 摘要调用）
    from drbrain.parser.pageindex.sdk_backend import configure_tree_backend
    from drbrain.parser.pageindex_parser import TreeConfig, md_to_tree

    tree_error: str | None = None
    try:
        pageindex_cfg = TreeConfig(
            if_thinning=False,
            if_add_node_summary=True,
            if_add_doc_description=True,
            if_add_node_text=False,
            if_add_node_id=True,
            max_node_tokens=10000,
            summary_token_threshold=2000,
        )
        configure_tree_backend(pageindex_cfg, cfg.get("pageindex"))
        from drbrain.storage.paths import raw_md_path

        doc_tree = asyncio.run(
            md_to_tree(raw_md_path(paper_dir_path), config=pageindex_cfg, models=llm_models)
        )
        _atomic_write_text(paper_dir_path, "tree.json", doc_tree.to_json())
        _publish_generation_marker(paper_dir_path, uuid.uuid4().hex)
        n_sections = len(doc_tree.structure)
        if n_sections <= 0:
            tree_error = "empty tree"
    except Exception as e:  # noqa: BLE001
        logger.warning("[scibase] tree failed {}: {}", local_id, _safe_pipeline_error(e, cfg))
        n_sections = 0
        tree_error = f"tree: {_safe_pipeline_error(e, cfg)}"

    db.set_paper_status(local_id, "uploaded")
    return {
        "ok": tree_error is None,
        "local_id": local_id,
        "error": tree_error,
        "report": {"sections": n_sections, "md_len": len(md), "secs": _time.monotonic() - t0},
    }


def _extract_llm(
    cleaned: dict,
    cfg: dict,
    local_id: str,
    root: str | Path | None = None,
) -> dict:
    """worker 进程：只做 LLM 抽取（paper type + tree），不碰 db（避免锁冲突）。

    写 raw.md + tree.json 到 data/papers/<local_id>/，返回 paper_type。

    ``root`` is optional for compatibility with callers that invoke this
    helper directly.  The multiprocessing path supplies the parent-selected
    runtime root so a worker cannot resolve ``config.build.yaml`` against a
    different working directory.
    """
    import time as _time

    from loguru import logger

    t0 = _time.monotonic()
    md = cleaned["markdown"]
    papers_dir = Path(cfg.get("dirs", {}).get("papers", "data/papers"))
    paper_dir_path = paper_dir(papers_dir, local_id)
    paper_dir_path.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(paper_dir_path, "raw.md", md)

    llm_models = cfg.get("llm", {}).get("models", [])
    if not llm_models:
        return {"ok": False, "error": "no llm models"}

    paper_type = "paper"
    try:
        from drbrain.extractor.detection import detect_paper_type_async

        blocks = filter_sections(md)
        first_page = blocks[0] if blocks else None
        paper_type = (
            asyncio.run(
                detect_paper_type_async(
                    title=cleaned.get("title") or "",
                    abstract=None,
                    first_page=first_page,
                    models=llm_models,
                )
            )
            or "paper"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[scibase] paper-type failed {}: {}", local_id, _safe_pipeline_error(e, cfg))

    from drbrain.parser.pageindex.sdk_backend import configure_tree_backend
    from drbrain.parser.pageindex_parser import TreeConfig, md_to_tree

    n_sections = 0
    tree_error: str | None = None
    try:
        pageindex_cfg = TreeConfig(
            if_thinning=False,
            if_add_node_summary=True,
            if_add_doc_description=False,  # doc description 单独用 hy3 生成（ox-alpha-free 返回空）
            if_add_node_text=False,
            if_add_node_id=True,
            max_node_tokens=10000,
            summary_token_threshold=2000,
        )
        configure_tree_backend(pageindex_cfg, cfg.get("pageindex"))
        from drbrain.storage.paths import raw_md_path

        doc_tree = asyncio.run(
            md_to_tree(raw_md_path(paper_dir_path), config=pageindex_cfg, models=llm_models)
        )
        # doc description 单独用 hy3 生成（ox-alpha-free 对纯文本描述返回空 content）
        try:
            from drbrain.parser.pageindex.retrieval import _create_clean_structure_for_description
            from drbrain.parser.pageindex.summary import _generate_doc_description

            if root is None:
                # Preserve the legacy direct-call behavior for integrations
                # that do not provide an invocation root.
                build_cfg = load_cfg("config.build.yaml")
            else:
                build_cfg = load_cfg("config.build.yaml", root=root)
            # hy3 在前
            hy3_models = build_cfg.get("llm", {}).get("models", [])
            clean_struct = _create_clean_structure_for_description(doc_tree.structure)
            if isinstance(clean_struct, list):
                clean_struct = {"structure": clean_struct}
            doc_desc = asyncio.run(_generate_doc_description(clean_struct, hy3_models))
            if doc_desc:
                doc_tree.doc_description = doc_desc
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[scibase] doc-description failed {}: {}",
                local_id,
                _safe_pipeline_error(e, cfg),
            )
        _atomic_write_text(paper_dir_path, "tree.json", doc_tree.to_json())
        _publish_generation_marker(paper_dir_path, uuid.uuid4().hex)
        n_sections = len(doc_tree.structure)
        if n_sections <= 0:
            tree_error = "empty tree"
    except Exception as e:  # noqa: BLE001
        logger.warning("[scibase] tree failed {}: {}", local_id, _safe_pipeline_error(e, cfg))
        tree_error = f"tree: {_safe_pipeline_error(e, cfg)}"

    return {
        "ok": tree_error is None,
        "paper_type": paper_type,
        "sections": n_sections,
        "error": tree_error,
        "secs": _time.monotonic() - t0,
    }


def _worker(args: tuple) -> dict:
    """多进程 worker：主进程先 identify 生成 local_id，worker 只做 LLM 抽取（不碰 db）。"""
    if len(args) == 4:
        cleaned, cfg, local_id, root = args
    else:
        # Keep the three-item shape accepted by older embedding callers and
        # unit tests.  New pipeline invocations always pass the fourth item.
        cleaned, cfg, local_id = args
        root = None
    try:
        doi = (cleaned.get("doi") or _parse_doi_file(cleaned["_file"])).strip()
        ids = _paper_ids(cleaned, doi or None)
        identity_fields = {
            "id_scheme": "canonical-v1",
            "doi": ids.doi,
            "arxiv": ids.arxiv,
            "s2_id": ids.s2_id,
            "openalex_id": ids.openalex_id,
            "title": cleaned.get("title") or "",
            "year": cleaned.get("year"),
        }
    except TimeoutError as e:
        return {
            "file": cleaned.get("_file", ""),
            "ok": False,
            "timeout": True,
            "local_id": local_id,
            **identity_fields,
            "error": _safe_pipeline_error(e, cfg),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "file": cleaned.get("_file", ""),
            "ok": False,
            "local_id": local_id,
            "error": f"identity: {_safe_pipeline_error(e, cfg)}",
        }
    try:
        if root is None:
            r = _extract_llm(cleaned, cfg, local_id)
        else:
            r = _extract_llm(cleaned, cfg, local_id, root=root)
        return {
            "file": cleaned.get("_file", ""),
            "ok": r["ok"],
            "local_id": local_id,
            **identity_fields,
            "paper_type": r.get("paper_type", "paper"),
            "sections": r.get("sections", 0),
            "error": r.get("error"),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "file": cleaned.get("_file", ""),
            "ok": False,
            "local_id": local_id,
            **identity_fields,
            "error": _safe_pipeline_error(e, cfg),
        }


def _resolve_identity(
    cleaned: dict, db: Database, dedup: DedupEngine
) -> tuple[PaperIDs, str, bool]:
    """Resolve a paper identity without mutating the database."""
    doi = (cleaned.get("doi") or _parse_doi_file(cleaned["_file"])).strip().lower()
    title = cleaned.get("title") or ""
    year = cleaned.get("year")
    if year and isinstance(year, str) and year.isdigit():
        year = int(year)
    ids = _paper_ids(cleaned, doi or None)
    local_id = dedup.resolve(ids, title=title, year=year)
    is_new = local_id is None
    if local_id is None:
        local_id = canonical_paper_id(ids, title=title, year=year, source_key=cleaned["_file"])
        if db.get_paper(local_id) is not None:
            raise ValueError(
                f"canonical paper ID collision for {local_id!r}; refusing to merge records"
            )
    return ids, local_id, is_new


def _persist_identity(cleaned: dict, local_id: str, ids: PaperIDs, db: Database) -> None:
    """Persist a resolved identity after its worker artifacts are complete."""
    title = cleaned.get("title") or ids.doi or ""
    year = cleaned.get("year")
    if isinstance(year, str) and year.isdigit():
        year = int(year)
    if db.get_paper(local_id) is None:
        db.insert_paper(local_id, title, year, "uploaded", paper_type="paper", strict=True)
    db.insert_paper_ids(
        local_id,
        doi=ids.doi,
        arxiv=ids.arxiv,
        s2_id=ids.s2_id,
        openalex_id=ids.openalex_id,
        strict=True,
    )


def _identify(cleaned: dict, db: Database, dedup: DedupEngine, *, persist: bool = True) -> str:
    """Resolve a local ID; optionally retain legacy immediate-write behavior."""
    ids, local_id, _ = _resolve_identity(cleaned, db, dedup)
    if persist:
        savepoint = f"identify_{uuid.uuid4().hex}"
        had_transaction = db.conn.in_transaction
        db.conn.execute(f"SAVEPOINT {savepoint}")
        try:
            _persist_identity(cleaned, local_id, ids, db)
            db.conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            if not had_transaction:
                db.commit()
        except Exception:
            try:
                db.conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            finally:
                db.conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            if not had_transaction:
                db.conn.rollback()
            raise
    return local_id


def _file_sha256(path: Path) -> str:
    """Hash a material in bounded memory for artifact fingerprints."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_db(
    rec: dict,
    db: Database,
    cfg: dict | None = None,
    *,
    allow_partial: bool = False,
) -> None:
    """Persist worker output after artifacts are written.

    A failed worker may still have produced ``raw.md``.  Keep that durable
    input and its identity in the shard, while marking the failed tree stage so
    a later resume can retry only the missing work.
    """
    local_id = rec.get("local_id")
    if not local_id or (not rec.get("ok") and not allow_partial):
        return
    ids = _paper_ids(rec, rec.get("doi"))
    papers_root = Path(db.path).parent / "papers"
    # The configured path is preferred; this fallback keeps the cache loader
    # useful for data-only runtimes where the DB sits beside ``papers/``.
    configured_root = rec.get("papers_root") or ((cfg or {}).get("dirs", {}) or {}).get("papers")
    if configured_root:
        papers_root = Path(configured_root)
    paper_path = paper_dir(papers_root, local_id)
    raw_path = paper_path / "raw.md"
    tree_path = paper_path / "tree.json"
    if not raw_path.is_file():
        return
    _persist_identity(rec, local_id, ids, db)
    if rec.get("paper_type"):
        db.set_paper_type(local_id, rec["paper_type"])
    db.set_paper_status(local_id, "uploaded")
    db.upsert_paper_artifact(
        local_id,
        "raw",
        "ready",
        fingerprint=_file_sha256(raw_path),
        metadata_json=json.dumps({"source": "scibase"}),
    )
    if tree_path.is_file():
        db.upsert_paper_artifact(
            local_id,
            "tree",
            "ready",
            fingerprint=hashlib.sha256(tree_path.read_bytes()).hexdigest(),
            metadata_json=json.dumps({"sections": rec.get("sections", 0)}),
        )
    else:
        db.upsert_paper_artifact(
            local_id,
            "tree",
            "degraded" if rec.get("ok") else "failed",
            error=str(rec.get("error") or "tree.json missing"),
        )
    db.commit()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # Resolve the default after parsing so an embedded caller that changes
    # DRBRAIN_ROOT after importing this module still uses the active root.
    ap.add_argument("--source", type=str, default=None)
    ap.add_argument("--db", type=str, required=True)
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--no-db",
        action="store_true",
        help="只缓存文件（raw.md + tree.json），不写 db；local_id 使用 canonical identity",
    )
    args = ap.parse_args()

    cfg = None
    try:
        root = runtime_root()
        cfg = load_cfg(args.config, root=root)
        # ``--source`` is an explicitly read-only input boundary and may live
        # on a shared corpus volume; generated papers, manifests, and DB
        # writes remain rooted by the runtime config below.
        # ``None`` means "use the default"; an explicitly empty argument is
        # invalid and must not silently redirect ingestion to another corpus.
        if args.source == "":
            raise ValueError("source path must not be empty")
        source_arg = (
            args.source
            if args.source is not None
            else str(root / "data" / "fulltext-cleaned-20260806")
        )
        source = runtime_path(source_arg, root, allow_external=True)
        manifest = runtime_path(args.manifest, root)
        db_path = runtime_path(args.db, root)
    except (OSError, TypeError, ValueError) as exc:
        print(
            f"[scibase] runtime/config error: {_safe_pipeline_error(exc, cfg)}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    if not source.is_dir():
        print("[scibase] source directory missing", file=sys.stderr, flush=True)
        return 1
    files = sorted(source.glob("*.json"))
    if args.limit and args.limit > 0:
        files = files[: args.limit]
    if not files:
        print("[scibase] source contains no JSON records", file=sys.stderr, flush=True)
        return 1

    # Validate the complete resume manifest and every selected source record
    # before creating a destination DB or appending a new line.  This keeps a
    # single corrupt input from leaving a partially ingestible shard behind.
    done, manifest_errors = _preflight_manifest(manifest)
    if manifest_errors:
        print(
            f"[scibase] manifest validation failed: {len(manifest_errors)} error(s)",
            file=sys.stderr,
            flush=True,
        )
        for error in manifest_errors[:10]:
            print(f"  {error}", file=sys.stderr, flush=True)
        return 1
    records: list[dict] = []
    source_errors: list[str] = []
    for path in files:
        record, error = _preflight_source_file(path)
        if error:
            source_errors.append(f"{path.name}: {error}")
        elif record is not None:
            records.append(record)
    if source_errors:
        print(
            f"[scibase] source validation failed: {len(source_errors)} error(s)",
            file=sys.stderr,
            flush=True,
        )
        for error in source_errors[:10]:
            print(f"  {error}", file=sys.stderr, flush=True)
        return 1
    if not records:
        print("[scibase] no valid source records", file=sys.stderr, flush=True)
        return 1
    print(f"[scibase] source records: {len(records)}")
    if done:
        print(f"[resume] 已跳过 {len(done)} 篇")

    try:
        concurrency = int(os.environ.get("INGEST_CONCURRENCY", "1"))
        worker_timeout = _process_worker_timeout()
    except (TypeError, ValueError):
        print(
            "[scibase] INGEST_CONCURRENCY must be a positive integer and "
            "INGEST_PAPER_TIMEOUT must be a finite positive number",
            file=sys.stderr,
        )
        return 1
    if concurrency <= 0:
        print("[scibase] INGEST_CONCURRENCY must be a positive integer", file=sys.stderr)
        return 1
    pending_records = [record for record in records if record["_file"] not in done]
    # A fully completed, validated manifest is a legitimate idempotent no-op;
    # avoid opening the DB or touching the manifest in that case.
    if not pending_records:
        print("[scibase] all source records already completed")
        return 0

    stats = Counter()
    bad: list[dict] = []
    t0 = time.monotonic()
    # ``--no-db`` is a cache-only mode: do not even create/open a shard DB.
    # This keeps concurrent cache preparation independent from database
    # writers and prevents an accidental empty DB from being mistaken for a
    # completed ingest.
    db = None
    manifest_f = None
    try:
        if not args.no_db:
            db = Database(db_path)
            dedup = DedupEngine(db)
        else:
            dedup = None
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest_f = open(manifest, "a", encoding="utf-8")
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        if db is not None:
            db.close()
        print(
            f"[scibase] unable to open outputs: {_safe_pipeline_error(exc, cfg)}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    try:
        # 主进程 identify（写 papers 表）→ worker 抽 LLM（不碰 db）→ 主进程写 db
        # --no-db 模式：local_id 使用 canonical-v1（确定性去重），完全不写 db，只缓存文件
        pending: list[tuple] = []
        planned_ids: set[str] = set()
        for cleaned in pending_records:
            fn_name = cleaned["_file"]
            try:
                if args.no_db:
                    doi = (cleaned.get("doi") or _parse_doi_file(fn_name)).strip().lower()
                    ids = _paper_ids(cleaned, doi or None)
                    local_id = canonical_paper_id(
                        ids,
                        title=cleaned.get("title") or "",
                        year=cleaned.get("year"),
                        source_key=fn_name,
                    )
                else:
                    local_id = _identify(cleaned, db, dedup, persist=False)
                if local_id in planned_ids:
                    raise ValueError(
                        f"duplicate pending paper identity {local_id!r}; refusing concurrent overwrite"
                    )
                planned_ids.add(local_id)
            except Exception as e:  # noqa: BLE001
                rec = {
                    "file": fn_name,
                    "ok": False,
                    "local_id": None,
                    "error": f"identify: {_safe_pipeline_error(e, cfg)}",
                }
                bad.append(rec)
                manifest_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            # Pass the invocation root explicitly.  ProcessPool workers may
            # inherit a different cwd under spawn, and must still load the
            # same build overlay and write to the same paper namespace.
            pending.append((cleaned, cfg, local_id, root))

        def consume_worker_result(rec: dict) -> None:
            """Publish one worker result, keeping DB writes after artifacts."""
            if not args.no_db and db is not None and rec.get("local_id"):
                try:
                    _write_db(rec, db, cfg, allow_partial=True)
                except Exception as exc:  # noqa: BLE001
                    if db is not None:
                        db.conn.rollback()
                    rec = {
                        **rec,
                        "ok": False,
                        "error": f"database: {_safe_pipeline_error(exc, cfg)}",
                    }
            stats["ok" if rec["ok"] else "fail"] += 1
            manifest_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            manifest_f.flush()
            if not rec["ok"]:
                bad.append(rec)

        def _timeout_record(task: tuple, seconds: float) -> dict:
            cleaned = task[0] if task and isinstance(task[0], dict) else {}
            return {
                "file": cleaned.get("_file", ""),
                "ok": False,
                "timeout": True,
                "local_id": task[2] if len(task) > 2 else None,
                "error": f"timeout>{seconds:g}s",
            }

        def _exception_record(task: tuple, exc: Exception) -> dict:
            cleaned = task[0] if task and isinstance(task[0], dict) else {}
            cfg_for_error = task[1] if len(task) > 1 else cfg
            return {
                "file": cleaned.get("_file", ""),
                "ok": False,
                "local_id": task[2] if len(task) > 2 else None,
                "error": _safe_pipeline_error(exc, cfg_for_error),
            }

        completed = 0

        def _consume_pool_result(rec: dict) -> None:
            nonlocal completed
            completed += 1
            consume_worker_result(rec)
            if completed % 10 == 0 or completed == len(pending):
                print(
                    f"[{completed}/{len(pending)}] ok={stats['ok']} fail={stats['fail']} "
                    f"elapsed={time.monotonic() - t0:.0f}s",
                    flush=True,
                )

        if pending and concurrency <= 1 and not _can_pickle_process_task(_worker, pending[0]):
            # Embedded callers occasionally replace the worker with a local
            # function that cannot cross a ProcessPool boundary.  Keep that
            # API working while enforcing a main-thread signal deadline.
            for i, task in enumerate(pending, 1):
                rec = _run_serial_worker_with_timeout(
                    _worker,
                    task,
                    timeout=worker_timeout,
                    timeout_result=_timeout_record,
                    exception_result=_exception_record,
                )
                consume_worker_result(rec)
                if rec.get("timeout") is True:
                    # A serial timeout has the same stage contract as a pool
                    # timeout: stop before starting another paper.
                    break
                if i % 10 == 0 or i == len(pending):
                    print(
                        f"[{i}/{len(pending)}] ok={stats['ok']} fail={stats['fail']} "
                        f"elapsed={time.monotonic() - t0:.0f}s",
                        flush=True,
                    )
        else:
            try:
                _run_process_pool_fail_fast(
                    pending,
                    _worker,
                    max_workers=concurrency,
                    timeout=worker_timeout,
                    on_result=_consume_pool_result,
                    timeout_result=_timeout_record,
                    exception_result=_exception_record,
                )
            except Exception as exc:  # noqa: BLE001
                # Executor construction/shutdown failures are pipeline
                # failures too; keep them bounded and visible in the manifest.
                consume_worker_result(
                    {
                        "file": "",
                        "ok": False,
                        "local_id": None,
                        "error": f"executor: {_safe_pipeline_error(exc, cfg)}",
                    }
                )
    finally:
        if manifest_f is not None:
            manifest_f.close()
        if db is not None:
            db.close()

    print(f"\n完成: ok={stats['ok']} fail={stats['fail']} ({time.monotonic() - t0:.0f}s)")
    for r in bad[:20]:
        print(f"  FAIL {r['file']}: {_safe_pipeline_error(r.get('error'), cfg)}")

    # A manifest is useful for resume, but it must not turn a partially
    # successful ingest into a successful process exit.  The shard wrappers
    # use this status to stop before build/load/embed and leave DONE absent.
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
