#!/usr/bin/env python
"""批量 build：对已 ingest 的论文跑 5-stage 因果推理抽取。

jsonl-out 模式：只抽 LLM 结果写 jsonl（不写 db，并发无锁），入库单独跑
（load_build.py）。并发：BUILD_CONCURRENCY 个线程 × 每篇内部叶子并发。

用法:
    uv run python scripts/pipeline/build.py --from-manifest data/shards/shard0.ingest.jsonl \
        --db data/shards/shard0.db --manifest data/shards/shard0.build.jsonl \
        --jsonl-out data/shards/shard0.build.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

# A runtime root may be a fresh data-only directory.  Import project code from
# the checkout containing this script before resolving the runtime namespace.
SOURCE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_SRC = SOURCE_ROOT / "src"
for _import_root in (SOURCE_ROOT, SOURCE_SRC):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from drbrain.runtime import runtime_root  # noqa: E402
from drbrain.security import configured_secret_values, safe_error  # noqa: E402
from drbrain.storage.database import Database  # noqa: E402
from drbrain.storage.paths import paper_dir, paper_fs_key, writable_artifact_path  # noqa: E402

# Compatibility snapshot for callers that imported this module directly.  Do
# not resolve ``DRBRAIN_ROOT`` while importing: command entrypoints validate it
# at invocation time and can return a concise non-zero status.
ROOT = SOURCE_ROOT

from scripts.pipeline.common import (  # noqa: E402
    can_pickle_process_task,
    load_cfg,
    run_process_pool_fail_fast,
    run_serial_worker_with_timeout,
    runtime_path,
)

VALID_TYPES = {"Problem", "Method", "Conclusion", "Debate", "Gap", "Actor"}


def _safe_pipeline_error(value: object, cfg: object | None = None) -> str:
    """Render a bounded provider/worker error without persisting credentials."""

    try:
        secrets = (
            configured_secret_values(cfg)
            if cfg is not None
            else configured_secret_values(os.environ)
        )
    except Exception:  # noqa: BLE001 - an error boundary must never mask itself
        secrets = ()
    if isinstance(value, BaseException):
        value = f"{type(value).__name__}: {value}"
    return safe_error(value, secrets=secrets)


def _persist_build_payload(local_id: str, result: dict, db: Database) -> dict:
    """Commit one staged extraction result atomically in the parent process.

    Workers intentionally never receive the destination ``Database``.  A
    savepoint keeps a malformed relation from leaving concepts/status rows
    behind while preserving any transaction owned by the caller.
    """

    savepoint = f"build_stage_{uuid.uuid4().hex}"
    had_transaction = bool(db.conn.in_transaction)
    db.conn.execute(f"SAVEPOINT {savepoint}")
    try:
        concepts = result.get("concepts", [])
        relations = result.get("relations", [])
        if not isinstance(concepts, list) or not isinstance(relations, list):
            raise ValueError("staged extraction fields must be lists")
        valid_count = 0
        rejected = 0
        for concept in concepts:
            if not isinstance(concept, dict):
                rejected += 1
                continue
            ctype = concept.get("type", "")
            label = concept.get("label", "")
            confidence = concept.get("confidence", 0.5)
            if ctype not in VALID_TYPES or not label:
                rejected += 1
                continue
            db.insert_concept(
                local_id,
                ctype,
                label,
                confidence,
                section=concept.get("section", ""),
                node_id=concept.get("node_id", ""),
            )
            valid_count += 1

        for index, relation in enumerate(relations):
            if not isinstance(relation, dict):
                raise ValueError(f"relation {index} is not an object")
            head = relation.get("head", "")
            rel = relation.get("rel", "")
            tail = relation.get("tail", "")
            if not (head and rel and tail):
                continue
            db.insert_edge(
                head,
                tail,
                rel,
                local_id,
                node_id=relation.get("node_id", ""),
                section=relation.get("section", ""),
            )
        db.set_paper_status(local_id, "extracted")
        db.conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        if not had_transaction:
            db.commit()
        return {
            "concepts": valid_count,
            "relations": len(relations),
            "rejected": rejected,
        }
    except Exception:
        try:
            db.conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        finally:
            db.conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        if not had_transaction:
            db.rollback()
        raise


def build_one(
    local_id: str,
    cfg: dict,
    db: Database | None,
    skip_refine: bool,
    jsonl_only: bool = False,
    *,
    root: Path | None = None,
    timeout_seconds: float | None = None,
) -> dict:
    """复刻 build_cmd 核心：tree → 5-stage 抽取 → 插入 concepts/edges。

    jsonl_only=True 时只返回抽取结果不写 db（并发 build 无锁，入库单独跑）。
    """
    import json as _json

    from loguru import logger

    from drbrain.extractor.cache import ApiCache
    from drbrain.extractor.concept import build_graph_from_tree

    active_root = Path(root if root is not None else runtime_root()).expanduser().resolve()
    papers_dir = Path(cfg.get("dirs", {}).get("papers", "data/papers"))
    if not papers_dir.is_absolute():
        papers_dir = active_root / papers_dir
    paper_path = paper_dir(papers_dir, local_id)
    # Artifact files are read from an untrusted cache.  Reuse the same lexical
    # symlink checks used by write paths so a stale tree/raw.md alias cannot
    # make a worker read another runtime's corpus.
    tree_path = writable_artifact_path(paper_path, "tree.json")
    md_path = writable_artifact_path(paper_path, "raw.md")
    if not tree_path.is_file() or not md_path.is_file():
        return {"ok": False, "local_id": local_id, "error": "tree/raw.md 缺失"}
    tree = _json.loads(tree_path.read_text(encoding="utf-8"))
    structure = tree.get("structure", [])
    if not structure:
        return {"ok": False, "local_id": local_id, "error": "empty tree"}

    llm_models = cfg.get("llm", {}).get("models", [])
    # The cache is mutable runtime state; derive it from the invocation root
    # instead of the import-time checkout path.
    cache = ApiCache(
        str(active_root / "data" / "spool" / "llm_cache"),
        secrets=configured_secret_values(cfg),
    )
    t0 = time.monotonic()

    async def _run_extraction():
        extraction = build_graph_from_tree(
            md_path, structure, llm_models, skip_refine=skip_refine, cache=cache
        )
        if timeout_seconds is None:
            return await extraction
        return await asyncio.wait_for(extraction, timeout=float(timeout_seconds))

    try:
        result = asyncio.run(_run_extraction())
    except TimeoutError:
        limit = timeout_seconds if timeout_seconds is not None else 0
        return {
            "ok": False,
            "local_id": local_id,
            "error": f"timeout>{limit}s",
            "timeout": True,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "local_id": local_id,
            "error": _safe_pipeline_error(e, cfg),
        }

    concepts = result.get("concepts", [])
    relations = result.get("relations", [])
    merges = result.get("merges", [])
    corrections = result.get("corrections", [])

    if jsonl_only:
        return {
            "ok": True,
            "local_id": local_id,
            "concepts": concepts,
            "relations": relations,
            "merges": merges,
            "corrections": corrections,
            "report": {
                "concepts": len(concepts),
                "relations": len(relations),
                "merges": len(merges),
                "corrections": len(corrections),
                "secs": time.monotonic() - t0,
            },
        }

    valid_count = 0
    rejected = 0
    for c in concepts:
        ctype = c.get("type", "")
        label = c.get("label", "")
        conf = c.get("confidence", 0.5)
        if ctype not in VALID_TYPES or not label:
            rejected += 1
            continue
        db.insert_concept(
            local_id, ctype, label, conf, section=c.get("section", ""), node_id=c.get("node_id", "")
        )
        valid_count += 1

    for index, relation in enumerate(relations):
        if not isinstance(relation, dict):
            try:
                db.conn.rollback()
            except Exception:  # noqa: BLE001
                pass
            return {
                "ok": False,
                "local_id": local_id,
                "error": f"relation {index} is not an object",
            }
        head = relation.get("head", "")
        rel = relation.get("rel", "")
        tail = relation.get("tail", "")
        if not (head and rel and tail):
            continue
        try:
            db.insert_edge(
                head,
                tail,
                rel,
                local_id,
                node_id=relation.get("node_id", ""),
                section=relation.get("section", ""),
            )
        except Exception as exc:  # noqa: BLE001 - a partial graph is unsafe
            try:
                db.conn.rollback()
            except Exception:  # noqa: BLE001 - preserve the relation error
                pass
            return {
                "ok": False,
                "local_id": local_id,
                "error": f"relation {index} failed: {_safe_pipeline_error(exc, cfg)}",
            }

    db.set_paper_status(local_id, "extracted")
    db.commit()
    logger.info(
        f"[build] {local_id} done in {time.monotonic() - t0:.0f}s — "
        f"concepts={valid_count} relations={len(relations)} merges={len(merges)} "
        f"corrections={len(corrections)} rejected={rejected}"
    )
    return {
        "ok": True,
        "local_id": local_id,
        "report": {
            "concepts": valid_count,
            "relations": len(relations),
            "merges": len(merges),
            "corrections": len(corrections),
            "secs": time.monotonic() - t0,
        },
    }


def _run_build_worker(
    local_id: str,
    cfg: dict,
    skip_refine: bool,
    root: Path,
    timeout_seconds: float,
) -> dict:
    """Run one extraction without ever opening the destination database."""

    try:
        result = build_one(
            local_id,
            cfg,
            None,
            skip_refine=skip_refine,
            jsonl_only=True,
            root=root,
            timeout_seconds=timeout_seconds,
        )
        return {
            "local_id": local_id,
            "ok": result["ok"],
            "error": result.get("error"),
            "timeout": bool(result.get("timeout")),
            "concepts": result.get("concepts", []),
            "relations": result.get("relations", []),
            "merges": result.get("merges", []),
            "corrections": result.get("corrections", []),
            "report": result.get("report"),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "local_id": local_id,
            "ok": False,
            "error": _safe_pipeline_error(exc, cfg),
        }


def _build_process_worker(task: tuple) -> dict:
    """Unpack one process-pool task at a pickle-friendly module boundary."""

    local_id, cfg, skip_refine, root, timeout_seconds = task
    return _run_build_worker(local_id, cfg, skip_refine, root, timeout_seconds)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--from-manifest",
        type=str,
        required=True,
        help="ingest manifest（成功入库的 local_id 列表）",
    )
    ap.add_argument("--db", type=str, required=True)
    ap.add_argument(
        "--manifest", type=str, required=True, help="build manifest 输出路径（断点续传）"
    )
    ap.add_argument(
        "--jsonl-out", type=str, default=None, help="只抽 LLM 结果写 jsonl（不写 db），入库单独跑"
    )
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-refine", action="store_true", default=True)
    args = ap.parse_args()

    try:
        root = runtime_root()
        cfg = load_cfg(args.config, root=root)
        db_path = str(runtime_path(args.db, root))
        manifest_path = runtime_path(args.manifest, root)
        jsonl_out_path = runtime_path(args.jsonl_out, root) if args.jsonl_out else None
        src = runtime_path(args.from_manifest, root)
    except (OSError, TypeError, ValueError) as exc:
        print(f"[build] runtime/config error: {_safe_pipeline_error(exc)}", file=sys.stderr)
        return 1

    # 从 ingest manifest 收集成功入库的 local_id。该文件是上游阶段的
    # 完成证明：缺失、损坏或含失败记录时必须在创建输出 manifest/DB 之前
    # 停止，否则空 build 产物会被下游误认为成功。
    ids: set[str] = set()
    if not src.is_file() or src.is_symlink():
        print(f"[build] from-manifest 不存在: {src}", file=sys.stderr)
        return 1
    manifest_errors: list[str] = []
    failed_upstream = 0
    try:
        source_lines = src.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(
            f"[build] from-manifest 读取失败: {src}: {_safe_pipeline_error(exc)}",
            file=sys.stderr,
        )
        return 1
    for line_no, line in enumerate(source_lines, 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            manifest_errors.append(f"{src}:{line_no}: invalid JSON ({_safe_pipeline_error(exc)})")
            continue
        if not isinstance(rec, dict):
            manifest_errors.append(f"{src}:{line_no}: record must be an object")
            continue
        if not isinstance(rec.get("ok"), bool):
            manifest_errors.append(f"{src}:{line_no}: ok must be boolean")
            continue
        if not rec["ok"]:
            failed_upstream += 1
            continue
        local_id = rec.get("local_id")
        if not isinstance(local_id, str) or not local_id.strip():
            manifest_errors.append(f"{src}:{line_no}: successful record missing local_id")
            continue
        try:
            paper_fs_key(local_id)
        except (TypeError, ValueError) as exc:
            manifest_errors.append(
                f"{src}:{line_no}: invalid local_id: {_safe_pipeline_error(exc)}"
            )
            continue
        ids.add(local_id)
    if manifest_errors:
        print(f"[build] from-manifest 校验失败: {len(manifest_errors)} 条", file=sys.stderr)
        for error in manifest_errors[:10]:
            print(f"  {error}", file=sys.stderr)
        return 1
    if failed_upstream:
        print(f"[build] from-manifest 含失败记录: {failed_upstream} 条", file=sys.stderr)
        return 1
    if not ids:
        print(f"[build] from-manifest 没有成功记录: {src}", file=sys.stderr)
        return 1
    papers = [{"local_id": lid} for lid in sorted(ids)]
    print(f"[build] from-manifest: {len(papers)} 篇")

    done: set[str] = set()
    # An existing output is still an input boundary: even a non-resume run
    # must not append after a truncated/corrupt JSONL record.  Only successful
    # records are used for skipping when --resume is explicitly requested.
    if manifest_path.exists():
        try:
            resume_lines = manifest_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            print(
                f"[build] resume manifest 读取失败: {manifest_path}: {_safe_pipeline_error(exc)}",
                file=sys.stderr,
            )
            return 1
        resume_errors: list[str] = []
        for line_no, line in enumerate(resume_lines, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                resume_errors.append(
                    f"{manifest_path}:{line_no}: invalid JSON ({_safe_pipeline_error(exc)})"
                )
                continue
            if not isinstance(rec, dict):
                resume_errors.append(f"{manifest_path}:{line_no}: record must be an object")
                continue
            if rec.get("ok") is True:
                local_id = rec.get("local_id")
                if isinstance(local_id, str) and local_id.strip():
                    try:
                        paper_fs_key(local_id)
                    except (TypeError, ValueError) as exc:
                        resume_errors.append(
                            f"{manifest_path}:{line_no}: invalid local_id: {_safe_pipeline_error(exc)}"
                        )
                    else:
                        if args.resume:
                            done.add(local_id)
                else:
                    resume_errors.append(
                        f"{manifest_path}:{line_no}: successful record missing local_id"
                    )
            elif rec.get("ok") is not False:
                resume_errors.append(f"{manifest_path}:{line_no}: ok must be boolean")
        if resume_errors:
            print(f"[build] resume manifest 校验失败: {len(resume_errors)} 条", file=sys.stderr)
            for error in resume_errors[:10]:
                print(f"  {error}", file=sys.stderr)
            return 1
        if args.resume:
            print(f"[resume] 已跳过 {len(done)} 篇")

    pending = [p["local_id"] for p in papers if p["local_id"] not in done]
    # A validated resume manifest may describe a fully completed batch.  Keep
    # this idempotent no-op free of DB/file side effects.
    if not pending:
        print("[build] all papers already completed")
        return 0

    try:
        concurrency = int(os.environ.get("BUILD_CONCURRENCY", "4"))
        per_paper_timeout = float(os.environ.get("BUILD_PAPER_TIMEOUT", "900"))
    except (TypeError, ValueError):
        print(
            "[build] BUILD_CONCURRENCY and BUILD_PAPER_TIMEOUT must be positive integers",
            file=sys.stderr,
        )
        return 1
    if concurrency <= 0 or not math.isfinite(per_paper_timeout) or per_paper_timeout <= 0:
        print(
            "[build] BUILD_CONCURRENCY and BUILD_PAPER_TIMEOUT must be positive integers",
            file=sys.stderr,
        )
        return 1

    if manifest_path.is_symlink() or (jsonl_out_path is not None and jsonl_out_path.is_symlink()):
        print("[build] output paths must not be symlinks", file=sys.stderr)
        return 1
    stats = Counter()
    bad: list[dict] = []
    t0 = time.monotonic()
    manifest_f = None
    output_f = None
    target_db: Database | None = None
    manifest_lock = threading.Lock()
    completed_count = 0
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_f = open(manifest_path, "a", encoding="utf-8")
        if jsonl_out_path is not None and jsonl_out_path != manifest_path:
            jsonl_out_path.parent.mkdir(parents=True, exist_ok=True)
            output_f = open(jsonl_out_path, "a", encoding="utf-8")

        def _emit(rec: dict) -> None:
            nonlocal completed_count
            nonlocal target_db
            completed_count += 1
            if rec.get("local_id") not in pending:
                rec = {
                    "local_id": "__protocol__",
                    "ok": False,
                    "error": "worker returned an unexpected paper identity",
                }
            # Open the destination only after the process pool has produced a
            # result.  Forked workers therefore never inherit a live SQLite
            # connection, and an all-failed batch does not create a DB as a
            # misleading success artifact.
            if rec.get("ok") is True and jsonl_out_path is None:
                try:
                    if target_db is None:
                        target_db = Database(db_path)
                    persisted = _persist_build_payload(rec["local_id"], rec, target_db)
                    report = rec.get("report")
                    if isinstance(report, dict):
                        report = {**report, **persisted}
                    else:
                        report = persisted
                    rec = {**rec, "report": report}
                except Exception as exc:  # noqa: BLE001
                    # A failed parent commit must be visible in the manifest;
                    # the helper has already rolled its savepoint back.
                    rec = {
                        **rec,
                        "ok": False,
                        "error": f"database: {_safe_pipeline_error(exc, cfg)}",
                    }
            stats["ok" if rec.get("ok") is True else "fail"] += 1
            with manifest_lock:
                serialized = json.dumps(rec, ensure_ascii=False) + "\n"
                if output_f is not None:
                    output_f.write(serialized)
                    output_f.flush()
                    os.fsync(output_f.fileno())
                manifest_f.write(serialized)
                manifest_f.flush()
                os.fsync(manifest_f.fileno())
            if not rec["ok"]:
                bad.append(rec)
            if completed_count % 5 == 0 or completed_count == len(pending):
                print(
                    f"[{completed_count}/{len(pending)}] ok={stats['ok']} fail={stats['fail']} "
                    f"elapsed={time.monotonic() - t0:.0f}s",
                    flush=True,
                )

        tasks = [(lid, cfg, args.skip_refine, root, per_paper_timeout) for lid in pending]

        def _timeout_record(task: tuple, seconds: float) -> dict:
            return {
                "local_id": task[0] if task else "__unknown__",
                "ok": False,
                "error": f"timeout>{seconds:g}s",
            }

        def _exception_record(task: tuple, exc: Exception) -> dict:
            return {
                "local_id": task[0] if task else "__unknown__",
                "ok": False,
                "error": _safe_pipeline_error(exc, cfg),
            }

        worker = _build_process_worker
        if not can_pickle_process_task(worker, tasks[0]):
            if concurrency > 1:
                raise ValueError("build worker/config is not pickleable for concurrent execution")
            # Direct embedded callers may monkeypatch a local worker.  Keep a
            # bounded compatibility path for one-at-a-time execution.
            for task in tasks:
                rec = run_serial_worker_with_timeout(
                    worker,
                    task,
                    timeout=per_paper_timeout,
                    timeout_result=_timeout_record,
                    exception_result=_exception_record,
                )
                _emit(rec)
                if str(rec.get("error", "")).startswith("timeout>"):
                    break
        else:
            run_process_pool_fail_fast(
                tasks,
                worker,
                max_workers=concurrency,
                timeout=per_paper_timeout,
                on_result=_emit,
                timeout_result=_timeout_record,
                exception_result=_exception_record,
            )
    except Exception as exc:  # noqa: BLE001
        bad.append(
            {
                "local_id": "__pipeline__",
                "ok": False,
                "error": f"pipeline: {_safe_pipeline_error(exc, cfg)}",
            }
        )
    finally:
        if manifest_f is not None:
            manifest_f.close()
        if output_f is not None:
            output_f.close()
        if target_db is not None:
            target_db.close()

    print(f"\n完成: ok={stats['ok']} fail={stats['fail']} ({time.monotonic() - t0:.0f}s)")
    for r in bad[:15]:
        print(f"  FAIL {r['local_id']}: {_safe_pipeline_error(r.get('error'), cfg)}")
    if bad:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
