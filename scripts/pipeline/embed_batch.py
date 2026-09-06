#!/usr/bin/env python
"""批量 tree 向量嵌入：从文件读 local_id 列表，逐篇生成 vectors+summaries。

用法:
    uv run python scripts/pipeline/embed_batch.py --ids-file data/.runtime/embed_q1.txt \
        --config config.embed1.yaml --db data/drbrain.db
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import threading
from pathlib import Path

# Import code from the checkout, independently from the runtime data root.
SOURCE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_SRC = SOURCE_ROOT / "src"
for _import_root in (SOURCE_ROOT, SOURCE_SRC):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from drbrain.runtime import RuntimeContext, runtime_root  # noqa: E402
from drbrain.security import configured_secret_values, safe_error  # noqa: E402

# Compatibility snapshot; runtime data paths are resolved inside ``main``.
ROOT = SOURCE_ROOT

from drbrain.storage.database import Database  # noqa: E402
from drbrain.storage.paths import paper_dir, paper_fs_key, tree_json_path  # noqa: E402
from scripts.pipeline.common import (  # noqa: E402
    can_pickle_process_task,
    load_cfg,
    run_process_pool_fail_fast,
    run_serial_worker_with_timeout,
    runtime_path,
)


def _safe_pipeline_error(value: object, cfg: object | None = None) -> str:
    """Render a bounded provider/worker error without exposing credentials."""

    try:
        secrets = (
            configured_secret_values(cfg)
            if cfg is not None
            else configured_secret_values(os.environ)
        )
    except Exception:  # noqa: BLE001 - error handling must not mask the failure
        secrets = ()
    if isinstance(value, BaseException):
        value = f"{type(value).__name__}: {value}"
    return safe_error(value, secrets=secrets)


def _staging_root(root: Path) -> Path:
    """Return a private, runtime-contained directory for one-paper DBs."""

    runtime = RuntimeContext.create(root)
    # Validate the complete lexical path before mkdir.  Creating first would
    # follow a pre-existing ``data``/``.runtime`` symlink and publish staging
    # databases outside the selected worktree.
    staging = runtime.assert_within_root(
        root / "data" / ".runtime" / "embed-staging",
        label="embedding staging directory",
    )
    staging.mkdir(parents=True, exist_ok=True)
    staging = runtime.assert_within_root(staging, label="embedding staging directory")
    if staging.is_symlink() or not staging.is_dir():
        raise ValueError(f"embedding staging directory is unsafe: {staging}")
    return staging


def _new_staging_db(root: Path) -> tuple[Path, Path]:
    """Create a unique staging SQLite path and its owning directory."""

    directory = Path(tempfile.mkdtemp(prefix="embed-", dir=_staging_root(root)))
    return directory / "stage.db", directory


def _cleanup_staging_dir(directory: Path | None) -> None:
    """Best-effort cleanup for a completed staging job."""

    if directory is None:
        return
    try:
        import shutil

        shutil.rmtree(directory)
    except (FileNotFoundError, OSError):
        pass


def _import_staged_rows(
    stage_path: Path,
    target_db: Database,
    paper_id: str,
    *,
    include_raptor: bool,
) -> int:
    """Atomically import one worker's vector rows into the parent DB.

    The staging DB is treated as untrusted worker output.  Only rows whose
    ``paper_id`` matches the requested paper are accepted, and a savepoint
    prevents a malformed row from publishing a partial paper update.
    """

    import sqlite3

    conn = sqlite3.connect(str(stage_path))
    try:
        vector_layers = ("pageindex",) if not include_raptor else None
        if vector_layers is None:
            vector_rows = conn.execute(
                "SELECT node_id, paper_id, embedding, content_hash, tree_layer "
                "FROM tree_vectors WHERE paper_id = ?",
                (paper_id,),
            ).fetchall()
            summary_rows = conn.execute(
                "SELECT node_id, paper_id, summary_text, source_node_ids, tree_layer "
                "FROM tree_summaries WHERE paper_id = ?",
                (paper_id,),
            ).fetchall()
        else:
            vector_rows = conn.execute(
                "SELECT node_id, paper_id, embedding, content_hash, tree_layer "
                "FROM tree_vectors WHERE paper_id = ? AND tree_layer = ?",
                (paper_id, vector_layers[0]),
            ).fetchall()
            summary_rows = []
        for row in (*vector_rows, *summary_rows):
            if row[1] != paper_id:
                raise ValueError("staging row paper_id mismatch")
    finally:
        conn.close()

    imported = target_db.upsert_tree_rows(vector_rows, summary_rows)

    # Keep the optional sqlite-vec mirror aligned when it is available.
    # The base tables above are authoritative if the extension is absent or a
    # dimension mismatch makes the mirror unusable.
    if vector_rows:
        try:
            from drbrain.storage import vector_index as vi

            if vi.ensure_vec_table(target_db.conn, len(vector_rows[0][2]) // 4):
                for node_id, _paper_id, embedding, _hash, _layer in vector_rows:
                    try:
                        vi.vec_upsert(target_db.conn, node_id, embedding)
                    except Exception:  # noqa: BLE001 - base row is durable
                        pass
        except Exception:  # noqa: BLE001 - sqlite-vec is optional
            pass
    return imported


def _embed_process_worker(task: tuple) -> dict:
    """Build one paper's vectors in an isolated child process.

    The destination DB is intentionally absent from this task.  A worker may
    leave partial rows in its private staging DB, but the parent only imports
    a result whose call returned successfully and whose paper/path identity
    matches the submitted task.
    """

    (
        local_id,
        stage_path,
        papers_dir,
        embed_cfg,
        llm_models,
        skip_raptor,
        collect_raptor,
        per_timeout,
        cfg,
        root,
    ) = task
    sink: list[dict] | None = [] if collect_raptor else None
    try:
        pdir = paper_dir(Path(papers_dir), local_id)
        tree_path = tree_json_path(pdir)
        if not tree_path.is_file():
            return {
                "local_id": local_id,
                "count": 0,
                "error": "no tree.json",
                "ok": False,
                "sink": [],
                "stage_path": str(stage_path),
            }

        # Initialise only the isolated staging database.  The embedding
        # service writes its rows there; no child receives the destination DB.
        stage_db = Database(Path(stage_path))
        stage_db.close()
        if skip_raptor:
            from drbrain.services.embedding import build_tree_vectors

            count = build_tree_vectors(
                Path(stage_path),
                pdir,
                embed_cfg,
                paper_id=local_id,
            )
        else:
            from drbrain.extractor.cache import ApiCache

            bridge = __import__("drbrain.services.embedding", fromlist=["build_paper_tree_vectors"])
            cache = ApiCache(
                str(Path(root) / "data" / "spool" / "llm_cache"),
                secrets=configured_secret_values(cfg),
            )

            async def _run_embedding():
                return await asyncio.wait_for(
                    bridge.build_paper_tree_vectors(
                        pdir,
                        Path(stage_path),
                        embed_cfg,
                        llm_models,
                        sink=sink,
                        cache=cache,
                        paper_id=local_id,
                    ),
                    timeout=float(per_timeout),
                )

            count = asyncio.run(_run_embedding())
        return {
            "local_id": local_id,
            "count": int(count or 0),
            "ok": True,
            "error": None,
            "sink": sink or [],
            "stage_path": str(stage_path),
        }
    except Exception as exc:  # noqa: BLE001 - parent records a bounded error
        return {
            "local_id": local_id,
            "count": 0,
            "error": _safe_pipeline_error(exc, cfg),
            "ok": False,
            "timeout": isinstance(exc, TimeoutError),
            "sink": [],
            "stage_path": str(stage_path),
        }


def _load_cfg(config_name: str, *, root: Path | None = None) -> dict:
    """Load an embedding config against the root selected for this call."""

    return load_cfg(config_name, root=root if root is not None else runtime_root())


def _safe_input_file(value: str, root: Path) -> Path:
    """Resolve the IDs file without following a symlink alias."""

    candidate = runtime_path(value, root)
    lexical = Path(value).expanduser()
    if not lexical.is_absolute():
        lexical = root / lexical
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"ids file escapes runtime root {root}: {lexical}") from exc
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"ids file must not contain symlink components: {lexical}")
    if not candidate.is_file():
        raise ValueError(f"ids file is not a regular file: {candidate}")
    return candidate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids-file", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--db", type=str, default="data/drbrain.db")
    ap.add_argument(
        "--skip-raptor",
        action="store_true",
        help="只生成 PageIndex 向量，跳过 RAPTOR（RAPTOR 需要 LLM 摘要，与 build 抢 key）",
    )
    ap.add_argument(
        "--raptor-out",
        type=str,
        default=None,
        help="RAPTOR 结果缓存到 jsonl（先缓存后入库，不写 db），入库用 load_raptor.py",
    )
    args = ap.parse_args()

    try:
        root = runtime_root()
        cfg = _load_cfg(args.config, root=root)
        ids_path = _safe_input_file(args.ids_file, root)
        db_path = runtime_path(args.db, root)
        raptor_path = runtime_path(args.raptor_out, root) if args.raptor_out else None
    except (OSError, TypeError, ValueError) as exc:
        print(f"runtime/config error: {_safe_pipeline_error(exc)}", file=sys.stderr)
        return 1
    try:
        ids_text = ids_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"ids file read error: {_safe_pipeline_error(exc, cfg)}", file=sys.stderr)
        return 1
    ids = [x.strip() for x in ids_text.replace(",", "\n").splitlines() if x.strip()]
    if not ids:
        print("ids file contains no paper IDs", file=sys.stderr)
        return 1
    if len(set(ids)) != len(ids):
        print(
            "ids file contains duplicate paper IDs; refusing concurrent overwrite", file=sys.stderr
        )
        return 1
    invalid_ids = []
    for lid in ids:
        try:
            paper_fs_key(lid)
        except (TypeError, ValueError):
            invalid_ids.append(lid)
    if invalid_ids:
        print(
            f"ids file contains {len(invalid_ids)} invalid paper IDs",
            file=sys.stderr,
        )
        return 1
    print(f"待嵌入: {len(ids)} 篇")

    try:
        papers_dir = RuntimeContext.create(root).assert_within_root(
            Path(cfg.get("dirs", {}).get("papers", root / "data" / "papers")),
            label="papers directory",
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"papers path error: {_safe_pipeline_error(exc, cfg)}", file=sys.stderr)
        return 1
    llm_models = cfg.get("llm", {}).get("models", [])
    embed_cfg = cfg.get("embed", {})
    from drbrain.config import EmbedConfig

    if isinstance(embed_cfg, dict):
        embed_cfg = EmbedConfig(**embed_cfg)

    import json as _json
    import os

    # append 模式:重启/续跑不覆盖已缓存结果(分片清单负责排除已完成篇)
    if raptor_path is not None:
        raptor_path.parent.mkdir(parents=True, exist_ok=True)
    raptor_f = None
    _json_lock = threading.Lock()
    try:
        workers = int(os.environ.get("EMBED_WORKERS", "8"))
        per_timeout = float(os.environ.get("EMBED_PAPER_TIMEOUT", "900"))
    except (TypeError, ValueError):
        print("EMBED_WORKERS and EMBED_PAPER_TIMEOUT must be positive integers", file=sys.stderr)
        return 1
    import math

    if workers <= 0 or not math.isfinite(per_timeout) or per_timeout <= 0:
        print("EMBED_WORKERS and EMBED_PAPER_TIMEOUT must be positive integers", file=sys.stderr)
        return 1

    total_vec = done = 0
    fails: list[tuple[str, str]] = []
    staging: dict[str, tuple[Path, Path]] = {}
    db: Database | None = None

    def _task_iter():
        for lid in ids:
            stage_path, stage_dir = _new_staging_db(root)
            staging[lid] = (stage_path, stage_dir)
            yield (
                lid,
                stage_path,
                papers_dir,
                embed_cfg,
                llm_models,
                args.skip_raptor,
                raptor_path is not None,
                per_timeout,
                cfg,
                root,
            )

    def _timeout_result(task: tuple, seconds: float) -> dict:
        return {
            "local_id": task[0],
            "count": 0,
            "error": f"timeout>{seconds:g}s",
            "timeout": True,
            "sink": [],
        }

    def _exception_result(task: tuple, exc: Exception) -> dict:
        return {
            "local_id": task[0],
            "count": 0,
            "error": _safe_pipeline_error(exc, cfg),
            "transport_error": True,
            "sink": [],
        }

    def _record_result(rec: dict) -> None:
        nonlocal total_vec, done, db, raptor_f
        lid = rec.get("local_id")
        stage_info = staging.get(lid)
        if stage_info is None:
            fails.append((str(lid), "worker returned an unknown paper ID"))
            return
        stage_path, stage_dir = stage_info
        err = rec.get("error")
        count = 0
        sink = rec.get("sink") if isinstance(rec.get("sink"), list) else []
        try:
            returned = Path(rec.get("stage_path", stage_path))
            if returned != stage_path:
                raise ValueError("worker returned an unexpected staging path")
            # Transport failures and explicit worker errors may have left
            # partial rows.  Never import those staging files.
            if rec.get("ok") is not True or err:
                fails.append((lid, _safe_pipeline_error(err, cfg)))
            else:
                if db is None:
                    db = Database(db_path)
                count = _import_staged_rows(stage_path, db, lid, include_raptor=raptor_path is None)
                if sink and raptor_path is not None:
                    if raptor_f is None:
                        raptor_f = open(raptor_path, "a", encoding="utf-8")
                    with _json_lock:
                        for item in sink:
                            raptor_f.write(_json.dumps(item, ensure_ascii=False) + "\n")
                        raptor_f.flush()
        except Exception as exc:  # noqa: BLE001
            fails.append((lid, _safe_pipeline_error(exc, cfg)))
            count = 0
        finally:
            total_vec += count
            done += 1
            if done % 500 == 0 or done == len(ids):
                print(f"[{done}/{len(ids)}] done={done} vec={total_vec}", flush=True)
            if not rec.get("timeout"):
                _cleanup_staging_dir(stage_dir)

    try:
        tasks = _task_iter()
        first_task = next(tasks)
        all_tasks = __import__("itertools").chain((first_task,), tasks)
        worker = _embed_process_worker
        if not can_pickle_process_task(worker, first_task):
            if workers > 1:
                raise ValueError(
                    "embedding worker/config is not pickleable for concurrent execution"
                )
            for task in all_tasks:
                rec = run_serial_worker_with_timeout(
                    worker,
                    task,
                    timeout=per_timeout,
                    timeout_result=_timeout_result,
                    exception_result=_exception_result,
                )
                _record_result(rec)
                if rec.get("timeout"):
                    break
        else:
            run_process_pool_fail_fast(
                all_tasks,
                worker,
                max_workers=workers,
                timeout=per_timeout,
                on_result=_record_result,
                timeout_result=_timeout_result,
                exception_result=_exception_result,
            )
    except Exception as exc:  # noqa: BLE001
        fails.append(("__pipeline__", _safe_pipeline_error(exc, cfg)))
    finally:
        for _path, stage_dir in staging.values():
            _cleanup_staging_dir(stage_dir)
        if raptor_f is not None:
            raptor_f.close()
        if db is not None:
            db.close()
    print(f"\n完成: {done} 篇, {total_vec} vectors, fail={len(fails)}")
    for lid, e in fails[:10]:
        print(f"  FAIL {lid}: {_safe_pipeline_error(e, cfg)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
