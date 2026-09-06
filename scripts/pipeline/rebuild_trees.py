#!/usr/bin/env python
"""空树重建：对 tree.json 结构为空的论文重跑 md_to_tree + doc description。

输入: data/.runtime/empty_trees.txt（每行一个 local_id）
用法: uv run python scripts/pipeline/rebuild_trees.py [--workers 8]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

# Keep imports tied to this source checkout.  Runtime data is selected at
# invocation time and must not change the module search path.
SOURCE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_SRC = SOURCE_ROOT / "src"
for _import_root in (SOURCE_ROOT, SOURCE_SRC):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from drbrain.runtime import RuntimeContext, runtime_root  # noqa: E402
from drbrain.security import configured_secret_values, safe_error  # noqa: E402

# Compatibility snapshot only; worker/main paths use runtime_root() afresh.
ROOT = SOURCE_ROOT

from drbrain.parser.pageindex.sdk_backend import configure_tree_backend  # noqa: E402
from drbrain.parser.pageindex_parser import TreeConfig, md_to_tree  # noqa: E402
from drbrain.storage.paths import (  # noqa: E402
    paper_dir,
    paper_fs_key,
    raw_md_path,
    writable_artifact_path,
)
from scripts.pipeline.common import load_cfg, runtime_path  # noqa: E402
from scripts.pipeline.ingest_scibase import (  # noqa: E402
    _process_worker_timeout,
    _run_process_pool_fail_fast,
)


def _safe_pipeline_error(value: object, cfg: object | None = None) -> str:
    """Bound and redact worker/config errors before printing them."""

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


def _safe_input_file(value: str, root: Path) -> Path:
    """Resolve the paper-ID list without following a symlink alias."""

    candidate = runtime_path(value, root)
    lexical = Path(value).expanduser()
    if not lexical.is_absolute():
        lexical = root / lexical
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"paper list escapes runtime root {root}: {lexical}") from exc
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"paper list must not contain symlink components: {lexical}")
    if not candidate.is_file():
        raise ValueError(f"paper list is not a regular file: {candidate}")
    return candidate


def _write_artifact_atomically(paper_path: Path, filename: str, content: str) -> None:
    """Write a generated paper artifact without following a stale symlink."""

    target = writable_artifact_path(paper_path, filename)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{filename}.", dir=paper_path)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)


def rebuild_one(args: tuple) -> dict:
    # Keep the three-item form for callers that use this worker directly;
    # main passes the invocation root explicitly so forked workers cannot
    # accidentally observe a different environment namespace.
    if len(args) == 4:
        lid, cfg, build_cfg, root_value = args
        # Re-validate the parent-selected root in the child.  Calling
        # ``Path.resolve`` directly would turn a symlink alias into a
        # seemingly safe path and bypass RuntimeContext's isolation checks.
        active_root = RuntimeContext.create(root_value).root
    else:
        lid, cfg, build_cfg = args
        active_root = runtime_root()
    from loguru import logger

    configured_papers_root = cfg.get("dirs", {}).get("papers")
    papers_root = Path(configured_papers_root or active_root / "data/papers")
    if not papers_root.is_absolute():
        papers_root = active_root / papers_root
    papers_root = papers_root.resolve()
    try:
        papers_root = RuntimeContext.create(active_root).assert_within_root(
            papers_root, label="papers directory"
        )
    except (OSError, TypeError, ValueError) as exc:
        return {
            "lid": lid,
            "ok": False,
            "error": f"invalid papers directory: {_safe_pipeline_error(exc, cfg)}",
        }
    try:
        pdir = paper_dir(papers_root, lid)
    except (OSError, TypeError, ValueError) as exc:
        return {
            "lid": lid,
            "ok": False,
            "error": f"invalid paper path: {_safe_pipeline_error(exc, cfg)}",
        }
    try:
        md_path = raw_md_path(pdir)
    except (OSError, TypeError, ValueError) as exc:
        return {
            "lid": lid,
            "ok": False,
            "error": f"invalid raw.md path: {_safe_pipeline_error(exc, cfg)}",
        }
    if not md_path.is_file():
        return {"lid": lid, "ok": False, "error": "no raw.md"}
    llm_models = cfg.get("llm", {}).get("models", [])
    try:
        pageindex_cfg = TreeConfig(
            if_thinning=False,
            if_add_node_summary=True,
            if_add_doc_description=False,  # doc description 单独用 hy3（ox 返回空）
            if_add_node_text=False,
            if_add_node_id=True,
            max_node_tokens=10000,
            summary_token_threshold=2000,
        )
        configure_tree_backend(pageindex_cfg, cfg.get("pageindex"))
        doc_tree = asyncio.run(md_to_tree(md_path, config=pageindex_cfg, models=llm_models))
        # 无 markdown 标题的纯文本片段（书摘/表格等）切不出章节——
        # 合成单节点全文树，保证可进向量检索
        if not doc_tree.structure:
            text = md_path.read_text(encoding="utf-8")
            doc_tree.structure = [
                {
                    "title": "Full Text",
                    "node_id": "0000",
                    "summary": text[:8000],
                }
            ]
        try:
            from drbrain.parser.pageindex.retrieval import (
                _create_clean_structure_for_description,
            )
            from drbrain.parser.pageindex.summary import _generate_doc_description

            hy3_models = build_cfg.get("llm", {}).get("models", [])
            clean = _create_clean_structure_for_description(doc_tree.structure)
            if isinstance(clean, list):
                clean = {"structure": clean}
            desc = asyncio.run(_generate_doc_description(clean, hy3_models))
            if desc:
                doc_tree.doc_description = desc
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[rebuild] doc-desc failed {}: {}",
                lid,
                _safe_pipeline_error(e, cfg),
            )
        _write_artifact_atomically(pdir, "tree.json", doc_tree.to_json())
        return {"lid": lid, "ok": True, "sections": len(doc_tree.structure)}
    except Exception as e:  # noqa: BLE001
        return {
            "lid": lid,
            "ok": False,
            "error": _safe_pipeline_error(e, cfg),
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--list", type=str, default=None)
    args = ap.parse_args()

    if args.workers <= 0:
        ap.error("--workers must be greater than zero")
    try:
        root = runtime_root()
        if args.list == "":
            raise ValueError("paper list path must not be empty")
        list_value = (
            args.list
            if args.list is not None
            else str(root / "data" / ".runtime" / "empty_trees.txt")
        )
        list_path = _safe_input_file(list_value, root)
        ids = [x.strip() for x in list_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    except (OSError, TypeError, ValueError) as exc:
        print(f"runtime/list path error: {_safe_pipeline_error(exc)}", file=sys.stderr)
        return 1
    if not ids:
        print("没有待重建的论文", file=sys.stderr)
        return 1
    invalid_ids: list[str] = []
    for lid in ids:
        try:
            paper_fs_key(lid)
        except (TypeError, ValueError):
            invalid_ids.append(lid)
    if invalid_ids:
        print(
            f"paper list contains {len(invalid_ids)} invalid paper IDs (first: {invalid_ids[0]!r})",
            file=sys.stderr,
        )
        return 1
    print(f"待重建树: {len(ids)} 篇")
    try:
        cfg = load_cfg(None, root=root)
        build_cfg = load_cfg("config.build.yaml", root=root)
    except (OSError, TypeError, ValueError) as exc:
        print(f"runtime/config error: {_safe_pipeline_error(exc)}", file=sys.stderr)
        return 1

    try:
        worker_timeout = _process_worker_timeout(
            "REBUILD_TREE_TIMEOUT",
            fallback_env="REBUILD_WORKER_TIMEOUT",
        )
    except (TypeError, ValueError) as exc:
        print(
            f"worker timeout configuration error: {_safe_pipeline_error(exc, cfg)}", file=sys.stderr
        )
        return 1

    tasks = [(lid, cfg, build_cfg, str(root)) for lid in ids]
    ok = fail = 0
    t0 = time.monotonic()
    completed = 0

    def consume_result(rec: dict) -> None:
        nonlocal completed, ok, fail
        completed += 1
        if rec.get("ok"):
            ok += 1
        else:
            fail += 1
        if completed % 50 == 0 or completed == len(ids):
            print(
                f"[{completed}/{len(ids)}] ok={ok} fail={fail} elapsed={time.monotonic() - t0:.0f}s",
                flush=True,
            )

    def timeout_result(task: tuple, seconds: float) -> dict:
        lid = task[0] if task else ""
        return {"lid": lid, "ok": False, "error": f"timeout>{seconds:g}s"}

    def exception_result(task: tuple, exc: Exception) -> dict:
        lid = task[0] if task else ""
        task_cfg = task[1] if len(task) > 1 else cfg
        return {"lid": lid, "ok": False, "error": _safe_pipeline_error(exc, task_cfg)}

    try:
        _run_process_pool_fail_fast(
            tasks,
            rebuild_one,
            max_workers=args.workers,
            timeout=worker_timeout,
            on_result=consume_result,
            timeout_result=timeout_result,
            exception_result=exception_result,
        )
    except Exception as exc:  # noqa: BLE001
        # Executor startup/shutdown failures must produce a non-zero status,
        # while keeping the diagnostic bounded and free of configured secrets.
        consume_result(
            {"lid": "__pipeline__", "ok": False, "error": _safe_pipeline_error(exc, cfg)}
        )
    print(f"\n重建完成: ok={ok} fail={fail} ({time.monotonic() - t0:.0f}s)")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
