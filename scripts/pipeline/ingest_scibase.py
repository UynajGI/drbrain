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
import json
import os
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

from drbrain.dedup.resolver import DedupEngine, PaperIDs
from drbrain.parser.mineru.parser import filter_sections
from drbrain.storage.database import Database
from drbrain.storage.paths import paper_dir

ROOT = Path("/home/jiangyuan/drbrain")
sys.path.insert(0, str(ROOT))

from scripts.pipeline.common import load_cfg  # noqa: E402

CLEANED = ROOT / "data/fulltext-cleaned-20260806"
MIN_MD = 500


def _parse_doi_file(fn: str) -> str:
    """cleaned json 文件名是 DOI 的 url 转义（_ → /）。"""
    return fn[:-5].replace("_", "/")


def _load_cleaned(fn: str) -> dict | None:
    try:
        d = json.loads((CLEANED / fn).read_text(encoding="utf-8"))
        if d.get("markdown") and len(d["markdown"]) >= MIN_MD:
            return d
    except Exception:  # noqa: BLE001
        pass
    return None


def ingest_from_md(cleaned: dict, cfg: dict, db: Database, dedup: DedupEngine) -> dict:
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
    ids = PaperIDs(doi=doi or None, arxiv=None)
    local_id = dedup.resolve(ids, title=title, year=year)
    is_new = local_id is None
    if is_new:
        local_id = f"p{uuid.uuid4().hex[:6]}"
        db.insert_paper(local_id, title or doi, year, "uploaded", paper_type="paper")
        db.insert_paper_ids(local_id, doi=doi, arxiv=None, s2_id=None, openalex_id=None)
    db.commit()

    # 写 raw.md
    paper_dir_path = paper_dir(Path(cfg.get("dirs", {}).get("papers", "data/papers")), local_id)
    paper_dir_path.mkdir(parents=True, exist_ok=True)
    (paper_dir_path / "raw.md").write_text(md, encoding="utf-8")

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
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[scibase] paper-type failed {local_id}: {e}")

    # tree（LLM 摘要；短节点(<2000 token)原文当摘要不调 LLM，省 ~80% 摘要调用）
    from drbrain.parser.pageindex_parser import TreeConfig, md_to_tree

    tree_path = paper_dir_path / "tree.json"
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
        doc_tree = asyncio.run(
            md_to_tree(paper_dir_path / "raw.md", config=pageindex_cfg, models=llm_models)
        )
        tree_path.write_text(doc_tree.to_json(), encoding="utf-8")
        n_sections = len(doc_tree.structure)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[scibase] tree failed {local_id}: {e}")
        n_sections = 0

    db.set_paper_status(local_id, "uploaded")
    db.commit()
    return {
        "ok": True,
        "local_id": local_id,
        "report": {"sections": n_sections, "md_len": len(md), "secs": _time.monotonic() - t0},
    }


def _extract_llm(cleaned: dict, cfg: dict, local_id: str) -> dict:
    """worker 进程：只做 LLM 抽取（paper type + tree），不碰 db（避免锁冲突）。

    写 raw.md + tree.json 到 data/papers/<local_id>/，返回 paper_type。
    """
    import time as _time

    from loguru import logger

    t0 = _time.monotonic()
    md = cleaned["markdown"]
    papers_dir = Path(cfg.get("dirs", {}).get("papers", "data/papers"))
    paper_dir_path = paper_dir(papers_dir, local_id)
    paper_dir_path.mkdir(parents=True, exist_ok=True)
    (paper_dir_path / "raw.md").write_text(md, encoding="utf-8")

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
        logger.warning(f"[scibase] paper-type failed {local_id}: {e}")

    from drbrain.parser.pageindex_parser import TreeConfig, md_to_tree

    tree_path = paper_dir_path / "tree.json"
    n_sections = 0
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
        doc_tree = asyncio.run(
            md_to_tree(paper_dir_path / "raw.md", config=pageindex_cfg, models=llm_models)
        )
        # doc description 单独用 hy3 生成（ox-alpha-free 对纯文本描述返回空 content）
        try:
            from drbrain.parser.pageindex.retrieval import _create_clean_structure_for_description
            from drbrain.parser.pageindex.summary import _generate_doc_description

            build_cfg = load_cfg("config.build.yaml")  # hy3 在前
            hy3_models = build_cfg.get("llm", {}).get("models", [])
            clean_struct = _create_clean_structure_for_description(doc_tree.structure)
            if isinstance(clean_struct, list):
                clean_struct = {"structure": clean_struct}
            doc_desc = asyncio.run(_generate_doc_description(clean_struct, hy3_models))
            if doc_desc:
                doc_tree.doc_description = doc_desc
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[scibase] doc-description failed {local_id}: {e}")
        tree_path.write_text(doc_tree.to_json(), encoding="utf-8")
        n_sections = len(doc_tree.structure)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[scibase] tree failed {local_id}: {e}")

    return {
        "ok": True,
        "paper_type": paper_type,
        "sections": n_sections,
        "secs": _time.monotonic() - t0,
    }


def _worker(args: tuple) -> dict:
    """多进程 worker：主进程先 identify 生成 local_id，worker 只做 LLM 抽取（不碰 db）。"""
    cleaned, cfg, local_id = args
    try:
        r = _extract_llm(cleaned, cfg, local_id)
        return {
            "file": cleaned.get("_file", ""),
            "ok": r["ok"],
            "local_id": local_id,
            "paper_type": r.get("paper_type", "paper"),
            "sections": r.get("sections", 0),
            "error": r.get("error"),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "file": cleaned.get("_file", ""),
            "ok": False,
            "local_id": local_id,
            "error": f"{type(e).__name__}: {e}",
        }


def _identify(cleaned: dict, db: Database, dedup: DedupEngine) -> str:
    """主进程：identify → local_id（写 papers/paper_ids 表）。"""
    doi = (cleaned.get("doi") or _parse_doi_file(cleaned["_file"])).strip().lower()
    title = cleaned.get("title") or ""
    year = cleaned.get("year")
    if year and isinstance(year, str) and year.isdigit():
        year = int(year)
    ids = PaperIDs(doi=doi or None, arxiv=None)
    local_id = dedup.resolve(ids, title=title, year=year)
    if local_id is None:
        local_id = f"p{uuid.uuid4().hex[:6]}"
        db.insert_paper(local_id, title or doi, year, "uploaded", paper_type="paper")
        db.insert_paper_ids(local_id, doi=doi, arxiv=None, s2_id=None, openalex_id=None)
    db.commit()
    return local_id


def _write_db(rec: dict, db: Database) -> None:
    """主进程：写 paper_type/status（串行，无锁冲突）。"""
    if not rec.get("ok"):
        return
    db.set_paper_type(rec["local_id"], rec.get("paper_type", "paper"))
    db.set_paper_status(rec["local_id"], "uploaded")
    db.commit()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=str, default=str(CLEANED))
    ap.add_argument("--db", type=str, required=True)
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--no-db",
        action="store_true",
        help="只缓存文件（raw.md + tree.json），不写 db；local_id 用 DOI 哈希",
    )
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    manifest = Path(args.manifest)
    files = sorted(Path(args.source).glob("*.json"))
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    done: set[str] = set()
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                if rec.get("ok"):
                    done.add(rec["file"])
            except json.JSONDecodeError:
                continue
        print(f"[resume] 已跳过 {len(done)} 篇")

    concurrency = int(os.environ.get("INGEST_CONCURRENCY", "1"))
    stats = Counter()
    bad: list[dict] = []
    t0 = time.monotonic()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_f = open(manifest, "a", encoding="utf-8")
    db = Database(args.db)
    dedup = DedupEngine(db)
    try:
        # 主进程 identify（写 papers 表）→ worker 抽 LLM（不碰 db）→ 主进程写 db
        # --no-db 模式：local_id 用 DOI 哈希（确定性去重），完全不写 db，只缓存文件
        pending: list[tuple] = []
        for fn in files:
            if fn.name in done:
                continue
            cleaned = _load_cleaned(fn.name)
            if cleaned is None:
                rec = {"file": fn.name, "ok": False, "error": "no-markdown"}
                bad.append(rec)
                manifest_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            cleaned["_file"] = fn.name
            try:
                if args.no_db:
                    import hashlib as _hl

                    doi = (cleaned.get("doi") or _parse_doi_file(fn.name)).strip().lower()
                    local_id = f"p{_hl.md5(doi.encode()).hexdigest()[:8]}"
                else:
                    local_id = _identify(cleaned, db, dedup)
            except Exception as e:  # noqa: BLE001
                rec = {
                    "file": fn.name,
                    "ok": False,
                    "local_id": None,
                    "error": f"identify: {type(e).__name__}: {e}",
                }
                bad.append(rec)
                manifest_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            pending.append((cleaned, cfg, local_id))

        if concurrency <= 1:
            for i, task in enumerate(pending, 1):
                rec = _worker(task)
                if not args.no_db:
                    _write_db(rec, db)
                stats["ok" if rec["ok"] else "fail"] += 1
                manifest_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                manifest_f.flush()
                if not rec["ok"]:
                    bad.append(rec)
                if i % 10 == 0 or i == len(pending):
                    print(
                        f"[{i}/{len(pending)}] ok={stats['ok']} fail={stats['fail']} "
                        f"elapsed={time.monotonic() - t0:.0f}s",
                        flush=True,
                    )
        else:
            import concurrent.futures

            with concurrent.futures.ProcessPoolExecutor(max_workers=concurrency) as ex:
                for i, rec in enumerate(ex.map(_worker, pending), 1):
                    if not args.no_db:
                        _write_db(rec, db)
                    stats["ok" if rec["ok"] else "fail"] += 1
                    manifest_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    manifest_f.flush()
                    if not rec["ok"]:
                        bad.append(rec)
                    if i % 10 == 0 or i == len(pending):
                        print(
                            f"[{i}/{len(pending)}] ok={stats['ok']} fail={stats['fail']} "
                            f"elapsed={time.monotonic() - t0:.0f}s",
                            flush=True,
                        )
    finally:
        manifest_f.close()
        db.close()

    print(f"\n完成: ok={stats['ok']} fail={stats['fail']} ({time.monotonic() - t0:.0f}s)")
    for r in bad[:20]:
        print(f"  FAIL {r['file']}: {str(r.get('error'))[:160]}")


if __name__ == "__main__":
    main()
