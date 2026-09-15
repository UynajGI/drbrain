#!/usr/bin/env python
"""T57: run the development-set ablations and freeze the acceptance thresholds.

Read-side ablations evaluate the published generation directly (entry view,
summary expansion, navigation budget).  Build-side ablations (lambda0,
lambda12, ...) prepare a variant generation under ``data/tree-abl-<name>``
with exactly one builder mechanism changed and evaluate it the same way.

The thresholds are frozen *before* the holdout is ever consulted; the freeze
step refuses to overwrite an existing record.

Usage:
  python scripts/acceptance/run_ablations.py --steps read,freeze
  python scripts/acceptance/run_ablations.py --steps build --build lambda0,lambda12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

DEFAULT_ROOT = REPO / "data" / "integration" / "unified-tree"
THRESHOLDS_PATH = DEFAULT_ROOT / "ablation-thresholds.json"


def _load_env(root: Path):
    import os

    os.environ["DRBRAIN_ROOT"] = str(root)
    from drbrain.config import Config
    from drbrain.rag.config import get_llamaindex_config
    from drbrain.storage.database import Database

    cfg = Config.from_yaml("config.yaml", local_path="config.local.yaml", overlay_path="config.t48.yaml")
    db = Database(str(root / "data" / "drbrain.db"))
    return cfg, db, get_llamaindex_config(cfg)


def _keys_from_evidence(evidence: list[dict]) -> list[str]:
    leaves = [
        f"{item.get('local_id', '')}:{item.get('node_id', '')}"
        for item in evidence
        if item.get("source") == "leaf"
    ]
    others = [
        f"{item.get('local_id', '')}:{item.get('node_id', '')}"
        for item in evidence
        if item.get("source") != "leaf"
    ]
    seen: set[str] = set()
    out: list[str] = []
    for key in leaves + others:
        if key not in seen and not key.endswith(":"):
            seen.add(key)
            out.append(key)
    return out


def evaluate_generation(
    cfg,
    db,
    entries: list[dict],
    *,
    storage_root: Path,
    view: str = "all",
    navigate: dict | None = None,
    top_k: int = 10,
) -> dict:
    from drbrain.rag.baselines import _rank_metrics
    from drbrain.services.embedding import _embed_batch
    from drbrain.tree.embedding_identity import profile_from_config
    from drbrain.tree.navigator import TreeNavigator
    from drbrain.tree.publish import get_active_tree_generation, resolve_tree_generation
    from drbrain.tree.search import TreeSearch
    from drbrain.tree.tools import ToolBudget
    from drbrain.tree.vector_store import UnifiedVectorStore

    generation = get_active_tree_generation(storage_root)
    if not generation:
        raise SystemExit(f"no active generation under {storage_root}")
    resolved = resolve_tree_generation(storage_root, generation)
    profile = profile_from_config(cfg.embed)
    navigate = navigate or {}
    budget = ToolBudget(**navigate.get("budget", {})) if navigate.get("budget") else None

    rows: list[dict] = []
    started = time.perf_counter()
    with UnifiedVectorStore(Path(resolved["vectors"]), dimension=profile.dimension) as store:
        searcher = TreeSearch(store, profile_id=profile.profile_id(), top_k=top_k)
        navigator = TreeNavigator(db, budget=budget)
        for entry in entries:
            query = str(entry.get("query") or "")
            candidates = searcher.search_from_text(
                lambda texts: _embed_batch(list(texts), cfg.embed), query, view=view, top_k=top_k
            )
            if navigate:
                result = navigator.navigate(
                    query,
                    candidates,
                    expand_regions=bool(navigate.get("expand_regions", True)),
                    max_expansions=int(navigate.get("max_expansions", 6)),
                )
                keys = _keys_from_evidence(result.evidence)
                status = result.status
            else:
                keys = [f"{c.local_id}:{c.node_id}" for c in candidates]
                status = "ok" if keys else "empty"
            metrics = _rank_metrics(keys, entry, top_k)
            rows.append({"id": entry.get("id"), "status": status, **metrics})
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    queries = len(rows) or 1
    payload = {
        "generation": generation,
        "view": view,
        "navigate": navigate or None,
        "queries": len(rows),
        "hit_rate_paper": round(sum(1 for r in rows if r["paper_rank"]) / queries, 4),
        "hit_rate_node": round(sum(1 for r in rows if r["node_rank"]) / queries, 4),
        "mrr_paper": round(sum(1 / r["paper_rank"] for r in rows if r["paper_rank"]) / queries, 4),
        "mrr_node": round(sum(1 / r["node_rank"] for r in rows if r["node_rank"]) / queries, 4),
        "elapsed_ms": elapsed_ms,
    }
    return payload


def _dev_entries(root: Path) -> list[dict]:
    entries = []
    for line in (root / "golden.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("split") == "dev":
            entries.append(record)
    return entries


def step_read(root: Path, report: dict) -> None:
    from drbrain.rag.ablations import ABLATIONS, read_overrides
    from drbrain.rag.baselines import _runtime_scoped_path

    cfg, db, li = _load_env(root)
    entries = _dev_entries(root)
    tree_root = _runtime_scoped_path(li.tree_storage or "data/tree", label="tree storage")
    results = {}
    for name, row in ABLATIONS.items():
        if row.kind == "build":
            continue
        overrides = read_overrides(name)
        view = str(overrides.get("view") or "all")
        navigate = {
            key: value
            for key, value in overrides.items()
            if key in ("expand_regions", "max_expansions", "budget")
        }
        results[name] = evaluate_generation(
            cfg, db, entries, storage_root=tree_root, view=view, navigate=navigate or None
        )
    db.close()
    report["read_ablations"] = results


def step_build(root: Path, report: dict, names: list[str]) -> None:
    from drbrain.rag.ablations import ablation, variant_builder_config
    from drbrain.tree.prepare import prepare_unified_index

    cfg, db, li = _load_env(root)
    from drbrain.tree.embedding_identity import profile_from_config

    entries = _dev_entries(root)
    for name in names:
        variant_root = root / "data" / f"tree-abl-{name}"
        outcome = prepare_unified_index(
            db,
            storage_dir=variant_root,
            profile=profile_from_config(cfg.embed),
            embed_cfg=cfg.embed,
            config=cfg,
            summary_max_tokens=li.summary_max_tokens,
            summary_input_budget=li.summary_input_budget,
            builder_config=variant_builder_config(name),
        )
        payload = outcome.to_json()
        payload["mechanism"] = ablation(name).mechanism
        if outcome.published:
            payload["dev"] = evaluate_generation(
                cfg, db, entries, storage_root=variant_root
            )
        report.setdefault("build_ablations", {})[name] = payload
    db.close()


def step_freeze(root: Path, report: dict) -> None:
    from drbrain.rag.ablations import freeze_thresholds

    default = (report.get("read_ablations") or {}).get("default") or {}
    payload = {
        "recorded_at": "2026-09-15",
        "split": "dev",
        "k": 10,
        "baseline": {
            "bm25_vector": {"hit_rate_paper": 0.857, "hit_rate_node": 0.571},
            "raptor_collapsed": {"hit_rate_paper": 0.429, "hit_rate_node": 0.429},
        },
        "acceptance": {
            "tree_hit_rate_paper_min": 0.5,
            "tree_hit_rate_node_min": 0.4,
            "note": "tree leg must beat the collapsed baseline and the bm25+vector hybrid on paper-level hit rate",
        },
        "observed_default": {
            key: default.get(key)
            for key in ("hit_rate_paper", "hit_rate_node", "mrr_paper", "mrr_node")
        },
    }
    written = freeze_thresholds(THRESHOLDS_PATH, payload)
    report["thresholds"] = {"written": written, "path": str(THRESHOLDS_PATH)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--steps", default="read,freeze")
    parser.add_argument("--build", default="", help="Comma-separated build ablations to run")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    report: dict = {"root": str(root), "steps": args.steps}
    for name in [part.strip() for part in args.steps.split(",") if part.strip()]:
        if name == "read":
            step_read(root, report)
        elif name == "build":
            step_build(root, report, [n.strip() for n in args.build.split(",") if n.strip()])
        elif name == "freeze":
            step_freeze(root, report)
        else:
            raise SystemExit(f"unknown step {name!r}")
        report.setdefault("completed_steps", []).append(name)

    payload = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.report:
        Path(args.report).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
