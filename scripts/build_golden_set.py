#!/usr/bin/env python
"""Build the LlamaIndex eval golden set (T7) from the test-run corpus.

Semi-automated annotation: curated queries (title/abstract-derived questions
over materials-science topics) + hand-picked relevant papers + relevant nodes
derived from each paper's ``tree.json`` structure. Splits into dev/val/test
60/20/20 deterministically (fixed seed). Idempotent — re-running without
``--force`` leaves an existing golden file untouched.

Usage (from repo root):
    python scripts/build_golden_set.py --papers-dir test-run/papers
    python scripts/build_golden_set.py --papers-dir test-run/papers --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--papers-dir",
        default=None,
        help="Corpus root with one dir per paper (default: config dirs.papers). "
        "Relative paths resolve against the CWD.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output JSONL path (default: config llamaindex.eval.golden_set).",
    )
    ap.add_argument(
        "--query-id",
        action="append",
        dest="query_ids",
        default=None,
        help="Only build these curated query ids (repeatable; testing use).",
    )
    ap.add_argument("--force", action="store_true", help="Rebuild even if the file exists")
    args = ap.parse_args(argv)

    from drbrain.config import load_config

    cfg = load_config()

    papers_dir = Path(args.papers_dir).resolve() if args.papers_dir else None
    out_path = Path(args.out).resolve() if args.out else None

    from drbrain.rag.eval import build_golden_set

    stats = build_golden_set(
        cfg,
        papers_dir=papers_dir,
        force=args.force,
        out_path=out_path,
        query_ids=args.query_ids,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if stats.get("status") == "ok":
        # Show a peek of the split counts per query for manual verification.
        golden_path = Path(stats["path"])
        print(f"\nGolden set written to {golden_path}")
        print(f"Splits: {stats.get('splits')}  (papers: {len(stats.get('papers') or [])})")
        if stats.get("missing_papers"):
            print(f"WARNING missing paper dirs: {stats['missing_papers']}", file=sys.stderr)
    elif stats.get("status") == "exists":
        print(f"Golden set already exists at {stats['path']} (use --force to rebuild)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
