#!/usr/bin/env python
"""Autoresearch director driver — AutoScientists-style continuous research loop.

Runs the ResearchDirector on a topic until stagnation (or ``--max-cycles``),
keeping a checkpointed champion/rejected/results state and a running report.
This is the "24h continuous research" entry point: resume it to keep going.

Usage:
    uv run python tests/integration/run_director.py \
        --topic "topological flat band" --max-cycles 3 --stagnation 2
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="topological flat band")
    ap.add_argument("--db", default=str(ROOT / "data" / "drbrain.db"))
    ap.add_argument("--plugins", default=str(ROOT / "research" / "plugins"))
    ap.add_argument("--models", default=str(ROOT / "research" / "models"))
    ap.add_argument("--run-dir", default=str(ROOT / "workspace" / "autoresearch"))
    ap.add_argument("--max-cycles", type=int, default=3)
    ap.add_argument("--stagnation", type=int, default=2)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    os.environ.setdefault("DRBRAIN_DB", args.db)
    os.environ.setdefault("DRBRAIN_MODELS_DIR", args.models)

    from drbrain.config import load_config
    from drbrain.loop import ResearchDirector
    from drbrain.storage.database import Database

    cfg = load_config(str(ROOT / "config.yaml"), str(ROOT / "config.local.yaml"))
    db = Database(args.db)
    try:
        director = ResearchDirector(
            cfg, db=db, plugins_dir=args.plugins, run_dir=args.run_dir
        )
        state = director.run_sync(
            args.topic, max_cycles=args.max_cycles, stagnation_cycles=args.stagnation
        )
    finally:
        db.close()

    print("\n" + "=" * 72)
    print("DIRECTOR RESULT")
    print("=" * 72)
    print(f"cycles={state['cycles']} champion={len(state['champion'])} "
          f"rejected={len(state['rejected'])} no_gain={state['consecutive_no_gain']}")
    for c in state["champion"]:
        print(f"  [champion, cycle {c['cycle']}] {c['statement']}")
    print(f"\nworkspace: {args.run_dir}/topological-flat-band/")
    print("  champion.md | dead_ends.md | knowledge/patterns.md | results/cycle-*.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
