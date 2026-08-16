#!/usr/bin/env python3
"""Walk through main-project commands against the realdata fulltext test set.

Temporarily points config.local.yaml at the test DB / test papers dir, runs each
command in the matrix with a timeout, records {command, exit, seconds, ok, tail}
to a report file, then restores config.local.yaml in a finally block.

Usage:
    python scripts/walkthrough.py [--db data/realdata_fulltext.db] [--papers-dir data/test_papers]
                                  [--commands "stats,list,index,query"] [--timeout 300]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_LOCAL = ROOT / "config.local.yaml"
BAK = ROOT / "config.local.yaml.walkthrough.bak"

# (label, argv, timeout_s)
DEFAULT_COMMANDS = [
    ("stats", ["stats"], 120),
    ("list", ["list", "--limit", "5"], 120),
    ("index", ["index"], 300),
    ("query-bm25", ["query", "materials"], 300),
    ("search", ["search", "perovskite"], 120),
    ("hybrid", ["hybrid", "machine learning"], 300),
    ("ask", ["ask", "what are the key synthesis methods?"], 300),
    ("analyze", ["analyze", "--query", "synthesis"], 300),
    ("landscape", ["landscape"], 300),
    ("frontier", ["frontier"], 300),
    ("evolve", ["evolve", "-s", "perovskite", "-d", "descendants"], 120),
    ("graph-neighbors", ["graph", "neighbors", "perovskite"], 120),
    ("graph-path", ["graph", "path", "perovskite", "oxide"], 120),
    ("citations", ["citations", "--title", "machine learning"], 120),
    ("export-bibtex", ["export", "--format", "bibtex"], 120),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/realdata_fulltext.db")
    ap.add_argument("--papers-dir", default="data/test_papers")
    ap.add_argument("--commands", default=None, help="comma-separated labels to run (default: all)")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--report", default="docs/realdata-walkthrough.jsonl")
    args = ap.parse_args()

    report = ROOT / args.report
    report.parent.mkdir(parents=True, exist_ok=True)

    # --- point config at the test set (backup first) ---
    import yaml  # noqa: PLC0415

    orig = CONFIG_LOCAL.read_text(encoding="utf-8") if CONFIG_LOCAL.exists() else None
    shutil.copy2(CONFIG_LOCAL, BAK) if orig is not None else None
    cfg = yaml.safe_load(orig) if orig else {}
    cfg["db"] = {"path": str(ROOT / args.db)}
    cfg["dirs"] = {**cfg.get("dirs", {}), "papers": str(ROOT / args.papers_dir)}
    CONFIG_LOCAL.write_text(yaml.dump(cfg, allow_unicode=True, default_flow_style=False), encoding="utf-8")
    print(f"[walkthrough] config.local.yaml -> test DB {args.db} (backup at {BAK.name})")

    labels = args.commands.split(",") if args.commands else [c[0] for c in DEFAULT_COMMANDS]
    matrix = {c[0]: c for c in DEFAULT_COMMANDS}
    try:
        for label in labels:
            if label not in matrix:
                print(f"[walkthrough] unknown command label: {label}")
                continue
            _, argv, default_to = matrix[label]
            to = default_to if args.commands is None else args.timeout
            print(f"[walkthrough] >>> drbrain {' '.join(argv)} (timeout {to}s)")
            t0 = time.time()
            try:
                p = subprocess.run(
                    [sys.executable, "-m", "drbrain.cli.main", *argv],
                    capture_output=True, text=True, timeout=to, cwd=ROOT,
                )
                code, out = p.returncode, (p.stdout or "") + (p.stderr or "")
            except subprocess.TimeoutExpired as e:
                code, out = 124, f"TIMEOUT after {to}s" + ((e.stdout or "")[-2000:])
            dt = time.time() - t0
            ok = code == 0
            tail = "\n".join(out.strip().splitlines()[-5:])[:2000]
            rec = {"command": label, "exit": code, "seconds": round(dt, 1), "ok": ok, "tail": tail}
            with report.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[walkthrough] {label}: exit={code} {dt:.0f}s {'OK' if ok else 'FAIL'}")
            if not ok:
                print(f"  {tail[:600]}")
    finally:
        # --- restore config.local.yaml ---
        if orig is not None:
            CONFIG_LOCAL.write_text(orig, encoding="utf-8")
            print("[walkthrough] config.local.yaml restored")

    print(f"[walkthrough] DONE — report: {report}")


if __name__ == "__main__":
    main()
