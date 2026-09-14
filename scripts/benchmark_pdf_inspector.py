"""Pilot benchmark for the CPU pdf-inspector path on local physics PDFs."""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from drbrain.parser.pdf_inspector_backend import extract_pdf_inspector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="physics/data/arxiv-pdf")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    files = sorted(Path(args.root).glob("*.pdf"))[: args.limit]
    started = time.perf_counter()

    def one(path: Path) -> dict:
        t = time.perf_counter()
        result = extract_pdf_inspector(path)
        return {"path": str(path), "ok": result is not None, "seconds": round(time.perf_counter() - t, 4), "bytes": len(result["markdown"]) if result else 0, "pdf_type": result.get("pdf_type", "") if result else ""}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(one, files))
    print(json.dumps({"root": os.path.abspath(args.root), "count": len(rows), "workers": args.workers, "elapsed_seconds": round(time.perf_counter() - started, 3), "ok": sum(r["ok"] for r in rows), "failed": sum(not r["ok"] for r in rows), "avg_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 4) if rows else 0, "rows": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
