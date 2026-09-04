#!/usr/bin/env python3
"""Ingest scholarweave/arxiv-latex parquet shards into the drbrain library.

Physics corpus path (review §6): the raw arXiv LaTeX source is converted to
``raw.md`` with :mod:`drbrain.parser.latex_md` (math atoms preserved, ``\\cite``
keys extracted, ``\\section`` headings → PageIndex tree built in code — no LLM
pass), then inserted as regular papers with ``categories`` metadata for
category-scoped retrieval. Citation keys land in ``paper_cite_keys``; a
``--resolve-citations`` pass maps them to in-corpus local_ids once the corpus
is complete.

Usage:
  python scripts/pipeline/ingest_arxiv_latex.py ingest --limit 100 physics/data/arxiv-latex/physics_arxiv_part_0001.parquet
  python scripts/pipeline/ingest_arxiv_latex.py ingest physics/data/arxiv-latex/physics_arxiv_part_*.parquet
  python scripts/pipeline/ingest_arxiv_latex.py resolve-citations
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

PAPERS_ROOT = REPO / "data" / "papers"


def _safe_paper_dir(arxiv_id: str) -> str:
    """arXiv ids contain ``/`` (old-style ``hep-lat/9107001``) — flatten it."""
    return arxiv_id.replace("/", "_")


def _iter_rows(shards: list[Path], limit: int):
    import polars as pl

    for shard in shards:
        wanted = [
            "id",
            "title",
            "abstract",
            "categories",
            "latex",
            "doi",
            "update_date",
            "journal-ref",
            "authors",
        ]
        schema_names = set(pl.read_parquet_schema(shard))
        df = pl.read_parquet(shard, columns=[c for c in wanted if c in schema_names])
        if limit == 0:
            yield from df.iter_rows(named=True)
            continue
        for row in df.iter_rows(named=True):
            yield row
            limit -= 1
            if limit <= 0:
                return


def cmd_ingest(args: argparse.Namespace) -> None:
    from drbrain.parser.latex_md import latex_to_document, markdown_to_tree
    from drbrain.storage.database import Database

    shards = [Path(p) for p in args.shards]
    if not shards:
        print("[ingest] no shards given", flush=True)
        return

    db_path = args.db
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = Database(str(db_path))
    papers_root = args.papers_root
    papers_root.mkdir(parents=True, exist_ok=True)

    imported = skipped = 0
    t0 = time.time()
    for row in _iter_rows(shards, args.limit):
        arxiv_id = str(row.get("id") or "").strip()
        latex = row.get("latex") or ""
        if not arxiv_id or not latex.strip():
            skipped += 1
            continue
        paper_dir = papers_root / _safe_paper_dir(arxiv_id)
        if (paper_dir / "raw.md").exists():
            skipped += 1
            continue
        try:
            doc = latex_to_document(latex)
        except Exception as exc:  # noqa: BLE001 — one bad source must not stop the corpus
            print(f"[ingest] latex convert failed {arxiv_id}: {exc}", flush=True)
            skipped += 1
            continue
        if len(doc.markdown.strip()) < args.min_chars:
            skipped += 1
            continue

        paper_dir.mkdir(parents=True, exist_ok=True)
        (paper_dir / "raw.md").write_text(doc.markdown, encoding="utf-8")
        (paper_dir / "tree.json").write_text(
            _json_dumps(markdown_to_tree(doc.markdown)), encoding="utf-8"
        )

        year = None
        m = re.match(r"(\d{4})-", str(row.get("update_date") or ""))
        if m:
            year = int(m.group(1))
        db.insert_paper(
            local_id=arxiv_id,
            title=str(row.get("title") or arxiv_id).strip(),
            year=year,
            status="uploaded",
            paper_type="preprint",
            journal=str(row.get("journal-ref") or ""),
            authors=str(row.get("authors") or ""),
            categories=str(row.get("categories") or ""),
        )
        db.insert_paper_ids(local_id=arxiv_id, doi=row.get("doi") or None, arxiv=arxiv_id)
        db.set_paper_abstract(arxiv_id, str(row.get("abstract") or ""))
        db.insert_paper_cite_keys(arxiv_id, doc.citations)
        db.commit()
        imported += 1
        if imported % args.report_every == 0:
            print(
                f"[ingest] {imported} papers ({skipped} skipped) in {time.time() - t0:.0f}s",
                flush=True,
            )
    print(f"[ingest] done: {imported} imported, {skipped} skipped", flush=True)
    db.close()


def cmd_resolve(args: argparse.Namespace) -> None:
    """Map ``paper_cite_keys.cited_key`` → in-corpus ``cited_local_id``."""
    from drbrain.storage.database import Database

    db = Database(str(args.db))
    key_map: dict[str, str] = {}
    for local_id, _title in [
        (r[0], r[1]) for r in db.execute("SELECT local_id, title FROM papers").fetchall()
    ]:
        key_map.setdefault(local_id, local_id)
        key_map.setdefault(_safe_paper_dir(local_id), local_id)
    for (arxiv_id,) in db.execute("SELECT arxiv FROM paper_ids WHERE arxiv IS NOT NULL").fetchall():
        key_map.setdefault(str(arxiv_id), str(arxiv_id))
        key_map.setdefault(_safe_paper_dir(str(arxiv_id)), str(arxiv_id))
    resolved = db.resolve_paper_cite_keys(key_map)
    print(f"[resolve] {resolved} citations resolved to in-corpus papers", flush=True)
    db.close()


def _json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="import parquet shards into the library")
    p_ingest.add_argument(
        "--db", type=Path, default=REPO / "data" / "drbrain.db", help="main library DB"
    )
    p_ingest.add_argument("--papers-root", type=Path, default=PAPERS_ROOT, help="paper dir root")
    p_ingest.add_argument("--limit", type=int, default=0, help="max rows (0 = all)")
    p_ingest.add_argument("--min-chars", type=int, default=2000)
    p_ingest.add_argument("--report-every", type=int, default=500)
    p_ingest.add_argument("shards", nargs="+", help="parquet shard paths")

    p_resolve = sub.add_parser("resolve-citations", help="map citation keys to in-corpus papers")
    p_resolve.add_argument(
        "--db", type=Path, default=REPO / "data" / "drbrain.db", help="main library DB"
    )

    args = ap.parse_args()
    if args.command == "ingest":
        cmd_ingest(args)
    else:
        cmd_resolve(args)


if __name__ == "__main__":
    main()
