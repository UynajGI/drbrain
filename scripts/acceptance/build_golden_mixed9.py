#!/usr/bin/env python
"""T55: fixed golden materials over the 9-sample acceptance corpus.

Every entry pins document revision 1 and one or more exact source quotes; the
builder locates the leaf whose text contains each quote and records its
canonical node id -- quotes always come from leaf text, never from generated
region summaries.  The dev/holdout split is assigned here, before any
evaluation runs, and the holdout half is not looked at until T57 freezes the
parameters.

Usage:
  python scripts/acceptance/build_golden_mixed9.py [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

DEFAULT_ROOT = REPO / "data" / "integration" / "unified-tree"
GOLDEN_PATH = DEFAULT_ROOT / "golden.jsonl"

PDF_OPENPHASE = "pf79e63fbe4b2e7fa96525228"
PDF_ANALYTICS = "p454e8e4e33ab44f6a69b446d"
PDF_PENDULAR = "p7f57aba32a6ae382b1e773c8"
TEX_SEMIFINAL = "p2fe42ddf5dccbef29a81fc1d"
TEX_EVORANK = "p23d82b4812380be90b1d2db6"
TEX_QUBIT = "pa47bbb49343d55a667519fe4"
MD_SUNPY = "pc4125827eaa55564688b4ece"
MD_LEGACY_OPTICS = "scibase-legacy-1"
MD_LEGACY_GRAVITY = "scibase-legacy-2"

#: Hand-curated queries; every quote is excerpted from the source text and
#: was checked against the canonical leaf that contains it.
ENTRIES: list[dict] = [
    {
        "id": "pdf-pendular-term",
        "query": "How is a pendular liquid ring described in this paper?",
        "kind": "term",
        "material": "pdf",
        "split": "dev",
        "evidence": [
            {
                "paper_id": PDF_PENDULAR,
                "quote": "formation of a pendular ring (trapped liquid at the contact of two solid particles) from an initially flooded condition",
            }
        ],
    },
    {
        "id": "pdf-openphase-term",
        "query": "How is the OpenPhase software package run?",
        "kind": "term",
        "material": "pdf",
        "split": "dev",
        "evidence": [
            {
                "paper_id": PDF_OPENPHASE,
                "quote": "and-line, non-interactive tool; it anticipates that the simulation input parameters will be provided in the form of an input file",
            }
        ],
    },
    {
        "id": "pdf-analytics-structure",
        "query": "Which chapter of the document introduces Analytics as a topic?",
        "kind": "structure",
        "material": "pdf",
        "split": "dev",
        "evidence": [{"paper_id": PDF_ANALYTICS, "quote": "# Analytics"}],
    },
    {
        "id": "tex-qubit-term",
        "query": "What is the minimal symmetric matter block in the light-matter transition paper?",
        "kind": "term",
        "material": "tex",
        "split": "dev",
        "evidence": [
            {
                "paper_id": TEX_QUBIT,
                "quote": "three identical qubits with uniform $ZZ$ and $ZZZ$ interactions form the minimal symmetric matter block",
            }
        ],
    },
    {
        "id": "tex-gapilot-term",
        "query": "What is Gapilot and which materials does it target?",
        "kind": "term",
        "material": "tex",
        "split": "dev",
        "evidence": [
            {
                "paper_id": TEX_SEMIFINAL,
                "quote": "本文提出 Gapilot,一个面向二维拓扑平带材料发现的文献驱动自主科研闭环",
            }
        ],
    },
    {
        "id": "tex-sac-formula",
        "query": "Which constant policy did the trained SAC controller converge to, and with which parameter values?",
        "kind": "formula",
        "material": "tex",
        "split": "dev",
        "evidence": [
            {
                "paper_id": TEX_EVORANK,
                "quote": "训练后的 SAC 策略收敛于动作空间的角点：2024 全年 $\\beta_t\\equiv0.800$、$\\lambda_t\\equiv0.100$",
            }
        ],
    },
    {
        "id": "md-sunpy-term",
        "query": "How did the authors locate and overplot the brightest pixel in the solar flare data?",
        "kind": "term",
        "material": "md",
        "split": "dev",
        "evidence": [
            {
                "paper_id": MD_SUNPY,
                "quote": "To find and overplot the location of the brightest pixel, we first created the Map using the FITS data and imported the coordinate functionality.",
            }
        ],
    },
    {
        "id": "md-optics-term",
        "query": "方形孔径相比圆形孔径在宏观傅里叶叠层成像中有哪些优势？",
        "kind": "term",
        "material": "md",
        "split": "holdout",
        "evidence": [
            {
                "paper_id": MD_LEGACY_OPTICS,
                "quote": "边长和直径相等的方形孔径与圆形孔径相比,方形孔径具有高光通量和宽传递函数的优势",
            }
        ],
    },
    {
        "id": "md-gravity-formula",
        "query": "Embedding Gravity 论文中的作用量主项写成什么形式？",
        "kind": "formula",
        "material": "md",
        "split": "holdout",
        "evidence": [
            {
                "paper_id": MD_LEGACY_GRAVITY,
                "quote": "S = \\int d ^ {4} x \\sqrt {- g} \\left(- \\frac {1}{2 \\varkappa} R + \\mathcal {L} _ {m}\\right)",
            }
        ],
    },
    {
        "id": "multi-software-tools",
        "query": "Which documents describe running command-line scientific software, and which tools do they name?",
        "kind": "multi",
        "material": "multi",
        "split": "holdout",
        "evidence": [
            {
                "paper_id": PDF_OPENPHASE,
                "quote": "and-line, non-interactive tool; it anticipates that the simulation input parameters will be provided in the form of an input file",
            },
            {
                "paper_id": MD_SUNPY,
                "quote": "To find and overplot the location of the brightest pixel, we first created the Map using the FITS data and imported the coordinate functionality.",
            },
        ],
    },
    {
        "id": "multi-textbook-chapters",
        "query": "Which two documents are chapters taken from a larger textbook?",
        "kind": "multi",
        "material": "multi",
        "split": "holdout",
        "evidence": [
            {"paper_id": PDF_ANALYTICS, "quote": "# Analytics"},
            {
                "paper_id": PDF_OPENPHASE,
                "quote": "## 9.1 Introduction to the OpenPhase Software Package",
            },
        ],
    },
]


def _normalized(text: str) -> str:
    return " ".join(str(text).split())


def _find_leaf(db, paper_id: str, quote: str) -> str | None:
    needle = _normalized(quote)
    rows = db.conn.execute(
        "SELECT n.node_id, b.text FROM tree_nodes n JOIN content_blocks b ON b.block_id = n.block_id "
        "WHERE n.local_id = ? AND n.state = 'ready' AND n.kind = 'leaf' ORDER BY b.ordinal",
        (paper_id,),
    ).fetchall()
    for node_id, text in rows:
        if needle in _normalized(text):
            return str(node_id)
    return None


def build(db_path: Path) -> list[dict]:
    from drbrain.storage.database import Database

    db = Database(db_path)
    try:
        out: list[dict] = []
        for entry in ENTRIES:
            evidence = []
            for item in entry["evidence"]:
                node_id = _find_leaf(db, item["paper_id"], item["quote"])
                if not node_id:
                    raise SystemExit(
                        f"quote not found in {item['paper_id']}: {item['quote'][:90]!r}"
                    )
                evidence.append(
                    {
                        "paper_id": item["paper_id"],
                        "node_id": node_id,
                        "quote": item["quote"],
                        "revision": 1,
                    }
                )
            record = {
                "id": entry["id"],
                "query": entry["query"],
                "kind": entry["kind"],
                "material": entry["material"],
                "split": entry["split"],
                "relevant_papers": sorted({ev["paper_id"] for ev in evidence}),
                "relevant_nodes": [ev["node_id"] for ev in evidence],
                "evidence": evidence,
            }
            out.append(record)
        return out
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--check", action="store_true", help="Validate without writing")
    args = parser.parse_args()

    root = Path(args.root)
    records = build(root / "data" / "drbrain.db")
    if args.check:
        print(json.dumps({"entries": len(records), "ids": [r["id"] for r in records]}))
        return 0
    target = root / "golden.jsonl"
    lines = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    target.write_text(lines, encoding="utf-8")
    splits = {split: sum(1 for r in records if r["split"] == split) for split in ("dev", "holdout")}
    kinds: dict[str, int] = {}
    for record in records:
        kinds[record["kind"]] = kinds.get(record["kind"], 0) + 1
    print(
        json.dumps(
            {
                "golden": str(target),
                "entries": len(records),
                "splits": splits,
                "kinds": kinds,
                "papers": sorted({p for r in records for p in r["relevant_papers"]}),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
