#!/usr/bin/env python
"""T54 acceptance: 9 mixed PDF/TeX/MD samples through the real CLIs.

The nine samples (3 PDF + 3 TeX + 3 MD, MD includes two legacy dirs selected
by ``storage audit``) live in the isolated acceptance runtime; every step
below shells out to the real ``drbrain`` CLI and records a JSON report:

  audit      -- pre-migration ``storage audit``
  plan       -- ``storage migrate --dry-run`` twice (determinism)
  apply      -- ``storage migrate --apply`` then again (idempotent no-op)
  verify     -- post-migration reconciliation (counts, hashes, no fake ready)
  prepare    -- ``rag prepare --unified`` (incremental)
  ask        -- one question per format; answer sources must be verifiable

Usage:
  python scripts/acceptance/mixed9_acceptance.py --steps audit,plan,apply,verify \
      --report /tmp/t54.json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "data" / "integration" / "unified-tree"
DB_PATH = DEFAULT_ROOT / "data" / "drbrain.db"
LEGACY_SAMPLES = ("scibase-legacy-1", "scibase-legacy-2")
ASK_QUESTIONS = {
    "pdf": "What does the tutorial say about the OpenPhase software?",
    "tex": "What does the report say about flat band criteria?",
    "md": "What does the SunPy paper describe about solar flare data analysis?",
}


def drbrain(args: list[str], *, root: Path, expect_zero: bool = True) -> str:
    env = {**os.environ, "DRBRAIN_ROOT": str(root)}
    env.pop("DRBRAIN_ROOT_OVERRIDE", None)
    proc = subprocess.run(
        [str(REPO / ".venv" / "bin" / "drbrain"), *args],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=5400,
    )
    if expect_zero and proc.returncode != 0:
        raise SystemExit(
            f"drbrain {' '.join(args)} exited {proc.returncode}:\n{proc.stderr[-2000:]}"
        )
    return proc.stdout


def json_tail(text: str) -> dict:
    """Parse the last JSON object printed by a command."""
    start = text.find("{")
    while start != -1:
        try:
            return json.loads(text[start:])
        except ValueError:
            start = text.find("{", start + 1)
    raise SystemExit(f"no JSON object in output: {text[-400:]}")


def _counts() -> dict:
    conn = sqlite3.connect(DB_PATH)
    try:
        q = conn.execute
        return {
            "documents": q("SELECT COUNT(*) FROM document_revisions").fetchone()[0],
            "papers": q("SELECT COUNT(*) FROM papers").fetchone()[0],
            "blocks": q("SELECT COUNT(*) FROM content_blocks").fetchone()[0],
            "leaves_ready": q(
                "SELECT COUNT(*) FROM tree_nodes WHERE kind='leaf' AND state='ready'"
            ).fetchone()[0],
            "regions_ready": q(
                "SELECT COUNT(*) FROM tree_nodes WHERE kind='region' AND state='ready'"
            ).fetchone()[0],
        }
    finally:
        conn.close()


def _body_matches_source(local_id: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT b.text FROM content_blocks b WHERE b.local_id = ? AND b.revision = 1 "
            "ORDER BY b.ordinal",
            (local_id,),
        ).fetchall()
        body = "".join(str(r[0]) for r in row)
    finally:
        conn.close()
    raw = (DEFAULT_ROOT / "data" / "papers" / local_id / "raw.md").read_text(
        encoding="utf-8", errors="replace"
    )
    return body == raw


def step_audit(report: dict, root: Path) -> None:
    before = json_tail(drbrain(["storage", "audit", "--json", "--sample", "3"], root=root))
    report["audit_before"] = {
        "counts": before.get("counts"),
        "legacy_raw_md": before.get("counts", {}).get("legacy_raw_md"),
        "findings": [f.get("category") for f in before.get("findings", [])],
    }


def step_plan(report: dict, root: Path) -> None:
    first = json_tail(drbrain(["storage", "migrate", "--dry-run", "--json"], root=root))
    second = json_tail(drbrain(["storage", "migrate", "--dry-run", "--json"], root=root))
    report["plan"] = {
        "plan_id": first.get("plan_id"),
        "deterministic": first == second,
        "summary": first.get("summary"),
        "items": [
            {"local_id": item["local_id"], "action": item["action"], "reason": item["reason"]}
            for item in first.get("items", [])
        ],
    }


def step_apply(report: dict, root: Path) -> None:
    # Controlled interruption: pause after one item, then resume to the end,
    # then run once more to prove re-apply is a no-op.
    paused = json_tail(
        drbrain(["storage", "migrate", "--apply", "--json", "--max-items", "1"], root=root)
    )
    resumed = json_tail(drbrain(["storage", "migrate", "--apply", "--json"], root=root))
    noop = json_tail(drbrain(["storage", "migrate", "--apply", "--json"], root=root))
    report["apply"] = {
        "paused": {
            "counts": paused.get("counts"),
            "checked_pause": paused.get("paused"),
        },
        "resumed": {
            "counts": resumed.get("counts"),
            "applied": [entry["local_id"] for entry in resumed.get("applied", [])],
            "failed": resumed.get("failed"),
        },
        "noop": {"counts": noop.get("counts"), "failed": noop.get("failed")},
    }


def step_verify(report: dict, root: Path) -> None:
    after = json_tail(drbrain(["storage", "audit", "--json", "--sample", "3"], root=root))
    consistent = {local_id: _body_matches_source(local_id) for local_id in LEGACY_SAMPLES}
    report["verify"] = {
        "counts": after.get("counts"),
        "findings": [f.get("category") for f in after.get("findings", [])],
        "legacy_body_verbatim": consistent,
        "papers": _counts(),
    }


def step_prepare(report: dict, root: Path) -> None:
    outcome = json_tail(
        drbrain(
            ["--config", "config.t48.yaml", "rag", "prepare", "--unified", "--json"],
            root=root,
        )
    )
    report["prepare"] = outcome


def _ask(question: str, root: Path, *extra: str) -> dict:
    return json_tail(
        drbrain(
            [
                "--config",
                "config.t48.yaml",
                "ask",
                question,
                "--json",
                "--top",
                "5",
                *extra,
            ],
            root=root,
        )
    )


def _ask_summary(payload: dict) -> dict:
    return {
        "status": payload.get("status", "ok"),
        "route": payload.get("route", {}),
        "sources": len(payload.get("sources", [])),
        "evidence_ids": payload.get("evidence_ids", [])[:5],
        "answer_head": str(payload.get("answer", ""))[:140],
        "truncated": payload.get("answer_truncated"),
    }


def step_ask(report: dict, root: Path) -> None:
    """Record the configured route (may degrade) and the available bm25 route.

    The SQL snapshot copies legacy ``tree_vectors``; the unified vectors live
    in the tree generation (T61 switches the reader).  Until then the vector
    and tree legs report ``source_unavailable`` while bm25 answers.
    """
    answers = {}
    for fmt, question in ASK_QUESTIONS.items():
        full = _ask(question, root)
        lean = _ask(question, root, "--legs", "bm25")
        answers[fmt] = {
            "full_route": _ask_summary(full),
            "bm25_route": _ask_summary(lean),
        }
    report["ask"] = answers


STEPS = {
    "audit": step_audit,
    "plan": step_plan,
    "apply": step_apply,
    "verify": step_verify,
    "prepare": step_prepare,
    "ask": step_ask,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--steps", default="audit,plan,apply,verify")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    report: dict = {"root": str(root), "steps": args.steps}
    for name in [part.strip() for part in args.steps.split(",") if part.strip()]:
        if name not in STEPS:
            raise SystemExit(f"unknown step {name!r}; expected {sorted(STEPS)}")
        STEPS[name](report, root)
        report.setdefault("completed_steps", []).append(name)

    payload = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.report:
        Path(args.report).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
