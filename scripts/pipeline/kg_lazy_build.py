#!/usr/bin/env python3
"""Lazy knowledge-graph construction (review §6.3, L1/L2 strategy layer).

L0 (full, cheap) is already covered by ``ragdb_fill.py`` (FTS5 + vectors) and
``ingest_arxiv_latex.py`` (citation keys + categories). This script adds the
two upper layers so the 0.8M-paper corpus does NOT need a front-loaded
5-stage extraction pass:

  l1              -- full corpus, cheap: for papers that have an abstract but
                     no concepts yet, extract 3-5 key concepts from the
                     abstract (+ the "## Conclusion" section of raw.md when
                     available) and insert them into the ``concepts`` table.
                     Pluggable extractor: ``--extractor heuristic`` (TF/keyword
                     rules, zero-model, default) or ``--extractor spark4b``
                     (local XHToken/Spark-X2.5-4B via transformers; best-effort
                     per paper, clean exit when the model is missing).
  l2              -- on demand: run the FULL 5-stage extraction pipeline for
                     selected papers (``--papers`` or the pending entries of
                     ``data/kg_worklist.json``) by delegating to the existing
                     ``scripts/pipeline/build.py::build_one``.
  mark-retrieved  -- append a "retrieved-but-not-yet-fully-extracted" paper id
                     to the worklist for L2 to consume.

Both build passes are idempotent: papers that already have concepts are
skipped; ``--limit`` bounds each batch.

Usage:
  python scripts/pipeline/kg_lazy_build.py l1 --extractor heuristic --limit 1000
  python scripts/pipeline/kg_lazy_build.py l1 --extractor spark4b
  python scripts/pipeline/kg_lazy_build.py mark-retrieved 2401.01234
  python scripts/pipeline/kg_lazy_build.py l2 --papers 2401.01234 2311.09999
  python scripts/pipeline/kg_lazy_build.py l2            # drain the worklist
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

MAIN_DB = REPO / "data" / "drbrain.db"
PAPERS_ROOT = REPO / "data" / "papers"
WORKLIST = REPO / "data" / "kg_worklist.json"
SPARK_MODEL_DIR = Path.home() / ".cache" / "modelscope" / "models" / "XHToken--Spark-X2.5-4B"

VALID_TYPES = frozenset({"Problem", "Method", "Conclusion", "Debate", "Gap", "Actor"})

MIN_CONCEPTS = 3
MAX_CONCEPTS = 5

# ── shared helpers ───────────────────────────────────────────────────────────


def _safe_paper_dir(arxiv_id: str) -> str:
    """arXiv ids contain ``/`` (old-style ``hep-lat/9107001``) — flatten it.

    Same rule as ``ingest_arxiv_latex.py`` so raw.md lookup matches on disk.
    """
    return arxiv_id.replace("/", "_")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── conclusion-section extraction from raw.md ────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_CONCLUSION_TITLES = ("conclusion", "conclusions", "concluding remarks", "summary", "outlook")


def _extract_conclusion_section(raw_md: str, max_chars: int = 4000) -> str:
    """Return the paragraph block under the "## Conclusion" heading (or a
    close synonym), stopping at the next markdown heading. Empty string when
    the document has no such section."""
    lines = raw_md.splitlines()
    headings: list[tuple[int, str]] = []
    for lineno, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m:
            headings.append((lineno, m.group(2).lower()))
    for i, (lineno, title) in enumerate(headings):
        if any(t in title for t in _CONCLUSION_TITLES):
            end = headings[i + 1][0] if i + 1 < len(headings) else len(lines)
            body = "\n".join(lines[lineno + 1 : end]).strip()
            return body[:max_chars]
    return ""


# ── extractor: heuristic (TF / keyword rules, zero-model) ────────────────────

_STOPWORDS = frozenset(
    """a an the and or of in on for to with by from as is are was were be been being
    am do does did doing have has had having
    this that these those it its we our us you your they their he she his her hers
    not no nor but if then than so such can could may might must shall should will would
    at into onto over under between among during after before above below up down out off
    how what when where which who whom whose why whether although because however therefore
    thus moreover furthermore also more most other others some any all both each few many
    much very quite rather only just even still yet ever never always often sometimes
    using used use uses based paper study studies result results show shows shown
    suggest suggests suggested proposed propose new novel first second third
    high low large small great little different various several possible important
    within without upon about along across toward towards due give given gives
    find finds found observed observe observed obtain obtained obtain consider considered
    become becomes became remain remains remained known unknown
    via per etc ie eg""".split()
)

_TYPE_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Conclusion",
        (
            "we find", "we found", "we show", "we demonstrate", "we conclude",
            "results show", "our results", "our findings", "our analysis",
            "indicates", "indicate", "demonstrates", "demonstrate", "reveals",
            "reveal", "in summary", "to summarize", "these findings",
            "conclude", "conclusion", "overall",
        ),
    ),
    (
        "Method",
        (
            "we propose", "we present", "we introduce", "we develop", "we derive",
            "we study", "we investigate", "we construct", "we calculate",
            "we perform", "we analyze", "we extend", "we formulate",
            "method", "approach", "framework", "formalism", "algorithm",
            "simulation", "calculation", "formulation", "scheme", "technique",
            "we model", "monte carlo", "mean field", "perturbation",
        ),
    ),
    (
        "Problem",
        (
            "problem", "challenge", "open question", "unsolved", "unresolved",
            "limitation", "difficulty", "remains unclear", "remains an open",
            "puzzle", "paradox", "issue", "obstacle", "lack of",
            "little is known", "poorly understood", "open problem",
        ),
    ),
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")
_ACRONYM_RE = re.compile(r"^[A-Z]{2,6}$")


def _sentence_type(sentence: str) -> str:
    low = sentence.lower()
    for ctype, cues in _TYPE_CUES:
        if any(cue in low for cue in cues):
            return ctype
    return "Method"  # neutral descriptive sentences default to Method


def heuristic_extract(
    chunks: list[tuple[str, str]],
    min_concepts: int = MIN_CONCEPTS,
    max_concepts: int = MAX_CONCEPTS,
) -> list[dict]:
    """Extract 3-5 key concepts from (text, section) chunks with pure TF /
    keyword rules — no model required, fully deterministic.

    Returns ``[{"label", "type", "confidence", "section"}]`` with types drawn
    from the ``concepts.type`` enum.
    """
    # 1. sentences with their source section
    sentences: list[tuple[str, str]] = []
    for text, section in chunks:
        for sent in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")):
            sent = sent.strip()
            if len(sent) >= 20:  # skip fragments / heading leftovers
                sentences.append((sent, section))

    # 2. candidate terms: content unigrams + bigrams + acronyms
    uni: Counter[str] = Counter()
    bi: Counter[str] = Counter()
    acronyms: Counter[str] = Counter()
    for sent, _ in sentences:
        toks = _WORD_RE.findall(sent)
        for tok in toks:
            if _ACRONYM_RE.match(tok):
                acronyms[tok] += 1
        content = [
            t.lower()
            if (len(t) >= 3 and t.lower() not in _STOPWORDS) or _ACRONYM_RE.match(t)
            else None
            for t in toks
        ]
        for i, t in enumerate(content):
            if t:
                uni[t] += 1
                if i + 1 < len(content) and content[i + 1]:
                    bi[f"{t} {content[i + 1]}"] += 1

    # 3. score: multiword terms and acronyms are more concept-like
    scored: list[tuple[float, str]] = []
    scored += [(cnt * 1.5, label) for label, cnt in bi.items()]
    scored += [(cnt * 1.0, label) for label, cnt in uni.items()]
    scored += [(cnt * 2.0, label) for label, cnt in acronyms.items()]
    scored.sort(key=lambda item: (-item[0], item[1]))

    # 4. greedy top-k with substring dedup (prefer the richer label)
    selected: list[dict] = []
    seen_labels: list[str] = []
    for score, label in scored:
        if len(selected) >= max_concepts:
            break
        low = label.lower()
        if any(low in prev or prev in low for prev in seen_labels):
            continue
        # attach the type of the first sentence that mentions the label
        ctype, section = "Method", ""
        for sent, sec in sentences:
            if low in sent.lower():
                ctype, section = _sentence_type(sent), sec
                break
        seen_labels.append(low)
        selected.append(
            {
                "label": label,
                "type": ctype if ctype in VALID_TYPES else "Method",
                "confidence": round(min(0.9, 0.4 + 0.05 * min(score, 10.0)), 2),
                "section": section,
            }
        )
    return selected


# ── extractor: spark4b (local XHToken/Spark-X2.5-4B, best-effort) ────────────

_spark_state: tuple | None = None

_SPARK_PROMPT = (
    "You are a scientific knowledge-graph assistant. From the paper text below, "
    "extract {min_n} to {max_n} key concepts capturing the research problem, the "
    "main method, and the main conclusion. Respond with ONLY a JSON array, each "
    'item of the form {{"label": "<short concept name>", "type": "Problem"|'
    '"Method"|"Conclusion"|"Debate"|"Gap"|"Actor", "confidence": 0.0}}.\n\n'
    "Paper text:\n{text}"
)


def _load_spark() -> tuple:
    """Load Spark-X2.5-4B lazily; clean exit with a clear message when absent."""
    global _spark_state
    if _spark_state is not None:
        return _spark_state
    if not SPARK_MODEL_DIR.is_dir():
        print(
            f"[kg-lazy] spark4b model not found at {SPARK_MODEL_DIR}; "
            "download it first (e.g. `modelscope download --model XHToken/Spark-X2.5-4B`) "
            "or use --extractor heuristic",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[kg-lazy] loading spark4b from {SPARK_MODEL_DIR} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(SPARK_MODEL_DIR), trust_remote_code=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(SPARK_MODEL_DIR), torch_dtype=dtype, trust_remote_code=True
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    _spark_state = (model, tokenizer)
    return _spark_state


def spark4b_extract(
    chunks: list[tuple[str, str]],
    min_concepts: int = MIN_CONCEPTS,
    max_concepts: int = MAX_CONCEPTS,
) -> list[dict]:
    """Best-effort L1 extraction with the local Spark-X2.5-4B model.

    A per-paper failure yields ``[]`` (caller logs and moves on); a missing
    model aborts the run via :func:`_load_spark`.
    """
    model, tokenizer = _load_spark()
    import torch

    text = "\n\n".join(t for t, _ in chunks).strip()[:6000]
    if not text:
        return []
    try:
        from drbrain.utils.llm_json import parse_llm_json

        messages = [
            {
                "role": "user",
                "content": _SPARK_PROMPT.format(
                    min_n=min_concepts, max_n=max_concepts, text=text
                ),
            }
        ]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                inputs, max_new_tokens=300, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        raw = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        parsed = parse_llm_json(raw)
    except Exception as exc:  # noqa: BLE001 — best-effort per paper
        print(f"[kg-lazy] spark4b extraction failed: {type(exc).__name__}: {exc}", flush=True)
        return []

    items = parsed if isinstance(parsed, list) else []
    concepts: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        ctype = str(item.get("type") or "Method")
        if ctype not in VALID_TYPES:
            ctype = "Method"
        try:
            conf = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        concepts.append({"label": label, "type": ctype, "confidence": max(0.0, min(conf, 1.0))})
        if len(concepts) >= max_concepts:
            break
    return concepts


EXTRACTORS = {
    "heuristic": heuristic_extract,
    "spark4b": spark4b_extract,
}


# ── worklist (L2 queue) ──────────────────────────────────────────────────────


def load_worklist(path: Path) -> dict:
    if not Path(path).exists():
        return {"pending": [], "done": []}
    wl = json.loads(Path(path).read_text(encoding="utf-8"))
    wl.setdefault("pending", [])
    wl.setdefault("done", [])
    return wl


def save_worklist(path: Path, wl: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(wl, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def mark_retrieved(paper_id: str, worklist_path: Path, db=None) -> bool:
    """Append a retrieved-but-not-yet-fully-extracted paper to the worklist.

    Idempotent: a paper already pending or done is not re-appended. Returns
    True when a new entry was added. ``db`` (optional) only enables a warning
    when the id is not yet in the library.
    """
    if db is not None and db.get_paper(paper_id) is None:
        print(f"[kg-lazy] warning: {paper_id} not in the library (marking anyway)", flush=True)
    wl = load_worklist(worklist_path)
    known = {e["paper_id"] for e in wl["pending"]} | {e["paper_id"] for e in wl["done"]}
    if paper_id in known:
        print(f"[kg-lazy] {paper_id} already in worklist", flush=True)
        return False
    wl["pending"].append({"paper_id": paper_id, "marked_at": _now_iso()})
    save_worklist(worklist_path, wl)
    print(f"[kg-lazy] marked retrieved: {paper_id} → {worklist_path}", flush=True)
    return True


def _worklist_pending_ids(wl: dict) -> list[str]:
    return [e["paper_id"] for e in wl["pending"]]


def _worklist_move_to_done(wl: dict, paper_id: str) -> None:
    wl["pending"] = [e for e in wl["pending"] if e["paper_id"] != paper_id]
    if not any(e["paper_id"] == paper_id for e in wl["done"]):
        wl["done"].append({"paper_id": paper_id, "extracted_at": _now_iso()})


# ── L1: full corpus, cheap (abstract + conclusion → 3-5 concepts) ────────────

_L1_SELECT = """
    SELECT p.local_id, p.title, p.abstract, p.year
    FROM papers p
    WHERE TRIM(COALESCE(p.abstract, '')) != ''
      AND NOT EXISTS (SELECT 1 FROM concepts c WHERE c.local_id = p.local_id)
    ORDER BY p.local_id
"""


def run_l1(
    db,
    papers_root: Path,
    extractor: str = "heuristic",
    limit: int = 0,
    min_concepts: int = MIN_CONCEPTS,
    max_concepts: int = MAX_CONCEPTS,
) -> dict:
    """Coarse KG pass over every paper that has an abstract but no concepts."""
    if extractor not in EXTRACTORS:
        print(f"[kg-lazy] unknown extractor: {extractor}", file=sys.stderr, flush=True)
        raise SystemExit(2)
    extract_fn = EXTRACTORS[extractor]

    rows = db.execute(_L1_SELECT).fetchall()
    if limit > 0:
        rows = rows[:limit]
    print(
        f"[kg-lazy] L1: {len(rows)} papers to process "
        f"(extractor={extractor}, root={papers_root})",
        flush=True,
    )

    stats = {"selected": len(rows), "processed": 0, "skipped": 0, "inserted": 0}
    t0 = time.time()
    for local_id, title, abstract, year in rows:
        if db.execute(
            "SELECT 1 FROM concepts WHERE local_id = ? LIMIT 1", (local_id,)
        ).fetchone():
            stats["skipped"] += 1
            continue
        chunks: list[tuple[str, str]] = [(abstract, "abstract")]
        raw_md_path = papers_root / _safe_paper_dir(local_id) / "raw.md"
        if raw_md_path.exists():
            conclusion = _extract_conclusion_section(
                raw_md_path.read_text(encoding="utf-8", errors="ignore")
            )
            if conclusion:
                chunks.append((conclusion, "conclusion"))
        concepts = extract_fn(chunks, min_concepts=min_concepts, max_concepts=max_concepts)
        n = 0
        for c in concepts:
            label = str(c.get("label") or "").strip()
            ctype = c.get("type") or "Method"
            if not label or ctype not in VALID_TYPES:
                continue
            db.insert_concept(
                local_id,
                ctype,
                label,
                float(c.get("confidence", 0.5)),
                year=year,
                section=str(c.get("section") or ""),
            )
            n += 1
        db.commit()
        stats["processed"] += 1
        stats["inserted"] += n
        if stats["processed"] % 200 == 0:
            print(
                f"[kg-lazy] L1 progress: {stats['processed']} papers, "
                f"{stats['inserted']} concepts ({time.time() - t0:.0f}s)",
                flush=True,
            )
    print(
        f"[kg-lazy] L1 done: processed={stats['processed']} skipped={stats['skipped']} "
        f"concepts={stats['inserted']} in {time.time() - t0:.0f}s",
        flush=True,
    )
    return stats


# ── L2: on-demand full 5-stage extraction ────────────────────────────────────


def _get_build_one():
    """Import ``scripts/pipeline/build.py::build_one`` — the existing 5-stage
    full-extraction entry (tree → ontology → entities → relations → coref →
    refine → concepts/edges 插入). Reused verbatim, not re-implemented."""
    import importlib.util

    build_path = REPO / "scripts" / "pipeline" / "build.py"
    if not build_path.exists():
        raise RuntimeError(f"full-extraction script missing: {build_path}")
    spec = importlib.util.spec_from_file_location("kg_lazy_build_pipeline_build", build_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_one


def _full_extract(db, paper_id: str, cfg: dict, papers_root: Path, skip_refine: bool = True) -> dict:
    """Delegate one paper to the existing full extraction pipeline."""
    build_one = _get_build_one()
    return build_one(paper_id, cfg, db, skip_refine=skip_refine)


def _load_cfg(config_path: str | None = None, papers_root: Path | None = None) -> dict:
    """Repo-local config.yaml (+ optional overlays) for the L2 pipeline.

    ``dirs.papers`` is forced to the absolute papers root so build_one resolves
    tree.json/raw.md regardless of CWD (scripts/pipeline/common.py pins a
    different repo path, hence the local implementation).
    """
    import yaml

    from drbrain.config import merge_dicts

    base = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8")) or {}
    cfg = base
    local_path = REPO / "config.local.yaml"
    if local_path.exists():
        cfg = merge_dicts(base, yaml.safe_load(local_path.read_text(encoding="utf-8")) or {})
    if config_path:
        cfg = merge_dicts(cfg, yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {})
    cfg.setdefault("dirs", {})
    if papers_root is not None:
        cfg["dirs"]["papers"] = str(papers_root)
    return cfg


def run_l2(
    db,
    papers_root: Path,
    paper_ids: list[str] | None = None,
    worklist_path: Path | None = None,
    cfg: dict | None = None,
    limit: int = 0,
    skip_refine: bool = False,
) -> dict:
    """Full 5-stage extraction for selected papers (or the worklist backlog)."""
    wl = load_worklist(worklist_path) if worklist_path else {"pending": [], "done": []}
    ids = list(paper_ids) if paper_ids else _worklist_pending_ids(wl)
    if limit > 0:
        ids = ids[:limit]
    if not ids:
        print("[kg-lazy] L2: no papers selected (empty --papers and empty worklist)", flush=True)
        return {"selected": 0, "ok": 0, "skipped": 0, "failed": 0}

    if cfg is None:
        cfg = _load_cfg(papers_root=papers_root)
    if not cfg.get("llm", {}).get("models", []):
        print(
            "[kg-lazy] L2 needs llm.models in config.yaml (run `drbrain setup`)",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2)

    print(f"[kg-lazy] L2: {len(ids)} papers for full extraction", flush=True)
    stats = {"selected": len(ids), "ok": 0, "skipped": 0, "failed": 0}
    worklist_changed = False
    for pid in ids:
        if db.get_paper(pid) is None:
            print(f"[kg-lazy] L2: {pid} not in library — keeping in worklist", flush=True)
            stats["failed"] += 1
            continue
        if db.execute("SELECT 1 FROM concepts WHERE local_id = ? LIMIT 1", (pid,)).fetchone():
            print(f"[kg-lazy] L2: {pid} already has concepts — skip", flush=True)
            stats["skipped"] += 1
            if worklist_path and pid in _worklist_pending_ids(wl):
                _worklist_move_to_done(wl, pid)
                worklist_changed = True
            continue
        result = _full_extract(db, pid, cfg, papers_root, skip_refine=skip_refine)
        if result.get("ok"):
            stats["ok"] += 1
            if worklist_path and pid in _worklist_pending_ids(wl):
                _worklist_move_to_done(wl, pid)
                worklist_changed = True
            print(f"[kg-lazy] L2: {pid} extracted", flush=True)
        else:
            stats["failed"] += 1
            print(f"[kg-lazy] L2: {pid} FAILED: {result.get('error')}", flush=True)
    if worklist_path and worklist_changed:
        save_worklist(worklist_path, wl)
    print(
        f"[kg-lazy] L2 done: ok={stats['ok']} skipped={stats['skipped']} "
        f"failed={stats['failed']}",
        flush=True,
    )
    return stats


# ── CLI ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    from drbrain.storage.database import Database

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p_l1 = sub.add_parser("l1", help="coarse pass: abstract+conclusion → 3-5 concepts")
    p_l1.add_argument("--db", type=Path, default=MAIN_DB)
    p_l1.add_argument("--papers-root", type=Path, default=PAPERS_ROOT)
    p_l1.add_argument("--extractor", choices=sorted(EXTRACTORS), default="heuristic")
    p_l1.add_argument("--limit", type=int, default=0, help="max papers this batch (0 = all)")

    p_l2 = sub.add_parser("l2", help="on-demand full 5-stage extraction")
    p_l2.add_argument("--db", type=Path, default=MAIN_DB)
    p_l2.add_argument("--papers-root", type=Path, default=PAPERS_ROOT)
    p_l2.add_argument("--papers", nargs="*", default=None, help="paper ids (default: worklist)")
    p_l2.add_argument("--worklist", type=Path, default=WORKLIST)
    p_l2.add_argument("--limit", type=int, default=0)
    p_l2.add_argument("--config", type=str, default=None, help="extra config overlay (yaml)")
    p_l2.add_argument("--skip-refine", action="store_true", help="skip stage 5 refinement")

    p_mark = sub.add_parser("mark-retrieved", help="queue a retrieved paper for L2")
    p_mark.add_argument("paper_id")
    p_mark.add_argument("--worklist", type=Path, default=WORKLIST)
    p_mark.add_argument("--db", type=Path, default=MAIN_DB, help="library DB (existence check)")

    args = ap.parse_args(argv)
    if args.command == "mark-retrieved":
        db = Database(str(args.db))
        try:
            mark_retrieved(args.paper_id, args.worklist, db=db)
        finally:
            db.close()
        return 0

    db = Database(str(args.db))
    try:
        if args.command == "l1":
            run_l1(
                db,
                args.papers_root,
                extractor=args.extractor,
                limit=args.limit,
            )
        else:
            cfg = _load_cfg(config_path=args.config, papers_root=args.papers_root)
            run_l2(
                db,
                args.papers_root,
                paper_ids=args.papers,
                worklist_path=args.worklist,
                cfg=cfg,
                limit=args.limit,
                skip_refine=args.skip_refine,
            )
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
