"""Quality alignment test: local vLLM vs glm-4.5-air concept extraction.

Runs the SAME lean prompt on the same sampled abstracts through both
endpoints, then reports per-paper concept overlap and normalisation
quality. Threshold for switching to local: overlap > 80%.

Usage:
    uv run python scripts/cg_quality_compare.py [--n 100] \
        --local-base http://localhost:8000/v1 --local-model Qwen/Qwen3.5-9B
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from drbrain.config import load_config

LEAN_PROMPT = open("prompts/extract_concepts_lean.txt", encoding="utf-8").read()

FILLER_WORDS = {"of", "the", "a", "an", "for", "in", "on", "and"}


def sample_abstracts(db_path: str, n: int, seed: int = 42) -> list[tuple[str, str]]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT title, abstract FROM papers "
        "WHERE abstract IS NOT NULL AND length(abstract) > 200 "
        "ORDER BY local_id LIMIT ?",
        (n * 3,),
    ).fetchall()
    conn.close()
    # deterministic spread across the corpus
    step = max(1, len(rows) // n)
    return [(t, a) for t, a in rows[::step]][:n]


def extract(client: OpenAI, model: str, text: str, extra: dict | None) -> list[str]:
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": LEAN_PROMPT},
            {"role": "user", "content": text[:6000]},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 400,
        "temperature": 0.1,
        "timeout": 120,
    }
    if extra:
        kwargs["extra_body"] = extra
    for attempt in range(2):
        try:
            r = client.chat.completions.create(**kwargs)
            data = json.loads(r.choices[0].message.content or "{}")
            return [str(c).strip().lower() for c in data.get("concepts", [])]
        except Exception:  # noqa: BLE001
            if attempt == 0:
                time.sleep(3)
    return []


def overlap(a: list[str], b: list[str]) -> float:
    """Symmetric containment: |A∩B| / min(|A|, |B|) — robust to length diff."""
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / min(len(sa), len(sb))


def norm_score(labels: list[str]) -> float:
    """Fraction of labels satisfying the normalisation rules."""
    if not labels:
        return 0.0
    ok = 0
    for lab in labels:
        words = lab.split()
        good = (
            lab == lab.lower()
            and 1 <= len(words) <= 5
            and not any(w in FILLER_WORDS for w in words)
            and not lab.endswith("s ")
        )
        ok += good
    return ok / len(labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--local-base", default="http://localhost:8000/v1")
    ap.add_argument("--local-model", default="qwen3.5-9b")
    ap.add_argument("--ref-base", default="https://api.deepseek.com")
    ap.add_argument("--ref-model", default="deepseek-v4-flash")
    ap.add_argument("--ref-key", default=os.environ.get("CG_REF_KEY", ""))
    args = ap.parse_args()
    if not args.ref_key:
        raise SystemExit("reference key required: --ref-key or CG_REF_KEY env var")

    cfg = load_config()
    glm_client = OpenAI(api_key=args.ref_key, base_url=args.ref_base, max_retries=1)
    ref_extra = (
        {"thinking": {"type": "disabled"}}
        if "deepseek" in args.ref_base
        else {"enable_thinking": False}
    )
    local_client = OpenAI(api_key="not-needed", base_url=args.local_base, max_retries=1)

    papers = sample_abstracts(cfg.db.path, args.n)
    print(f"comparing {len(papers)} papers: {args.local_model} vs {args.ref_model}", flush=True)

    t0 = time.time()
    overlaps: list[float] = []
    norm_local: list[float] = []
    norm_glm: list[float] = []
    empty_local = empty_glm = 0

    def work(i_text: tuple[int, tuple[str, str]]):
        i, (title, abstract) = i_text
        text = f"{title}\n\n{abstract}".strip()
        lc = extract(local_client, args.local_model, text, None)
        gc = extract(glm_client, args.ref_model, text, ref_extra)
        return i, lc, gc

    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, lc, gc in pool.map(work, enumerate(papers)):
            if not lc:
                empty_local += 1
            if not gc:
                empty_glm += 1
            overlaps.append(overlap(lc, gc))
            norm_local.append(norm_score(lc))
            norm_glm.append(norm_score(gc))
            if (i + 1) % 25 == 0:
                print(f"[{i + 1}/{len(papers)}] elapsed {time.time() - t0:.0f}s", flush=True)

    n = len(papers)
    mean_ov = sum(overlaps) / n
    over80 = sum(1 for o in overlaps if o >= 0.8) / n
    print(f"\n=== results ({n} papers, {time.time() - t0:.0f}s) ===")
    print(f"mean overlap (containment): {mean_ov:.1%}")
    print(f"papers with overlap >= 80%:  {over80:.1%}")
    print(f"normalisation quality: local={sum(norm_local) / n:.1%} ref={sum(norm_glm) / n:.1%}")
    print(f"empty results: local={empty_local} ref={empty_glm}")
    verdict = "PASS: switch to local" if over80 >= 0.6 and mean_ov >= 0.8 else "REVIEW manually"
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
