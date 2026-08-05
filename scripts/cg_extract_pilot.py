"""Pilot: compare concept-extraction cost & quality across prompts/models.

Sample N abstracts from the library, run (A) the heavy drbrain prompt and
(B) a paper-style lean prompt on the cheapest available models, and print
token usage, latency, and per-paper cost. No writes to the DB.

Usage:
    uv run python scripts/cg_extract_pilot.py [--n 20] [--model glm-4.5-air]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time

from openai import OpenAI

from drbrain.config import load_config

LEAN_PROMPT = """Extract key research concepts from the paper title+abstract. Output STRICT JSON {"concepts": ["label", ...]}.
Rules:
- 8-20 short noun phrases (1-5 words each), lowercase.
- Normalize: singular form, drop filler words like "of"/"the", expand abbreviations only if the long form appears.
- Include materials, chemical formulae, methods, techniques, properties, and phenomena.
- Only concepts actually grounded in the text; no invented terms."""


def sample_abstracts(db_path: str, n: int) -> list[tuple[str, str]]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT title, abstract FROM papers "
        "WHERE abstract IS NOT NULL AND length(abstract) > 200 "
        "ORDER BY random() LIMIT ?",
        (n,),
    ).fetchall()
    conn.close()
    return [(t, a) for t, a in rows]


def run_one(client: OpenAI, model: str, system: str, text: str) -> dict:
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text[:8000]},
            ],
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": False},
            max_tokens=600,
            temperature=0.1,
            timeout=60,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "in": 0,
            "out": 0,
            "secs": round(time.time() - t0, 1),
            "ok": False,
            "n_concepts": 0,
            "sample": [],
            "error": str(exc)[:80],
        }
    dt = time.time() - t0
    content = r.choices[0].message.content or "{}"
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = None
    u = r.usage
    return {
        "in": u.prompt_tokens,
        "out": u.completion_tokens,
        "secs": round(dt, 1),
        "ok": parsed is not None,
        "n_concepts": len(parsed.get("concepts", parsed.get("methods", []))) if parsed else 0,
        "sample": (parsed or {}).get("concepts", [])[:6],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--model", default="glm-4.5-air")
    args = ap.parse_args()

    cfg = load_config()
    primary = cfg.llm.models[0]
    client = OpenAI(api_key=primary["api_key"], base_url=primary["base_url"], max_retries=1)
    heavy = (cfg.llm.models and open("prompts/extract_concepts.txt").read()) or ""

    papers = sample_abstracts(cfg.db.path, args.n)
    n = len(papers)
    print(f"pilot: {n} abstracts, model={args.model}\n")

    totals = {
        "lean": {"in": 0, "out": 0, "secs": 0.0, "ok": 0, "concepts": 0},
        "heavy": {"in": 0, "out": 0, "secs": 0.0, "ok": 0, "concepts": 0},
    }

    for i, (title, abstract) in enumerate(papers):
        text = f"{title}\n\n{abstract}"
        lean = run_one(client, args.model, LEAN_PROMPT, text)
        hv = run_one(client, args.model, heavy, text)
        print(
            f"[{i + 1}/{n}] lean={lean['secs']}s/{lean['out']}tok heavy={hv['secs']}s/{hv['out']}tok"
            + (f" ERR:{lean.get('error', '')}" if not lean["ok"] else ""),
            flush=True,
        )
        for key, res in (("lean", lean), ("heavy", hv)):
            t = totals[key]
            t["in"] += res["in"]
            t["out"] += res["out"]
            t["secs"] += res["secs"]
            t["ok"] += res["ok"]
            t["concepts"] += res["n_concepts"]
        if i < 3:
            print(f"--- paper {i + 1}: {title[:60]}")
            print(f"  lean : {lean['out']:>4} tok {lean['secs']:>5}s -> {lean['sample']}")
            print(f"  heavy: {hv['out']:>4} tok {hv['secs']:>5}s -> {hv['n_concepts']} items")

    print("\n=== summary (per-paper avg) ===")
    for key, t in totals.items():
        print(
            f"{key:5s}: in={t['in'] / n:.0f} out={t['out'] / n:.0f} tok, "
            f"{t['secs'] / n:.1f}s, parse_ok={t['ok']}/{n}, concepts/paper={t['concepts'] / n:.1f}"
        )
    est_lean = totals["lean"]["out"] / n
    est_heavy = totals["heavy"]["out"] / n
    print(
        f"\n22w papers output-token estimate: lean~{est_lean * 220000 / 1e6:.1f}M vs heavy~{est_heavy * 220000 / 1e6:.1f}M"
    )


if __name__ == "__main__":
    main()
