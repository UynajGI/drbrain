"""WebUI service baseline: fixed machine / data scale timings (design §7.1).

Run: .venv/bin/python scripts/webui_baseline.py  (not a pytest file)
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

root = Path(tempfile.mkdtemp()) / "root"
(root / "data").mkdir(parents=True)
os.environ["DRBRAIN_ROOT"] = str(root)

from drbrain.app import service  # noqa: E402
from drbrain.loop.store import RunLedger  # noqa: E402
from drbrain.storage.database import Database  # noqa: E402

cfg = {
    "db": {"path": "data/test.db"},
    "llm": {"models": []},
    "dirs": {"papers": "data/papers"},
    "autoresearch": {"enabled": False, "run_dir": "workspace/runs", "plugins_dir": ""},
}

PAPERS = 2000
EVENTS = 10000

db = Database(root / "data" / "test.db")
for index in range(PAPERS):
    db.insert_paper(
        f"p-{index:05d}",
        f"研究论文标题 {index} — flat band mechanisms in kagome systems",
        2024,
        "extracted",
        authors="作者甲, 作者乙",
        journal="Physical Review B",
    )
db.commit()
db.close()

ledger = RunLedger(service.ledger_path(cfg))
run = ledger.get_or_create_run("baseline topic", project_id="prj-default")
start = time.perf_counter()
with ledger.transaction() as conn:
    for index in range(EVENTS):
        ledger.append_event(
            conn,
            run.run_id,
            actor="analyst",
            event_type="proposal_recorded",
            payload={"index": index, "statement": f"claim {index}"},
        )
print(f"seed: {PAPERS} papers + {EVENTS} events in {time.perf_counter() - start:.1f}s")


def timed(label: str, fn):
    start = time.perf_counter()
    result = fn()
    elapsed = (time.perf_counter() - start) * 1000
    print(f"{label}: {elapsed:.0f} ms")
    return result, elapsed


page, _ = timed("papers first page (20)", lambda: service.papers(cfg, limit=20))
assert page["next_cursor"] and page["total"] == PAPERS


def walk():
    cursor = None
    count = 0
    pages = 0
    while True:
        data = service.papers(cfg, limit=100, cursor=cursor)
        count += len(data["items"])
        pages += 1
        cursor = data["next_cursor"]
        if not cursor:
            return count, pages


(count, pages), _ = timed(f"papers full walk ({PAPERS})", walk)
assert count == PAPERS, count
print(f"  walked {count} papers over {pages} cursor pages")

sessions_data, _ = timed("sessions list (100)", lambda: service.sessions(cfg, limit=100))
assert sessions_data["total"] == 0  # corpus-only baseline: no conversations seeded
events, _ = timed(
    "run events first page (200)", lambda: service.run_events(cfg, run.run_id, limit=200)
)
assert len(events) == 200
detail, _ = timed(
    "run detail (claim/experiment joins)", lambda: service.run_detail(cfg, run.run_id)
)
assert detail["events"] == EVENTS + 1
runs, _ = timed("runs list (limit 200)", lambda: service.runs(cfg, limit=200))
print("json size:", len(json.dumps({"page": page, "detail": detail}, default=str)), "bytes")
