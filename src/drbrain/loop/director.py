"""Research director — AutoScientists-style continuous research loop.

The 12-node :class:`ResearchLoopWorkflow` is one *cycle*. The director runs
cycles repeatedly on a topic until stagnation, keeping a **file-based task
workspace** (mirroring AutoScientists: ``champion.md`` / ``dead_ends.md`` /
``knowledge/patterns.md`` / ``results/cycle-*.md`` — not a single JSON blob)
so the run is resumable, auditable, and each semantic object has its own file.

The architecture here is **domain-agnostic**: the director only orchestrates and
persists. Concrete capabilities (models, software, data) are injected by the
*agent itself* through the plugin protocol at runtime — the director never
imports a domain plugin.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from loguru import logger

from drbrain.loop.events import ResearchState
from drbrain.loop.workflow import (
    CRITIQUE_DISCARD_SCORE,
    ResearchLoopWorkflow,
    _job_log_has_number,
)


def _slug(topic: str) -> str:
    """Filesystem-safe slug for a topic (run dir name)."""
    s = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "-", topic.strip()).strip("-")
    return (s or "research")[:80]


# T-janitor: an async job that claimed compute but produced no numeric log within
# this many seconds is flagged stale (AutoScientists' monitor re-claim, simplified:
# the director scans once per cycle instead of running a monitor process).
JANITOR_STALE_SECONDS = 1800  # 30 minutes

# T8/endorsement: a structural change (Phase 4 pivot) needs the critic's
# independent veto on top of stagnation. Same bar as the T5 review gate
# (CRITIQUE_DISCARD_SCORE) — a mean score below it means the critic no longer
# backs the current hypothesis direction.
CRITIC_ENDORSEMENT_SCORE = CRITIQUE_DISCARD_SCORE


def _default_state(topic: str) -> dict[str, Any]:
    """Fresh in-memory state (exported for tests / first-run bootstrap)."""
    now = time.time()
    return {
        "topic": topic,
        "cycles": 0,
        "champion": [],
        "rejected": [],
        "results": [],
        "consecutive_no_gain": 0,
        "adaptations": 0,  # Phase 4: 停滞转向次数
        "started_at": now,
        "updated_at": now,
    }


# ── markdown frontmatter helpers ──────────────────────────────────────────────


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a ``---\\nkey: val\\n---`` header from the body (best-effort)."""
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    if len(lines) < 2 or lines[1] != "" or "---" not in lines[2:]:
        # header is a single `---` on line 0? tolerate either form
        pass
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return {}, text
    fm: dict[str, Any] = {}
    for ln in lines[1:end]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            fm[k.strip()] = v.strip()
    return fm, "\n".join(lines[end + 1 :]).strip()


def _render_frontmatter(fm: dict[str, Any], body: str) -> str:
    head = "\n".join(f"{k}: {v}" for k, v in fm.items())
    return f"---\n{head}\n---\n\n{body}".strip() + "\n"


# ── run.json (runtime-only state: cycle counter + stagnation) ─────────────────


class ResearchDirector:
    """Runs the research loop continuously on one topic into a file workspace.

    Workspace layout per topic (``run_dir/<topic>/``):

    - ``task.md`` — the research task (bootstrap, written once)
    - ``champion.md`` — confirmed conclusions (frontmatter ``count``)
    - ``dead_ends.md`` — rejected hypotheses (frontmatter ``count``)
    - ``knowledge/patterns.md`` — winning patterns + dead ends + exhausted axes
    - ``knowledge/role-{critic|verifier}.md`` — per-role cross-cycle memory (T7)
    - ``knowledge/proposals.md`` — proposal board, one post per hypothesis (T8)
    - ``knowledge/reviews.md`` — the critic's non-author reviews of proposals (T8)
    - ``results/cycle-NNN.md`` — per-cycle evidence (verified/predictions/hypotheses/report)
    - ``run.json`` — runtime-only resume state (cycles, no-gain counter, timestamps)
    """

    def __init__(
        self,
        cfg: Any,
        *,
        db: Any = None,
        graph: Any = None,
        plugins_dir: str | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
        run_dir: str | Path = "workspace/autoresearch",
        n_critics: int = 3,
    ) -> None:
        self._cfg = cfg
        self._db = db
        self._graph = graph
        self._plugins_dir = plugins_dir
        self._mcp_servers = mcp_servers
        self._run_dir = Path(run_dir)
        self._n_critics = max(1, int(n_critics))

    # ── workspace paths ───────────────────────────────────────────────────────

    def _topic_dir(self, topic: str) -> Path:
        d = self._run_dir / _slug(topic)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _run_json(self, topic: str) -> Path:
        return self._topic_dir(topic) / "run.json"

    def _champion_md(self, topic: str) -> Path:
        return self._topic_dir(topic) / "champion.md"

    def _dead_ends_md(self, topic: str) -> Path:
        return self._topic_dir(topic) / "dead_ends.md"

    def _patterns_md(self, topic: str) -> Path:
        return self._topic_dir(topic) / "knowledge" / "patterns.md"

    def _role_memory_md(self, topic: str, role: str) -> Path:
        """T7: per-role cross-cycle memory file (``knowledge/role-{critic|verifier}.md``)."""
        return self._topic_dir(topic) / "knowledge" / f"role-{role}.md"

    def _result_md(self, topic: str, cycle: int) -> Path:
        return self._topic_dir(topic) / "results" / f"cycle-{cycle:03d}.md"

    def _roster_md(self, topic: str) -> Path:
        """Team roster file (AutoScientists ``teams/roster.md``)."""
        return self._topic_dir(topic) / "teams" / "roster.md"

    # ── load / save (derived from the files, not a JSON blob) ─────────────────

    def _load_state(self, topic: str) -> dict[str, Any]:
        """Reconstruct the in-memory state from the workspace files."""
        run: dict[str, Any] = {
            "cycles": 0,
            "consecutive_no_gain": 0,
            "started_at": time.time(),
            "updated_at": time.time(),
        }
        if self._run_json(topic).exists():
            try:
                run.update(json.loads(self._run_json(topic).read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                pass

        champion: list[dict[str, Any]] = []
        if self._champion_md(topic).exists():
            _, body = _parse_frontmatter(self._champion_md(topic).read_text(encoding="utf-8"))
            for ln in body.splitlines():
                m = re.match(r"^- \[cycle (\d+)\] (.*)$", ln.strip())
                if m:
                    champion.append(
                        {"statement": m.group(2), "cycle": int(m.group(1)), "confidence": 1.0}
                    )

        rejected: list[str] = []
        if self._dead_ends_md(topic).exists():
            _, body = _parse_frontmatter(self._dead_ends_md(topic).read_text(encoding="utf-8"))
            rejected = [ln[2:].strip() for ln in body.splitlines() if ln.startswith("- ")]

        return {
            "topic": topic,
            "cycles": run.get("cycles", 0),
            "champion": champion,
            "rejected": rejected,
            "results": [],  # reconstructed lazily from results/*.md (kept for report)
            "consecutive_no_gain": run.get("consecutive_no_gain", 0),
            "adaptations": run.get("adaptations", 0),
            "pending": run.get("pending", []),
            "mode": run.get("mode", "execute"),
            "started_at": run.get("started_at", time.time()),
            "updated_at": run.get("updated_at", time.time()),
        }

    def _save_state(self, topic: str, state: dict[str, Any]) -> None:
        """Persist the semantic state to its canonical files (single-writer)."""
        state["updated_at"] = time.time()

        # task.md (bootstrap)
        task_path = self._topic_dir(topic) / "task.md"
        if not task_path.exists():
            task_path.write_text(f"# 研究任务\n\n{state['topic']}\n", encoding="utf-8")

        # champion.md
        body = "\n".join(f"- [cycle {c['cycle']}] {c['statement']}" for c in state["champion"])
        self._champion_md(topic).write_text(
            _render_frontmatter({"count": len(state["champion"])}, body or "（尚无）"),
            encoding="utf-8",
        )

        # dead_ends.md
        body = "\n".join(f"- {h}" for h in state["rejected"])
        self._dead_ends_md(topic).write_text(
            _render_frontmatter({"count": len(state["rejected"])}, body or "（尚无）"),
            encoding="utf-8",
        )

        # knowledge/patterns.md — winning patterns (champion) + dead ends + exhausted axes
        self._patterns_md(topic).parent.mkdir(parents=True, exist_ok=True)
        lines = ["# 知识 / 模式", ""]
        lines.append("## 已验证结论（winning patterns）")
        lines.extend(f"- {c['statement']}" for c in state["champion"]) if state[
            "champion"
        ] else lines.append("（尚无）")
        lines.append("\n## 已否定假设（dead ends）")
        lines.extend(f"- {h}" for h in state["rejected"]) if state["rejected"] else lines.append(
            "（尚无）"
        )
        lines.append("\n## 已耗尽方向（exhausted axes）")
        lines.append(f"- 连续无进展轮次：{state['consecutive_no_gain']}")
        lines.append(f"- 已转向次数：{state.get('adaptations', 0)}")
        self._patterns_md(topic).write_text("\n".join(lines), encoding="utf-8")

        # runtime-only resume state
        self._run_json(topic).write_text(
            json.dumps(
                {
                    "cycles": state["cycles"],
                    "consecutive_no_gain": state["consecutive_no_gain"],
                    "adaptations": state.get("adaptations", 0),
                    "pending": state.get("pending", []),
                    "mode": state.get("mode", "execute"),
                    "started_at": state["started_at"],
                    "updated_at": state["updated_at"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _save_cycle_result(self, topic: str, result: dict[str, Any]) -> None:
        """Write one cycle's evidence to ``results/cycle-NNN.md`` (append-only)."""
        path = self._result_md(topic, result["cycle"])
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# 第 {result['cycle']} 轮", ""]
        lines.append(f"- 验证结论（KEEP）：{result.get('verified') or '（无）'}")
        lines.append(f"- 证伪假设（DISCARD）：{result.get('falsified') or '（无）'}")
        lines.append(f"- 预测：{result.get('predictions') or '（无）'}")
        lines.append(f"- 假设：{result.get('hypotheses') or '（无）'}")
        verifs = result.get("verifications") or []
        if verifs:
            lines.append("\n## 核验计数（Supports/Refutes/Orthogonal）\n")
            for v in verifs:
                lines.append(
                    f"- {v.get('statement')}：supports={v.get('supports')}, "
                    f"refutes={v.get('refutes')}, orthogonal={v.get('orthogonal')} "
                    f"→ {v.get('status')}"
                    + (f"，实算={v.get('computed')}" if v.get("computed") else "")
                )
        rep = (result.get("report") or "").strip()
        if rep:
            lines.append("\n## 本轮报告\n")
            lines.append(rep)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _save_cycle_trace(self, topic: str, cycle_no: int, rs: ResearchState | None) -> None:
        """Serialize one cycle's full ResearchState to ``traces/cycle-NNN.json``.

        Captures every intermediate product the workflow nodes left in the
        shared state (candidates, entities, gaps, hypotheses, scores, …) so a
        single cycle is fully auditable after the fact. Best-effort: a
        serialization or write failure must never break the loop.
        """
        if rs is None:
            return
        path = self._topic_dir(topic) / "traces" / f"cycle-{cycle_no:03d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            payload = rs.model_dump()
        except Exception:  # noqa: BLE001 — fall back to a minimal dict
            payload = {"task": getattr(rs, "task", None)}
        try:
            path.write_text(
                json.dumps(payload, ensure_ascii=False, default=str, indent=2),
                encoding="utf-8",
            )
        except OSError:
            logger.warning("[director] trace write failed for cycle %d", cycle_no)

    # ── T7: per-role cross-cycle memory (director is the single writer) ─────────

    def _save_role_memories(self, topic: str, cycle_no: int, rs: ResearchState | None) -> None:
        """Append this cycle's critic/verifier history to ``knowledge/role-*.md``.

        AutoScientists gives every agent its own ``memory/MEMORY.md`` so each
        session remembers its own judgments. Here the two role-bearing nodes —
        the critic and the verifier — keep their own cross-cycle history, and a
        later cycle injects the recent tail into the node prompt (see
        :func:`drbrain.loop.workflow._read_role_history`). The director is the
        single writer; the nodes only read. Append-only, one judgment per line.
        """
        if rs is None:
            return
        critic_lines = []
        for h in rs.hypotheses:
            verdict = "DISCARD" if h.status == "discarded" else "KEEP"
            critic_lines.append(
                f"- [cycle {cycle_no}] {h.statement}（score={h.score:.2f}, verdict={verdict}）"
            )
        if critic_lines:
            path = self._role_memory_md(topic, "critic")
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write("\n".join(critic_lines) + "\n")
            except OSError:
                logger.warning("[director] role-critic memory write failed")

        verifier_lines = []
        for v in rs.verifications:
            verifier_lines.append(
                f"- [cycle {cycle_no}] {v.statement}：supports={v.supports}, "
                f"refutes={v.refutes}, orthogonal={v.orthogonal} → {v.status}"
            )
        if verifier_lines:
            path = self._role_memory_md(topic, "verifier")
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write("\n".join(verifier_lines) + "\n")
            except OSError:
                logger.warning("[director] role-verifier memory write failed")

    # ── T8: proposal board + critic reviews (Discussion-Before-Queuing) ─────────

    def _save_discussion(self, topic: str, cycle_no: int, rs: ResearchState | None) -> None:
        """Persist this cycle's proposals and their critic reviews to ``knowledge/``.

        Discussion-Before-Queuing in file-workspace form (AutoScientists' message
        board, without the Node service): every hypothesis the proposing role
        (identify_gaps) raised this cycle is a ``[PROPOSAL]`` post appended to
        ``knowledge/proposals.md``; the critic — a *separate role* from the
        proposer (T1 role differentiation, driven by ``CRITIC_SYSTEM_PROMPT``),
        i.e. the non-author reviewer — appends one review per proposal to
        ``knowledge/reviews.md``, linked to its proposal by the verbatim
        statement. The critic's counter-argument is structurally encoded in
        ``score`` + ``verdict`` (its JSON contract carries no free-text reason);
        DISCARDed proposals never enter verification (T5 gate — unchanged, this
        only persists the discussion). Append-only, one line per object; the
        director is the single writer, the nodes only read.
        """
        if rs is None:
            return
        knowledge = self._topic_dir(topic) / "knowledge"
        knowledge.mkdir(parents=True, exist_ok=True)
        proposal_lines = [f"- [cycle {cycle_no}] {h.statement}" for h in rs.hypotheses]
        review_lines = []
        for h in rs.hypotheses:
            if h.status == "proposed":
                # discussion_pending：未获非作者评论 → 没有 review 可落盘（它还没
                # 被 critic 评审过，等下一轮补评论）。写一行 KEEP 会误导成"已评审"。
                continue
            verdict = "DISCARD" if h.status == "discarded" else "KEEP"
            review_lines.append(
                f"- [cycle {cycle_no}] {h.statement}（reviewer=critic, "
                f"score={h.score:.2f}, verdict={verdict}）"
            )
        try:
            if proposal_lines:
                with open(knowledge / "proposals.md", "a", encoding="utf-8") as f:
                    f.write("\n".join(proposal_lines) + "\n")
            if review_lines:
                with open(knowledge / "reviews.md", "a", encoding="utf-8") as f:
                    f.write("\n".join(review_lines) + "\n")
        except OSError:
            logger.warning("[director] discussion write failed")

    def _save_roster(self, topic: str) -> None:
        """Write the fixed team roster (AutoScientists ``teams/roster.md``).

        The discussion layer's agent identities — the proposer (``analyst``)
        and the ``n_critics`` independent reviewers (``critic-1..N``) — come
        from here; the compute/verifier roles round out the team. In-process,
        so the roster is a fixed team (no cold-start self-organization yet), but
        the file is the same single source of truth agents read.
        """
        path = self._roster_md(topic)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# 团队名单（roster）", ""]
        lines.append("- analyst（identify_gaps：提案者）")
        for i in range(1, self._n_critics + 1):
            lines.append(f"- critic-{i}（critique：独立评审者）")
        lines.append("- compute（compute：实算）")
        lines.append("- verifier（verify：证据核验）")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ── logging (AutoScientists: single canonical JSONL log, orchestrator writes) ─

    def _logs_dir(self, topic: str) -> Path:
        d = self._topic_dir(topic) / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _log_experiment(self, topic: str, result: dict[str, Any]) -> None:
        """Append one line to ``logs/experiments.jsonl`` — the canonical log.

        Single source of truth for per-cycle outcomes (KEEP/DISCARD), written
        ONLY by the director (agents never touch it). This is what the
        stagnation check reads, mirroring AutoScientists' ``experiments.jsonl``.
        """
        entry = {
            "cycle": result["cycle"],
            "topic": topic,
            "outcome": result.get("outcome"),
            "champion_before": result.get("champion_before"),
            "champion_after": result.get("champion_after"),
            "verified": result.get("verified", []),
            "falsified": result.get("falsified", []),
            "predictions": result.get("predictions", []),
            "hypotheses": result.get("hypotheses", []),
            "verifications": result.get("verifications", []),
            "started_at": result.get("started_at"),
            "completed_at": result.get("completed_at"),
            "duration_seconds": result.get("duration_seconds"),
        }
        with open(self._logs_dir(topic) / "experiments.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _log_session(
        self, topic: str, state: dict[str, Any], status: str, started_at: float
    ) -> None:
        """Append one line to ``logs/sessions.jsonl`` — one per director run."""
        entry = {
            "session_id": str(int(started_at)),
            "topic": topic,
            "started_at": started_at,
            "ended_at": time.time(),
            "duration_seconds": round(time.time() - started_at, 2),
            "cycles_run": state["cycles"],
            "champion_count": len(state["champion"]),
            "rejected_count": len(state["rejected"]),
            "adaptations": state.get("adaptations", 0),
            "status": status,
        }
        with open(self._logs_dir(topic) / "sessions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _append_janitor_log(self, topic: str, entry: dict[str, Any]) -> None:
        """Append one line to ``logs/janitor.jsonl`` (stale-job audit trail)."""
        with open(self._logs_dir(topic) / "janitor.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── T-janitor: stale async-job re-claim (simplified monitor) ───────────────

    def _janitor_scan(self, topic: str, state: dict[str, Any]) -> None:
        """Flag async jobs that claimed compute but never produced a result.

        Mirrors AutoScientists' monitor re-claim: a job whose metadata says it
        started more than ``JANITOR_STALE_SECONDS`` ago but whose log still
        carries no numeric output is *stale* — the claim is released and its
        evidence is not trusted. Simplified: no monitor process; the director
        scans the ``jobs/`` directory once per cycle.

        When a stale job's hypothesis was recorded as verified anywhere in the
        run (a contradiction — verified without real values), a warning is
        logged so the next cycle does not trust that job's evidence.
        """
        jobs_dir = self._topic_dir(topic) / "jobs"
        if not jobs_dir.is_dir():
            return
        now = time.time()
        stale: list[tuple[str, float]] = []
        for meta_path in sorted(jobs_dir.glob("*.json")):
            job_id = meta_path.stem
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(meta, dict):
                continue
            if _job_log_has_number(str(jobs_dir), job_id):
                continue  # log carries a numeric result → completed, not stale
            try:
                started_raw = meta.get("started_at")
                started = float(started_raw) if isinstance(started_raw, (int, float)) else 0.0
            except (TypeError, ValueError):
                started = meta_path.stat().st_mtime  # fall back to meta file mtime
            if now - started > JANITOR_STALE_SECONDS:
                stale.append((job_id, started))
        if not stale:
            return
        # Map stale jobs back to hypotheses that were ever recorded as verified
        # (verifications are accumulated per cycle in state["results"]).
        verified_by_job: dict[str, list[str]] = {}
        for res in state.get("results", []):
            for v in res.get("verifications", []):
                if isinstance(v, dict) and v.get("status") == "verified" and v.get("job_id"):
                    verified_by_job.setdefault(str(v["job_id"]), []).append(
                        str(v.get("statement", ""))
                    )
        for job_id, started in stale:
            logger.warning(
                "[janitor] job %s stale (started %.0fs ago, no result)", job_id, now - started
            )
            self._append_janitor_log(
                topic,
                {
                    "event": f"[janitor] job {job_id} stale",
                    "job_id": job_id,
                    "started_at": started,
                    "stale_seconds": round(now - started, 1),
                    "cycle": state["cycles"],
                },
            )
            for statement in verified_by_job.get(job_id, []):
                logger.warning(
                    "[janitor] job %s has no real result but its hypothesis %r was "
                    "recorded verified — do not trust this job's evidence next cycle",
                    job_id,
                    statement,
                )
                self._append_janitor_log(
                    topic,
                    {
                        "event": "[janitor] stale job trusted",
                        "job_id": job_id,
                        "statement": statement,
                        "cycle": state["cycles"],
                    },
                )

    # ── prior-context (shared memory across cycles) ───────────────────────────

    @staticmethod
    def _build_prior_context(state: dict[str, Any]) -> str:
        parts: list[str] = []
        if state.get("champion"):
            parts.append("已确认结论：" + "；".join(c["statement"] for c in state["champion"]))
        if state.get("rejected"):
            parts.append("已否定假设（不要重复提出）：" + "；".join(state["rejected"]))
        if state.get("pending"):
            parts.append(
                "上轮提出但未讨论完的假设（本轮 critic 需重新独立评审）："
                + "；".join(state["pending"])
            )
        return "\n".join(parts)

    @staticmethod
    def _critic_vetoes_direction(rs: ResearchState | None) -> bool:
        """T8: does the critic independently veto the current hypothesis direction?

        Endorsement gate for structural change (Phase 4 adapt). The critic — a
        separate role from the proposer (T1, ``CRITIC_SYSTEM_PROMPT``) — vetoes
        the direction when its latest review round shows the direction is below
        the bar: mean ``score < CRITIC_ENDORSEMENT_SCORE`` or every hypothesis
        of the round was DISCARDed. An absent critic opinion (no hypotheses / no
        state) is NOT a veto — a structural change needs an explicit independent
        objection, so the loop keeps cycling instead of pivoting.
        """
        if rs is None or not rs.hypotheses:
            return False
        mean_score = sum(h.score for h in rs.hypotheses) / len(rs.hypotheses)
        all_discarded = all(h.status == "discarded" for h in rs.hypotheses)
        return all_discarded or mean_score < CRITIC_ENDORSEMENT_SCORE

    # ── one cycle ─────────────────────────────────────────────────────────────

    async def _run_cycle(
        self,
        topic: str,
        prior_context: str,
        prior_champion: list[str] | None = None,
        prior_rejected: list[str] | None = None,
    ) -> tuple[str, ResearchState | None]:
        wf = ResearchLoopWorkflow(
            cfg=self._cfg,
            db=self._db,
            graph=self._graph,
            plugins_dir=self._plugins_dir,
            mcp_servers=self._mcp_servers,
            n_critics=self._n_critics,
        )
        # T6: pass structured prior champion/dead-ends so the proposal gate can
        # reject duplicates in code (not just via prompt text).
        # T7: pass the knowledge/ dir so the critic/verifier nodes can inject
        # their own cross-cycle history into the prompt.
        handler = wf.run(
            task=topic,
            prior_context=prior_context,
            prior_champion=prior_champion or [],
            prior_rejected=prior_rejected or [],
            role_memory_dir=str(self._topic_dir(topic) / "knowledge"),
        )
        report = await handler
        rs = await handler.ctx.store.get("research_state", default=None)
        return report, rs

    def _absorb(
        self, state: dict[str, Any], report: str, rs: ResearchState | None
    ) -> dict[str, Any]:
        """Classify one cycle's outcome into champion vs dead-ends (KEEP/DISCARD).

        AutoScientists three-way: ``verified`` → champion (KEEP), ``falsified``
        → dead-ends (DISCARD). Hypotheses that were neither verified nor
        falsified stay *unresolved* (not a dead end — the evidence just wasn't
        conclusive), so the next cycle can re-attempt them with prior context.

        T3: when the verifier produced structured ``Verification`` records
        (Supports/Refutes/Orthogonal counts), the three-way classification is
        re-derived here from those counts — the same code rule that the verify
        node used — instead of trusting whatever strings landed in
        ``rs.verified`` / ``rs.falsified``. Legacy states without verification
        records keep the list-based path.
        """
        verified = list(rs.verified) if rs else []
        falsified = list(rs.falsified) if rs else []
        predictions = list(rs.predictions) if rs else []
        hypotheses = [h.statement for h in (rs.hypotheses if rs else [])]
        verifications = list(rs.verifications) if rs else []

        # T3: count-driven classification is the source of truth when present.
        if verifications:
            verified = [v.statement for v in verifications if v.status == "verified"]
            falsified = [v.statement for v in verifications if v.status == "falsified"]
            predictions = [v.statement for v in verifications if v.status == "prediction"]

        champion_statements = {c["statement"] for c in state["champion"]}
        rejected = set(state["rejected"])

        new_champion = [v for v in verified if v not in champion_statements]
        new_rejected = [f for f in falsified if f not in rejected and f not in champion_statements]

        cycle_no = state["cycles"] + 1
        for v in new_champion:
            state["champion"].append({"statement": v, "cycle": cycle_no, "confidence": 1.0})
        state["rejected"].extend(new_rejected)
        state["rejected"] = list(dict.fromkeys(state["rejected"]))  # dedupe, keep order

        cycle_result = {
            "cycle": cycle_no,
            "verified": verified,
            "falsified": falsified,
            "predictions": predictions,
            "hypotheses": hypotheses,
            "verifications": [v.model_dump() for v in verifications],
            "report": report,
        }
        state["results"].append(cycle_result)
        state["cycles"] = cycle_no
        state["consecutive_no_gain"] = 0 if new_champion else state["consecutive_no_gain"] + 1
        return cycle_result

    # ── the continuous loop ───────────────────────────────────────────────────

    async def run(
        self,
        topic: str,
        *,
        max_cycles: int = 10,
        stagnation_cycles: int = 3,
        max_adaptations: int = 2,
    ) -> dict[str, Any]:
        """Run research cycles until stagnation or ``max_cycles``; return the state.

        On stagnation, AutoScientists **Phase 4 (adapt)** fires: the exhausted
        direction is recorded as a dead end and the no-gain counter resets so
        the loop *pivots* and keeps going — only after ``max_adaptations``
        pivots does it stop.
        """
        state = self._load_state(topic)
        session_started = time.time()
        # Agent 自写的后台作业（run_python mode=async）落进本课题工作区，check_job
        # 从同一目录轮询——长 DFT/计算因此随课题一起沉淀、可审计、可续跑。
        os.environ["DRBRAIN_RUN_DIR"] = str(self._topic_dir(topic) / "jobs")
        # Team roster (AutoScientists teams/roster.md) — written once per run so
        # the discussion layer's agent identities are visible & auditable.
        self._save_roster(topic)
        logger.info(
            "[director] resume: topic=%r cycles=%d champion=%d rejected=%d",
            topic,
            state["cycles"],
            len(state["champion"]),
            len(state["rejected"]),
        )

        stop_status = "max_cycles"
        while state["cycles"] < max_cycles:
            prior = self._build_prior_context(state)
            cycle_started = time.time()
            report, rs = await self._run_cycle(
                topic,
                prior,
                prior_champion=[c["statement"] for c in state["champion"]],
                prior_rejected=list(state["rejected"]),
            )
            champion_before = len(state["champion"])
            cycle_result = self._absorb(state, report, rs)
            # Mode Selector: hypotheses the critic left undiscussed
            # (status == "proposed" == discussion_pending) are carried into the
            # next cycle's prior context so a later critic re-reviews them
            # instead of dropping them silently (AutoScientists keeps pending
            # proposals on the board until a non-author comment lands).
            pending = [h.statement for h in (rs.hypotheses if rs else []) if h.status == "proposed"]
            state["pending"] = pending
            state["mode"] = "discussion" if pending else "execute"
            self._save_cycle_trace(topic, state["cycles"], rs)
            champion_after = len(state["champion"])
            cycle_result.update(
                {
                    "outcome": "KEEP" if champion_after > champion_before else "NO_GAIN",
                    "champion_before": champion_before,
                    "champion_after": champion_after,
                    "started_at": cycle_started,
                    "completed_at": time.time(),
                    "duration_seconds": round(time.time() - cycle_started, 2),
                }
            )
            # single-writer: persist semantic objects + per-cycle evidence + canonical log
            self._save_state(topic, state)
            self._save_cycle_result(topic, cycle_result)
            self._log_experiment(topic, cycle_result)
            # T7: append this cycle's critic/verifier history to per-role memory files.
            self._save_role_memories(topic, state["cycles"], rs)
            # T8: proposal board + critic reviews (Discussion-Before-Queuing, file form).
            self._save_discussion(topic, state["cycles"], rs)
            # T-janitor: flag async jobs that claimed compute but produced no result.
            self._janitor_scan(topic, state)
            logger.info(
                "[director] cycle=%d champion=%d rejected=%d no_gain=%d",
                state["cycles"],
                len(state["champion"]),
                len(state["rejected"]),
                state["consecutive_no_gain"],
            )
            if state["consecutive_no_gain"] >= stagnation_cycles:
                if not self._critic_vetoes_direction(rs):
                    # T8/endorsement: stagnation alone does not make a structural
                    # change — the critic (an independent role from the proposer)
                    # must also veto the current direction. It still endorses →
                    # keep cycling instead of pivoting.
                    logger.info(
                        "[director] no-gain=%d but critic endorses direction — keep cycling",
                        state["consecutive_no_gain"],
                    )
                    continue
                # Phase 4: adapt — record the exhausted direction, then pivot.
                state["adaptations"] = state.get("adaptations", 0) + 1
                state["rejected"].append(
                    f"[stagnation] 连续 {stagnation_cycles} 轮无新结论，当前方向已耗尽"
                    f"（第 {state['adaptations']} 次转向）"
                )
                state["rejected"] = list(dict.fromkeys(state["rejected"]))
                state["consecutive_no_gain"] = 0
                self._save_state(topic, state)
                logger.info("[director] adapt #%d: pivot direction", state["adaptations"])
                if state["adaptations"] >= max_adaptations:
                    stop_status = "max_adaptations"
                    logger.info("[director] max adaptations reached — stop")
                    break

        self._log_session(topic, state, stop_status, session_started)
        return state

    def run_sync(self, topic: str, **kwargs: Any) -> dict[str, Any]:
        """Convenience sync wrapper (``asyncio.run``)."""
        return asyncio.run(self.run(topic, **kwargs))
