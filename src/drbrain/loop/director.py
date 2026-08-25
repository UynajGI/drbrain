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
import hashlib
import json
import os
import re
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from loguru import logger

from drbrain.loop.checkpointing import (
    CheckpointError,
    CheckpointManifest,
    CheckpointRestoreError,
    WorkflowCheckpointService,
)
from drbrain.loop.durable_execution import DurableExecution
from drbrain.loop.events import ResearchState
from drbrain.loop.front_half import DurableFrontHalf
from drbrain.loop.governance import RunGovernance
from drbrain.loop.policy import ToolPolicy
from drbrain.loop.store import LedgerEvent, RunExecutionBlockedError, RunLedger
from drbrain.loop.tool_broker import ToolBroker, redact
from drbrain.loop.transitions import LeaseUnavailableError, TransitionService
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


_MCP_CONTRACT_FIELDS = (
    "id",
    "name",
    "command",
    "url",
    "args",
    "timeout_seconds",
    "trusted",
    "allowed_tools",
    "required_capabilities",
    "side_effect",
    "code_digest",
    "version",
    "max_output_bytes",
    "cost_hint",
    "supports_idempotency",
    "supports_reconcile",
    "supports_cancel",
    "sandbox_profile",
    "approval_policy",
)
_UNORDERED_MCP_FIELDS = frozenset({"allowed_tools", "required_capabilities"})
_IGNORED_PLUGIN_SOURCE_PARTS = frozenset({".git", ".venv", "venv", "__pycache__", "site-packages"})


def _file_sha256(path: Path) -> str | None:
    """Return a source digest without retaining the source or its contents."""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _plugin_source_contract(plugins_dir: str | Path | None) -> dict[str, Any] | None:
    """Fingerprint discoverable plugin sources for safe checkpoint resumption."""
    if not plugins_dir:
        return None
    root = Path(plugins_dir).resolve()
    if not root.is_dir():
        return {"path": str(root), "state": "missing_or_not_directory", "sources": []}
    sources: list[dict[str, str | None]] = []
    for path in sorted(root.rglob("*.py"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root)
        if any(part in _IGNORED_PLUGIN_SOURCE_PARTS for part in relative.parts):
            continue
        sources.append({"path": relative.as_posix(), "sha256": _file_sha256(path)})
    return {"path": str(root), "state": "directory", "sources": sources}


def _mcp_contract(server: dict[str, Any]) -> dict[str, Any]:
    """Project every execution/policy-relevant MCP field without secrets."""
    contract: dict[str, Any] = {}
    for field in _MCP_CONTRACT_FIELDS:
        value = server.get(field)
        if value is None:
            continue
        if isinstance(value, str | int | float | bool):
            contract[field] = value
        elif isinstance(value, (list, tuple, set, frozenset)) and all(
            isinstance(item, str | int | float | bool) for item in value
        ):
            items = list(value)
            if field in _UNORDERED_MCP_FIELDS:
                items.sort(key=lambda item: json.dumps(item, sort_keys=True))
            contract[field] = items
    env = server.get("env")
    if isinstance(env, dict):
        contract["env_keys"] = sorted(str(key) for key in env)
    return contract


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
        lease_seconds: float = 900.0,
        tool_policy: ToolPolicy | None = None,
        noise_band: float = 0.0,
        required_repeats: int = 2,
    ) -> None:
        self._cfg = cfg
        self._db = db
        self._graph = graph
        self._plugins_dir = plugins_dir
        self._mcp_servers = mcp_servers
        self._run_dir = Path(run_dir)
        self._n_critics = max(1, int(n_critics))
        self._lease_seconds = max(1.0, float(lease_seconds))
        self._tool_policy = tool_policy
        self._noise_band = max(0.0, float(noise_band))
        self._required_repeats = max(1, int(required_repeats))
        self._worker_id = uuid.uuid4().hex
        self._active_checkpoint: WorkflowCheckpointService | None = None
        self._active_checkpoint_id: str | None = None
        self._rag_generation: str | None = None

    def _ledger(self) -> RunLedger:
        """Return this director's isolated, SQLite-only run ledger."""
        return RunLedger(self._run_dir / "ledger.sqlite3")

    def _checkpoint_manifest(self) -> CheckpointManifest:
        """Build a secret-free contract for safe Context restoration.

        The RAG generation is frozen when the durable run is created and shared
        with its workflow, so a later atomic RAG publication cannot redirect an
        in-flight cycle or make a resumed checkpoint describe different data.
        """
        model_manifest: dict[str, Any] = {"config_type": type(self._cfg).__name__}
        for name in ("provider", "model", "model_name", "llm_model", "embedding_model"):
            value = getattr(self._cfg, name, None)
            if isinstance(value, str | int | float | bool) or value is None:
                if value is not None:
                    model_manifest[name] = value
        # The normal CLI passes ``Config`` whose fallback chain lives at
        # ``cfg.llm.models``.  Keep the historical top-level ``models`` shape
        # for lightweight callers, and accept the equivalent dict form used by
        # tests and integrations.  The chain order is part of the execution
        # contract: a resumed hypothesis must not silently switch to a
        # different fallback model.
        if isinstance(self._cfg, dict):
            llm_cfg = self._cfg.get("llm")
            models = llm_cfg.get("models") if isinstance(llm_cfg, dict) else None
            if not isinstance(models, list):
                models = self._cfg.get("models")
        else:
            llm_cfg = getattr(self._cfg, "llm", None)
            models = getattr(llm_cfg, "models", None)
            if not isinstance(models, list):
                models = getattr(self._cfg, "models", None)
        if isinstance(models, list):
            public_models: list[dict[str, Any]] = []
            for item in models:
                fields: dict[str, Any] = {}
                # Preserve model identity and routing without persisting
                # credentials.  These are the stable, non-secret fields used
                # by the project's LLM fallback configuration.
                for name in (
                    "provider",
                    "model",
                    "model_name",
                    "base_url",
                    "api_base",
                    "deployment",
                    "api_version",
                    "temperature",
                    "max_tokens",
                    "max_output_tokens",
                ):
                    value = getattr(item, name, None)
                    if isinstance(item, dict):
                        value = item.get(name)
                    if isinstance(value, str | int | float | bool):
                        fields[name] = value
                if fields:
                    public_models.append(fields)
            if public_models:
                model_manifest["models"] = public_models

        servers: list[dict[str, Any]] = []
        for server in self._mcp_servers or []:
            if not isinstance(server, dict):
                continue
            servers.append(_mcp_contract(server))
        tool_manifest = {
            "plugins_dir": str(Path(self._plugins_dir).resolve()) if self._plugins_dir else None,
            "plugin_source_contract": _plugin_source_contract(self._plugins_dir),
            "mcp_servers": sorted(servers, key=lambda item: json.dumps(item, sort_keys=True)),
            "tool_policy": self._tool_policy.to_manifest()
            if self._tool_policy is not None
            else None,
        }
        return CheckpointManifest(
            workflow_version="research-loop-v1",
            model_manifest=model_manifest,
            tool_manifest=tool_manifest,
            rag_generation=self._rag_generation,
        )

    # ── workspace paths ───────────────────────────────────────────────────────

    def _topic_dir(self, topic: str) -> Path:
        base = _slug(topic)
        d = self._run_dir / base
        # Distinct topics that sanitize to the same slug (e.g. "a/b" vs "a b")
        # must not share a workspace — disambiguate with a short hash.
        run_json = d / "run.json"
        if d.exists() and run_json.exists():
            try:
                other = json.loads(run_json.read_text(encoding="utf-8")).get("topic")
            except (json.JSONDecodeError, OSError):
                other = None
            if other is not None and other != topic:
                d = self._run_dir / f"{base}-{hashlib.sha1(topic.encode()).hexdigest()[:8]}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _run_json(self, topic: str) -> Path:
        return self._topic_dir(topic) / "run.json"

    def _has_legacy_projection(self, topic: str) -> bool:
        """Whether a topic predates the ledger and has workspace state to import."""
        return any(
            path.exists()
            for path in (
                self._run_json(topic),
                self._champion_md(topic),
                self._dead_ends_md(topic),
                self._patterns_md(topic),
            )
        )

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

    def _project_cycle(
        self,
        topic: str,
        state: dict[str, Any],
        cycle_result: dict[str, Any],
        research_state: ResearchState | None,
    ) -> None:
        """Write one committed cycle to the existing compatibility workspace."""
        self._save_cycle_trace(topic, state["cycles"], research_state)
        self._save_state(topic, state)
        self._save_cycle_result(topic, cycle_result)
        self._log_experiment(topic, cycle_result)
        self._save_role_memories(topic, state["cycles"], research_state)
        self._save_discussion(topic, state["cycles"], research_state)
        self._janitor_scan(topic, state)

    def _project_pending_ledger_events(self, topic: str, ledger: RunLedger, run_id: str) -> None:
        """Replay ledger commits that reached SQLite before their file projection."""
        for event in ledger.pending_projection_events(run_id):
            self._project_ledger_event(topic, event)
            ledger.mark_projected(run_id, event.event_seq)

    def _project_ledger_event(self, topic: str, event: LedgerEvent) -> None:
        """Materialize one replayable ledger event into backward-compatible files."""
        state = event.payload.get("state")
        if not isinstance(state, dict):
            raise ValueError(f"ledger event {event.event_seq} has no state snapshot")

        if event.event_type == "state_snapshot":
            self._save_state(topic, state)
            return

        cycle_result = event.payload.get("cycle_result")
        if not isinstance(cycle_result, dict):
            raise ValueError(f"ledger event {event.event_seq} has no cycle result")
        raw_research_state = event.payload.get("research_state")
        research_state = (
            ResearchState.model_validate(raw_research_state)
            if raw_research_state is not None
            else None
        )
        self._project_cycle(topic, state, cycle_result, research_state)

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

    @staticmethod
    def _append_missing_lines(path: Path, lines: list[str]) -> None:
        """Append only absent lines so replaying a committed cycle is idempotent."""
        existing = set(path.read_text(encoding="utf-8").splitlines()) if path.exists() else set()
        missing = list(dict.fromkeys(line for line in lines if line not in existing))
        if not missing:
            return
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(missing) + "\n")

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
                self._append_missing_lines(path, critic_lines)
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
                self._append_missing_lines(path, verifier_lines)
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
                self._append_missing_lines(knowledge / "proposals.md", proposal_lines)
            if review_lines:
                self._append_missing_lines(knowledge / "reviews.md", review_lines)
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
        path = self._logs_dir(topic) / "experiments.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if existing.get("cycle") == entry["cycle"]:
                    return
        with open(path, "a", encoding="utf-8") as f:
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
        """Append one distinct line to ``logs/janitor.jsonl``.

        A committed cycle can be replayed after a process exits between its
        workspace projection and ``mark_projected``.  ``stale_seconds`` varies
        on replay, so deduplicate on the stable identity of a janitor finding
        rather than on the complete serialized JSON object.
        """
        path = self._logs_dir(topic) / "janitor.jsonl"
        identity_fields = ("event", "job_id", "statement", "cycle", "started_at")
        identity = tuple(entry.get(field) for field in identity_fields)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    isinstance(existing, dict)
                    and tuple(existing.get(field) for field in identity_fields) == identity
                ):
                    return
        with open(path, "a", encoding="utf-8") as f:
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
        checkpoint = self._active_checkpoint
        tool_broker = None
        evidence_recorder = None
        durable_front_half = None
        durable_execution = None
        governance = None
        if checkpoint is not None:
            governance = RunGovernance(checkpoint.ledger)
            durable_front_half = DurableFrontHalf(
                TransitionService(checkpoint.ledger), checkpoint.run_id
            )
            await asyncio.to_thread(durable_front_half.ensure_node_contracts)

            def evidence_recorder(bundle: Mapping[str, Any]) -> None:
                checkpoint.ledger.record_evidence_bundle(
                    run_id=checkpoint.run_id,
                    step_id=checkpoint.step_id,
                    attempt_id=checkpoint.attempt_id,
                    worker_id=checkpoint.worker_id,
                    bundle=redact(dict(bundle)),
                )

        if self._tool_policy is not None:
            if checkpoint is None:
                raise RuntimeError("durable tool policy requires an active checkpoint attempt")
            tool_broker = ToolBroker(
                ledger=checkpoint.ledger,
                run_id=checkpoint.run_id,
                step_id=checkpoint.step_id,
                attempt_id=checkpoint.attempt_id,
                worker_id=checkpoint.worker_id,
                lease_seconds=checkpoint.lease_seconds,
                policy=self._tool_policy,
            )
        if checkpoint is not None and tool_broker is not None:
            durable_execution = DurableExecution(
                TransitionService(checkpoint.ledger),
                checkpoint.run_id,
                step_id=checkpoint.step_id,
                attempt_id=checkpoint.attempt_id,
                worker_id=checkpoint.worker_id,
                noise_band=self._noise_band,
                required_repeats=self._required_repeats,
            )
            await asyncio.to_thread(durable_execution.ensure_node_contracts)
        wf = ResearchLoopWorkflow(
            cfg=self._cfg,
            db=self._db,
            graph=self._graph,
            plugins_dir=self._plugins_dir,
            mcp_servers=self._mcp_servers,
            n_critics=self._n_critics,
            # Per-run job dir (not process-global env): concurrent directors on
            # different topics keep their own compute evidence (review P1).
            jobs_dir=str(self._topic_dir(topic) / "jobs"),
            tool_broker=tool_broker,
            tool_policy=self._tool_policy,
            rag_generation=self._rag_generation,
            evidence_recorder=evidence_recorder,
            durable_front_half=durable_front_half,
            durable_execution=durable_execution,
            budget_reserver=(
                (lambda amounts: governance.reserve(checkpoint.run_id, amounts))
                if governance is not None and checkpoint is not None
                else None
            ),
        )
        if checkpoint is not None and checkpoint.checkpoint is not None:
            # Restoring Context replays only the scheduler's pending events; it
            # does not invent a new StartEvent or rerun successful prior nodes.
            handler = wf.run(ctx=checkpoint.restore_context(wf))
        else:
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
        if checkpoint is not None:
            async for event in handler.stream_events(expose_internal=True):
                try:
                    captured = checkpoint.capture_if_safe(ctx=handler.ctx, workflow=wf, event=event)
                except Exception:
                    # A failed durable write must not let the in-memory workflow
                    # keep issuing tools after the director has lost its recovery
                    # boundary. The handler performs a graceful cancellation.
                    await handler.cancel_run()
                    raise
                if captured is not None:
                    self._active_checkpoint_id = captured.checkpoint_id
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
            # Reconcile repeated records: one statement may carry conflicting
            # statuses across records — first-wins per statement keeps a claim
            # out of both champion and dead-ends (review P1).
            verdicts: dict[str, str] = {}
            for v in verifications:
                verdicts.setdefault(v.statement, v.status)
            verified = [s for s, st in verdicts.items() if st == "verified"]
            falsified = [s for s, st in verdicts.items() if st == "falsified"]
            predictions = [s for s, st in verdicts.items() if st == "prediction"]

        champion_statements = {c["statement"] for c in state["champion"]}
        rejected = set(state["rejected"])

        new_champion = list(dict.fromkeys(v for v in verified if v not in champion_statements))
        new_rejected = [f for f in falsified if f not in rejected and f not in champion_statements]

        cycle_no = state["cycles"] + 1
        for stmt in new_champion:
            state["champion"].append({"statement": stmt, "cycle": cycle_no, "confidence": 1.0})
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
        budget: Mapping[str, int | float] | None = None,
    ) -> dict[str, Any]:
        """Run research cycles until stagnation or ``max_cycles``; return the state.

        On stagnation, AutoScientists **Phase 4 (adapt)** fires: the exhausted
        direction is recorded as a dead end and the no-gain counter resets so
        the loop *pivots* and keeps going — only after ``max_adaptations``
        pivots does it stop.
        """
        ledger = self._ledger()
        try:
            from drbrain.rag.indexer import capture_index_generation

            captured_generation = capture_index_generation(self._cfg)
        except Exception as exc:  # noqa: BLE001 - RAG is optional for existing loop users
            logger.warning(
                "[director] cannot capture RAG generation; disabling RAG evidence: %s", exc
            )
            captured_generation = None
        existing_run = ledger.get_run(topic)
        if existing_run is not None:
            # A prior process may have committed a cycle before its Markdown / JSONL
            # projection finished. Replay that committed fact before deriving the
            # current in-memory state from the compatibility workspace.
            self._project_pending_ledger_events(topic, ledger, existing_run.run_id)

        legacy_projection = existing_run is None and self._has_legacy_projection(topic)
        state = self._load_state(topic)
        config = {"n_critics": self._n_critics, "rag_generation": captured_generation}
        effective_budget: dict[str, int | float] = {
            "max_cycles": max_cycles,
            "stagnation_cycles": stagnation_cycles,
            "max_adaptations": max_adaptations,
        }
        if budget is not None:
            budget_aliases = {
                "attempts": "max_attempts",
                "tool_calls": "max_tool_calls",
                "rag_calls": "max_rag_calls",
                "model_calls": "max_model_calls",
                "wall_seconds": "max_wall_seconds",
            }
            supported_budget_keys = set(budget_aliases) | set(budget_aliases.values())
            unknown_budget_keys = sorted(
                str(name) for name in budget if str(name) not in supported_budget_keys
            )
            if unknown_budget_keys:
                raise ValueError("unknown budget limit(s): " + ", ".join(unknown_budget_keys))
            invalid_budget_values = {
                str(name): value
                for name, value in budget.items()
                if not isinstance(value, int | float) or isinstance(value, bool) or value < 0
            }
            if invalid_budget_values:
                name, value = next(iter(invalid_budget_values.items()))
                raise ValueError(f"invalid budget limit for {name!r}: {value!r}")
            provided_names: dict[str, set[str]] = {}
            for name in budget:
                provided = str(name)
                canonical = budget_aliases.get(provided, provided)
                provided_names.setdefault(canonical, set()).add(provided)
            ambiguous_budget_limits = {
                canonical: sorted(names)
                for canonical, names in provided_names.items()
                if len(names) > 1
            }
            if ambiguous_budget_limits:
                canonical, names = next(iter(ambiguous_budget_limits.items()))
                raise ValueError(f"ambiguous budget limit for {canonical!r}: " + ", ".join(names))
            effective_budget.update(
                {budget_aliases.get(str(name), str(name)): value for name, value in budget.items()}
            )
        run = ledger.get_or_create_run(
            topic,
            config=config,
            budget=effective_budget,
            legacy_snapshot=state if legacy_projection else None,
        )
        stored_generation = run.config.get("rag_generation")
        self._rag_generation = (
            str(stored_generation)
            if isinstance(stored_generation, str) and stored_generation
            else captured_generation
        )
        if self._rag_generation:
            attempted_generation = self._rag_generation
            try:
                from drbrain.rag.indexer import retain_index_generation

                retain_index_generation(self._cfg, self._rag_generation, run.run_id)
            except Exception as exc:  # noqa: BLE001 - unavailable retention disables RAG evidence
                logger.warning(
                    "[director] cannot retain RAG generation; disabling RAG evidence: %s", exc
                )
                try:
                    ledger.record_rag_evidence_disabled(
                        run.run_id,
                        generation=attempted_generation,
                    )
                except Exception as audit_exc:  # noqa: BLE001 - preserve optional RAG behavior
                    logger.error("[director] cannot record RAG evidence downgrade: %s", audit_exc)
                self._rag_generation = None
        effective_config = {"n_critics": self._n_critics, "rag_generation": self._rag_generation}
        if existing_run is not None:
            ledger.record_resume(run.run_id, config=effective_config, budget=effective_budget)
        transitions = TransitionService(ledger)
        transitions.reconcile_incomplete_cycles(run.run_id)
        transitions.start_run(run.run_id)
        active_steps = ledger.active_leased_steps(run.run_id)
        if active_steps:
            # Another director is still working. Do not pause its run or start a
            # second cycle: the SQLite lease is the single-writer boundary.
            raise LeaseUnavailableError(
                f"run {run.run_id} is active under another worker: {active_steps[0]}"
            )

        resumed_checkpoint: WorkflowCheckpointService | None = None
        recovery_steps = (
            ledger.recoverable_step_ids(run.run_id) if state["cycles"] < max_cycles else []
        )
        if len(recovery_steps) > 1:
            for stale_step_id in recovery_steps:
                transitions.mark_manual_review(
                    run.run_id,
                    step_id=stale_step_id,
                    reason="multiple interrupted cycles require operator reconciliation",
                )
            transitions.pause_run(run.run_id, reason="checkpoint_manual_review")
            raise CheckpointRestoreError("multiple interrupted cycles require manual review")
        if recovery_steps:
            recovery_step_id = recovery_steps[0]
            inflight_node = ledger.inflight_workflow_step(recovery_step_id)
            if WorkflowCheckpointService.requires_manual_recovery(inflight_node):
                transitions.mark_manual_review(
                    run.run_id,
                    step_id=recovery_step_id,
                    reason=(
                        "interrupted external-side-effect node requires operator reconciliation: "
                        f"{inflight_node}"
                    ),
                )
                transitions.pause_run(run.run_id, reason="checkpoint_manual_review")
                raise CheckpointRestoreError(
                    f"interrupted external-side-effect node requires manual review: {inflight_node}"
                )
            checkpoint = ledger.latest_checkpoint_for_step(recovery_step_id)
            if checkpoint is None:
                transitions.mark_manual_review(
                    run.run_id,
                    step_id=recovery_step_id,
                    reason="interrupted cycle has no JSON checkpoint",
                )
                transitions.pause_run(run.run_id, reason="checkpoint_manual_review")
                raise CheckpointRestoreError("interrupted cycle has no JSON checkpoint")
            checkpoint_generation = checkpoint.manifest.get("rag_generation")
            if isinstance(checkpoint_generation, str) and checkpoint_generation:
                self._rag_generation = checkpoint_generation
            manifest = self._checkpoint_manifest()
            preview = WorkflowCheckpointService(
                ledger=ledger,
                run_id=run.run_id,
                step_id=recovery_step_id,
                attempt_id=checkpoint.attempt_id,
                worker_id=self._worker_id,
                manifest=manifest,
                lease_seconds=self._lease_seconds,
                checkpoint=checkpoint,
            )
            try:
                preview.validate_checkpoint()
            except CheckpointError as exc:
                transitions.mark_manual_review(
                    run.run_id,
                    step_id=recovery_step_id,
                    reason=str(exc),
                )
                transitions.pause_run(run.run_id, reason="checkpoint_manual_review")
                raise
            resumed_attempt_id = transitions.resume_cycle(
                run.run_id,
                step_id=recovery_step_id,
                checkpoint_id=checkpoint.checkpoint_id,
                worker_id=self._worker_id,
                lease_seconds=self._lease_seconds,
            )
            resumed_checkpoint = WorkflowCheckpointService(
                ledger=ledger,
                run_id=run.run_id,
                step_id=recovery_step_id,
                attempt_id=resumed_attempt_id,
                worker_id=self._worker_id,
                manifest=manifest,
                lease_seconds=self._lease_seconds,
                checkpoint=checkpoint,
            )
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
            if resumed_checkpoint is not None:
                step_id = resumed_checkpoint.step_id
                checkpoint_service = resumed_checkpoint
                resumed_checkpoint = None
            else:
                try:
                    ledger.reserve_budget(run.run_id, {"attempts": 1})
                except RunExecutionBlockedError as exc:
                    current = ledger.get_run_by_id(run.run_id)
                    stop_status = (
                        current.status
                        if current is not None and current.status in {"paused", "cancelled"}
                        else "budget_exhausted"
                    )
                    logger.info("[director] stop before a new cycle: %s", exc)
                    break
                try:
                    step_id = transitions.begin_cycle(
                        run.run_id,
                        cycle=state["cycles"] + 1,
                        worker_id=self._worker_id,
                        lease_seconds=self._lease_seconds,
                    )
                except Exception:
                    # Reservation occurs before the durable attempt exists so a
                    # zero budget cannot create a live lease. Compensate only
                    # this setup failure; once begin_cycle succeeds, the cycle
                    # itself is a real counted research attempt.
                    try:
                        ledger.release_budget(run.run_id, {"attempts": 1})
                    except Exception as release_exc:  # noqa: BLE001 - preserve setup cause
                        logger.error(
                            "[director] could not release unstarted attempt: %s", release_exc
                        )
                    raise
                active_attempt_id = ledger.active_attempt_id(step_id)
                if (
                    active_attempt_id is None
                ):  # pragma: no cover - protected by begin_cycle transaction
                    raise RuntimeError(f"cycle {step_id} was created without an active attempt")
                checkpoint_service = WorkflowCheckpointService(
                    ledger=ledger,
                    run_id=run.run_id,
                    step_id=step_id,
                    attempt_id=active_attempt_id,
                    worker_id=self._worker_id,
                    manifest=self._checkpoint_manifest(),
                    lease_seconds=self._lease_seconds,
                )
            self._active_checkpoint = checkpoint_service
            self._active_checkpoint_id = (
                checkpoint_service.checkpoint.checkpoint_id
                if checkpoint_service.checkpoint is not None
                else None
            )
            checkpoint_id: str | None = None
            try:
                report, rs = await self._run_cycle(
                    topic,
                    prior,
                    prior_champion=[c["statement"] for c in state["champion"]],
                    prior_rejected=list(state["rejected"]),
                )
                checkpoint_id = self._active_checkpoint_id
            except RunExecutionBlockedError as exc:
                current = ledger.get_run_by_id(run.run_id)
                if current is not None and current.status == "failed":
                    transitions.fail_cycle(
                        run.run_id,
                        step_id=step_id,
                        error=exc,
                        worker_id=self._worker_id,
                    )
                    stop_status = "budget_exhausted"
                    logger.info("[director] runtime budget exhausted: %s", exc)
                    break
                if current is not None and current.status == "cancelled":
                    stop_status = "cancelled"
                    logger.info("[director] cycle cancelled while executing: %s", exc)
                    break
                if current is not None and current.status == "paused":
                    stop_status = "paused"
                    logger.info("[director] run paused while executing: %s", exc)
                    break
                raise
            except CheckpointError as exc:
                transitions.mark_manual_review(run.run_id, step_id=step_id, reason=str(exc))
                transitions.pause_run(run.run_id, reason="checkpoint_manual_review")
                raise
            except Exception as exc:
                transitions.fail_cycle(
                    run.run_id,
                    step_id=step_id,
                    error=exc,
                    worker_id=self._worker_id,
                )
                transitions.pause_run(run.run_id, reason="cycle_failed")
                raise
            finally:
                self._active_checkpoint = None
                self._active_checkpoint_id = None
            current = ledger.get_run_by_id(run.run_id)
            if current is None:  # pragma: no cover - run identity is durable
                raise RuntimeError(f"durable run {run.run_id!r} disappeared")
            if current.status in {"failed", "cancelled"}:
                if current.status == "failed":
                    transitions.fail_cycle(
                        run.run_id,
                        step_id=step_id,
                        error=RunExecutionBlockedError(
                            "run failed at a governed execution boundary"
                        ),
                        worker_id=self._worker_id,
                    )
                    stop_status = "budget_exhausted"
                elif current.status == "cancelled":
                    stop_status = "cancelled"
                logger.info("[director] do not project terminal run state: %s", current.status)
                break
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
            try:
                event = transitions.complete_cycle(
                    run.run_id,
                    step_id=step_id,
                    cycle_result=cycle_result,
                    state_snapshot=state,
                    research_state=rs.model_dump() if rs is not None else None,
                    checkpoint_id=checkpoint_id,
                    worker_id=self._worker_id,
                )
            except RunExecutionBlockedError:
                current = ledger.get_run_by_id(run.run_id)
                if current is not None and current.status in {"failed", "cancelled", "paused"}:
                    stop_status = (
                        "budget_exhausted" if current.status == "failed" else current.status
                    )
                    logger.info("[director] terminal run won cycle completion: %s", current.status)
                    break
                raise
            # SQLite is the durable source of truth. Only after its transaction
            # commits do we update the pre-existing file workspace.
            self._project_cycle(topic, state, cycle_result, rs)
            ledger.mark_projected(run.run_id, event.event_seq)
            current = ledger.get_run_by_id(run.run_id)
            if current is not None and current.status == "paused":
                stop_status = "paused"
                logger.info("[director] paused after committing active cycle")
                break
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
                event = transitions.record_state_snapshot(
                    run.run_id, reason="stagnation_adaptation", state_snapshot=state
                )
                self._save_state(topic, state)
                ledger.mark_projected(run.run_id, event.event_seq)
                logger.info("[director] adapt #%d: pivot direction", state["adaptations"])
                if state["adaptations"] >= max_adaptations:
                    stop_status = "max_adaptations"
                    logger.info("[director] max adaptations reached — stop")
                    break

        current_run = ledger.get_run_by_id(run.run_id)
        if current_run is not None and current_run.status == "running":
            transitions.pause_run(run.run_id, reason=stop_status)
        self._log_session(topic, state, stop_status, session_started)
        return state

    def run_sync(self, topic: str, **kwargs: Any) -> dict[str, Any]:
        """Convenience sync wrapper (``asyncio.run``)."""
        return asyncio.run(self.run(topic, **kwargs))
