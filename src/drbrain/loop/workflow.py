"""Research loop orchestration — deterministic pipeline + AutoScientists semantics.

The third layer of the three-in-one architecture. A LlamaIndex ``Workflow``
runs a **deterministic 12-node pipeline** (task → retrieve → … → report); the
AutoScientists loop semantics (hypothesis proposal, critique-before-compute,
shared success/failure) are folded into specific nodes, not free-form chat.

P0 is the skeleton: every node advances the shared :class:`ResearchState` and
emits the next typed event. The conditional loop (retrieve-again on
insufficient candidates) is wired and bounded by ``MAX_RETRIEVE_ATTEMPTS``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from llama_index.core.workflow import (
    Context,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from loguru import logger

from drbrain.loop.events import (
    Critiqued,
    Extracted,
    Filtered,
    Fused,
    GapsIdentified,
    Hypothesis,
    Normalized,
    Parsed,
    ResearchState,
    RetrieveAgain,
    Retrieved,
    Settled,
    TaskPlanned,
    Verification,
    Verified,
)

_STATE_KEY = "research_state"
MAX_RETRIEVE_ATTEMPTS = 3

# ── T2 / T3 / T5 thresholds (code-level gates, domain-agnostic) ──────────────
VERIFY_LOW_SCORE = 0.5  # T2: below this a hypothesis needs stronger evidence to be verified
STRONG_SUPPORTS = 2  # T2: low-score verified bar (supports >= 2 and refutes == 0)
CRITIQUE_DISCARD_SCORE = 0.4  # T5: at/below this the critic DISCARDs (never enters verify)
FALSIFY_REFUTES = 2  # T3: refutes >= 2 and supports == 0 → falsified

# T4: tool names that mark the environment as having compute capability.
_COMPUTE_TOOL_NAMES = ("run_python", "check_job")


def _normalize_statement(s: str) -> str:
    """Lowercase + strip punctuation/whitespace for duplicate detection (T6)."""
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", (s or "").lower())


def _parse_prior_context(text: str) -> list[str]:
    """Best-effort parse of the director's ``prior_context`` text blob (T6).

    ``_build_prior_context`` renders ``已确认结论：A；B`` / ``已否定假设（不要重复提
    出）：X；Y`` — split on separators and strip the section labels. Only used as
    a fallback when no structured prior lists were passed to the workflow.
    """
    out: list[str] = []
    for chunk in re.split(r"[；;\n]", text or ""):
        chunk = chunk.strip()
        if not chunk or re.search(r"已确认结论|已否定假设|不要重复", chunk):
            continue
        if "：" in chunk or ":" in chunk:
            chunk = re.split(r"[:：]", chunk, maxsplit=1)[-1].strip()
        if chunk:
            out.append(chunk)
    return out


def _is_duplicate_proposal(statement: str, prior_statements: list[str]) -> bool:
    """T6 dedup gate: does ``statement`` repeat a prior champion/dead-end?

    Exact match after normalization, or containment either way (guarded by a
    minimum length so short strings don't false-positive).
    """
    norm = _normalize_statement(statement)
    if not norm:
        return False
    for prior in prior_statements:
        pn = _normalize_statement(prior)
        if not pn:
            continue
        if norm == pn:
            return True
        if len(norm) >= 8 and len(pn) >= 8 and (norm in pn or pn in norm):
            return True
    return False


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _job_log_has_number(run_dir: str | None, job_id: str) -> bool:
    """T4 evidence gate: an async job's on-disk artifacts prove a real computation.

    The agent's ``computed`` / ``value`` strings are human-readable summaries and
    are NOT trusted — the only evidence that a computation actually ran is the
    job directory: ``<job_id>.json`` (meta carrying pid / log_path) must exist
    and the ``<job_id>.log`` it points at (the job's captured stdout) must
    contain a parseable number. Domain-neutral: any numeric stdout qualifies.
    """
    if not run_dir or not job_id:
        return False
    jobs = Path(run_dir)
    meta_path = jobs / f"{job_id}.json"
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(meta, dict):
        return False
    log_path = meta.get("log_path")
    log_file = Path(log_path) if log_path else jobs / f"{job_id}.log"
    if not log_file.is_file():
        return False
    try:
        text = log_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return re.search(r"-?\d+\.?\d*", text) is not None


def _classify_verification(
    ver: Verification, score: float, has_compute: bool, run_dir: str | None = None
) -> str:
    """Code-based verdict from the evidence counts (T3), with T2/T4 gates.

    - ``refutes >= FALSIFY_REFUTES and supports == 0`` → falsified
    - ``supports >= 1 and refutes == 0`` → verified, unless
      - T2: ``score < VERIFY_LOW_SCORE`` requires ``supports >= STRONG_SUPPORTS``;
      - T4: compute tools exist but the verification carries no on-disk job
        evidence (``job_id`` empty, ``<job_id>.json``/``<job_id>.log`` missing,
        or the log contains no parseable number).
    - anything else → prediction (evidence insufficient / mixed)
    """
    if ver.refutes >= FALSIFY_REFUTES and ver.supports == 0:
        return "falsified"
    if ver.supports >= 1 and ver.refutes == 0:
        if score < VERIFY_LOW_SCORE and ver.supports < STRONG_SUPPORTS:
            return "prediction"
        if has_compute and not _job_log_has_number(run_dir, ver.job_id):
            return "prediction"
        return "verified"
    return "prediction"


def _parse_json_lenient(text: str) -> Any:
    """Extract a JSON object from an agent's free-text answer (lenient).

    Handles markdown fences and leading/trailing prose; returns ``None`` when
    no valid JSON object is present.
    """
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


class ResearchLoopWorkflow(Workflow):
    """Deterministic research loop (P0 skeleton).

    Run with ``await workflow.run(task="…")``; the result is the final report
    string (``StopEvent.result``). ``plugins_dir`` points at an external plugin
    directory (data search / DL models / software / CLI) exposed via
    :meth:`load_plugins` — the bridge an agent-backed node consumes.
    """

    def __init__(
        self,
        *,
        plugins_dir: str | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
        cfg: Any = None,
        db: Any = None,
        graph: Any = None,
        timeout: float | None = 600.0,
        **kwargs: Any,
    ) -> None:
        # Agent-backed nodes make several LLM round-trips (2–15s each) inside a
        # single step, so the LlamaIndex default per-step timeout (45s) is too
        # tight for this pipeline. 600s gives a bounded but comfortable budget.
        super().__init__(timeout=timeout, **kwargs)
        self._plugins_dir = plugins_dir
        self._mcp_servers = mcp_servers
        self._cfg = cfg
        self._db = db
        self._graph = graph
        self._plugin_registry: Any = None

    def load_plugins(self) -> Any:
        """Discover external plugins (data / software) for agent tools.

        Lazy and graceful: returns a :class:`PluginRegistry` even when
        ``plugins_dir`` is absent; a broken directory never raises.
        """
        if self._plugin_registry is None:
            from drbrain.plugins.registry import PluginRegistry

            self._plugin_registry = PluginRegistry()
            if self._plugins_dir:
                self._plugin_registry.discover(self._plugins_dir)
        return self._plugin_registry

    def _direct_search(self, query: str, limit: int = 10) -> list[str]:
        """Deterministic candidate fetch via the local KG ``search_papers`` plugin.

        Used as the reliable path in :meth:`retrieve`: the agent distills the
        task into a query, then this calls the plugin directly instead of
        parsing free-form agent output. Returns paper titles, or ``[]`` when
        the plugin is absent or finds nothing.
        """
        registry = self.load_plugins()
        try:
            result = registry.call("search_papers", {"query": query, "limit": limit})
        except Exception:  # noqa: BLE001 — plugin absence must not break the loop
            return []
        if not getattr(result, "ok", False) or not isinstance(getattr(result, "data", None), dict):
            return []
        papers = result.data.get("papers", []) or []
        return [str(p.get("title", "")).strip() for p in papers if p.get("title")]

    @staticmethod
    def _fallback_query(task: str) -> str:
        """Best-effort English keyword extraction when the agent can't distill.

        Research tasks are often Chinese prose with an English topic embedded
        (e.g. 「…拓扑平带（topological flat band）…」); pull the longest English
        phrase so a deterministic ``LIKE`` search still hits real titles.
        """
        phrases = re.findall(r"[A-Za-z][A-Za-z0-9\- ]{2,}", task or "")
        if not phrases:
            return task or ""
        return max(phrases, key=len).strip()

    @staticmethod
    def _build_template_report(state: ResearchState) -> str:
        """Deterministic report assembled from the structured state.

        Used as the reliable baseline in :meth:`report`: the agent-authored
        report wins when it produces text, but the loop must still emit a real
        report when the agent comes back empty.
        """
        lines = ["# 研究报告", ""]
        lines.append(f"## 任务\n\n{state.task or '(未提供)'}\n")

        lines.append("## 候选文献")
        if state.candidates:
            for i, c in enumerate(state.candidates, 1):
                lines.append(f"{i}. {c}")
        else:
            lines.append("（无）")

        lines.append("\n## 提取的关键特征 / 机制 / 实体")
        if state.entities:
            lines.append("；".join(state.entities))
        else:
            lines.append("（无）")

        lines.append("\n## 研究 gap")
        if state.gaps:
            lines.extend(f"- {g}" for g in state.gaps)
        else:
            lines.append("（无）")

        lines.append("\n## 假设（可证伪：prediction / falsification）")
        if state.hypotheses:
            for h in state.hypotheses:
                lines.append(f"- {h.statement}（score {h.score:.2f}）")
                if h.prediction:
                    lines.append(f"  - 支持证据：{h.prediction}")
                if h.falsification:
                    lines.append(f"  - 证伪标准：{h.falsification}")
        else:
            lines.append("（无）")

        lines.append("\n## 已验证结论（KEEP）")
        if state.verified:
            lines.extend(f"- {v}" for v in state.verified)
        else:
            lines.append("（无）")

        lines.append("\n## 已证伪假设（DISCARD）")
        if state.falsified:
            lines.extend(f"- {f}" for f in state.falsified)
        else:
            lines.append("（无）")

        lines.append("\n## 核验计数（Supports/Refutes/Orthogonal，代码判定）")
        if state.verifications:
            for v in state.verifications:
                computed = f"，实算={v.computed}" if v.computed else ""
                lines.append(
                    f"- {v.statement}：supports={v.supports}, refutes={v.refutes}, "
                    f"orthogonal={v.orthogonal} → {v.status}{computed}"
                )
        else:
            lines.append("（无）")

        lines.append("\n## 预测 / 后续验证建议")
        if state.predictions:
            lines.extend(f"- {p}" for p in state.predictions)
        else:
            lines.append("（无）")

        return "\n".join(lines)

    def build_node_agent(self, *, plugins_dir: str | None = None, role: str | None = None) -> Any:
        """Assemble the loop's :class:`FunctionAgent` with the full tool surface.

        Reuses :func:`drbrain.rag.agent.build_agent` (7 graph tools + fused
        retrieval + external plugins). Returns ``None`` when no config is
        supplied or llama-index is unavailable — callers must not assume an
        agent is always present.

        ``role`` (T1) swaps in a role-differentiated system prompt (critic /
        verifier) instead of the generic assistant template; ``None`` keeps the
        default prompt for the neutral nodes (retrieve / extract / gaps /
        report).
        """
        if self._cfg is None:
            return None
        from drbrain.rag.agent import build_agent

        agent = build_agent(
            self._cfg,
            self._db,
            graph=self._graph,
            plugins_dir=plugins_dir if plugins_dir is not None else self._plugins_dir,
            mcp_servers=self._mcp_servers,
        )
        if agent is not None and role:
            from drbrain.loop.roles import ROLE_SYSTEM_PROMPTS

            prompt = ROLE_SYSTEM_PROMPTS.get(role)
            if prompt:
                # FunctionAgent reads system_prompt at call time (workflow step),
                # so mutating it after build_agent is sufficient for the role swap.
                agent.system_prompt = prompt
        return agent

    def _has_compute_tools(self, agent: Any | None = None) -> bool:
        """T4: does the environment expose compute tools (run_python / check_job)?

        Checks plugin registry names plus (when an agent was built) the agent's
        tool surface — which also covers MCP-served tools. Best-effort: a plugin
        that fails to list must not raise.
        """
        names: list[str] = []
        try:
            names += [p.name for p in self.load_plugins().list_plugins()]
        except Exception:  # noqa: BLE001 — capability probe must never break a node
            pass
        if agent is not None:
            try:
                names += [t.metadata.name for t in agent.tools]
            except Exception:  # noqa: BLE001
                pass
        lowered = [n.lower() for n in names]
        return any(n in _COMPUTE_TOOL_NAMES or "materials" in n or "compute" in n for n in lowered)

    async def run_agent(
        self,
        agent: Any,
        user_msg: str,
        *,
        max_iterations: int = 5,
    ) -> str | None:
        """Run ``agent`` with ``user_msg`` and return its text answer.

        The answer is extracted from the agent result's ``response.content``
        (mirrors :func:`reason_llamaindex`). Returns ``None`` for a missing
        agent so a node can fall back to its deterministic path.
        """
        if agent is None:
            return None
        handler = agent.run(user_msg=user_msg, max_iterations=max(1, int(max_iterations)))
        result = await handler
        response = getattr(result, "response", None)
        answer = ""
        if response is not None:
            answer = response.content or ""
        return answer or "No answer generated."

    async def run_agent_json(
        self,
        agent: Any,
        user_msg: str,
        *,
        max_iterations: int = 5,
        retries: int = 1,
    ) -> Any:
        """Run ``agent`` and parse its answer as a JSON object (lenient).

        Returns the parsed dict/list, or ``None`` when the agent is missing or
        its answer carries no valid JSON. Retries once with a stricter "JSON
        only" nudge — the LLM is non-deterministic, so a single malformed
        answer must not silently empty a node.
        """
        for attempt in range(retries + 1):
            answer = await self.run_agent(agent, user_msg, max_iterations=max_iterations)
            if answer and answer != "No answer generated.":
                data = _parse_json_lenient(answer)
                if data is not None:
                    return data
            if attempt < retries:
                user_msg += (
                    "\n\n（上次没返回合法 JSON，这次务必只输出一个 JSON 对象，不要任何其他文字。）"
                )
        return None

    async def _get_state(self, ctx: Context) -> ResearchState:
        state = await ctx.store.get(_STATE_KEY, default=None)
        if state is None:
            state = ResearchState()
            await ctx.store.set(_STATE_KEY, state)
        return state

    async def _set_state(self, ctx: Context, state: ResearchState) -> None:
        await ctx.store.set(_STATE_KEY, state)

    # 1. 任务规划
    @step
    async def plan_task(self, ctx: Context, ev: StartEvent) -> TaskPlanned:
        state = await self._get_state(ctx)
        state.task = getattr(ev, "task", "") or ""
        state.prior_context = getattr(ev, "prior_context", "") or ""
        # T6: structured prior champion/dead-ends (director passes them explicitly;
        # direct ``wf.run`` callers fall back to parsing ``prior_context`` text).
        state.prior_champion = list(getattr(ev, "prior_champion", None) or [])
        state.prior_rejected = list(getattr(ev, "prior_rejected", None) or [])
        await self._set_state(ctx, state)
        logger.info("[loop] plan_task: %r", state.task)
        return TaskPlanned(task=state.task)

    # 2. 文献检索（接受首次 TaskPlanned 与回环 RetrieveAgain）
    @step
    async def retrieve(self, ctx: Context, ev: TaskPlanned | RetrieveAgain) -> Retrieved:
        state = await self._get_state(ctx)
        attempt = ev.attempt if isinstance(ev, RetrieveAgain) else 1
        candidates = list(state.candidates)
        # First attempt only (the RetrieveAgain loop bounds re-runs): the agent
        # distills the task into a focused query, then a deterministic plugin
        # search fetches the candidate papers — no free-form output to parse.
        if attempt == 1 and state.task:
            query = state.task
            agent = self.build_node_agent()
            if agent is not None:
                q = await self.run_agent_json(
                    agent,
                    "把下面的研究任务提炼成 2~4 个最核心的检索关键词（空格分隔），只返回 JSON："
                    '{"query": "..."}。'
                    f"任务：{state.task}",
                )
                if isinstance(q, dict) and str(q.get("query", "")).strip():
                    query = str(q["query"]).strip()
                else:
                    query = self._fallback_query(state.task)
            candidates = self._direct_search(query)
        state.candidates = candidates
        state.retrieve_candidates = candidates
        await self._set_state(ctx, state)
        logger.info("[loop] retrieve: %d candidates", len(candidates))
        return Retrieved(candidates=candidates, attempt=attempt)

    # 3. 文献筛选（不足则回检索，超次则放行）
    @step
    async def filter(self, ctx: Context, ev: Retrieved) -> Filtered | RetrieveAgain:
        state = await self._get_state(ctx)
        state.candidates = ev.candidates
        await self._set_state(ctx, state)
        if not ev.candidates and ev.attempt < MAX_RETRIEVE_ATTEMPTS:
            return RetrieveAgain(attempt=ev.attempt + 1, reason="candidates empty")
        return Filtered(selected=ev.candidates)

    # 4. PDF 解析
    @step
    async def parse_pdf(self, ctx: Context, ev: Filtered) -> Parsed:
        state = await self._get_state(ctx)
        state.parsed = ev.selected
        await self._set_state(ctx, state)
        return Parsed(docs=ev.selected)

    # 5. 知识抽取（agent-backed：从候选文献提取概念/特征/机制/材料）
    @step
    async def extract(self, ctx: Context, ev: Parsed) -> Extracted:
        state = await self._get_state(ctx)
        entities = list(state.entities)
        agent = self.build_node_agent()
        if agent is not None and ev.docs:
            data = await self.run_agent_json(
                agent,
                "从以下候选文献中提取关键概念、特征、机制与实体。"
                "不要调用任何检索工具，直接基于给定的候选文献推理，只返回 JSON："
                '{"entities": ["...", "..."]}。'
                f"候选文献：{ev.docs}",
            )
            if isinstance(data, dict) and data.get("entities"):
                entities = [str(e) for e in data["entities"]]
        # deterministic fallback: when extraction yields nothing, the candidates
        # themselves are the extracted surface, so downstream nodes still get input.
        if not entities:
            entities = list(ev.docs)
        state.entities = entities
        await self._set_state(ctx, state)
        logger.info("[loop] extract: %d entities", len(entities))
        return Extracted(entities=entities)

    # 6. 实体规范化
    @step
    async def normalize(self, ctx: Context, ev: Extracted) -> Normalized:
        state = await self._get_state(ctx)
        state.entities = ev.entities
        await self._set_state(ctx, state)
        return Normalized(entities=ev.entities)

    # 7. 跨文献融合
    @step
    async def fuse(self, ctx: Context, ev: Normalized) -> Fused:
        state = await self._get_state(ctx)
        state.entities = ev.entities
        await self._set_state(ctx, state)
        return Fused(entities=ev.entities)

    # 8. Gap 识别 → 假设提出（AutoScientists：从 gap 生成候选假设）
    @step
    async def identify_gaps(self, ctx: Context, ev: Fused) -> GapsIdentified:
        state = await self._get_state(ctx)
        gaps = list(state.gaps)
        hypotheses = list(state.hypotheses)
        # T6: dedup gate — every proposed hypothesis is checked against prior
        # champion/dead-ends before it is allowed in; already confirmed or
        # already falsified hypotheses are not re-proposed.
        prior_statements = list(state.prior_champion) + list(state.prior_rejected)
        if not prior_statements:
            prior_statements = _parse_prior_context(state.prior_context)

        # agent-backed: the loop's agent proposes gaps + hypotheses (structured JSON).
        agent = self.build_node_agent()
        if agent is not None:
            prior = state.prior_context
            context_line = (
                f"\n\n此前已确认结论与已否定假设（禁止重复提出，先对照再提案）：{prior}"
                if prior
                else ""
            )
            data = await self.run_agent_json(
                agent,
                "基于以下实体，识别研究 gap 并提出**可证伪**假设。"
                "提案前必须对照 prior_context：已确认结论与已否定假设一律不得重复提出。"
                "不要调用任何检索工具，直接基于给定实体推理，只返回 JSON："
                '{"gaps": ["...", ...], "hypotheses": [{"statement": "...", '
                '"prediction": "什么证据会支持它", "falsification": "什么证据会证伪它", '
                '"conditions": {}}]}。'
                f"实体：{state.entities}{context_line}",
            )
            if isinstance(data, dict):
                gaps = [str(g) for g in data.get("gaps", [])]
                hypotheses = [
                    Hypothesis(
                        statement=str(h.get("statement", "")).strip(),
                        conditions=h.get("conditions") or {},
                        prediction=str(h.get("prediction", "")).strip(),
                        falsification=str(h.get("falsification", "")).strip(),
                    )
                    for h in data.get("hypotheses", [])
                    if isinstance(h, dict) and str(h.get("statement", "")).strip()
                ]
        # T6: drop proposals that duplicate prior champion/dead-ends (code gate).
        if prior_statements:
            kept = [
                h for h in hypotheses if not _is_duplicate_proposal(h.statement, prior_statements)
            ]
            dropped = len(hypotheses) - len(kept)
            if dropped:
                logger.info("[loop] identify_gaps: dropped %d duplicate hypothesis(es)", dropped)
            hypotheses = kept
        # deterministic fallback: never let a cycle go empty — if the agent
        # yields nothing, derive a default gap + hypothesis from the entities so
        # critique/verify/settle still have input (AutoScientists: no dead cycle).
        if not gaps and state.entities:
            gaps = [f"缺少关于「{state.entities[0]}」的机制与候选验证"]
        if not hypotheses and gaps:
            hypotheses = [
                Hypothesis(
                    statement=f"假设：{gaps[0]} 可通过进一步检索与数值验证来确认或证伪",
                    conditions={},
                    prediction="检索到支持性证据或数值验证通过",
                    falsification="检索与数值验证均无法支持",
                )
            ]
        state.gaps = gaps
        state.hypotheses = hypotheses
        await self._set_state(ctx, state)
        logger.info("[loop] identify_gaps: %d gaps, %d hypotheses", len(gaps), len(hypotheses))
        return GapsIdentified(gaps=gaps, hypotheses=hypotheses)

    # 9. 假设互评（AutoScientists：critique-before-compute；T1 批判者角色 + T5 评审门）
    @step
    async def critique(self, ctx: Context, ev: GapsIdentified) -> Critiqued:
        state = await self._get_state(ctx)
        # T5: default status critiqued (KEEP); the critic may DISCARD weak ones.
        hypotheses = [h.model_copy(update={"status": "critiqued"}) for h in ev.hypotheses]
        # agent-backed: the critic (独立批判者 role, T1) scores each hypothesis
        # and finds flaws BEFORE any compute is spent. Low scores are DISCARDed
        # right here — they never enter verification (T5 proposal review gate).
        agent = self.build_node_agent(role="critic")
        if agent is not None and ev.hypotheses:
            data = await self.run_agent_json(
                agent,
                "作为批判者独立评审以下假设：给每个假设打分(0~1)并找漏洞/反驳。"
                '同时裁决是否值得进入核验阶段：值得验证 → "KEEP"；'
                '分数过低或明显不成立 → "DISCARD"（DISCARD 的假设将被直接过滤，'
                "不消耗任何计算资源）。"
                "不要调用任何检索/计算工具，直接基于假设本身推理，只返回 JSON："
                '{"hypotheses": [{"statement": "...", "score": 0.8, "verdict": "KEEP"}]}。'
                f"假设：{[h.statement for h in ev.hypotheses]}",
            )
            if isinstance(data, dict):
                scored = {
                    str(h.get("statement", "")): h
                    for h in data.get("hypotheses", [])
                    if isinstance(h, dict)
                }
                hypotheses = []
                for h in ev.hypotheses:
                    entry = scored.get(h.statement, {})
                    score = float(entry.get("score", 0.0) or 0.0)
                    verdict = str(entry.get("verdict", "") or "KEEP").upper()
                    # T5: code gate — low score or explicit DISCARD → filtered out.
                    if score < CRITIQUE_DISCARD_SCORE or verdict == "DISCARD":
                        status = "discarded"
                    else:
                        status = "critiqued"
                    hypotheses.append(h.model_copy(update={"status": status, "score": score}))
        state.hypotheses = hypotheses
        state.scores = [h.score for h in hypotheses]
        await self._set_state(ctx, state)
        return Critiqued(hypotheses=hypotheses)

    # 10. 证据核验（T1 核验者角色 + T2 分数消费 + T3 三角验证代码化 + T4 实算门）
    @step
    async def verify(self, ctx: Context, ev: Critiqued) -> Verified:
        state = await self._get_state(ctx)
        # T5: only hypotheses that survived the critic enter verification.
        candidates = [h for h in ev.hypotheses if h.status == "critiqued"]
        verified: list[str] = []
        falsified = list(state.falsified)
        predictions = list(state.predictions)
        verifications = list(state.verifications)
        agent = self.build_node_agent(role="verifier")
        handled = False
        if agent is not None and candidates:
            has_compute = self._has_compute_tools(agent)
            # agent-backed dual-path: the agent searches evidence (RAG), calls
            # the tools it discovers (model/software plugins), and — when no
            # ready tool fits — writes its own code. T3: it reports structured
            # Supports/Refutes/Orthogonal counts; verdicts are derived in code.
            data = await self.run_agent_json(agent, self._verify_prompt(candidates, has_compute))
            if isinstance(data, dict):
                raw_vs = data.get("verifications")
                if isinstance(raw_vs, list):
                    handled = True
                    vs_by_stmt = {h.statement: h for h in candidates}
                    # T4: evidence lives in the on-disk job directory the
                    # director points DRBRAIN_RUN_DIR at (run_python async).
                    run_dir = os.environ.get("DRBRAIN_RUN_DIR") or None
                    for raw in raw_vs:
                        if not isinstance(raw, dict) or not str(raw.get("statement", "")).strip():
                            continue
                        stmt = str(raw["statement"]).strip()
                        h = vs_by_stmt.get(stmt)
                        if h is None:
                            continue  # not one of the proposed hypotheses — ignore
                        ver = Verification(
                            statement=stmt,
                            supports=_to_int(raw.get("supports")),
                            refutes=_to_int(raw.get("refutes")),
                            orthogonal=_to_int(raw.get("orthogonal")),
                            evidence=str(raw.get("evidence") or ""),
                            computed=str(raw.get("computed") or ""),
                            value=_to_float(raw.get("value")),
                            unit=str(raw.get("unit") or ""),
                            job_id=str(raw.get("job_id") or ""),
                        )
                        ver.status = _classify_verification(ver, h.score, has_compute, run_dir)
                        verifications.append(ver)
                        if ver.status == "verified":
                            verified.append(stmt)
                        elif ver.status == "falsified":
                            falsified.append(stmt)
                        else:
                            predictions.append(stmt)
        if not handled:
            # T3/T4: without structured verification counts there is no evidence —
            # nothing may be verified. Every candidate becomes a prediction. The
            # legacy {"verified": [...]} path is removed so a real agent cannot
            # bypass the evidence/compute gate by returning the old format.
            predictions = [h.statement for h in candidates] + predictions
        state.verified = verified
        state.falsified = falsified
        state.predictions = predictions
        state.verifications = verifications
        await self._set_state(ctx, state)
        return Verified(
            verified=verified,
            falsified=falsified,
            predictions=predictions,
            verifications=verifications,
        )

    def _verify_prompt(self, candidates: list[Hypothesis], has_compute: bool) -> str:
        """The verifier's user message (T3 structured counts + T4 compute gate).

        Domain-neutral: requires a real computed numeric result when compute
        tools exist, without naming any domain (materials/DFT specifics come
        from the task + skills, never from this prompt).
        """
        compute_line = (
            "\n环境提供计算类工具（run_python / check_job / 数值计算插件）。"
            "对每个假设，你必须用 run_python(mode=\"async\") 实际启动一次计算，"
            "用 check_job 轮询到作业跑完，并把返回的 job_id 填进该字段；"
            "computed/value 只是给人看的摘要，代码只认 job_id 对应的作业文件"
            "（$DRBRAIN_RUN_DIR/<job_id>.json 与 <job_id>.log，且日志含数值）。"
            "没有真实作业文件支撑的 computed 一律不认。"
            if has_compute
            else "\n环境无计算类工具：可只做文献证据核验，computed 留空即可。"
        )
        return (
            "核验以下假设：用检索/证据工具收集文献证据，对每个假设统计证据计数："
            "supports（支持其 prediction 的证据条数）、refutes（反驳的证据条数）、"
            "orthogonal（与假设无关/无法判定的证据条数），并写 evidence 摘要。"
            f"{compute_line}"
            "不要自行下结论，只报证据计数与数值；判定由下游代码完成。只返回 JSON："
            '{"verifications": [{"statement": "...", "supports": N, "refutes": N, '
            '"orthogonal": N, "evidence": "...", '
            '"job_id": "run_python(async) 返回的作业 id，无实算则为空", '
            '"computed": "实际数值结果或空", '
            '"value": 数值或 null, "unit": "单位"}]}。'
            f"假设：{[h.statement for h in candidates]}"
        )

    # 11. 沉淀（AutoScientists：共享成败 → 写回共享记忆）
    @step
    async def settle(self, ctx: Context, ev: Verified) -> Settled:
        state = await self._get_state(ctx)
        state.verified = ev.verified
        state.falsified = ev.falsified
        state.predictions = ev.predictions
        state.verifications = ev.verifications
        self._persist_claims(state)
        await self._set_state(ctx, state)
        return Settled(verified=ev.verified, falsified=ev.falsified)

    def _persist_claims(self, state: ResearchState) -> None:
        """闭环沉淀：把核验结论/证伪/预测写回 KG（``claims`` 表）。

        KEEP（verified）→ ``Conclusion``；DISCARD（falsified）→ ``Rejected``
        （负结论也是知识）；预测 → ``Prediction``。Idempotent via
        ``record_claim`` (stable claim_id hash). Degrades to a no-op when no DB
        is supplied; a DB write failure must never break the loop.
        """
        if self._db is None:
            return
        try:
            for statement in state.verified:
                self._db.record_claim(
                    state.task or "research-loop",
                    statement,
                    claim_type="Conclusion",
                    authority="research-loop",
                    provenance="research-loop",
                    confidence=1.0,
                )
                self._record_evidence_for(statement)
            for statement in state.falsified:
                self._db.record_claim(
                    state.task or "research-loop",
                    statement,
                    claim_type="Rejected",
                    authority="research-loop",
                    provenance="research-loop",
                    confidence=1.0,
                )
                self._record_evidence_for(statement)
            for prediction in state.predictions:
                self._db.record_claim(
                    state.task or "research-loop",
                    prediction,
                    claim_type="Prediction",
                    authority="research-loop",
                    provenance="research-loop",
                    confidence=1.0,
                )
                self._record_evidence_for(prediction)
        except Exception as exc:  # noqa: BLE001 — persistence must not break the loop
            logger.warning("[loop] settle persist failed: %s", exc)

    def _record_evidence_for(self, statement: str) -> None:
        """闭环沉淀：给一条结论写 first-class 证据行（可追溯）。

        ``evidence_id`` 由 statement 哈希而来（幂等）；paper/node 留空（结论是
        跨文献综合，非单篇摘录），snippet 记录结论原文，provenance 标记来源。
        """
        import hashlib

        eid = "evidence_" + hashlib.sha1(statement.encode()).hexdigest()[:16]
        self._db.record_evidence(
            paper_id="",
            snippet=statement,
            value="1.0",
            provenance="research-loop",
            authority="research-loop",
            evidence_id=eid,
        )

    # 12. 报告生成（agent-backed：把整条链路的累积状态写成结构化报告）
    @step
    async def report(self, ctx: Context, ev: Settled) -> StopEvent:
        state = await self._get_state(ctx)
        summary = (
            f"task={state.task!r}; candidates={len(state.candidates)}; "
            f"gaps={len(state.gaps)}; hypotheses={len(state.hypotheses)}; "
            f"verified={len(state.verified)}; falsified={len(state.falsified)}"
        )
        report = self._build_template_report(state)
        agent = self.build_node_agent()
        if agent is not None:
            text = await self.run_agent(
                agent,
                "基于以下研究状态，撰写一份结构化研究报告（markdown），涵盖："
                "① 从文献中提取的关键特征与机制；② 候选方案/实体及其依据；"
                "③ 研究 gap 与假设（含互评分数）；④ 已验证结论与预测；⑤ 后续验证建议。"
                f"\n\n任务：{state.task}\n\n候选文献：{state.candidates}\n\n"
                f"提取实体：{state.entities}\n\ngap：{state.gaps}\n\n"
                f"假设：{[h.statement for h in state.hypotheses]}\n\n"
                f"已验证结论：{state.verified}\n\n预测：{state.predictions}",
            )
            if text and text != "No answer generated.":
                report = text
        final = f"{summary}\n\n{report}"
        state.report = final
        await self._set_state(ctx, state)
        logger.info("[loop] report: %s", summary)
        return StopEvent(result=final)
