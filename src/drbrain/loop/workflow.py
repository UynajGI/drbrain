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
import re
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
    Verified,
)

_STATE_KEY = "research_state"
MAX_RETRIEVE_ATTEMPTS = 3


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

        lines.append("\n## 预测 / 后续验证建议")
        if state.predictions:
            lines.extend(f"- {p}" for p in state.predictions)
        else:
            lines.append("（无）")

        return "\n".join(lines)

    def build_node_agent(self, *, plugins_dir: str | None = None) -> Any:
        """Assemble the loop's :class:`FunctionAgent` with the full tool surface.

        Reuses :func:`drbrain.rag.agent.build_agent` (7 graph tools + fused
        retrieval + external plugins). Returns ``None`` when no config is
        supplied or llama-index is unavailable — callers must not assume an
        agent is always present.
        """
        if self._cfg is None:
            return None
        from drbrain.rag.agent import build_agent

        return build_agent(
            self._cfg,
            self._db,
            graph=self._graph,
            plugins_dir=plugins_dir if plugins_dir is not None else self._plugins_dir,
            mcp_servers=self._mcp_servers,
        )

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
                user_msg += "\n\n（上次没返回合法 JSON，这次务必只输出一个 JSON 对象，不要任何其他文字。）"
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
        # agent-backed: the loop's agent proposes gaps + hypotheses (structured JSON).
        agent = self.build_node_agent()
        if agent is not None:
            prior = state.prior_context
            context_line = (
                f"\n\n此前已确认结论与已否定假设（不要重复，基于此推进，提出新假设）：{prior}"
                if prior
                else ""
            )
            data = await self.run_agent_json(
                agent,
                "基于以下实体，识别研究 gap 并提出**可证伪**假设。"
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

    # 9. 假设互评（AutoScientists：critique-before-compute）
    @step
    async def critique(self, ctx: Context, ev: GapsIdentified) -> Critiqued:
        state = await self._get_state(ctx)
        hypotheses = [h.model_copy(update={"status": "critiqued"}) for h in ev.hypotheses]
        # agent-backed: the agent scores each hypothesis before compute is spent.
        agent = self.build_node_agent()
        if agent is not None and ev.hypotheses:
            data = await self.run_agent_json(
                agent,
                "互评以下假设，给每个假设打分(0~1)并判断是否值得验证。"
                "不要调用任何检索工具，直接打分，只返回 JSON："
                '{"hypotheses": [{"statement": "...", "score": 0.8}]}。'
                f"假设：{[h.statement for h in ev.hypotheses]}",
            )
            if isinstance(data, dict):
                scored = {
                    str(h.get("statement", "")): h
                    for h in data.get("hypotheses", [])
                    if isinstance(h, dict)
                }
                hypotheses = [
                    h.model_copy(
                        update={
                            "status": "critiqued",
                            "score": float(scored.get(h.statement, {}).get("score", 0.0) or 0.0),
                        }
                    )
                    for h in ev.hypotheses
                ]
        state.hypotheses = hypotheses
        state.scores = [h.score for h in hypotheses]
        await self._set_state(ctx, state)
        return Critiqued(hypotheses=hypotheses)

    # 10. 证据核验（双路：RAG 推理 + 插件计算）
    @step
    async def verify(self, ctx: Context, ev: Critiqued) -> Verified:
        state = await self._get_state(ctx)
        verified = [h.statement for h in ev.hypotheses if h.status == "critiqued"]
        falsified = list(state.falsified)
        predictions = list(state.predictions)
        # agent-backed dual-path: the agent searches evidence (RAG), calls the
        # tools it discovers (model/software plugins), and — when no ready tool
        # fits — writes its own code (self-extension). Each hypothesis is judged
        # verified (evidence supports the prediction) or falsified (evidence
        # refutes it). Domain specifics come from the task + skills.
        agent = self.build_node_agent()
        if agent is not None and ev.hypotheses:
            data = await self.run_agent_json(
                agent,
                "核验以下假设：用你手上的检索工具找文献证据，用计算/模型工具或"
                "自写代码做数值验证。对每个假设判定：verified（证据支持其 prediction）、"
                "falsified（证据证伪）、或无法判定则不入两类。"
                "只返回 JSON："
                "{'verified': ['...', ...], 'falsified': ['...', ...], 'predictions': ['...', ...]}。"
                f"假设：{[h.statement for h in ev.hypotheses]}",
            )
            if isinstance(data, dict):
                if data.get("verified"):
                    verified = [str(v) for v in data["verified"]]
                if data.get("falsified"):
                    falsified = [str(f) for f in data["falsified"]]
                if data.get("predictions"):
                    predictions = [str(p) for p in data["predictions"]]
        state.verified = verified
        state.falsified = falsified
        state.predictions = predictions
        await self._set_state(ctx, state)
        return Verified(verified=verified, falsified=falsified, predictions=predictions)

    # 11. 沉淀（AutoScientists：共享成败 → 写回共享记忆）
    @step
    async def settle(self, ctx: Context, ev: Verified) -> Settled:
        state = await self._get_state(ctx)
        state.verified = ev.verified
        state.falsified = ev.falsified
        state.predictions = ev.predictions
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
