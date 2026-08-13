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
import logging
import re
from typing import Any

from llama_index.core.workflow import (
    Context,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

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

log = logging.getLogger(__name__)

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
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
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
    ) -> Any:
        """Run ``agent`` and parse its answer as a JSON object (lenient).

        Returns the parsed dict/list, or ``None`` when the agent is missing or
        its answer carries no valid JSON.
        """
        answer = await self.run_agent(agent, user_msg, max_iterations=max_iterations)
        if not answer or answer == "No answer generated.":
            return None
        return _parse_json_lenient(answer)

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
        await self._set_state(ctx, state)
        log.info("[loop] plan_task: %r", state.task)
        return TaskPlanned(task=state.task)

    # 2. 文献检索（接受首次 TaskPlanned 与回环 RetrieveAgain）
    @step
    async def retrieve(self, ctx: Context, ev: TaskPlanned | RetrieveAgain) -> Retrieved:
        state = await self._get_state(ctx)
        attempt = ev.attempt if isinstance(ev, RetrieveAgain) else 1
        candidates = list(state.candidates)
        # agent-backed search (first attempt only): use the loop's agent to find
        # papers; fall back to the prior/empty candidates when no agent is present.
        if attempt == 1 and state.task:
            agent = self.build_node_agent()
            if agent is not None:
                answer = await self.run_agent(
                    agent,
                    f"检索关于「{state.task}」的学术文献，返回相关论文标题（每行一个，最多 10 个）。",
                )
                if answer and answer != "No answer generated.":
                    candidates = [ln.strip() for ln in answer.splitlines() if ln.strip()]
        state.candidates = candidates
        await self._set_state(ctx, state)
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

    # 5. 知识抽取
    @step
    async def extract(self, ctx: Context, ev: Parsed) -> Extracted:
        state = await self._get_state(ctx)
        # P0 stub: no LLM extraction.
        entities = list(state.entities)
        state.entities = entities
        await self._set_state(ctx, state)
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
            data = await self.run_agent_json(
                agent,
                "基于以下实体，识别研究 gap 并提出可验证假设，只返回 JSON："
                '{"gaps": ["...", ...], "hypotheses": [{"statement": "...", "conditions": {}}]}。'
                f"实体：{state.entities}",
            )
            if isinstance(data, dict):
                gaps = [str(g) for g in data.get("gaps", [])]
                hypotheses = [
                    Hypothesis(
                        statement=str(h.get("statement", "")).strip(),
                        conditions=h.get("conditions") or {},
                    )
                    for h in data.get("hypotheses", [])
                    if isinstance(h, dict) and str(h.get("statement", "")).strip()
                ]
        state.gaps = gaps
        state.hypotheses = hypotheses
        await self._set_state(ctx, state)
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
                "互评以下假设，给每个假设打分(0~1)并判断是否值得验证，只返回 JSON："
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
        await self._set_state(ctx, state)
        return Critiqued(hypotheses=hypotheses)

    # 10. 证据核验（双路：RAG 推理 + 插件计算）
    @step
    async def verify(self, ctx: Context, ev: Critiqued) -> Verified:
        state = await self._get_state(ctx)
        verified = [h.statement for h in ev.hypotheses if h.status == "critiqued"]
        predictions = list(state.predictions)
        # agent-backed dual-path: the agent searches evidence (RAG) and calls
        # compute plugins (DL/software) to verify each hypothesis.
        agent = self.build_node_agent()
        if agent is not None and ev.hypotheses:
            data = await self.run_agent_json(
                agent,
                "核验以下假设：用可用的检索工具找文献证据，用可用的计算插件做数值验证，"
                "只返回 JSON：{'verified': ['...', ...], 'predictions': ['...', ...]}。"
                f"假设：{[h.statement for h in ev.hypotheses]}",
            )
            if isinstance(data, dict):
                if data.get("verified"):
                    verified = [str(v) for v in data["verified"]]
                if data.get("predictions"):
                    predictions = [str(p) for p in data["predictions"]]
        state.verified = verified
        state.predictions = predictions
        await self._set_state(ctx, state)
        return Verified(verified=verified, predictions=predictions)

    # 11. 沉淀（AutoScientists：共享成败 → 写回共享记忆）
    @step
    async def settle(self, ctx: Context, ev: Verified) -> Settled:
        state = await self._get_state(ctx)
        state.verified = ev.verified
        state.predictions = ev.predictions
        await self._set_state(ctx, state)
        return Settled(verified=ev.verified)

    # 12. 报告生成
    @step
    async def report(self, ctx: Context, ev: Settled) -> StopEvent:
        state = await self._get_state(ctx)
        state.report = (
            f"task={state.task!r}; candidates={len(state.candidates)}; "
            f"gaps={len(state.gaps)}; hypotheses={len(state.hypotheses)}; "
            f"verified={len(state.verified)}"
        )
        await self._set_state(ctx, state)
        log.info("[loop] report: %s", state.report)
        return StopEvent(result=state.report)
