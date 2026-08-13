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

import logging

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


class ResearchLoopWorkflow(Workflow):
    """Deterministic research loop (P0 skeleton).

    Run with ``await workflow.run(task="…")``; the result is the final report
    string (``StopEvent.result``).
    """

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
        # P0 stub: no real retrieval — candidates stay empty, driving the loop.
        return Retrieved(candidates=list(state.candidates), attempt=attempt)

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
        # P0 stub: no gap detection — no hypotheses proposed yet.
        gaps = list(state.gaps)
        hypotheses = list(state.hypotheses)
        state.gaps = gaps
        state.hypotheses = hypotheses
        await self._set_state(ctx, state)
        return GapsIdentified(gaps=gaps, hypotheses=hypotheses)

    # 9. 假设互评（AutoScientists：critique-before-compute）
    @step
    async def critique(self, ctx: Context, ev: GapsIdentified) -> Critiqued:
        state = await self._get_state(ctx)
        hypotheses: list[Hypothesis] = []
        for h in ev.hypotheses:
            # P0 stub: no scoring — just advance the status.
            hypotheses.append(h.model_copy(update={"status": "critiqued"}))
        state.hypotheses = hypotheses
        await self._set_state(ctx, state)
        return Critiqued(hypotheses=hypotheses)

    # 10. 证据核验（双路：RAG 推理 + 插件计算 —— P0 为 stub）
    @step
    async def verify(self, ctx: Context, ev: Critiqued) -> Verified:
        state = await self._get_state(ctx)
        verified = [h.statement for h in ev.hypotheses if h.status == "critiqued"]
        predictions = list(state.predictions)
        state.verified = verified
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
