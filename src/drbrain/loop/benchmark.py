"""Small deterministic benchmarks for the adaptive frontier.

The benchmark is intentionally synthetic: it measures scheduler behavior
without pretending that an LLM judge is scientific ground truth. Hosts can
provide a real BranchWorker to replay a fixed task set later.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from drbrain.loop.frontier import (
    BranchOutcome,
    BranchSpec,
    BranchStatus,
    DoneContract,
    ResearchObjective,
)
from drbrain.loop.research_events import InMemoryEventLog
from drbrain.loop.supervisor import ResearchSupervisor, SupervisorConfig


async def run_frontier_benchmark(
    branches: Sequence[BranchSpec],
    worker: Callable[
        [BranchSpec, ResearchObjective, DoneContract], BranchOutcome | Awaitable[BranchOutcome]
    ],
    *,
    max_evaluations: int | None = None,
    max_parallel_branches: int = 2,
) -> dict[str, Any]:
    """Run a fixed branch set and return scheduler-only efficiency metrics."""
    started = time.perf_counter()
    log = InMemoryEventLog()
    supervisor = ResearchSupervisor(
        objective=ResearchObjective(question="frontier benchmark"),
        # Benchmarks should consume the requested budget so runs are
        # comparable across scheduler implementations, even when a synthetic
        # worker reports a retained branch early.
        done_contract=DoneContract(min_supported_evidence=(max_evaluations or len(branches)) + 1),
        worker=worker,
        event_log=log,
        config=SupervisorConfig(
            max_parallel_branches=max_parallel_branches,
            max_evaluations=max_evaluations or max(1, len(branches)),
            human_mode="autonomous",
        ),
    )
    result = await supervisor.run(list(branches))
    retained = sum(
        branch.status == BranchStatus.RETAINED for branch in result.frontier.branches.values()
    )
    pruned = sum(
        branch.status == BranchStatus.PRUNED for branch in result.frontier.branches.values()
    )
    return {
        "run_id": result.run_id,
        "completed": result.completed,
        "reason": result.reason,
        "evaluations": result.frontier.completed_evaluations,
        "retained": retained,
        "pruned": pruned,
        "event_count": len(result.events),
        "elapsed_seconds": time.perf_counter() - started,
    }


__all__ = ["run_frontier_benchmark"]
