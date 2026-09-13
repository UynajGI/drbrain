"""Research loop — the orchestration layer (third layer of the three-in-one).

Domain-agnostic: the loop schedules retrieval → extraction → gap → hypothesis →
critique → verification → report, while concrete capabilities (plugins, MCP,
APIs, CLIs, models and Skills) are discovered through the shared
:class:`LoopToolSpace`; literature understanding remains in the RAG layer
(:mod:`drbrain.rag`).
"""

from drbrain.loop.benchmark import run_frontier_benchmark
from drbrain.loop.director import ResearchDirector
from drbrain.loop.durable_execution import ChampionVersionConflictError, DurableExecution
from drbrain.loop.events import (
    Evidence,
    EvidenceBundle,
    Hypothesis,
    ResearchState,
)
from drbrain.loop.frontier import (
    ArtifactRef,
    BranchOutcome,
    BranchSpec,
    BranchStatus,
    DoneContract,
    EvidenceRef,
    ExperimentSpec,
    ResearchFrontier,
    ResearchObjective,
)
from drbrain.loop.governance import RunGovernance
from drbrain.loop.policy import ToolDefinition, ToolPolicy
from drbrain.loop.research_events import (
    EventEnvelope,
    EventLogSnapshotStore,
    EventSourcedState,
    InMemoryEventLog,
    InMemorySnapshotStore,
    RunLedgerEventLog,
)
from drbrain.loop.supervisor import (
    BranchWorker,
    ResearchSupervisor,
    SupervisorConfig,
    SupervisorResult,
    WorkflowBranchWorker,
)
from drbrain.loop.tool_broker import ToolBroker, ToolCallStatus, ToolObservation
from drbrain.loop.tool_space import LoopToolSpace
from drbrain.loop.workflow import ResearchLoopWorkflow

__all__ = [
    "Evidence",
    "EvidenceBundle",
    "ArtifactRef",
    "BranchOutcome",
    "BranchSpec",
    "BranchStatus",
    "ChampionVersionConflictError",
    "DurableExecution",
    "Hypothesis",
    "ResearchDirector",
    "ResearchLoopWorkflow",
    "ResearchState",
    "ResearchFrontier",
    "ResearchObjective",
    "DoneContract",
    "EvidenceRef",
    "ExperimentSpec",
    "RunGovernance",
    "ToolBroker",
    "ToolCallStatus",
    "ToolDefinition",
    "ToolObservation",
    "ToolPolicy",
    "LoopToolSpace",
    "EventEnvelope",
    "EventLogSnapshotStore",
    "EventSourcedState",
    "InMemoryEventLog",
    "InMemorySnapshotStore",
    "RunLedgerEventLog",
    "BranchWorker",
    "WorkflowBranchWorker",
    "ResearchSupervisor",
    "SupervisorConfig",
    "SupervisorResult",
    "run_frontier_benchmark",
]
