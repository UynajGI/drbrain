"""Research loop — the orchestration layer (third layer of the three-in-one).

Domain-agnostic: the loop schedules retrieval → extraction → gap → hypothesis →
critique → verification → report, while concrete capabilities (models, software)
are injected through the plugin layer (:mod:`drbrain.plugins`) and literature
understanding through the RAG layer (:mod:`drbrain.rag`).
"""

from drbrain.loop.director import ResearchDirector
from drbrain.loop.durable_execution import ChampionVersionConflictError, DurableExecution
from drbrain.loop.events import (
    Evidence,
    EvidenceBundle,
    Hypothesis,
    ResearchState,
)
from drbrain.loop.governance import RunGovernance
from drbrain.loop.policy import ToolDefinition, ToolPolicy
from drbrain.loop.tool_broker import ToolBroker, ToolCallStatus, ToolObservation
from drbrain.loop.workflow import ResearchLoopWorkflow

__all__ = [
    "Evidence",
    "EvidenceBundle",
    "ChampionVersionConflictError",
    "DurableExecution",
    "Hypothesis",
    "ResearchDirector",
    "ResearchLoopWorkflow",
    "ResearchState",
    "RunGovernance",
    "ToolBroker",
    "ToolCallStatus",
    "ToolDefinition",
    "ToolObservation",
    "ToolPolicy",
]
