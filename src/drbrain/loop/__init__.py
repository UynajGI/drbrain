"""Research loop — the orchestration layer (third layer of the three-in-one).

Domain-agnostic: the loop schedules retrieval → extraction → gap → hypothesis →
critique → verification → report, while concrete capabilities (models, software)
are injected through the plugin layer (:mod:`drbrain.plugins`) and literature
understanding through the RAG layer (:mod:`drbrain.rag`).
"""

from drbrain.loop.director import ResearchDirector
from drbrain.loop.events import (
    Evidence,
    Hypothesis,
    ResearchState,
)
from drbrain.loop.policy import ToolDefinition, ToolPolicy
from drbrain.loop.tool_broker import ToolBroker, ToolCallStatus, ToolObservation
from drbrain.loop.workflow import ResearchLoopWorkflow

__all__ = [
    "Evidence",
    "Hypothesis",
    "ResearchDirector",
    "ResearchLoopWorkflow",
    "ResearchState",
    "ToolBroker",
    "ToolCallStatus",
    "ToolDefinition",
    "ToolObservation",
    "ToolPolicy",
]
