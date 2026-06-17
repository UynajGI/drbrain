"""Base abstractions for the reasoning workflow engine.

A workflow is a deterministic pipeline of steps that compose symbolic
graph computation with LLM semantic judgment. Each step produces a
result stored in WorkflowContext, accessible to subsequent steps.

Design principles:
- Symbolic first: use graph/DB computation whenever possible
- LLM only for semantic judgment (synthesis, classification, narrative)
- Each step is independently testable
- Backward compatible: existing ReasonerAgent path unchanged
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from drbrain.extractor.cache import ApiCache

from loguru import logger


def _run_async(coro):
    """Run a coroutine safely, even inside a running event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        return loop.run_until_complete(coro)
    return asyncio.run(coro)


@dataclass
class WorkflowContext:
    """Container for intermediate results passed between workflow steps.

    Attributes:
        db: Open Database connection.
        graph: Loaded GraphEngine with edges.
        models: LLM model config list (provider/model/api_key).
        question: The user's natural-language question.
        results: Step outputs keyed by step name. Populated during execute().
    """

    db: Any  # Database — avoid hard import to prevent circular deps
    graph: Any  # GraphEngine
    models: list[dict]
    question: str
    cache: ApiCache | None = None
    results: dict[str, Any] = field(default_factory=dict)

    def get(self, step_name: str, default: Any = None) -> Any:
        """Retrieve a previous step's output by name."""
        return self.results.get(step_name, default)


class WorkflowStep(ABC):
    """A single step in a reasoning workflow pipeline.

    Subclasses implement ``run()`` to produce a result. Steps can read
    prior outputs via ``ctx.get(step_name)``.
    """

    name: str = ""
    requires_llm: bool = False

    @abstractmethod
    def run(self, ctx: WorkflowContext) -> Any:
        """Execute this step and return its output."""
        ...


class ReasoningWorkflow:
    """Orchestrates a sequence of WorkflowStep instances.

    Usage::

        wf = CausalWorkflow()
        ctx = WorkflowContext(db=db, graph=graph, models=models, question="...")
        results = wf.execute(ctx)
        explanation = results["synthesize_explanation"]
    """

    name: str = ""
    description: str = ""
    steps: list[WorkflowStep] = []

    def execute(self, ctx: WorkflowContext) -> dict[str, Any]:
        """Run all steps in order, storing each output in ctx.results.

        If ``ctx.cache`` is provided, a cache key is built from the workflow
        name, question, and graph/DB state fingerprint.  A hit skips all steps
        and returns the previously cached result dict.
        """
        # ── Workflow-level cache check ──────────────────────────────────
        if ctx.cache is not None:
            cache_key = self._build_cache_key(ctx)
            cached = ctx.cache.get(cache_key)
            if cached is not None:
                logger.info(
                    "[workflow:%s] cache hit — skipping %d steps", self.name, len(self.steps)
                )
                ctx.results = cached
                return cached

        # ── Run steps ───────────────────────────────────────────────────
        logger.info("[workflow:%s] starting — %d steps", self.name, len(self.steps))
        for step in self.steps:
            logger.info("[workflow:%s] step: %s (llm=%s)", self.name, step.name, step.requires_llm)
            try:
                result = step.run(ctx)
                ctx.results[step.name] = result
            except Exception as e:
                logger.warning("[workflow:%s] step %s failed: %s", self.name, step.name, e)
                ctx.results[step.name] = None
        logger.info("[workflow:%s] done — %d results", self.name, len(ctx.results))

        # ── Store in cache (only if all steps succeeded) ────────────────
        if ctx.cache is not None and all(v is not None for v in ctx.results.values()):
            try:
                import json as _json

                cacheable = {}
                for k, v in ctx.results.items():
                    try:
                        _json.dumps(v, default=str)
                        cacheable[k] = v
                    except (TypeError, ValueError):
                        cacheable[k] = str(v)
                ctx.cache.set(cache_key, cacheable)
            except Exception:
                logger.debug("[workflow:%s] cache store skipped", self.name)

        return ctx.results

    @staticmethod
    def _build_cache_key(ctx: WorkflowContext) -> str:
        """Build a deterministic cache key from workflow inputs + state."""
        graph_edge_count = ctx.graph.graph.number_of_edges() if hasattr(ctx.graph, "graph") else 0
        graph_node_count = ctx.graph.graph.number_of_nodes() if hasattr(ctx.graph, "graph") else 0
        try:
            db_paper_count = ctx.db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        except Exception:
            db_paper_count = 0
        return f"wf:{ctx.question}:{graph_edge_count}:{graph_node_count}:{db_paper_count}"


# ── Workflow registry ────────────────────────────────────────────────

_WORKFLOW_REGISTRY: dict[str, type[ReasoningWorkflow]] = {}


def register_workflow(name: str):
    """Decorator to register a workflow class in the global registry."""

    def decorator(cls: type[ReasoningWorkflow]) -> type[ReasoningWorkflow]:
        _WORKFLOW_REGISTRY[name] = cls
        return cls

    return decorator


def get_workflow(name: str) -> ReasoningWorkflow:
    """Look up and instantiate a workflow by name.

    Available workflows are registered lazily on first import of the
    workflow submodules.
    """
    # Lazy-load all workflow modules to populate the registry
    from drbrain.reasoning import (  # noqa: F401
        causal,
        contradiction,
        gap_analysis,
        hypothesis_wf,
        impact,
        review,
        temporal,
    )

    if name not in _WORKFLOW_REGISTRY:
        available = ", ".join(sorted(_WORKFLOW_REGISTRY.keys()))
        raise ValueError(f"Unknown workflow '{name}'. Available: {available}")
    return _WORKFLOW_REGISTRY[name]()


def list_workflows() -> list[dict[str, str]]:
    """Return metadata for all registered workflows."""
    from drbrain.reasoning import (  # noqa: F401
        causal,
        contradiction,
        gap_analysis,
        hypothesis_wf,
        impact,
        review,
        temporal,
    )

    return [
        {"name": cls.name, "description": cls.description} for cls in _WORKFLOW_REGISTRY.values()
    ]
