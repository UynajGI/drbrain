# Structured Reasoning Workflows

DrBrain ships 7 pre-built reasoning workflows that compose symbolic graph computation
with LLM semantic judgment. Each runs as a deterministic pipeline of steps accessible
via `drbrain reason --workflow`.

## Overview

Workflows address a common problem: answering research questions requires multiple
coordinated steps (collect data → analyze → synthesize), not a single LLM call.
Workflows automate this by chaining symbolic steps (fast, deterministic) with LLM
steps (for synthesis and narrative), caching results at each level.

A workflow is defined in `src/drbrain/reasoning/` as a sequence of `WorkflowStep`
instances orchestrated by a `ReasoningWorkflow`. Each step reads prior outputs via
`ctx.get(step_name)`, produces a named result, and the workflow returns all results
keyed by step name.

```
Question → [Step 1: Collect] → [Step 2: Analyze] → [Step 3: Classify] → [Step 4: LLM Synthesize] → Result
              ↑                                    ↑                      ↑
         Symbolic (fast)                    Symbolic (fast)          LLM (semantic)
```

All results are cached per (workflow, question, graph state) key.

## User Guide

### review

Generates a structured literature review from the knowledge graph.

**Pipeline**: collect papers → identify themes (seed detection) → extract causal chains → LLM synthesis

```bash
drbrain reason --workflow review "self-attention mechanisms in transformers"
drbrain reason --workflow review p6a321e    # focus on one paper
```

### gap-analysis

Identifies and prioritizes open research problems.

**Pipeline**: detect gaps → classify difficulty → score by impact → LLM research agenda

```bash
drbrain reason --workflow gap-analysis -w nlp
drbrain reason --workflow gap-analysis "graph neural networks" --session sess-xxx
```

### impact

Assesses the research impact of a concept, paper, or method.

**Pipeline**: find neighbors → trace causal influence → measure graph centrality → LLM impact report

```bash
drbrain reason --workflow impact "Transformer architecture"
drbrain reason --workflow impact --session sess-xxx "Dropout"
```

### compare

Compares two or more concepts, methods, or papers across multiple dimensions.

**Pipeline**: collect both sides → find shared/differing neighbors → detect debates → LLM comparison

```bash
drbrain reason --workflow compare "Transformer vs LSTM for sequence modeling"
drbrain reason --workflow compare "SGD" "Adam" --session sess-xxx
```

### frontier

Composite knowledge frontier report: seeds + debates + cliffs + difficulty + confidence collapse.

**Pipeline**: multi-signal scan → classify patterns → LLM frontier summary

```bash
drbrain reason --workflow frontier -w nlp
drbrain reason --workflow frontier "large language models"
```

### lineage

Traces the evolution of a concept through the knowledge graph.

**Pipeline**: BFS ancestors → BFS descendants → detect paradigm signals → LLM lineage narrative

```bash
drbrain reason --workflow lineage "Attention mechanism"
drbrain reason --workflow lineage "ReLU activation"
```

### paradigm

Detects paradigm shifts — replacement, explosion, or cross-domain invasion patterns.

**Pipeline**: paper-age analysis → cluster emerging/declining → classify shift type → LLM narrative

```bash
drbrain reason --workflow paradigm -w nlp
drbrain reason --workflow paradigm "deep learning"
```

## Workflow Internals

### Architecture

Each workflow is a `ReasoningWorkflow` subclass (in `src/drbrain/reasoning/`) composed
of `WorkflowStep` instances. Steps access a shared `WorkflowContext` that carries the DB
connection, graph engine, LLM models, user question, and all prior step outputs.

```python
# src/drbrain/reasoning/base.py
@dataclass
class WorkflowContext:
    db: Any                # Database connection
    graph: Any             # GraphEngine
    models: list[dict]     # LLM model config
    question: str          # User question
    cache: ApiCache | None # Optional result cache
    results: dict          # Step outputs keyed by step name

    def get(self, step_name: str, default=None) -> Any:
        """Retrieve a previous step's output."""

class WorkflowStep(ABC):
    name: str = ""
    requires_llm: bool = False

    @abstractmethod
    def run(self, ctx: WorkflowContext) -> Any: ...

class ReasoningWorkflow:
    name: str = ""
    description: str = ""
    steps: list[WorkflowStep] = []

    def execute(self, ctx: WorkflowContext) -> dict[str, Any]:
        """Run all steps, managing cache and error handling."""
```

### Step Execution

Steps run sequentially. Each step's `run()` receives the context with all prior outputs.
On step failure, the result is set to `None` and execution continues to subsequent steps.

### Result Caching

Two levels of caching:

1. **Workflow-level** (in `ReasoningWorkflow.execute()`): A cache key is built from
   workflow name + question + graph node/edge count + DB paper count. A full hit skips
   all steps. Store only if all steps succeeded.

2. **API-level** (in LLM calls): `ApiCache` checks before calling LLM. Cache is disabled
   when `temperature > 0` (non-deterministic outputs can't be cached).

```python
# Cache key construction
cache_key = f"wf:{question}:{edge_count}:{node_count}:{paper_count}"
```

### Workflow Registry

Workflows register via the `@register_workflow(name)` decorator. Lazy-loading on first
import keeps cold-start fast:

```python
@register_workflow("review")
class ReviewWorkflow(ReasoningWorkflow):
    name = "review"
    description = "Generate a structured survey/review from KG papers and evidence"
    steps = [_CollectPapersStep(), _IdentifyThemesStep(), _ExtractEvidenceStep(), _GenerateReviewStep()]
```

## Creating a New Workflow

### Step 1: Create the workflow module

Create `src/drbrain/reasoning/my_workflow.py`:

```python
from __future__ import annotations
from typing import Any
from drbrain.reasoning.base import ReasoningWorkflow, WorkflowContext, WorkflowStep, register_workflow


class _GatherDataStep(WorkflowStep):
    """Collect relevant data from the KG."""
    name = "gather_data"
    requires_llm = False

    def run(self, ctx: WorkflowContext) -> dict[str, Any]:
        concepts = ctx.db.conn.execute(
            "SELECT label, type FROM concepts WHERE type = ?", ("Method",)
        ).fetchall()
        return {"method_count": len(concepts), "methods": [r[0] for r in concepts[:20]]}


class _AnalyzeStep(WorkflowStep):
    """Analyze the gathered data."""
    name = "analyze"
    requires_llm = False

    def run(self, ctx: WorkflowContext) -> dict[str, Any]:
        data = ctx.get("gather_data", {})
        # ... compute analysis ...
        return {"top_findings": [...]}


class _SynthesizeStep(WorkflowStep):
    """LLM synthesizes final output."""
    name = "synthesize"
    requires_llm = True

    def run(self, ctx: WorkflowContext) -> str:
        from drbrain.extractor.llm_client import call_text_with_fallback
        data = ctx.get("gather_data", {})
        analysis = ctx.get("analyze", {})
        prompt = f"Based on this data: {data} and analysis: {analysis}..."
        return call_text_with_fallback(prompt, ctx.models, max_tokens=1024) or ""


@register_workflow("my-analysis")
class MyWorkflow(ReasoningWorkflow):
    name = "my-analysis"
    description = "My custom analysis workflow"
    steps = [_GatherDataStep(), _AnalyzeStep(), _SynthesizeStep()]
```

### Step 2: Register in `base.py`

Add the import in `get_workflow()` and `list_workflows()` functions (in `src/drbrain/reasoning/base.py`):

```python
from drbrain.reasoning import my_workflow  # noqa: F401
```

### Step 3: Add tests

Create `tests/test_my_workflow.py`:

```python
from drbrain.reasoning.my_workflow import MyWorkflow
from drbrain.reasoning.base import WorkflowContext
from drbrain.storage.database import Database
from drbrain.graph.engine import GraphEngine

def test_my_workflow():
    db = Database(":memory:")
    graph = GraphEngine()
    db.insert_concept("p1", "Method", "MyMethod", 0.9)
    graph.load_from_db(db)

    wf = MyWorkflow()
    ctx = WorkflowContext(db=db, graph=graph, models=[], question="test")
    results = wf.execute(ctx)

    assert "gather_data" in results
    assert results["gather_data"]["method_count"] == 1
```

### Step 4: Document

Add your workflow in this file under "User Guide" and mention it in `docs/cli-reference.md`
under `drbrain reason --workflow`.

### Design Principles

- **Symbolic first**: Use graph/DB computation whenever possible. LLM only for semantic judgment.
- **Each step is independently testable**: Steps can be tested with a DB + graph, no LLM needed for `requires_llm=False` steps.
- **Don't hardcode question handling**: The user's question is in `ctx.question`. Let the LLM interpret it.
- **Cache when deterministic**: Set `requires_llm=False` for steps that produce the same output given the same state.

## Workflow Visualizer

The visualizer (`--workflow` flag implicitly uses it) renders pipeline diagrams and
result summaries to help understand what each step produced.

Output example:
```
╔══════════════════════════════════════════╗
║  Workflow: review                       ║
╠══════════════════════════════════════════╣
║  Step 1: collect_papers    ✓ (0.02s)    ║
║  Step 2: identify_themes   ✓ (0.15s)    ║
║  Step 3: extract_evidence  ✓ (0.08s)    ║
║  Step 4: generate_review   ✓ (2.30s)    ║
╚══════════════════════════════════════════╝
```

Visualizer code is in `src/drbrain/reasoning/visualizer.py` (if present) or embedded
in the session agent output path.
