"""Workflow visualization — generates Mermaid flowcharts and result summaries.

Two modes:
1. **Pipeline visualization**: shows the step DAG before/after execution
2. **Result summarization**: condenses each step's output into a readable format

Usage::

    from drbrain.reasoning.visualizer import WorkflowVisualizer

    vz = WorkflowVisualizer(wf)
    print(vz.mermaid_flowchart())      # Mermaid diagram of the pipeline
    print(vz.text_flowchart())          # ASCII art flowchart
    print(vz.summarize_results(ctx))    # Human-readable result summary
"""

from __future__ import annotations

from typing import Any

from drbrain.reasoning.base import ReasoningWorkflow, WorkflowContext


class WorkflowVisualizer:
    """Generate visual representations of a workflow and its results."""

    def __init__(self, workflow: ReasoningWorkflow):
        self.wf = workflow

    # ── Mermaid ───────────────────────────────────────────────────────

    def mermaid_flowchart(self) -> str:
        """Generate a Mermaid flowchart of the workflow pipeline.

        Each step is a node. Symbolic steps are blue, LLM steps are orange.
        Arrows show execution order. Failed steps (if results available) are red.
        """
        lines = ["```mermaid", "flowchart TD"]

        for i, step in enumerate(self.wf.steps):
            node_id = f"s{i}"
            label = step.name.replace("_", " ")
            icon = "🤖" if step.requires_llm else "⚙️"

            # Determine style based on step type
            if step.requires_llm:
                lines.append(f"    {node_id}[{icon} {label}]")
                lines.append(f"    style {node_id} fill:#fff3e0,stroke:#ff9800")
            else:
                lines.append(f"    {node_id}[{icon} {label}]")
                lines.append(f"    style {node_id} fill:#e3f2fd,stroke:#2196f3")

            # Arrow to next step
            if i < len(self.wf.steps) - 1:
                lines.append(f"    {node_id} --> s{i + 1}")

        # Add start/end markers
        lines.insert(2, "    start((Question))")
        lines.insert(3, "    style start fill:#c8e6c9,stroke:#4caf50")
        if self.wf.steps:
            lines.append("    start --> s0")
            last_id = f"s{len(self.wf.steps) - 1}"
            lines.append(f"    {last_id} --> output{{Answer}}")
            lines.append("    style output fill:#c8e6c9,stroke:#4caf50")

        lines.append("```")
        return "\n".join(lines)

    # ── ASCII text flowchart ──────────────────────────────────────────

    def text_flowchart(self) -> str:
        """Generate an ASCII flowchart of the pipeline."""
        width = 50
        lines = [
            "",
            f"  ┌{'─' * width}┐",
            f"  │  Workflow: {self.wf.name:<{width - 13}}│",
            f"  │  {self.wf.description[:width]:<{width}}│",
            f"  └{'─' * width}┘",
            "           │",
            "           ▼",
        ]

        for i, step in enumerate(self.wf.steps):
            icon = "🤖 LLM" if step.requires_llm else "⚙️  SYM"
            label = step.name.replace("_", " ")
            is_last = i == len(self.wf.steps) - 1

            box_width = max(len(label) + 12, 20)
            inner = "─" * box_width
            lines.append(f"       ┌{inner}┐")
            lines.append(f"       │ {icon} │ {label:<{box_width - 8}}│")
            lines.append(f"       └{inner}┘")
            if not is_last:
                lines.append("           │")
                lines.append("           ▼")

        lines.append("           │")
        lines.append("           ▼")
        lines.append(f"       ┌{'─' * width}┐")
        lines.append(f"       │  ✅ Result{' ' * (width - 9)}│")
        lines.append(f"       └{'─' * width}┘")
        lines.append("")
        return "\n".join(lines)

    # ── Result summarization ──────────────────────────────────────────

    def summarize_results(self, ctx: WorkflowContext) -> str:
        """Generate a human-readable summary of each step's output.

        For each step, shows:
        - Step name and type (symbolic/LLM)
        - A condensed view of the result (not raw dump)
        - Execution status (success/failure)
        """
        lines = [
            "",
            f"  ╔{'═' * 56}╗",
            f"  ║  Pipeline Results: {self.wf.name:<35}║",
            f"  ╚{'═' * 56}╝",
            "",
        ]

        for step in self.wf.steps:
            result = ctx.results.get(step.name)
            icon = "🤖" if step.requires_llm else "⚙️ "
            status = "✅" if result is not None else "❌"

            lines.append(f"  {status} {icon} {step.name}")

            if result is None:
                lines.append("       → (no output / failed)")
            else:
                summary = self._summarize_value(result)
                for line in summary:
                    lines.append(f"       {line}")

            lines.append("")

        return "\n".join(lines)

    def _summarize_value(self, value: Any, max_items: int = 5) -> list[str]:
        """Condense a result value into readable summary lines."""
        if isinstance(value, str):
            # Truncate long strings
            if len(value) > 200:
                return [f"→ {value[:200]}..."]
            return [f"→ {value}"]

        if isinstance(value, list):
            count = len(value)
            if count == 0:
                return ["→ (empty)"]
            lines = [f"→ {count} item(s)"]
            for item in value[:max_items]:
                if isinstance(item, dict):
                    # Extract the most informative field
                    key = (
                        item.get("concept")
                        or item.get("label")
                        or item.get("node")
                        or item.get("description", "")
                    )
                    score = item.get("score", item.get("impact", ""))
                    suffix = f" (score: {score})" if score else ""
                    lines.append(f"   • {str(key)[:60]}{suffix}")
                else:
                    lines.append(f"   • {str(item)[:60]}")
            if count > max_items:
                lines.append(f"   ... and {count - max_items} more")
            return lines

        if isinstance(value, dict):
            lines = []
            for k, v in list(value.items())[:max_items]:
                if isinstance(v, (list, dict)):
                    lines.append(f"   {k}: {len(v)} items")
                elif isinstance(v, str) and len(v) > 60:
                    lines.append(f"   {k}: {v[:60]}...")
                else:
                    lines.append(f"   {k}: {v}")
            if len(value) > max_items:
                lines.append(f"   ... and {len(value) - max_items} more fields")
            return lines

        return [f"→ {str(value)[:100]}"]
