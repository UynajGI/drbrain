"""LLM Agent for graph reasoning with tool-calling loop."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from drbrain.extractor.agent_tools import (
    TOOL_DEFINITIONS,
    find_path,
    get_document_structure,
    get_neighbors,
    get_raptor_summaries,
    get_section_content,
    kg_validate,
    search_concepts,
    search_tree,
)

log = logging.getLogger(__name__)


class ReasonerAgent:
    """Agent that explores a knowledge graph using LLM tool-calling."""

    def __init__(self, db=None, graph_engine=None, models=None, closure_context: str = ""):
        self.db = db
        self.graph = graph_engine
        self.models = models or []
        self.closure_context = closure_context

    @property
    def _papers_dir(self) -> Path | None:
        """Resolve the papers data directory from DB config."""
        if not self.db:
            return None
        return self.db.path.parent / "papers"

    def tool_definitions(self) -> list[dict]:
        return list(TOOL_DEFINITIONS)

    # -- Tool wrappers (delegate to shared agent_tools handlers) --

    def _search_concepts(self, query: str, limit: int = 5) -> list[dict]:
        return search_concepts(self.db, query, limit)

    def _get_neighbors(self, node: str, hops: int = 1, direction: str = "both") -> list[dict]:
        return get_neighbors(self.graph, node, hops, direction)

    def _find_path(self, src: str, dst: str) -> dict | None:
        return find_path(self.graph, src, dst)

    def _get_document_structure(self, paper_id: str) -> list[dict]:
        return get_document_structure(self._papers_dir, paper_id)

    def _get_section_content(self, paper_id: str, node_id: str) -> str:
        return get_section_content(self._papers_dir, paper_id, node_id)

    def _search_tree(self, query: str) -> list[dict]:
        return search_tree(self.db, query)

    def _get_raptor_summaries(self, paper_id: str) -> list[dict]:
        return get_raptor_summaries(self.db, paper_id)

    async def reason(self, question: str, max_turns: int = 5) -> str:
        """Run LLM agent loop to reason about a question using graph tools."""
        import time as _rtime

        _rt0 = _rtime.monotonic()
        log.info("[reasoner] starting — question=%.80s max_turns=%d", question, max_turns)

        if not self.models:
            return "No LLM models configured."

        tools = self.tool_definitions()
        system_content = (
            "You are a knowledge graph reasoning assistant. "
            "Use the provided tools to explore the graph and answer questions. "
            "Explain your reasoning step by step."
        )
        if self.closure_context:
            system_content += (
                "\n\nInferred relations from logical closure (distinguished by --[inferred: ...]-->):\n"
                + self.closure_context
            )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question},
        ]

        for _ in range(max_turns):
            msg = None
            last_error = None
            for model in self.models:
                try:
                    import litellm

                    name = f"{model['provider']}/{model['model']}"
                    kwargs = {
                        "model": name,
                        "messages": messages,
                        "temperature": 0.3,
                        "max_tokens": 1024,
                        "timeout": 60,
                        "tools": tools,
                        "extra_body": {"thinking": {"type": "disabled"}},
                    }
                    if model.get("api_key"):
                        kwargs["api_key"] = model["api_key"]
                    if model.get("base_url"):
                        kwargs["api_base"] = model["base_url"]

                    resp = await litellm.acompletion(**kwargs)
                    msg = resp.choices[0].message
                    break  # success
                except Exception as e:
                    last_error = e
                    log.warning("[reasoner] model %s failed: %s", model.get("model"), e)
                    continue

            if msg is None:
                return f"Reasoning error: {last_error}"

            if msg.tool_calls:
                _called = [tc.function.name for tc in msg.tool_calls]
                log.info("[reasoner] tool calls: %s", _called)
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    if tc.function.name == "search_concepts":
                        result = self._search_concepts(**args)
                    elif tc.function.name == "get_neighbors":
                        result = self._get_neighbors(**args)
                    elif tc.function.name == "find_path":
                        result = self._find_path(**args)
                    elif tc.function.name == "get_document_structure":
                        result = self._get_document_structure(**args)
                    elif tc.function.name == "get_section_content":
                        result = self._get_section_content(**args)
                    elif tc.function.name == "search_tree":
                        result = self._search_tree(**args)
                    elif tc.function.name == "get_raptor_summaries":
                        result = self._get_raptor_summaries(**args)
                    else:
                        result = []
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result, ensure_ascii=False, default=str),
                        }
                    )
            else:
                _content = msg.content or "No answer generated."
                log.info(
                    "[reasoner] done in %.1fs — answer=%d chars",
                    _rtime.monotonic() - _rt0,
                    len(_content),
                )
                return _content

        log.warning("[reasoner] max turns (%d) exhausted", max_turns)
        return "Unable to answer after maximum reasoning turns."

    def _call_llm(self, prompt: str, system: str | None = None) -> str | None:
        """Call LLM with a prompt and return text response (no tool-calling).

        Iterates through all configured models with fallback.

        Args:
            prompt: User message to send.
            system: Optional system prompt override.

        Returns:
            LLM text response or None on failure.
        """
        if not self.models:
            return None

        import litellm

        system_content = system or (
            "You are a knowledge graph reasoning assistant. Answer concisely based on evidence."
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

        for i, model in enumerate(self.models):
            name = f"{model['provider']}/{model['model']}"
            try:
                kwargs = {
                    "model": name,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 1024,
                    "timeout": 60,
                    "extra_body": {"thinking": {"type": "disabled"}},
                }
                if model.get("api_key"):
                    kwargs["api_key"] = model["api_key"]
                if model.get("base_url"):
                    kwargs["api_base"] = model["base_url"]
                resp = litellm.completion(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception:
                log.warning("Model %s failed (attempt %d/%d)", name, i + 1, len(self.models))

        log.error("All %d models failed in _call_llm", len(self.models))
        return None

    def _kg_validate(self, hypothesis: str) -> dict:
        """Check hypothesis against KG for consistency.

        Delegates to the shared kg_validate function in agent_tools.
        """
        return kg_validate(hypothesis, db=self.db, graph=self.graph)

    def reason_bidirectional(self, question: str, max_rounds: int = 3) -> dict:
        """Iterative LLM-KG reasoning loop.

        Each round:
        1. LLM proposes hypothesis based on question + previous KG feedback
        2. KG validates hypothesis via TBox/RBox consistency check
        3. If contradiction: feed contradiction back to LLM, repeat
        4. KG detects graph patterns (gaps, contradictions, debates) -> feeds to LLM

        Args:
            question: The research question to reason about.
            max_rounds: Maximum number of hypothesis-revision rounds.

        Returns:
            {"answer": str, "rounds": int, "hypotheses": [...], "kg_validations": [...]}
        """
        if not self.models:
            return {"error": "No LLM models configured."}

        hypotheses: list[str] = []
        validations: list[dict] = []
        previous_violations: list[str] = []
        previous_patterns: list[str] = []

        for round_num in range(1, max_rounds + 1):
            # Build prompt
            if round_num == 1:
                prompt = (
                    f"Question: {question}\n\n"
                    "Based on your knowledge, propose a clear hypothesis that answers "
                    "this question. Include specific entities and relationships "
                    "(e.g., 'Method X addresses Problem Y', 'Finding A supports Conclusion B')."
                )
                if self.closure_context:
                    prompt += (
                        "\n\nInferred relations from logical closure "
                        "(distinguished by --[inferred: ...]-->):\n" + self.closure_context
                    )
            else:
                violation_text = (
                    "\n".join(f"- {v}" for v in previous_violations)
                    if previous_violations
                    else "None"
                )
                pattern_text = (
                    "\n".join(f"- {p}" for p in previous_patterns) if previous_patterns else "None"
                )
                prompt = (
                    f"Question: {question}\n\n"
                    f"Your previous hypothesis was:\n"
                    f'  "{hypotheses[-1]}"\n\n'
                    f"The knowledge graph found these issues:\n"
                    f"  Violations: {violation_text}\n"
                    f"  Patterns: {pattern_text}\n\n"
                    "Please revise your hypothesis to address these issues. "
                    "Propose a new hypothesis that is consistent with the graph constraints. "
                    "If you cannot resolve the contradictions, acknowledge the uncertainty."
                )

            hypothesis = self._call_llm(prompt)
            if hypothesis is None:
                return {
                    "answer": "LLM call failed during bidirectional reasoning.",
                    "rounds": round_num,
                    "hypotheses": hypotheses,
                    "kg_validations": validations,
                }

            hypotheses.append(hypothesis)

            # Validate against KG
            validation = self._kg_validate(hypothesis)
            validations.append(validation)

            # Collect violations and patterns for next round
            previous_violations = [v["reason"] for v in validation["violations"]]
            previous_patterns = [p["description"] for p in validation["patterns"]]

            if validation["consistent"]:
                return {
                    "answer": hypothesis,
                    "rounds": round_num,
                    "hypotheses": hypotheses,
                    "kg_validations": validations,
                }

        # Exhausted all rounds
        best = hypotheses[-1] if hypotheses else "No hypothesis generated."
        return {
            "answer": best,
            "rounds": max_rounds,
            "hypotheses": hypotheses,
            "kg_validations": validations,
        }
