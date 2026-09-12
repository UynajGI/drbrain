"""Function-calling model adapter, independent of agent assembly and sessions."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from drbrain.config import Config
from drbrain.rag.agent_defaults import AGENT_MAX_TOKENS, AGENT_TEMPERATURE, CANONICAL_TOOL_SPECS
from drbrain.rag.llm import DrbrainLLM

try:
    from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
    from llama_index.core.llms.function_calling import FunctionCallingLLM
    from llama_index.core.llms.llm import ToolSelection
    from llama_index.core.tools import BaseTool
except ImportError:
    if not TYPE_CHECKING:
        ChatMessage = ChatResponse = MessageRole = ToolSelection = BaseTool = None
        FunctionCallingLLM = object


class AgentFunctionLLM(DrbrainLLM, FunctionCallingLLM):
    """``FunctionCallingLLM`` adapter over ``DrbrainLLM`` for the agent loop.

    ``DrbrainLLM`` (rag/llm.py) is T2-owned and deliberately not touched: it
    advertises ``is_function_calling_model=False`` and drops ``tools``. This
    subclass adds exactly the FunctionAgent contract — advertises function
    calling, forwards OpenAI-format tool specs through the same drbrain
    fallback chain (``llm_client.acall_with_messages``), and round-trips
    assistant ``tool_calls`` / tool ``tool_call_id`` messages — mirroring the
    equivalent LlamaIndex function-calling implementation.
    """

    def __init__(
        self,
        cfg: Config,
        temperature: float = AGENT_TEMPERATURE,
        max_tokens: int = AGENT_MAX_TOKENS,
        models_override: list[dict] | None = None,
        **kwargs: Any,
    ) -> None:
        # Explicit so mypy uses this signature instead of synthesizing one from
        # the (multiple-inheritance) pydantic base's fields.
        super().__init__(cfg, temperature=temperature, max_tokens=max_tokens, **kwargs)
        if models_override:
            # per-node 模型覆盖（llm.node_models[step]）：同样走 resolve_agent_key
            # 保持 key 轮换语义；覆盖 DrbrainLLM 从 cfg.llm.models 设置的全局链。
            from drbrain.extractor.llm_client import resolve_agent_key

            self._models = [resolve_agent_key(m) for m in models_override]

    @property
    def metadata(self) -> Any:
        md = super().metadata
        md.is_function_calling_model = True
        return md

    def _prepare_chat_with_tools(
        self,
        tools: Sequence[BaseTool],
        user_msg: str | ChatMessage | None = None,
        chat_history: list[ChatMessage] | None = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Prefer the canonical drbrain OpenAI-format specs (identical to what
        # the legacy ReasonerAgent sent); fall back to the tool's own schema.
        tool_specs = [
            CANONICAL_TOOL_SPECS.get(tool.metadata.name or "")
            or tool.metadata.to_openai_tool(skip_length_check=True)
            for tool in tools
        ]
        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)
        messages = list(chat_history or [])
        if user_msg is not None:
            messages.append(user_msg)
        return {"messages": messages, "tools": tool_specs or None}

    def get_tool_calls_from_response(
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> list[ToolSelection]:
        """Parse ``ChatResponse.message.additional_kwargs["tool_calls"]``."""
        tool_calls = response.message.additional_kwargs.get("tool_calls", [])
        if len(tool_calls) < 1:
            if error_on_no_tool_call:
                raise ValueError(f"Expected at least one tool call, but got {len(tool_calls)}.")
            return []
        selections: list[ToolSelection] = []
        for tool_call in tool_calls:
            if tool_call.get("type") != "function" or "function" not in tool_call:
                raise ValueError(f"Invalid tool call of type {tool_call.get('type')}")
            fn = tool_call.get("function", {})
            arguments = fn.get("arguments")
            try:
                argument_dict = json.loads(arguments) if arguments else {}
            except (ValueError, TypeError, json.JSONDecodeError):
                argument_dict = {}
            if fn.get("name"):
                selections.append(
                    ToolSelection(
                        tool_id=tool_call.get("id") or f"call_{len(selections)}",
                        tool_name=fn.get("name"),
                        tool_kwargs=argument_dict,
                    )
                )
        if not selections and error_on_no_tool_call:
            raise ValueError("No valid tool calls found.")
        return selections

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Any:
        """Forward ``tools`` (OpenAI-format) to the drbrain fallback chain."""
        tools = kwargs.pop("tools", None)
        if not tools:
            return await super().achat(messages, **kwargs)
        from drbrain.extractor.llm_client import acall_with_messages

        result = await acall_with_messages(
            self._to_openai_messages(messages),
            self._models,
            tools=tools,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            _cache=self._get_cache(),
        )
        return self._chat(result)

    @staticmethod
    def _to_openai_messages(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
        """OpenAI chat dicts preserving tool protocol fields + block content."""
        out: list[dict[str, Any]] = []
        for msg in messages:
            role = getattr(msg.role, "value", str(msg.role))
            content = msg.content
            if content is None:  # tool-result messages carry ContentBlocks
                parts = []
                for block in getattr(msg, "blocks", None) or []:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                content = "\n".join(parts)
            item: dict[str, Any] = {"role": role, "content": content or ""}
            ak = getattr(msg, "additional_kwargs", None) or {}
            if ak.get("tool_calls"):
                item["tool_calls"] = ak["tool_calls"]
            if ak.get("tool_call_id"):
                item["tool_call_id"] = ak["tool_call_id"]
            out.append(item)
        return out


# ── agent assembly ──────────────────────────────────────────────────────────
