"""Model-tool registry: register models, invoke them, expose them to the agent.

The registry is the single entry point an agent (or a workflow step) uses to
call a model. It keeps the model-specific handler *out* of the reasoning path:
the agent only ever sees :class:`ModelTool` descriptors and :class:`ModelResult`
envelopes.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

from drbrain.rag.plugins.protocol import (
    ModelResult,
    ModelTool,
    ResultStatus,
    make_evidence,
)

_JSON_TO_PY: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _input_schema_to_model(tool: ModelTool) -> type | None:
    """Build a pydantic model from ``tool.input_schema`` for ``fn_schema``.

    Handles the common flat ``object`` schema (typed top-level properties);
    returns ``None`` for anything more complex, in which case the bridge falls
    back to a bare tool (no schema validation).
    """
    try:
        from pydantic import create_model
    except ImportError:
        return None
    schema = tool.input_schema or {}
    properties = schema.get("properties") or {}
    if schema.get("type") != "object" or not properties:
        return None
    required = set(schema.get("required") or [])
    fields: dict[str, Any] = {}
    for key, prop in properties.items():
        ptype = _JSON_TO_PY.get(prop.get("type"), Any)
        fields[key] = (ptype, ... if key in required else None)
    return create_model(f"{tool.name}_Input", **fields)


class ModelToolRegistry:
    """Holds registered model tools and invokes them with degradation semantics.

    ``handler`` is ``Callable[[dict], Any]``: it takes the JSON arguments and
    returns the raw result data (or raises). The registry wraps it with a
    timeout and classifies exceptions into :class:`ResultStatus`.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ModelTool] = {}
        self._handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}

    def register(self, tool: ModelTool, handler: Callable[[dict[str, Any]], Any]) -> None:
        """Register a model tool and its handler (idempotent: re-register replaces)."""
        if not tool.name:
            raise ValueError("tool name must be non-empty")
        self._tools[tool.name] = tool
        self._handlers[tool.name] = handler

    def get(self, name: str) -> ModelTool:
        """Return a registered tool descriptor (KeyError if unknown)."""
        return self._tools[name]

    def list_tools(self) -> list[ModelTool]:
        """Return all registered tool descriptors (registration order)."""
        return list(self._tools.values())

    def call(self, name: str, arguments: dict[str, Any]) -> ModelResult:
        """Invoke ``name`` with ``arguments`` and wrap the outcome.

        Never raises for model-side failures: a broken/missing model yields a
        ``MODEL_UNAVAILABLE`` result (the caller abstains) instead of an
        exception propagating into the reasoning loop.
        """
        tool = self._tools[name]
        handler = self._handlers[name]
        evidence = make_evidence(tool, arguments)

        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(handler, arguments)
            data = future.result(timeout=tool.timeout_s)
        except (FutureTimeout, TimeoutError):
            return ModelResult(
                ResultStatus.TIMEOUT,
                evidence=evidence,
                error=f"调用超时(>{tool.timeout_s}s)",
            )
        except (ValueError, TypeError, KeyError) as exc:
            return ModelResult(
                ResultStatus.INVALID_INPUT,
                evidence=evidence,
                error=f"输入不符合 schema: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 — model failure must not propagate
            return ModelResult(
                ResultStatus.MODEL_UNAVAILABLE,
                evidence=evidence,
                error=str(exc),
            )
        finally:
            pool.shutdown(wait=False)

        if data is None:
            return ModelResult(ResultStatus.NO_RESULT, evidence=evidence, error="模型无输出")
        evidence["output"] = data
        return ModelResult(ResultStatus.OK, data=data, evidence=evidence)

    def to_llamaindex_tools(self) -> list:
        """Bridge every registered tool to a LlamaIndex ``FunctionTool``.

        Returns ``[]`` when llama-index is unavailable. Each tool's handler
        calls back into :meth:`call` so degradation semantics and evidence
        provenance are preserved end-to-end.
        """
        try:
            from llama_index.core.tools import FunctionTool
        except ImportError:
            return []

        tools = []
        for tool in self._tools.values():

            def _fn(**kwargs: Any) -> str:
                result = self.call(tool.name, dict(kwargs))
                return result.to_llm_message(tool)

            tools.append(
                FunctionTool.from_defaults(
                    fn=_fn,
                    name=tool.name,
                    description=tool.description,
                    fn_schema=_input_schema_to_model(tool),
                )
            )
        return tools
