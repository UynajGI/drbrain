"""Agent-facing tool spaces for the research loop.

The workflow owns *when* a node runs.  This module owns the much smaller
question of *which* capabilities that node may see.  A tool space is created
per node, while its capability catalog is shared for the lifetime of a
workflow, so discovery is stable and the model can remain autonomous inside a
host-owned boundary.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, cast

from drbrain.capabilities import CapabilityCatalog, CapabilityDescriptor
from drbrain.loop.policy import ToolDefinition, ToolPolicy

# These roles deliberately have a smaller surface than their prompts alone
# would imply.  Prompt instructions are useful guidance; this table is the
# enforceable boundary used while tools are assembled.
_ROLE_DENIED_SOURCES: dict[str, frozenset[str]] = {
    "compute": frozenset({"graph", "rag"}),
}
_TOOL_FREE_ROLES = frozenset({"analyst", "critic"})

_KNOWN_SIDE_EFFECTS = frozenset({"pure", "read", "write", "irreversible", "unspecified"})


class LoopToolSpace:
    """Resolve and bind one node's visible capabilities.

    ``catalog`` is optional for compatibility with the legacy RAG agent.  When
    supplied, catalog entries (including API, CLI, model and MCP adapters) are
    exposed through the same policy and broker boundary as graph/RAG tools.
    Skills remain instruction context rather than executable functions.
    """

    def __init__(
        self,
        *,
        step_name: str,
        role: str | None = None,
        policy: ToolPolicy | None = None,
        broker: Any = None,
        catalog: CapabilityCatalog | None = None,
    ) -> None:
        self.step_name = str(step_name or "")
        self.role = str(role or "").strip() or None
        self.policy = policy
        self.broker = broker
        self.catalog = catalog

    @classmethod
    def discover_catalog(
        cls,
        *,
        catalog: CapabilityCatalog | None = None,
        plugins_dir: str | Path | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
        skills_root: str | Path | None = None,
        adapters: Iterable[Any] | None = None,
        require_trusted_mcp: bool = False,
    ) -> CapabilityCatalog | None:
        """Build one shared catalog from all configured capability sources.

        Discovery is best effort at this boundary.  A broken optional source
        must not prevent graph/RAG tools from assembling; policy and execution
        still fail closed for anything that did not produce a descriptor.
        """
        source_requested = bool(
            catalog is not None or plugins_dir or mcp_servers or skills_root or adapters
        )
        if not source_requested:
            return None
        result = catalog or CapabilityCatalog()
        if plugins_dir:
            try:
                from drbrain.plugins.registry import PluginRegistry

                registry = PluginRegistry()
                registry.discover(plugins_dir)
                result.register_plugin_registry(registry, replace=True)
            except Exception:
                # Optional external sources are isolated; the caller keeps the
                # catalog entries that were already supplied explicitly.
                pass
        if mcp_servers:
            try:
                result.register_mcp_servers(
                    mcp_servers,
                    require_trusted=require_trusted_mcp,
                    replace=True,
                )
            except Exception:
                pass
        if skills_root:
            try:
                result.register_skills(str(skills_root), replace=True)
            except Exception:
                pass
        for adapter in adapters or ():
            try:
                result.register_adapter(adapter, replace=True)
            except Exception:
                pass
        return result

    def child(self, *, step_name: str, role: str | None = None) -> LoopToolSpace:
        """Return another node view over the same discovered catalog."""
        return type(self)(
            step_name=step_name,
            role=role,
            policy=self.policy,
            broker=self.broker,
            catalog=self.catalog,
        )

    @staticmethod
    def _source_for_descriptor(descriptor: CapabilityDescriptor) -> str:
        kind = descriptor.kind.strip().lower()
        return "mcp" if kind == "mcp_tool" else kind

    @staticmethod
    def _side_effect_for_descriptor(
        descriptor: CapabilityDescriptor,
    ) -> Literal["pure", "read", "write", "irreversible", "unspecified"]:
        raw = descriptor.metadata.get("side_effect")
        if isinstance(raw, str) and raw in _KNOWN_SIDE_EFFECTS:
            return cast(Literal["pure", "read", "write", "irreversible", "unspecified"], raw)
        if descriptor.annotations.destructive is True:
            return "irreversible"
        if descriptor.annotations.read_only is True:
            return "read"
        if descriptor.annotations.read_only is False:
            return "write"
        if descriptor.kind == "model":
            return "pure"
        return "unspecified"

    @classmethod
    def definition_for_descriptor(cls, descriptor: CapabilityDescriptor) -> ToolDefinition:
        """Project a neutral descriptor onto the loop's durable policy shape."""
        source = cls._source_for_descriptor(descriptor)
        required = tuple(descriptor.permissions) or (f"{source}:{descriptor.name}",)
        metadata = descriptor.metadata
        allowed = metadata.get("allowed_tools", ())
        if isinstance(allowed, str):
            allowed = (allowed,)
        elif not isinstance(allowed, (list, tuple, set, frozenset)):
            allowed = ()
        return ToolDefinition(
            name=descriptor.id,
            source=source,
            input_schema=descriptor.input_schema,
            side_effect=cls._side_effect_for_descriptor(descriptor),
            required_capabilities=required,
            code_digest=descriptor.provenance.code_digest,
            version=descriptor.version,
            resource_scope=descriptor.resource_scope,
            secret_refs=tuple(str(item) for item in metadata.get("secret_refs", ()) or ()),
            max_output_bytes=(
                int(metadata["max_output_bytes"])
                if isinstance(metadata.get("max_output_bytes"), int)
                and not isinstance(metadata.get("max_output_bytes"), bool)
                else None
            ),
            cost_hint=(
                float(metadata["cost_hint"])
                if isinstance(metadata.get("cost_hint"), int | float)
                and not isinstance(metadata.get("cost_hint"), bool)
                else None
            ),
            supports_idempotency=descriptor.execution.supports_idempotency,
            supports_reconcile=descriptor.execution.supports_reconcile,
            supports_cancel=descriptor.execution.supports_cancel,
            sandbox_profile=str(metadata.get("sandbox_profile") or ""),
            approval_policy=str(metadata.get("approval_policy") or "default"),
            trusted=metadata.get("trusted") is True,
            allowed_tools=tuple(str(item) for item in allowed),
            timeout_s=descriptor.execution.timeout_seconds,
        )

    def is_visible(self, definition: ToolDefinition) -> bool:
        """Return whether a concrete tool belongs on this node's surface."""
        # Analyst and critic are deliberately tool-free. Checking the role
        # before the source list also closes the escape hatch for a future
        # adapter that introduces a new, otherwise unknown capability kind.
        if self.role in _TOOL_FREE_ROLES:
            return False
        if (
            self.role in _ROLE_DENIED_SOURCES
            and definition.source in _ROLE_DENIED_SOURCES[self.role]
        ):
            return False
        if self.policy is None:
            return True
        return self.policy.is_visible(node_name=self.step_name, definition=definition)

    def descriptor_is_visible(self, descriptor: CapabilityDescriptor) -> bool:
        return self.is_visible(self.definition_for_descriptor(descriptor))

    def recommend(
        self,
        query: str,
        *,
        kinds: Iterable[str] | None = None,
        limit: int = 10,
    ) -> list[CapabilityDescriptor]:
        """Recommend only capabilities the current node could actually use."""
        if self.catalog is None:
            return []
        # Rank the complete candidate set before filtering. A large number of
        # denied capabilities must not crowd a usable one out of the top-N
        # merely because it shared more query terms.
        candidates = self.catalog.recommend(
            query,
            kinds=kinds,
            limit=max(len(self.catalog.list()), max(0, limit)),
        )
        return [item for item in candidates if self.descriptor_is_visible(item)][: max(0, limit)]

    def skill_context(self, *, max_chars: int = 12000) -> str:
        """Render bounded Skill instructions for the current agent prompt."""
        if self.catalog is None:
            return ""
        chunks: list[str] = []
        used = 0
        for descriptor in self.catalog.list(kind="skill"):
            body = str(descriptor.metadata.get("body") or "").strip()
            if not body:
                continue
            block = f"\n## Skill: {descriptor.name}\n{body}\n"
            if used + len(block) > max_chars:
                remaining = max_chars - used
                if remaining > 80:
                    chunks.append(block[:remaining])
                break
            chunks.append(block)
            used += len(block)
        return "".join(chunks).strip()

    async def _invoke_catalog(
        self,
        descriptor: CapabilityDescriptor,
        arguments: dict[str, Any],
    ) -> str:
        if self.catalog is None:
            return "能力目录不可用"
        if self.broker is None:
            return (await self.catalog.ainvoke(descriptor.id, arguments)).to_llm_message()
        observation = await self.broker.execute(
            node_name=self.step_name,
            definition=self.definition_for_descriptor(descriptor),
            arguments=arguments,
            executor=lambda: self.catalog.ainvoke(descriptor.id, arguments),
        )
        return observation.to_llm_message()

    def catalog_tools(self) -> list[Any]:
        """Bind visible catalog capabilities to provider-safe FunctionTools."""
        if self.catalog is None:
            return []
        # Skills are prompt resources, never executable tools.
        kinds = {descriptor.kind for descriptor in self.catalog.list()} - {"skill"}
        if not kinds:
            return []
        return self.catalog.to_llamaindex_tools(
            kinds=kinds,
            include=self.descriptor_is_visible,
            call_override=self._invoke_catalog,
            # Keep the historical plugin tool spelling for existing loop
            # callers while MCP/API/model IDs remain visibly namespaced.
            name_for=lambda descriptor: (
                descriptor.name if descriptor.kind == "plugin" else descriptor.id
            ),
        )

    def manifest(self) -> dict[str, Any]:
        """Return a secret-free diagnostic view of this node's tool space."""
        descriptors = self.catalog.list() if self.catalog is not None else []
        return {
            "step": self.step_name,
            "role": self.role,
            "policy": self.policy.to_manifest() if self.policy is not None else None,
            "capabilities": [
                {
                    "id": item.id,
                    "kind": item.kind,
                    "visible": self.descriptor_is_visible(item),
                }
                for item in descriptors
            ],
        }

    def manifest_json(self) -> str:
        return json.dumps(self.manifest(), ensure_ascii=False, sort_keys=True)


__all__ = ["LoopToolSpace"]
