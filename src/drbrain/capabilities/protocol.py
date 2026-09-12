"""Stable, transport-neutral capability and invocation contracts."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

# Known values are documented for policy UIs, but the wire contract stays
# extensible so a new protocol can register a new kind without core changes.
CapabilityKind = str


@runtime_checkable
class CapabilityAdapter(Protocol):
    """Minimal extension point for a future tool protocol."""

    def descriptor(self) -> CapabilityDescriptor: ...

    def invoke(self, arguments: dict[str, Any]) -> Any: ...


class InvocationStatus(StrEnum):
    """Outcome values understood by every capability adapter."""

    OK = "ok"
    NO_RESULT = "no_result"
    INVALID_INPUT = "invalid_input"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    DENIED = "denied"
    ERROR = "error"
    PENDING = "pending"
    RUNNING = "running"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class CapabilityAnnotations:
    """Portable safety hints; adapter-specific annotations stay in metadata."""

    read_only: bool | None = None
    destructive: bool | None = None
    idempotent: bool | None = None
    open_world: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "readOnly": self.read_only,
                "destructive": self.destructive,
                "idempotent": self.idempotent,
                "openWorld": self.open_world,
            }.items()
            if value is not None
        }

    @classmethod
    def from_dict(cls, value: Any) -> CapabilityAnnotations:
        if not isinstance(value, dict):
            return cls()
        return cls(
            read_only=value.get("readOnly", value.get("readOnlyHint", value.get("read_only"))),
            destructive=value.get("destructive", value.get("destructiveHint")),
            idempotent=value.get("idempotent", value.get("idempotentHint")),
            open_world=value.get("openWorld", value.get("openWorldHint", value.get("open_world"))),
        )


@dataclass(frozen=True)
class CapabilityExecution:
    """Execution and job semantics independent of the underlying transport."""

    mode: Literal["sync", "async"] = "sync"
    timeout_seconds: float | None = None
    supports_cancel: bool = False
    supports_idempotency: bool = False
    supports_reconcile: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "timeoutSeconds": self.timeout_seconds,
            "supportsCancel": self.supports_cancel,
            "supportsIdempotency": self.supports_idempotency,
            "supportsReconcile": self.supports_reconcile,
        }

    @classmethod
    def from_dict(cls, value: Any) -> CapabilityExecution:
        if not isinstance(value, dict):
            return cls()
        mode = value.get("mode", "sync")
        if mode not in {"sync", "async"}:
            mode = "sync"
        timeout = value.get("timeoutSeconds", value.get("timeout_seconds"))
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            timeout = None
        return cls(
            mode=mode,
            timeout_seconds=float(timeout) if timeout is not None else None,
            supports_cancel=value.get("supportsCancel", value.get("supports_cancel")) is True,
            supports_idempotency=value.get("supportsIdempotency", value.get("supports_idempotency"))
            is True,
            supports_reconcile=value.get("supportsReconcile", value.get("supports_reconcile"))
            is True,
        )


@dataclass(frozen=True)
class CapabilityJobMethods:
    """Optional async job contract shared by local plugins and remote tasks."""

    submit: Any
    poll: Any
    cancel: Any

    def __post_init__(self) -> None:
        if not all(callable(method) for method in (self.submit, self.poll, self.cancel)):
            raise TypeError("job methods submit/poll/cancel must be callable")


_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def valid_job_id(value: Any) -> bool:
    """Accept only bounded, path-safe IDs suitable for durable job records."""
    return isinstance(value, str) and _JOB_ID_RE.fullmatch(value) is not None


@dataclass(frozen=True)
class CapabilityProvenance:
    """Evidence needed to identify the implementation used for an invocation."""

    source: str = ""
    version: str = ""
    code_digest: str = ""
    resource_digests: tuple[str, ...] = ()
    runtime: str = ""
    dependency_digest: str = ""
    deterministic: bool | None = None
    seed: int | str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "version": self.version,
            "codeDigest": self.code_digest,
            "resourceDigests": list(self.resource_digests),
            "runtime": self.runtime,
            "dependencyDigest": self.dependency_digest,
            "deterministic": self.deterministic,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, value: Any) -> CapabilityProvenance:
        if not isinstance(value, dict):
            return cls()
        digests = value.get("resourceDigests", value.get("resource_digests", ()))
        if not isinstance(digests, (list, tuple)):
            digests = ()
        return cls(
            source=str(value.get("source") or ""),
            version=str(value.get("version") or ""),
            code_digest=str(value.get("codeDigest", value.get("code_digest")) or ""),
            resource_digests=tuple(str(item) for item in digests),
            runtime=str(value.get("runtime") or ""),
            dependency_digest=str(
                value.get("dependencyDigest", value.get("dependency_digest")) or ""
            ),
            deterministic=value.get("deterministic")
            if isinstance(value.get("deterministic"), bool)
            else None,
            seed=value.get("seed"),
        )


@dataclass(frozen=True)
class CapabilityDescriptor:
    """Canonical description consumed by discovery, policy, and audit code."""

    id: str
    name: str
    description: str
    kind: CapabilityKind
    version: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    annotations: CapabilityAnnotations = field(default_factory=CapabilityAnnotations)
    execution: CapabilityExecution = field(default_factory=CapabilityExecution)
    permissions: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: CapabilityProvenance = field(default_factory=CapabilityProvenance)
    resource_scope: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("capability id must be non-empty")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("capability name must be non-empty")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("capability kind must be a non-empty string")
        if not isinstance(self.input_schema, dict):
            raise TypeError("input_schema must be a JSON Schema object")
        if self.execution.mode not in {"sync", "async"}:
            raise ValueError("execution mode must be 'sync' or 'async'")
        timeout = self.execution.timeout_seconds
        if timeout is not None and (
            isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0
        ):
            raise ValueError("execution timeout_seconds must be > 0")
        object.__setattr__(self, "permissions", tuple(str(item) for item in self.permissions))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "resource_scope", dict(self.resource_scope))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, stable descriptor representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "kind": self.kind,
            "version": self.version,
            "inputSchema": self.input_schema,
            "outputSchema": self.output_schema,
            "annotations": self.annotations.to_dict(),
            "execution": self.execution.to_dict(),
            "permissions": list(self.permissions),
            "metadata": self.metadata,
            "provenance": self.provenance.to_dict(),
            "resourceScope": self.resource_scope,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CapabilityDescriptor:
        if not isinstance(value, dict):
            raise TypeError("capability descriptor must be a mapping")
        missing = [key for key in ("id", "name") if key not in value]
        if missing:
            fields = ", ".join(missing)
            raise ValueError(f"capability descriptor missing required field(s): {fields}")
        return cls(
            id=str(value["id"]),
            name=str(value["name"]),
            description=str(value.get("description") or ""),
            kind=value.get("kind", "plugin"),
            version=str(value.get("version") or ""),
            input_schema=dict(value.get("inputSchema") or value.get("input_schema") or {}),
            output_schema=(
                dict(value["outputSchema"]) if isinstance(value.get("outputSchema"), dict) else None
            ),
            annotations=CapabilityAnnotations.from_dict(value.get("annotations")),
            execution=CapabilityExecution.from_dict(value.get("execution")),
            permissions=tuple(value.get("permissions") or ()),
            metadata=dict(value.get("metadata") or {}),
            provenance=CapabilityProvenance.from_dict(value.get("provenance")),
            resource_scope=dict(value.get("resourceScope") or value.get("resource_scope") or {}),
        )


@dataclass(frozen=True)
class InvocationResult:
    """Transport-neutral result envelope preserving structured and rich output."""

    status: InvocationStatus | str
    data: Any = None
    content: tuple[dict[str, Any], ...] = ()
    structured_content: Any = None
    error: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    job_id: str | None = None
    artifacts: tuple[dict[str, Any], ...] = ()
    truncated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.status, (InvocationStatus, str)):
            raise TypeError("invocation status must be a string")
        if isinstance(self.status, str) and not isinstance(self.status, InvocationStatus):
            try:
                object.__setattr__(self, "status", InvocationStatus(self.status))
            except ValueError:
                pass  # preserve extension statuses from future protocols
        object.__setattr__(self, "content", tuple(dict(item) for item in self.content))
        object.__setattr__(self, "artifacts", tuple(dict(item) for item in self.artifacts))
        object.__setattr__(self, "evidence", dict(self.evidence))

    @property
    def completed(self) -> bool:
        return self.status in {InvocationStatus.OK, InvocationStatus.NO_RESULT}

    @property
    def ok(self) -> bool:
        """Compatibility predicate: only an actual successful output is ``True``."""
        return self.status is InvocationStatus.OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value
            if isinstance(self.status, InvocationStatus)
            else self.status,
            "data": self.data,
            "content": list(self.content),
            "structuredContent": self.structured_content,
            "error": self.error,
            "evidence": self.evidence,
            "jobId": self.job_id,
            "artifacts": list(self.artifacts),
            "truncated": self.truncated,
        }

    def to_llm_message(self) -> str:
        """Render a source-neutral message for a function-calling agent."""
        if self.status is not InvocationStatus.OK:
            return f"能力调用失败 [{self.status}]: {self.error or '无输出'}"
        if self.structured_content is not None:
            return json.dumps(self.structured_content, ensure_ascii=False, default=str)
        if self.data is not None:
            return json.dumps(self.data, ensure_ascii=False, default=str)
        texts = [str(item.get("text") or "") for item in self.content if item.get("type") == "text"]
        return "\n".join(texts) if texts else "能力调用成功但无输出。"


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data deterministically for hashing and logs."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def input_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def descriptor_id(source: str, name: str) -> str:
    """Build a stable namespaced identifier for an adapter-owned capability."""
    source = source.strip().replace(" ", "_")
    name = name.strip().replace(" ", "_")
    return f"{source}:{name}" if source else name


def function_tool_name(capability_id: str, used: set[str] | None = None) -> str:
    """Return a provider-safe function name while retaining the canonical ID elsewhere.

    Function-calling providers commonly allow only letters, digits, underscores,
    and hyphens, with a 64-character limit.  ``used`` makes sanitization
    collision-safe for a single tool surface (for example ``a:b`` and
    ``a_b``).
    """
    raw = str(capability_id)
    name = re.sub(r"[^A-Za-z0-9_-]", "_", raw).strip("_") or "capability"
    if not name[0].isalpha():
        name = f"cap_{name}"
    name = name[:64]
    candidate = name
    if used is not None:
        counter = 2
        while candidate in used:
            suffix = f"_{counter}"
            candidate = f"{name[: 64 - len(suffix)]}{suffix}"
            counter += 1
        used.add(candidate)
    return candidate


@lru_cache(maxsize=1)
def runtime_fingerprint() -> str:
    """Return a compact runtime marker suitable for provenance records."""
    implementation = getattr(sys.implementation, "cache_tag", "unknown")
    return f"python={platform.python_version()};implementation={implementation};platform={platform.platform()}"


def file_digest(path: str | Path) -> str:
    """Hash a resource when it exists; callers can leave missing resources blank."""
    candidate = Path(path)
    if not candidate.is_file():
        return ""
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


@lru_cache(maxsize=32)
def dependency_lock_digest(base: str | Path | None = None) -> str:
    """Hash the nearest dependency lock so results can be reproduced later."""
    directory = Path(base or Path.cwd()).resolve()
    for parent in (directory, *directory.parents):
        for filename in ("uv.lock", "poetry.lock", "requirements.lock"):
            digest = file_digest(parent / filename)
            if digest:
                return digest
    return ""
