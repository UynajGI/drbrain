"""Reusable adapters for API endpoints, fixed CLI commands, and model callables."""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from drbrain.capabilities.protocol import (
    CapabilityAnnotations,
    CapabilityDescriptor,
    CapabilityExecution,
    CapabilityProvenance,
    InvocationResult,
    InvocationStatus,
    dependency_lock_digest,
    descriptor_id,
    runtime_fingerprint,
)


@dataclass
class APIAdapter:
    """Call a host-configured HTTP API without exposing secrets in its descriptor."""

    name: str
    description: str
    url: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    method: str = "POST"
    provider: str = "endpoint"
    version: str = ""
    timeout_seconds: float = 30.0
    headers: Mapping[str, str] = field(default_factory=dict)
    request: Callable[..., tuple[int, Any]] | None = None

    def descriptor(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            id=descriptor_id("api", f"{self.provider}:{self.name}"),
            name=self.name,
            description=self.description,
            kind="api",
            version=self.version,
            input_schema=dict(self.input_schema),
            annotations=CapabilityAnnotations(read_only=self.method.upper() == "GET"),
            execution=CapabilityExecution(timeout_seconds=self.timeout_seconds),
            metadata={"method": self.method.upper(), "provider": self.provider},
            provenance=CapabilityProvenance(
                source=f"api:{self.provider}",
                version=self.version,
                runtime=runtime_fingerprint(),
                dependency_digest=dependency_lock_digest(),
            ),
        )

    def invoke(self, arguments: dict[str, Any]) -> InvocationResult:
        if self.request is not None:
            status_code, payload = self.request(
                method=self.method.upper(),
                url=self.url,
                headers=dict(self.headers),
                json=arguments,
                timeout=self.timeout_seconds,
            )
        else:
            status_code, payload = _http_json_request(
                self.method, self.url, arguments, self.headers, self.timeout_seconds
            )
        status = InvocationStatus.OK if 200 <= status_code < 400 else InvocationStatus.ERROR
        return InvocationResult(
            status,
            data=payload,
            structured_content=payload,
            error=None if status is InvocationStatus.OK else f"HTTP {status_code}",
            evidence={"http_status": status_code, "provider": self.provider},
        )


@dataclass
class CLIAdapter:
    """Invoke a fixed executable with JSON on stdin; never invokes a shell."""

    name: str
    description: str
    command: tuple[str, ...]
    input_schema: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 60.0
    version: str = ""
    env: Mapping[str, str] | None = None

    def descriptor(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            id=descriptor_id("cli", self.name),
            name=self.name,
            description=self.description,
            kind="cli",
            version=self.version,
            input_schema=dict(self.input_schema),
            annotations=CapabilityAnnotations(read_only=False),
            execution=CapabilityExecution(timeout_seconds=self.timeout_seconds),
            metadata={"argv": list(self.command)},
            provenance=CapabilityProvenance(
                source=f"cli:{self.command[0] if self.command else ''}",
                version=self.version,
                dependency_digest=dependency_lock_digest(),
            ),
        )

    def invoke(self, arguments: dict[str, Any]) -> InvocationResult:
        if not self.command:
            raise ValueError("CLI command must be non-empty")
        environment = os.environ.copy()
        if self.env is not None:
            environment.update(self.env)
        try:
            completed = subprocess.run(
                list(self.command),
                input=json.dumps(arguments, ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
                env=environment,
            )
        except subprocess.TimeoutExpired:
            return InvocationResult(
                InvocationStatus.TIMEOUT,
                error=f"CLI command timed out after {self.timeout_seconds}s",
                evidence={"argv": list(self.command), "timeout": self.timeout_seconds},
            )
        output = completed.stdout.strip()
        try:
            payload: Any = json.loads(output) if output else None
        except json.JSONDecodeError:
            payload = output
        status = InvocationStatus.OK if completed.returncode == 0 else InvocationStatus.ERROR
        return InvocationResult(
            status,
            data=payload,
            structured_content=payload,
            error=None if status is InvocationStatus.OK else completed.stderr.strip(),
            evidence={"returncode": completed.returncode, "argv": list(self.command)},
        )


@dataclass
class ModelAdapter:
    """Wrap a locally trained model or hosted model client as a regular tool."""

    name: str
    description: str
    predict: Callable[[dict[str, Any]], Any]
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    version: str = ""
    model_digest: str = ""
    deterministic: bool | None = None

    def descriptor(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            id=descriptor_id("model", self.name),
            name=self.name,
            description=self.description,
            kind="model",
            version=self.version,
            input_schema=dict(self.input_schema),
            output_schema=dict(self.output_schema) if self.output_schema else None,
            annotations=CapabilityAnnotations(read_only=True),
            execution=CapabilityExecution(),
            provenance=CapabilityProvenance(
                source=f"model:{self.name}",
                version=self.version,
                code_digest=self.model_digest,
                deterministic=self.deterministic,
                runtime=runtime_fingerprint(),
                dependency_digest=dependency_lock_digest(),
            ),
        )

    def invoke(self, arguments: dict[str, Any]) -> Any:
        return self.predict(arguments)


def _http_json_request(
    method: str,
    url: str,
    arguments: dict[str, Any],
    headers: Mapping[str, str],
    timeout: float,
) -> tuple[int, Any]:
    method = method.upper()
    request_url = url
    body: bytes | None = None
    if method in {"GET", "HEAD", "DELETE"}:
        query = urllib.parse.urlencode(arguments, doseq=True)
        if query:
            request_url += ("&" if "?" in request_url else "?") + query
    else:
        body = json.dumps(arguments, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        request_url,
        data=body,
        headers={"Content-Type": "application/json", **dict(headers)},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = int(exc.code)
    try:
        payload: Any = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        payload = raw
    return status, payload
