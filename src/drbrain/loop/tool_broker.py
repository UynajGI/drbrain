"""Durable, policy-controlled execution boundary for autoresearch tools."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from drbrain.loop.policy import PolicyDisposition, ToolDefinition, ToolPolicy, ToolProposal
from drbrain.loop.store import RunLedger

_SENSITIVE_KEY = re.compile(r"(?:api[_-]?key|authorization|cookie|password|secret|token)", re.I)
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|authorization|cookie|password|secret|token)\s*[:=]\s*"
    r"(?:bearer\s+)?[^,\s]+"
)


class ToolCallStatus(StrEnum):
    """Durable states for one tool proposal after broker handling."""

    INTENT = "intent"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    UNKNOWN = "unknown"
    DENIED = "denied"
    WAITING_APPROVAL = "waiting_approval"


@dataclass(frozen=True)
class ToolObservation:
    """Result returned to the workflow while its redacted form enters the ledger."""

    tool_call_id: str
    status: ToolCallStatus
    output: Any = None
    error: str | None = None
    attempts: int = 0
    reused: bool = False

    @property
    def ok(self) -> bool:
        return self.status is ToolCallStatus.SUCCEEDED

    def to_llm_message(self) -> str:
        """Render a conservative tool result for FunctionAgent callers."""
        if self.ok:
            # The broker's durable observation is redacted; preserve that same
            # trust boundary in the model-facing trajectory.  A tool result is
            # often just as likely to carry an access token as its arguments.
            output = _redact(self.output)
            if isinstance(output, str):
                return output
            return json.dumps(output, ensure_ascii=False, default=str)
        if self.status is ToolCallStatus.WAITING_APPROVAL:
            return f"工具调用等待审批: {self.error or 'approval required'}"
        if self.status is ToolCallStatus.UNKNOWN:
            return f"工具调用状态未知，不能假定已执行或可重试: {self.error or ''}".strip()
        return f"工具调用未执行/失败 [{self.status.value}]: {self.error or 'no observation'}"


@dataclass(frozen=True)
class _ExecutionOutcome:
    status: ToolCallStatus
    output: Any = None
    error: str | None = None
    evidence: Mapping[str, Any] | None = None


class ToolBroker:
    """Serialize policy approval, durable intent, execution, and observation.

    A broker is bound to one leased ledger attempt.  It deliberately accepts an
    executor closure rather than importing graph, plugin, RAG, or MCP layers;
    callers therefore share one control plane without creating a dependency
    cycle between those integrations.
    """

    def __init__(
        self,
        *,
        ledger: RunLedger,
        run_id: str,
        step_id: str,
        attempt_id: str,
        worker_id: str,
        lease_seconds: float,
        policy: ToolPolicy,
    ) -> None:
        self._ledger = ledger
        self.run_id = run_id
        self.step_id = step_id
        self.attempt_id = attempt_id
        self.worker_id = worker_id
        self.lease_seconds = max(1.0, float(lease_seconds))
        self.policy = policy

    async def execute(
        self,
        *,
        node_name: str,
        definition: ToolDefinition,
        arguments: Mapping[str, Any],
        executor: Callable[[], Any],
        idempotency_key: str | None = None,
        approved: bool = False,
    ) -> ToolObservation:
        """Evaluate and run a proposal, never invoking a denied handler."""
        tool_call_id = uuid.uuid4().hex
        normalized_arguments = dict(arguments)
        key = self._safe_explicit_idempotency_key(idempotency_key)
        if key is None:
            key = self._derived_idempotency_key(
                node_name=node_name,
                definition=definition,
                arguments=normalized_arguments,
            )
        proposal = ToolProposal(
            tool_call_id=tool_call_id,
            node_name=node_name,
            definition=definition,
            arguments=normalized_arguments,
            idempotency_key=key,
            approved=approved,
        )
        safe_proposal = self._safe_proposal(proposal)
        decision = self.policy.evaluate(proposal)
        self._renew_lease()
        if not decision.allowed:
            status = (
                ToolCallStatus.WAITING_APPROVAL
                if decision.disposition is PolicyDisposition.WAITING_APPROVAL
                else ToolCallStatus.DENIED
            )
            self._ledger.record_tool_decision(
                tool_call_id=tool_call_id,
                run_id=self.run_id,
                step_id=self.step_id,
                attempt_id=self.attempt_id,
                worker_id=self.worker_id,
                tool_name=definition.name,
                source=definition.source,
                side_effect=definition.side_effect,
                node_name=node_name,
                proposal=safe_proposal,
                idempotency_key=key,
                status=status.value,
                reason=decision.reason,
            )
            return ToolObservation(tool_call_id=tool_call_id, status=status, error=decision.reason)

        if key:
            previous = self._ledger.latest_tool_call_for_idempotency(
                self.run_id,
                step_id=self.step_id,
                idempotency_key=key,
            )
            if previous is not None and previous.status == ToolCallStatus.SUCCEEDED:
                return ToolObservation(
                    tool_call_id=previous.tool_call_id,
                    status=ToolCallStatus.SUCCEEDED,
                    output=previous.observation.get("output"),
                    attempts=int(previous.observation.get("attempts") or 0),
                    reused=True,
                )
            if (
                previous is not None
                and previous.status == ToolCallStatus.INTENT
                and definition.side_effect
                in {
                    "write",
                    "irreversible",
                }
            ):
                return ToolObservation(
                    tool_call_id=previous.tool_call_id,
                    status=ToolCallStatus.UNKNOWN,
                    error="matching side-effect intent lacks a durable observation; reconcile manually",
                    reused=True,
                )

        self._ledger.record_tool_intent(
            tool_call_id=tool_call_id,
            run_id=self.run_id,
            step_id=self.step_id,
            attempt_id=self.attempt_id,
            worker_id=self.worker_id,
            tool_name=definition.name,
            source=definition.source,
            side_effect=definition.side_effect,
            node_name=node_name,
            proposal=safe_proposal,
            idempotency_key=key,
            lease_seconds=self.lease_seconds,
        )

        keepalive = asyncio.create_task(self._maintain_lease())
        try:
            outcome = _ExecutionOutcome(
                ToolCallStatus.FAILED, error="tool executor produced no result"
            )
            attempts = 0
            for attempts in range(1, decision.max_attempts + 1):
                outcome = await self._execute_once(executor, definition)
                if (
                    outcome.status is not ToolCallStatus.TIMED_OUT
                    or attempts >= decision.max_attempts
                ):
                    break
        finally:
            keepalive.cancel()
            try:
                await keepalive
            except asyncio.CancelledError:
                pass

        self._renew_lease()

        status = outcome.status
        if status is ToolCallStatus.TIMED_OUT and definition.side_effect in {
            "write",
            "irreversible",
        }:
            status = ToolCallStatus.UNKNOWN
        safe_observation = {
            "output": _bounded(_redact(outcome.output), definition.max_output_bytes),
            "error": _redact_text(outcome.error) if outcome.error else None,
            "evidence": _bounded(
                _redact(dict(outcome.evidence or {})), definition.max_output_bytes
            ),
            "attempts": attempts,
        }
        self._ledger.record_tool_observation(
            tool_call_id=tool_call_id,
            run_id=self.run_id,
            step_id=self.step_id,
            attempt_id=self.attempt_id,
            worker_id=self.worker_id,
            status=status.value,
            observation=safe_observation,
            lease_seconds=self.lease_seconds,
        )
        return ToolObservation(
            tool_call_id=tool_call_id,
            status=status,
            output=outcome.output,
            error=outcome.error,
            attempts=attempts,
        )

    def record_evidence_bundle(self, bundle: Mapping[str, Any]) -> None:
        """Persist a safe RAG evidence bundle under this broker's active lease."""
        self._ledger.record_evidence_bundle(
            run_id=self.run_id,
            step_id=self.step_id,
            attempt_id=self.attempt_id,
            worker_id=self.worker_id,
            bundle=_redact(dict(bundle)),
        )

    def tool_call_id_for_output(self, value: str) -> str:
        """Resolve a durable compute-tool locator for a returned output value.

        Compute agents commonly expose only a job id in their final JSON.  This
        resolves the job id against SQLite-backed observations instead of trusting
        an agent-supplied tool-call id, and therefore also works after resume.
        """
        needle = str(value)
        if not needle:
            return ""
        for call in reversed(self._ledger.tool_calls(self.run_id)):
            if call.node_name != "compute" or call.status != ToolCallStatus.SUCCEEDED:
                continue
            if _contains_value(call.observation.get("output"), needle):
                return call.tool_call_id
        return ""

    def tool_call_proposal(self, tool_call_id: str) -> dict[str, Any]:
        """Return the already-redacted durable proposal for an artifact pointer."""
        for call in self._ledger.tool_calls(self.run_id):
            if call.tool_call_id == tool_call_id:
                return dict(call.proposal)
        return {}

    def _renew_lease(self) -> None:
        self._ledger.renew_lease(
            run_id=self.run_id,
            step_id=self.step_id,
            attempt_id=self.attempt_id,
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
        )

    async def _maintain_lease(self) -> None:
        """Keep ownership alive while an external tool may still be running."""
        interval = max(0.1, min(self.lease_seconds / 3.0, 30.0))
        while True:
            await asyncio.sleep(interval)
            self._renew_lease()

    async def _execute_once(
        self,
        executor: Callable[[], Any],
        definition: ToolDefinition,
    ) -> _ExecutionOutcome:
        try:
            if definition.timeout_s is None:
                result = executor()
                if inspect.isawaitable(result):
                    result = await result
            else:
                # A synchronous plugin/MCP adapter can block just as surely as
                # an async one. Invoke it off-loop so the broker's timeout
                # policy remains real; cancellation cannot stop the underlying
                # side effect, which is why write timeouts settle as UNKNOWN.
                timeout = max(0.001, definition.timeout_s)
                result = await asyncio.wait_for(asyncio.to_thread(executor), timeout=timeout)
                if inspect.isawaitable(result):
                    result = await asyncio.wait_for(result, timeout=timeout)
            return _normalize_result(result)
        except TimeoutError as exc:
            return _ExecutionOutcome(ToolCallStatus.TIMED_OUT, error=str(exc) or "tool timeout")
        except Exception as exc:  # noqa: BLE001 - normalize arbitrary tool failures
            return _ExecutionOutcome(ToolCallStatus.FAILED, error=f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _safe_proposal(proposal: ToolProposal) -> dict[str, Any]:
        definition = proposal.definition
        return {
            "tool_name": definition.name,
            "source": definition.source,
            "side_effect": definition.side_effect,
            "node_name": proposal.node_name,
            "arguments": _redact(dict(proposal.arguments)),
            "required_capabilities": list(definition.required_capabilities),
            "code_digest": definition.code_digest,
            "version": definition.version,
            "resource_scope": _redact(dict(definition.resource_scope)),
            "secret_refs": list(definition.secret_refs),
            "supports_idempotency": definition.supports_idempotency,
            "supports_reconcile": definition.supports_reconcile,
            "supports_cancel": definition.supports_cancel,
            "sandbox_profile": definition.sandbox_profile,
            "approval_policy": definition.approval_policy,
            "idempotency_key": proposal.idempotency_key,
        }

    @staticmethod
    def _derived_idempotency_key(
        *,
        node_name: str,
        definition: ToolDefinition,
        arguments: Mapping[str, Any],
    ) -> str | None:
        if not definition.supports_idempotency:
            return None
        payload = {
            "node_name": node_name,
            "tool_name": definition.name,
            "source": definition.source,
            # Hash the full canonical call identity locally.  Only the digest is
            # persisted, so redaction must not collapse credentials or tenants
            # into one reusable idempotency key.
            "arguments": dict(arguments),
        }
        encoded = json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _safe_explicit_idempotency_key(value: str | None) -> str | None:
        """Hash caller-provided keys before they reach durable audit records."""
        if not value:
            return None
        return hashlib.sha256(f"explicit:{value}".encode()).hexdigest()


def _normalize_result(result: Any) -> _ExecutionOutcome:
    """Adapt PluginResult-like envelopes without importing the plugin package."""
    raw_status = getattr(result, "status", None)
    status = getattr(raw_status, "value", raw_status)
    if status is None:
        return _ExecutionOutcome(ToolCallStatus.SUCCEEDED, output=result)
    if status in {"ok", "no_result"}:
        return _ExecutionOutcome(
            ToolCallStatus.SUCCEEDED,
            output=getattr(result, "data", None),
            evidence=getattr(result, "evidence", None),
            error=getattr(result, "error", None),
        )
    if status == "timeout":
        return _ExecutionOutcome(
            ToolCallStatus.TIMED_OUT,
            error=getattr(result, "error", None) or "tool timeout",
            evidence=getattr(result, "evidence", None),
        )
    return _ExecutionOutcome(
        ToolCallStatus.FAILED,
        error=getattr(result, "error", None) or f"tool result status {status!r}",
        evidence=getattr(result, "evidence", None),
    )


def _contains_value(value: Any, needle: str) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_value(item, needle) for item in value.values())
    if isinstance(value, list | tuple | set | frozenset):
        return any(_contains_value(item, needle) for item in value)
    return str(value) == needle


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if _SENSITIVE_KEY.search(str(key)) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple | set | frozenset):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return _redact_text(str(value))


def redact(value: Any) -> Any:
    """Redact a durable payload at the shared loop trust boundary."""
    return _redact(value)


def _redact_text(value: str | None) -> str | None:
    if value is None:
        return None
    return _SENSITIVE_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)


def _bounded(value: Any, max_output_bytes: int | None) -> Any:
    if max_output_bytes is None or max_output_bytes <= 0:
        return value
    encoded = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
    if len(encoded) <= max_output_bytes:
        return value
    return {
        "truncated": True,
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "preview": encoded[:max_output_bytes].decode("utf-8", errors="ignore"),
    }
