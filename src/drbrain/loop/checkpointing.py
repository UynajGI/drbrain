"""JSON-only checkpoints for safe autoresearch workflow resumption.

The ledger owns durability and leases; this module owns the LlamaIndex-specific
boundary.  It deliberately persists only ``Context.to_dict(JsonSerializer())``
plus the two workflow-owned in-memory collaboration objects.  Pickle is never
used, so a checkpoint stays inspectable and can be rejected on an explicit
manifest mismatch instead of silently replaying under a different environment.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from llama_index.core.workflow import Context, JsonSerializer
from workflows.events import StepState, StepStateChanged

from drbrain.loop.store import LedgerCheckpoint, RunLedger

_EXTERNAL_SIDE_EFFECT_NODES = frozenset({"compute"})


class CheckpointError(RuntimeError):
    """Base class for an unsafe or unusable workflow checkpoint."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when a checkpoint was created under a different manifest."""


class CheckpointSerializationError(CheckpointError):
    """Raised when a Context cannot be represented as durable JSON."""


class CheckpointRestoreError(CheckpointError):
    """Raised when a previously valid JSON checkpoint cannot be restored."""


@dataclass(frozen=True)
class CheckpointManifest:
    """The execution contract required to continue a checkpoint safely."""

    workflow_version: str
    model_manifest: Mapping[str, Any]
    tool_manifest: Mapping[str, Any]
    rag_generation: str | None
    require_rag_evidence: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_version": self.workflow_version,
            "model_manifest": dict(self.model_manifest),
            "tool_manifest": dict(self.tool_manifest),
            "rag_generation": self.rag_generation,
            "require_rag_evidence": self.require_rag_evidence,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CheckpointManifest:
        return cls(
            workflow_version=str(value.get("workflow_version", "")),
            model_manifest=dict(value.get("model_manifest", {})),
            tool_manifest=dict(value.get("tool_manifest", {})),
            rag_generation=(
                str(value["rag_generation"]) if value.get("rag_generation") is not None else None
            ),
            require_rag_evidence=bool(value.get("require_rag_evidence", False)),
        )


class WorkflowCheckpointService:
    """Capture, validate, and restore one leased workflow attempt."""

    def __init__(
        self,
        *,
        ledger: RunLedger,
        run_id: str,
        step_id: str,
        attempt_id: str,
        worker_id: str,
        manifest: CheckpointManifest,
        lease_seconds: float,
        checkpoint: LedgerCheckpoint | None = None,
    ) -> None:
        self._ledger = ledger
        self.run_id = run_id
        self.step_id = step_id
        self.attempt_id = attempt_id
        self.worker_id = worker_id
        self.manifest = manifest
        self.lease_seconds = lease_seconds
        self.checkpoint = checkpoint

    @property
    def ledger(self) -> RunLedger:
        """Expose the leased ledger to additive control-plane integrations."""
        return self._ledger

    def capture(self, *, ctx: Any, workflow: Any, step_name: str) -> LedgerCheckpoint:
        """Persist a safe Context boundary and renew the current worker lease."""
        try:
            context_payload = ctx.to_dict(serializer=JsonSerializer())
            workflow_state = workflow.checkpoint_state()
            # Validate now rather than allowing sqlite's permissive JSON helper to
            # stringify an opaque runtime object.  A later restore must get the
            # same structured payload, never a repr.
            json.dumps(context_payload, ensure_ascii=False)
            json.dumps(workflow_state, ensure_ascii=False)
        except (TypeError, ValueError, AttributeError) as exc:
            raise CheckpointSerializationError(
                f"workflow checkpoint at {step_name!r} is not JSON serializable"
            ) from exc
        if not isinstance(context_payload, Mapping) or not isinstance(workflow_state, Mapping):
            raise CheckpointSerializationError("workflow checkpoint must serialize to JSON objects")
        return self._ledger.record_checkpoint(
            run_id=self.run_id,
            step_id=self.step_id,
            attempt_id=self.attempt_id,
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
            step_name=step_name,
            context_payload=dict(context_payload),
            workflow_state=dict(workflow_state),
            manifest=self.manifest.to_dict(),
        )

    def capture_if_safe(self, *, ctx: Any, workflow: Any, event: Any) -> LedgerCheckpoint | None:
        """Write node-start intent, then checkpoint only at a safe completion boundary."""
        if not isinstance(event, StepStateChanged):
            return None
        if event.step_state == StepState.RUNNING:
            self._ledger.record_workflow_step_started(
                run_id=self.run_id,
                step_id=self.step_id,
                attempt_id=self.attempt_id,
                worker_id=self.worker_id,
                lease_seconds=self.lease_seconds,
                node_name=str(event.name),
            )
            return None
        if event.step_state != StepState.NOT_RUNNING:
            return None
        return self.capture(ctx=ctx, workflow=workflow, step_name=str(event.name))

    @staticmethod
    def requires_manual_recovery(node_name: str | None) -> bool:
        """Whether a crash in this node may have crossed an external write boundary."""
        return node_name in _EXTERNAL_SIDE_EFFECT_NODES

    def validate_checkpoint(self) -> None:
        """Reject mismatched workflow/model/tool/RAG environments explicitly."""
        if self.checkpoint is None:
            raise CheckpointRestoreError("no checkpoint is bound to this resume attempt")
        if self.checkpoint.run_id != self.run_id or self.checkpoint.step_id != self.step_id:
            raise CheckpointCompatibilityError(
                "checkpoint belongs to a different run/step than this resume attempt"
            )
        expected = self.manifest.to_dict()
        actual = dict(self.checkpoint.manifest)
        # ``require_rag_evidence`` was added after the initial checkpoint
        # contract.  Its absent legacy value has always meant the compatible,
        # non-strict mode; normalize only that additive field and continue to
        # reject all other manifest drift exactly.
        actual.setdefault("require_rag_evidence", False)
        if actual != expected:
            raise CheckpointCompatibilityError(
                "checkpoint manifest is incompatible with this director session"
            )

    def restore_context(self, workflow: Any) -> Context:
        """Restore Context and workflow-owned collaboration state from JSON."""
        self.validate_checkpoint()
        assert self.checkpoint is not None  # narrowed by validate_checkpoint
        try:
            ctx = Context.from_dict(
                workflow,
                self.checkpoint.context_payload,
                serializer=JsonSerializer(),
            )
            workflow.restore_checkpoint_state(self.checkpoint.workflow_state)
            return ctx
        except Exception as exc:  # noqa: BLE001 - turn opaque library failures into a safe boundary
            raise CheckpointRestoreError(
                f"checkpoint {self.checkpoint.checkpoint_id} cannot be restored"
            ) from exc
