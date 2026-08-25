"""Durable experiment, artifact, and settlement facts for the research loop.

This is the second half counterpart to :mod:`drbrain.loop.front_half`.  The
workflow keeps its typed in-memory events for compatibility, while this facade
owns stable IDs and sends every canonical write through ``TransitionService``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drbrain.loop.transitions import ChampionVersionConflictError, TransitionService

_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class ExecutionNodeSpec:
    """Immutable execution-node contract persisted with a durable run."""

    name: str
    input_event: str
    output_event: str
    allowed_tools: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_schema": {"event": self.input_event},
            "output_schema": {"event": self.output_event},
            "allowed_tools": list(self.allowed_tools),
            "max_attempts": 3,
            "retry_class": "transient",
        }


EXECUTION_NODE_SPECS = (
    ExecutionNodeSpec("compute", "Critiqued", "Computed", ("run_python", "check_job")),
    ExecutionNodeSpec("verify", "Computed", "Verified", ("rag_retrieve",)),
    ExecutionNodeSpec("settle", "Verified", "Settled"),
    ExecutionNodeSpec("report", "Settled", "StopEvent"),
)


def _stable_id(prefix: str, *parts: str) -> str:
    value = "\x00".join(parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(value).hexdigest()[:24]}"


def experiment_id(run_id: str, claim_id: str) -> str:
    return _stable_id("exp", run_id, claim_id)


def settlement_id(experiment_id_value: str) -> str:
    return _stable_id("stl", experiment_id_value)


class DurableExecution:
    """Run-bound facade for experiment artifacts and structural settlement gates.

    ``noise_band`` and ``required_repeats`` are host configuration.  A verified
    result whose numeric value is inside the configured band cannot become a
    champion until a later repeat is represented as a separate experiment.
    """

    def __init__(
        self,
        transitions: TransitionService,
        run_id: str,
        *,
        step_id: str | None = None,
        attempt_id: str | None = None,
        worker_id: str | None = None,
        noise_band: float = 0.0,
        required_repeats: int = 2,
    ) -> None:
        self.transitions = transitions
        self.run_id = run_id
        self.step_id = step_id
        self.attempt_id = attempt_id
        self.worker_id = worker_id
        self.noise_band = max(0.0, float(noise_band))
        self.required_repeats = max(1, int(required_repeats))

    def ensure_node_contracts(self) -> None:
        self.transitions.register_execution_node_contracts(
            self.run_id, {spec.name: spec.to_dict() for spec in EXECUTION_NODE_SPECS}
        )

    def record_experiment(
        self,
        hypothesis: dict[str, Any],
        *,
        environment: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist the plan plus inline config/environment/seed/input artifacts."""
        claim_id = str(hypothesis.get("claim_id") or "").strip()
        if not claim_id:
            raise ValueError("durable experiment requires a stable claim_id")
        conditions = hypothesis.get("conditions")
        conditions = dict(conditions) if isinstance(conditions, dict) else {}
        raw_seed = conditions.get("seed")
        seed = raw_seed if isinstance(raw_seed, int) and not isinstance(raw_seed, bool) else None
        experiment = self.transitions.record_experiment(
            self.run_id,
            experiment_id=experiment_id(self.run_id, claim_id),
            proposal_id=str(hypothesis.get("proposal_id") or ""),
            claim_id=claim_id,
            plan=self._plan(hypothesis),
            environment=dict(environment),
            config=dict(config),
            seed=seed,
            producer_attempt_id=self.attempt_id,
            step_id=self.step_id,
            worker_id=self.worker_id,
        )
        exp_id = experiment["experiment_id"]
        self._record_inline(exp_id, "plan", self._plan(hypothesis))
        self._record_inline(exp_id, "config", dict(config))
        self._record_inline(exp_id, "environment", dict(environment))
        self._record_inline(exp_id, "seed", {"seed": seed})
        self._record_inline(exp_id, "input", conditions)
        return experiment

    def record_artifact(
        self,
        experiment_id_value: str,
        *,
        actor: str,
        kind: str,
        uri: str,
        payload: Any | None = None,
        path: Path | None = None,
        media_type: str = "application/json",
        metadata: dict[str, Any] | None = None,
        tool_call_id: str = "",
    ) -> dict[str, Any]:
        """Record an immutable artifact; the transition service enforces role isolation."""
        if path is not None:
            content = path.read_bytes() if path.is_file() else b""
            byte_size = len(content)
            sha256 = hashlib.sha256(content).hexdigest()
            if not uri:
                uri = str(path.resolve())
            artifact_metadata = dict(metadata or {})
        else:
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode(
                "utf-8"
            )
            byte_size = len(encoded)
            sha256 = hashlib.sha256(encoded).hexdigest()
            artifact_metadata = {**dict(metadata or {}), "inline_payload": payload}
        return self.transitions.record_execution_artifact(
            self.run_id,
            artifact_id=_stable_id("art", experiment_id_value, kind, sha256, uri),
            experiment_id=experiment_id_value,
            actor=actor,
            kind=kind,
            media_type=media_type,
            uri=uri,
            sha256=sha256,
            byte_size=byte_size,
            metadata=artifact_metadata,
            tool_call_id=tool_call_id,
            producer_attempt_id=self.attempt_id,
            step_id=self.step_id,
            worker_id=self.worker_id,
        )

    def record_compute_output(
        self,
        experiment_id_value: str,
        *,
        job_id: str,
        jobs_dir: str | Path,
        tool_call_id: str = "",
        code: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Materialize tool code-pointer and completed job output as artifacts."""
        jobs = Path(jobs_dir)
        meta_path = jobs / f"{job_id}.json"
        log_path = jobs / f"{job_id}.log"
        meta: dict[str, Any] = {}
        if meta_path.is_file():
            try:
                loaded = json.loads(meta_path.read_text(encoding="utf-8"))
                meta = dict(loaded) if isinstance(loaded, dict) else {}
            except (OSError, json.JSONDecodeError):
                meta = {}
        configured_log = meta.get("log_path")
        if isinstance(configured_log, str) and configured_log:
            log_path = Path(configured_log)

        self._record_inline(
            experiment_id_value,
            "code",
            {
                "tool_call_id": tool_call_id,
                "arguments": dict(code or {}),
                "proposal_ref": "tool_call.proposal.arguments",
            },
            uri=f"ledger://tool-calls/{tool_call_id or 'unresolved'}/proposal",
            tool_call_id=tool_call_id,
        )
        if meta_path.is_file():
            self.record_artifact(
                experiment_id_value,
                actor="compute",
                kind="output_metadata",
                uri=str(meta_path.resolve()),
                path=meta_path,
                media_type="application/json",
                metadata={"job_id": job_id},
                tool_call_id=tool_call_id,
            )
        numeric = False
        if log_path.is_file():
            try:
                numeric = _job_is_finished(meta) and bool(
                    _NUMBER.search(log_path.read_text(encoding="utf-8", errors="replace"))
                )
            except OSError:
                numeric = False
            output = self.record_artifact(
                experiment_id_value,
                actor="compute",
                kind="output",
                uri=str(log_path.resolve()),
                path=log_path,
                media_type="text/plain",
                metadata={"job_id": job_id, "numeric": numeric},
                tool_call_id=tool_call_id,
            )
        else:
            output = self._record_inline(
                experiment_id_value,
                "output",
                {"job_id": job_id, "missing": True},
                metadata={"job_id": job_id, "numeric": False},
                tool_call_id=tool_call_id,
            )
        return {"job_id": job_id, "numeric": numeric, "artifact_id": output["artifact_id"]}

    def settle_verification(
        self,
        experiment_id_value: str,
        *,
        verification: dict[str, Any],
        expected_champion_version: int | None,
    ) -> dict[str, Any]:
        """Settle one verifier result using only recorded artifacts and evidence IDs."""
        snapshot = self.snapshot()
        numeric = any(
            artifact["experiment_id"] == experiment_id_value
            and artifact["kind"] == "output"
            and bool(artifact["metadata"].get("numeric"))
            for artifact in snapshot["artifacts"]
        )
        evidence = verification.get("evidence_ids")
        evidence_ids = [str(value) for value in evidence] if isinstance(evidence, list) else []
        return self.transitions.settle_execution_claim(
            self.run_id,
            settlement_id=settlement_id(experiment_id_value),
            experiment_id=experiment_id_value,
            verification=dict(verification),
            evidence_ids=evidence_ids,
            has_numeric_artifact=numeric,
            expected_champion_version=expected_champion_version,
            noise_band=self.noise_band,
            required_repeats=self.required_repeats,
        )

    def snapshot(self) -> dict[str, Any]:
        return self.transitions.execution_snapshot(self.run_id)

    @staticmethod
    def _plan(hypothesis: dict[str, Any]) -> dict[str, Any]:
        return {
            key: hypothesis.get(key)
            for key in (
                "claim_id",
                "proposal_id",
                "statement",
                "prediction",
                "falsification",
                "conditions",
            )
        }

    def _record_inline(
        self,
        experiment_id_value: str,
        kind: str,
        payload: Any,
        *,
        uri: str | None = None,
        metadata: dict[str, Any] | None = None,
        tool_call_id: str = "",
    ) -> dict[str, Any]:
        return self.record_artifact(
            experiment_id_value,
            actor="compute",
            kind=kind,
            uri=uri or f"inline://experiments/{experiment_id_value}/{kind}",
            payload=payload,
            metadata=metadata,
            tool_call_id=tool_call_id,
        )


def _job_is_finished(metadata: dict[str, Any]) -> bool:
    """Reject partial stdout from a still-running asynchronous job."""
    pid = metadata.get("pid")
    if not pid:
        return True
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return True
    after = stat[stat.rfind(")") + 2 :]
    return bool(after) and after[0] in {"Z", "X"}


__all__ = [
    "ChampionVersionConflictError",
    "DurableExecution",
    "EXECUTION_NODE_SPECS",
    "ExecutionNodeSpec",
    "experiment_id",
    "settlement_id",
]
