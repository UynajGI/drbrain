"""Build-stage agents with idempotency, retry, and structured I/O contracts.

Each agent wraps a single build stage with a dedicated system prompt,
input/output validation, and idempotency guard via DB status tracking.
Agents communicate through structured intermediate artifacts, not raw
LLM context — inspired by 2511.11017's agent-based workflow.
"""

from __future__ import annotations

import enum
import json as _json
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

# -- Prompt paths (repo root / prompts) --
_PROMPTS = Path(__file__).parent.parent.parent.parent / "prompts"


# -- Stage status enum --
class StageStatus(enum.StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


# -- Shared contracts --
@dataclass
class AgentInput:
    """Input contract for a build-stage agent."""

    paper_id: str
    stage: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentOutput:
    """Output contract for a build-stage agent."""

    paper_id: str
    stage: str
    status: StageStatus
    data: dict[str, Any] = field(default_factory=dict)
    diff: dict[str, Any] | None = None  # refinement before/after


class BuildAgent(ABC):
    """Base agent for build pipeline stages.

    Each agent has:
      - name: stage identifier ("ontology", "entities", etc.)
      - system_prompt: loaded from prompts/<stage>.txt
      - input_schema / output_schema: validation contracts (dict shape)
      - run(): idempotency guard → build prompt → LLM call → validate → persist

    Subclasses implement _build_prompt() and _validate_output().
    """

    name: str
    prompt_file: str  # relative to prompts/

    def __init__(self) -> None:
        """Load the system prompt from the agent's prompt_file."""
        self.system_prompt = (_PROMPTS / self.prompt_file).read_text(encoding="utf-8")

    # -- Public API --

    async def run(
        self, input_data: AgentInput, models: list[dict], *, db=None, _cache=None
    ) -> AgentOutput:
        """Execute agent with idempotency guard.

        1. Check DB for existing stage output → skip if complete
        2. Build prompt from system_prompt + input_data
        3. Call LLM via acall_with_fallback
        4. Validate output
        5. Persist to DB with status
        6. Return AgentOutput
        """
        from drbrain.extractor.llm_client import acall_with_fallback

        # Idempotency: skip if already complete
        if db is not None and self._is_complete(db, input_data.paper_id):
            logger.info(f"[{self.name}] Already complete for {input_data.paper_id}, skipping")
            cached = self._load_cached(db, input_data.paper_id)
            if cached is not None:
                return AgentOutput(
                    paper_id=input_data.paper_id,
                    stage=self.name,
                    status=StageStatus.COMPLETE,
                    data=cached,
                )

        # Mark in-progress
        if db is not None:
            self._save_status(db, input_data.paper_id, StageStatus.IN_PROGRESS)

        # Build user prompt from structured input
        user_prompt = self._build_prompt(input_data)

        # Call LLM
        logger.info(f"[{self.name}] Running for {input_data.paper_id}")
        raw = await acall_with_fallback(
            prompt=user_prompt,
            models=models,
            system_prompt=self.system_prompt,
            _cache=_cache,
        )

        if not raw:
            if db is not None:
                self._save_status(db, input_data.paper_id, StageStatus.FAILED)
            return AgentOutput(
                paper_id=input_data.paper_id,
                stage=self.name,
                status=StageStatus.FAILED,
            )

        # Validate
        if not isinstance(raw, dict):
            raw = {}
        validated = self._validate_output(raw)

        # Persist
        if db is not None:
            self._save_result(db, input_data.paper_id, validated)
            self._save_status(db, input_data.paper_id, StageStatus.COMPLETE)

        logger.info(f"[{self.name}] Complete for {input_data.paper_id}")
        return AgentOutput(
            paper_id=input_data.paper_id,
            stage=self.name,
            status=StageStatus.COMPLETE,
            data=validated,
        )

    # -- Subclass contract --

    @abstractmethod
    def _build_prompt(self, input_data: AgentInput) -> str:
        """Build the user prompt from structured input data."""
        ...

    @abstractmethod
    def _validate_output(self, raw: dict) -> dict:
        """Validate and normalize LLM output. Raises ValueError on invalid."""
        ...

    # -- DB helpers (override for custom storage) --

    def _is_complete(self, db, paper_id: str) -> bool:
        """Check if this stage has already completed for the paper."""
        try:
            row = db.conn.execute(
                "SELECT status FROM build_stages WHERE paper_id = ? AND stage = ?",
                (paper_id, self.name),
            ).fetchone()
            return row is not None and row[0] == StageStatus.COMPLETE.value
        except sqlite3.Error as e:
            logger.warning(f"[agent] _is_complete failed for {paper_id}: {e}")
            return False

    def _save_status(self, db, paper_id: str, status: StageStatus) -> None:
        """Upsert stage status."""
        try:
            db.upsert_build_stage(paper_id, self.name, status.value)
            db.commit()
        except sqlite3.Error as e:
            logger.warning(f"[agent] _save_status failed for {paper_id}: {e}")

    def _save_result(self, db, paper_id: str, result: dict) -> None:
        """Persist validated output for idempotency replay."""
        try:
            db.upsert_build_stage(
                paper_id, self.name, StageStatus.COMPLETE.value, _json.dumps(result)
            )
            db.commit()
        except sqlite3.Error as e:
            logger.warning(f"[agent] _save_result failed for {paper_id}: {e}")

    def _load_cached(self, db, paper_id: str) -> dict | None:
        """Load cached result from a prior completed run."""
        try:
            row = db.conn.execute(
                "SELECT result_json FROM build_stages WHERE paper_id = ? AND stage = ?",
                (paper_id, self.name),
            ).fetchone()
            if row and row[0]:
                return _json.loads(row[0])
        except (sqlite3.Error, _json.JSONDecodeError) as e:
            logger.warning(f"[agent] _load_cached failed for {paper_id}: {e}")
        return None


# -- Stage 1: Ontology Agent --
class OntologyAgent(BuildAgent):
    """Maps TOC hierarchy to ontology classes under TBox 6 types."""

    name = "ontology"
    prompt_file = "ontology.txt"

    def _build_prompt(self, input_data: AgentInput) -> str:
        return input_data.data.get("prompt", "")

    def _validate_output(self, raw: dict) -> dict:
        valid_types = {"Problem", "Method", "Conclusion", "Gap", "Debate", "Actor"}
        result = {}
        for k, v in raw.items():
            if k in valid_types and isinstance(v, list):
                result[k] = [str(item) for item in v]
        return result


# -- Stage 2: Entity Agent --
class EntityAgent(BuildAgent):
    """Per leaf node: extracts concepts with subcategory labels and node_id provenance."""

    name = "entities"
    prompt_file = "entities.txt"

    def _build_prompt(self, input_data: AgentInput) -> str:
        return input_data.data.get("prompt", "")

    def _validate_output(self, raw: dict) -> dict:
        concepts = raw.get("concepts", [])
        if not isinstance(concepts, list):
            raise ValueError("entities output missing 'concepts' list")
        result = []
        for c in concepts:
            if not isinstance(c, dict):
                continue
            label = c.get("label", "").strip()
            ctype = c.get("type", "").strip()
            if not label or not ctype:
                continue
            result.append(
                {
                    "label": label,
                    "type": ctype,
                    "confidence": float(c.get("confidence", 0.5)),
                    "section": c.get("section", ""),
                    "node_id": c.get("node_id", ""),
                }
            )
        return {"concepts": result}


# -- Stage 3: Relation Agent --
class RelationAgent(BuildAgent):
    """Links concepts across sections with node_id provenance from source concept."""

    name = "relations"
    prompt_file = "relations.txt"

    def _build_prompt(self, input_data: AgentInput) -> str:
        return input_data.data.get("prompt", "")

    def _validate_output(self, raw: dict) -> dict:
        relations = raw.get("relations", [])
        if not isinstance(relations, list):
            raise ValueError("relations output missing 'relations' list")
        result = []
        for r in relations:
            if not isinstance(r, dict):
                continue
            head = r.get("head", "").strip()
            rel = r.get("rel", "").strip()
            tail = r.get("tail", "").strip()
            if not head or not rel or not tail:
                continue
            result.append(
                {
                    "head": head,
                    "rel": rel,
                    "tail": tail,
                    "node_id": r.get("node_id", ""),
                    "section": r.get("section", ""),
                }
            )
        return {"relations": result}


# -- Stage 4: Coreference Agent --
class CorefAgent(BuildAgent):
    """Merges duplicate concept labels across sections."""

    name = "coreference"
    prompt_file = "coreference.txt"

    def _build_prompt(self, input_data: AgentInput) -> str:
        return input_data.data.get("prompt", "")

    def _validate_output(self, raw: dict) -> dict:
        merges = raw.get("merges", [])
        if not isinstance(merges, list):
            raise ValueError("coreference output missing 'merges' list")
        result = []
        for m in merges:
            if not isinstance(m, dict):
                continue
            canonical = m.get("canonical", "").strip()
            variants = m.get("variants", [])
            if not canonical or not isinstance(variants, list):
                continue
            result.append(
                {
                    "canonical": canonical,
                    "variants": [str(v) for v in variants],
                }
            )
        return {"merges": result}


# -- Stage 5: Refine Agent --
class RefineAgent(BuildAgent):
    """Self-reviews extraction, outputs corrections with before/after diff."""

    name = "refine"
    prompt_file = "refine.txt"

    def __init__(self) -> None:
        """Initialize with empty pre-refinement snapshot."""
        super().__init__()
        self._pre_refine_snapshot: dict | None = None

    def set_snapshot(self, concepts: list[dict], relations: list[dict]) -> None:
        """Store pre-refinement snapshot for diff computation."""
        self._pre_refine_snapshot = {
            "concept_count": len(concepts),
            "relation_count": len(relations),
            "concept_labels": sorted(c.get("label", "") for c in concepts),
        }

    def _build_prompt(self, input_data: AgentInput) -> str:
        return input_data.data.get("prompt", "")

    def _validate_output(self, raw: dict) -> dict:
        corrections = raw.get("corrections", [])
        if not isinstance(corrections, list):
            raise ValueError("refine output missing 'corrections' list")
        diff = None
        if self._pre_refine_snapshot is not None:
            diff = {
                "before": self._pre_refine_snapshot,
                "after": {"correction_count": len(corrections)},
            }
        return {"corrections": [dict(c) for c in corrections if isinstance(c, dict)], "diff": diff}


# -- Agent factory --
_AGENTS: dict[str, BuildAgent] = {
    "ontology": OntologyAgent(),
    "entities": EntityAgent(),
    "relations": RelationAgent(),
    "coreference": CorefAgent(),
    "refine": RefineAgent(),
}


def get_agent(name: str) -> BuildAgent:
    """Get a build-stage agent by name."""
    agent = _AGENTS.get(name)
    if agent is None:
        raise ValueError(f"Unknown agent: {name}. Available: {list(_AGENTS)}")
    return agent
