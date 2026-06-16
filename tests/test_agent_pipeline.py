"""Tests for BuildAgent.run() — the 5-stage extraction orchestrator.

BuildAgent.run() had zero coverage despite being the core orchestrator for all
5 agent subclasses. These tests verify: LLM call + persist, LLM failure,
idempotency (skip if already complete), status transitions, and cache replay.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from drbrain.extractor.agent import (
    AgentInput,
    AgentOutput,
    CorefAgent,
    EntityAgent,
    OntologyAgent,
    RefineAgent,
    RelationAgent,
    StageStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_complete_with_result(db, paper_id: str, stage: str, result: dict) -> None:
    """Insert a COMPLETE row with result_json so idempotency guard returns cached data."""
    import json as _json

    db.conn.execute(
        "INSERT OR REPLACE INTO build_stages (paper_id, stage, status, result_json) VALUES (?, ?, ?, ?)",
        (paper_id, stage, "complete", _json.dumps(result)),
    )
    db.commit()


def _insert_status(db, paper_id: str, stage: str, status: str, result_json: str = "") -> None:
    """Insert a row into build_stages with given status and optional result_json."""
    db.conn.execute(
        "INSERT OR REPLACE INTO build_stages (paper_id, stage, status, result_json) VALUES (?, ?, ?, ?)",
        (paper_id, stage, status, result_json),
    )
    db.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOntologyAgentRun:
    """OntologyAgent.run() — Stage 1 orchestrator."""

    async def test_run_calls_llm_and_persists_result(self, tmp_db):
        """run() calls acall_with_fallback, validates output, and persists to build_stages."""
        agent = OntologyAgent()
        mock_response = {"Problem": ["efficiency"], "Method": ["transformer"]}

        with patch(
            "drbrain.extractor.llm_client.acall_with_fallback",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await agent.run(
                input_data=AgentInput(
                    paper_id="p1", stage="ontology", data={"prompt": "build ontology"}
                ),
                models=[{"provider": "test", "model": "m"}],
                db=tmp_db,
            )

        assert isinstance(result, AgentOutput)
        assert result.status == StageStatus.COMPLETE
        assert result.stage == "ontology"
        assert result.paper_id == "p1"
        assert "Method" in result.data
        assert result.data["Method"] == ["transformer"]

        # Verify persisted to build_stages with status 'complete'
        row = tmp_db.conn.execute(
            "SELECT status FROM build_stages WHERE paper_id = 'p1' AND stage = 'ontology'"
        ).fetchone()
        assert row is not None
        assert row[0] == "complete"

    async def test_run_returns_failed_on_llm_failure(self, tmp_db):
        """run() returns FAILED AgentOutput when LLM returns None."""
        agent = OntologyAgent()

        with patch(
            "drbrain.extractor.llm_client.acall_with_fallback",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await agent.run(
                input_data=AgentInput(paper_id="p1", stage="ontology", data={"prompt": "test"}),
                models=[{"provider": "test", "model": "m"}],
                db=tmp_db,
            )

        assert isinstance(result, AgentOutput)
        assert result.status == StageStatus.FAILED
        assert result.data == {}

        row = tmp_db.conn.execute(
            "SELECT status FROM build_stages WHERE paper_id = 'p1' AND stage = 'ontology'"
        ).fetchone()
        assert row is not None
        assert row[0] == "failed"

    async def test_run_returns_failed_on_empty_dict(self, tmp_db):
        """run() treats empty LLM response (empty dict is falsy) as failure."""
        agent = OntologyAgent()

        with patch(
            "drbrain.extractor.llm_client.acall_with_fallback",
            new_callable=AsyncMock,
            return_value={},
        ):
            result = await agent.run(
                input_data=AgentInput(paper_id="p1", stage="ontology", data={"prompt": "test"}),
                models=[{"provider": "test", "model": "m"}],
                db=tmp_db,
            )

        assert result.status == StageStatus.FAILED

    async def test_run_skips_if_complete_with_cached_result(self, tmp_db):
        """run() skips LLM call and returns cached result when COMPLETE + result_json exists."""
        cached_data = {"Problem": ["old"], "Method": ["old_method"]}
        _insert_complete_with_result(tmp_db, "p1", "ontology", cached_data)

        agent = OntologyAgent()
        with patch(
            "drbrain.extractor.llm_client.acall_with_fallback",
            new_callable=AsyncMock,
        ) as mock_llm:
            result = await agent.run(
                input_data=AgentInput(paper_id="p1", stage="ontology", data={"prompt": "test"}),
                models=[{"provider": "test", "model": "m"}],
                db=tmp_db,
            )

        mock_llm.assert_not_called()
        assert result.status == StageStatus.COMPLETE
        assert result.data == cached_data

    async def test_run_retries_if_complete_but_no_cached_result(self, tmp_db):
        """run() falls through to LLM when COMPLETE but result_json is empty (missing cache)."""
        # This happens when _save_status overwrites result_json after _save_result.
        # The code checks _is_complete → True, then _load_cached → None,
        # then falls through to re-run the LLM call.
        _insert_status(tmp_db, "p1", "ontology", "complete", result_json="")

        agent = OntologyAgent()
        mock_response = {"Method": ["re-run"]}

        with patch(
            "drbrain.extractor.llm_client.acall_with_fallback",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_llm:
            result = await agent.run(
                input_data=AgentInput(paper_id="p1", stage="ontology", data={"prompt": "test"}),
                models=[{"provider": "test", "model": "m"}],
                db=tmp_db,
            )

        # LLM should have been called (no cached data to return)
        mock_llm.assert_called_once()
        assert result.status == StageStatus.COMPLETE
        assert result.data["Method"] == ["re-run"]

    async def test_run_validates_and_filters_output(self, tmp_db):
        """run() filters invalid ontology types through _validate_output."""
        agent = OntologyAgent()
        mock_response = {
            "Method": ["GNN"],
            "Problem": ["Scalability"],
            "InvalidType": ["should be filtered"],
        }

        with patch(
            "drbrain.extractor.llm_client.acall_with_fallback",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await agent.run(
                input_data=AgentInput(paper_id="p1", stage="ontology", data={"prompt": "test"}),
                models=[{"provider": "test", "model": "m"}],
                db=tmp_db,
            )

        assert result.status == StageStatus.COMPLETE
        assert "InvalidType" not in result.data
        assert "Method" in result.data

    async def test_run_sets_in_progress_before_call(self, tmp_db):
        """run() sets status to IN_PROGRESS before LLM call, then COMPLETE after."""
        agent = OntologyAgent()
        call_states = []

        async def capture_state(*args, **kwargs):
            row = tmp_db.conn.execute(
                "SELECT status FROM build_stages WHERE paper_id = 'p1' AND stage = 'ontology'"
            ).fetchone()
            call_states.append(row[0] if row else None)
            return {"Method": ["test"]}

        with patch(
            "drbrain.extractor.llm_client.acall_with_fallback",
            new_callable=AsyncMock,
            side_effect=capture_state,
        ):
            await agent.run(
                input_data=AgentInput(paper_id="p1", stage="ontology", data={"prompt": "test"}),
                models=[{"provider": "test", "model": "m"}],
                db=tmp_db,
            )

        assert call_states == ["in_progress"]
        row = tmp_db.conn.execute(
            "SELECT status FROM build_stages WHERE paper_id = 'p1' AND stage = 'ontology'"
        ).fetchone()
        assert row[0] == "complete"

    async def test_run_without_db_still_works(self):
        """run() works without db — no persistence, no idempotency."""
        agent = OntologyAgent()

        with patch(
            "drbrain.extractor.llm_client.acall_with_fallback",
            new_callable=AsyncMock,
            return_value={"Method": ["X"]},
        ):
            result = await agent.run(
                input_data=AgentInput(paper_id="p1", stage="ontology", data={"prompt": "test"}),
                models=[{"provider": "test", "model": "m"}],
            )

        assert result.status == StageStatus.COMPLETE
        assert result.data == {"Method": ["X"]}


class TestEntityAgentRun:
    """EntityAgent.run() — Stage 2 orchestrator."""

    async def test_entity_run_validates_concepts(self, tmp_db):
        """EntityAgent.run() validates concepts through _validate_output."""
        agent = EntityAgent()
        mock_response = {
            "concepts": [
                {
                    "label": "GNN",
                    "type": "Method",
                    "confidence": 0.9,
                    "section": "3.1",
                    "node_id": "1.1",
                },
                {"label": "", "type": "Method"},  # filtered: empty label
                {"label": "Bad", "type": ""},  # filtered: empty type
            ],
        }

        with patch(
            "drbrain.extractor.llm_client.acall_with_fallback",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await agent.run(
                input_data=AgentInput(paper_id="p1", stage="entities", data={"prompt": "extract"}),
                models=[{"provider": "test", "model": "m"}],
                db=tmp_db,
            )

        assert result.status == StageStatus.COMPLETE
        assert len(result.data["concepts"]) == 1
        assert result.data["concepts"][0]["label"] == "GNN"

    async def test_entity_run_propagates_validation_error(self, tmp_db):
        """EntityAgent.run() propagates ValueError from _validate_output on bad input type."""
        agent = EntityAgent()

        # "concepts" key present but not a list → isinstance("not a list", list) is False
        with (
            patch(
                "drbrain.extractor.llm_client.acall_with_fallback",
                new_callable=AsyncMock,
                return_value={"concepts": "not a list"},
            ),
            pytest.raises(ValueError, match="missing 'concepts' list"),
        ):
            await agent.run(
                input_data=AgentInput(paper_id="p1", stage="entities", data={"prompt": "extract"}),
                models=[{"provider": "test", "model": "m"}],
                db=tmp_db,
            )


class TestRelationAgentRun:
    """RelationAgent.run() — Stage 3 orchestrator."""

    async def test_relation_run_validates_relations(self, tmp_db):
        """RelationAgent.run() validates relations through _validate_output."""
        agent = RelationAgent()
        mock_response = {
            "relations": [
                {"head": "GNN", "rel": "uses", "tail": "Dataset", "node_id": "1.1"},
                {"head": "", "rel": "uses", "tail": "X"},  # filtered: empty head
            ],
        }

        with patch(
            "drbrain.extractor.llm_client.acall_with_fallback",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await agent.run(
                input_data=AgentInput(paper_id="p1", stage="relations", data={"prompt": "link"}),
                models=[{"provider": "test", "model": "m"}],
                db=tmp_db,
            )

        assert result.status == StageStatus.COMPLETE
        assert len(result.data["relations"]) == 1
        assert result.data["relations"][0]["head"] == "GNN"


class TestCorefAgentRun:
    """CorefAgent.run() — Stage 4 orchestrator."""

    async def test_coref_run_validates_merges(self, tmp_db):
        """CorefAgent.run() validates merges through _validate_output."""
        agent = CorefAgent()
        mock_response = {
            "merges": [
                {"canonical": "GNN", "variants": ["Graph Neural Network"]},
                {"canonical": "", "variants": ["X"]},  # filtered: empty canonical
            ],
        }

        with patch(
            "drbrain.extractor.llm_client.acall_with_fallback",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await agent.run(
                input_data=AgentInput(paper_id="p1", stage="coreference", data={"prompt": "merge"}),
                models=[{"provider": "test", "model": "m"}],
                db=tmp_db,
            )

        assert result.status == StageStatus.COMPLETE
        assert len(result.data["merges"]) == 1
        assert result.data["merges"][0]["canonical"] == "GNN"


class TestRefineAgentRun:
    """RefineAgent.run() — Stage 5 orchestrator."""

    async def test_refine_run_includes_diff_when_snapshot_set(self, tmp_db):
        """RefineAgent.run() includes diff when set_snapshot() was called."""
        agent = RefineAgent()
        agent.set_snapshot(
            concepts=[{"label": "GNN"}, {"label": "CNN"}],
            relations=[{"head": "GNN", "rel": "uses", "tail": "Data"}],
        )
        mock_response = {"corrections": [{"type": "relabel", "old": "GNN", "new": "Graph NN"}]}

        with patch(
            "drbrain.extractor.llm_client.acall_with_fallback",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await agent.run(
                input_data=AgentInput(paper_id="p1", stage="refine", data={"prompt": "refine"}),
                models=[{"provider": "test", "model": "m"}],
                db=tmp_db,
            )

        assert result.status == StageStatus.COMPLETE
        assert result.data["diff"] is not None
        assert result.data["diff"]["before"]["concept_count"] == 2

    async def test_refine_run_without_snapshot_no_diff(self, tmp_db):
        """RefineAgent.run() has no diff when set_snapshot() was never called."""
        agent = RefineAgent()
        mock_response = {"corrections": [{"type": "relabel", "old": "A", "new": "B"}]}

        with patch(
            "drbrain.extractor.llm_client.acall_with_fallback",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await agent.run(
                input_data=AgentInput(paper_id="p1", stage="refine", data={"prompt": "refine"}),
                models=[{"provider": "test", "model": "m"}],
                db=tmp_db,
            )

        assert result.status == StageStatus.COMPLETE
        assert result.data["diff"] is None


class TestIdempotencyAllAgents:
    """Verify idempotency guard works for all 5 agent types."""

    @pytest.mark.parametrize(
        "agent_cls,stage",
        [
            (OntologyAgent, "ontology"),
            (EntityAgent, "entities"),
            (RelationAgent, "relations"),
            (CorefAgent, "coreference"),
            (RefineAgent, "refine"),
        ],
    )
    async def test_skip_if_complete_with_cache(self, tmp_db, agent_cls, stage):
        """All agent types skip LLM call when stage already COMPLETE with cached result."""
        cached = {"key": "cached_value"}
        _insert_complete_with_result(tmp_db, "p1", stage, cached)

        agent = agent_cls()
        with patch(
            "drbrain.extractor.llm_client.acall_with_fallback",
            new_callable=AsyncMock,
        ) as mock_llm:
            result = await agent.run(
                input_data=AgentInput(paper_id="p1", stage=stage, data={"prompt": "test"}),
                models=[{"provider": "test", "model": "m"}],
                db=tmp_db,
            )

        mock_llm.assert_not_called()
        assert result.status == StageStatus.COMPLETE
        assert result.data == cached
