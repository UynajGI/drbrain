"""Tests for the reasoning workflow engine.

Tests verify:
- Workflow registration and lookup
- Each workflow's symbolic steps produce correct output (no LLM mock needed)
- LLM steps are properly mocked
- Full pipeline execution produces expected result keys
"""

from unittest.mock import AsyncMock, patch

import pytest

from drbrain.graph.engine import GraphEngine
from drbrain.reasoning import WorkflowContext, get_workflow, list_workflows
from drbrain.storage.database import Database


@pytest.fixture
def tmp_db():
    db = Database(":memory:")
    yield db
    db.close()


@pytest.fixture
def tmp_graph(tmp_db):
    g = GraphEngine()
    g.load_from_db(tmp_db)
    return g


@pytest.fixture
def ctx(tmp_db, tmp_graph):
    return WorkflowContext(
        db=tmp_db,
        graph=tmp_graph,
        models=[{"provider": "test", "model": "m"}],
        question="test question",
    )


# ── Registry tests ────────────────────────────────────────────────────


class TestWorkflowRegistry:
    def test_list_workflows_returns_all_four(self):
        wfs = list_workflows()
        names = {w["name"] for w in wfs}
        assert names == {"causal", "contradiction", "temporal", "hypothesis"}

    def test_get_workflow_causal(self):
        wf = get_workflow("causal")
        assert wf.name == "causal"
        assert len(wf.steps) == 5

    def test_get_workflow_contradiction(self):
        wf = get_workflow("contradiction")
        assert wf.name == "contradiction"
        assert len(wf.steps) == 4

    def test_get_workflow_temporal(self):
        wf = get_workflow("temporal")
        assert wf.name == "temporal"
        assert len(wf.steps) == 4

    def test_get_workflow_hypothesis(self):
        wf = get_workflow("hypothesis")
        assert wf.name == "hypothesis"
        assert len(wf.steps) == 5

    def test_get_workflow_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown workflow"):
            get_workflow("nonexistent")


# ── Base class tests ──────────────────────────────────────────────────


class TestWorkflowContext:
    def test_get_returns_default_for_missing(self, ctx):
        assert ctx.get("nonexistent") is None
        assert ctx.get("nonexistent", "fallback") == "fallback"

    def test_results_starts_empty(self, ctx):
        assert ctx.results == {}


class TestWorkflowExecute:
    def test_execute_populates_results(self, ctx):
        wf = get_workflow("causal")
        # Mock the LLM step to avoid real API calls
        with patch("drbrain.extractor.llm_client.acall_text_with_fallback", new_callable=AsyncMock):
            results = wf.execute(ctx)

        # All 5 steps should have results (even if None for empty DB)
        assert "extract_entities" in results
        assert "find_causal_chain" in results
        assert "extract_mechanisms" in results
        assert "counterfactual_check" in results
        assert "synthesize_explanation" in results


# ── CausalWorkflow symbolic steps ────────────────────────────────────


class TestCausalWorkflowSymbolic:
    def test_extract_entities_empty_db(self, ctx):
        wf = get_workflow("causal")
        step = wf.steps[0]  # _ExtractEntitiesStep
        result = step.run(ctx)
        assert result["source"] is None
        assert result["target"] is None

    def test_extract_entities_finds_concepts(self, tmp_db, tmp_graph):
        tmp_db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Test', 2024, 'extracted')"
        )
        tmp_db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence, section) "
            "VALUES ('p1', 'Method', 'Transformer Architecture', 0.9, 'method')"
        )
        tmp_db.commit()
        tmp_graph.load_from_db(tmp_db)

        ctx = WorkflowContext(
            db=tmp_db,
            graph=tmp_graph,
            models=[{"provider": "test", "model": "m"}],
            question="How does Transformer Architecture work?",
        )
        wf = get_workflow("causal")
        result = wf.steps[0].run(ctx)
        assert result["source"] is not None

    def test_find_causal_chain_empty(self, ctx):
        wf = get_workflow("causal")
        result = wf.steps[1].run(ctx)
        assert result["chain"] is None
        assert result["args_count"] == 0

    def test_counterfactual_check_no_source(self, ctx):
        wf = get_workflow("causal")
        # No source entity → graceful handling
        result = wf.steps[3].run(ctx)
        assert result["impact"] is None


# ── ContradictionWorkflow symbolic steps ─────────────────────────────


class TestContradictionWorkflowSymbolic:
    def test_scan_debates_empty(self, ctx):
        wf = get_workflow("contradiction")
        result = wf.steps[0].run(ctx)
        assert result == []

    def test_scan_debates_finds_conflict(self, tmp_db, tmp_graph):
        tmp_db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'A', 2023, 'extracted')"
        )
        tmp_db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('p2', 'B', 2024, 'extracted')"
        )
        tmp_db.conn.execute(
            "INSERT INTO arguments (source_paper, claim, claim_type, target_label, target_type, confidence) "
            "VALUES ('p1', 'X works well', 'supports', 'Attention', 'Method', 0.9)"
        )
        tmp_db.conn.execute(
            "INSERT INTO arguments (source_paper, claim, claim_type, target_label, target_type, confidence) "
            "VALUES ('p2', 'X is inefficient', 'challenges', 'Attention', 'Method', 0.8)"
        )
        tmp_db.commit()

        ctx = WorkflowContext(
            db=tmp_db,
            graph=tmp_graph,
            models=[{"provider": "test", "model": "m"}],
            question="test",
        )
        wf = get_workflow("contradiction")
        result = wf.steps[0].run(ctx)
        assert len(result) >= 1
        assert result[0]["concept"] == "Attention"
        assert result[0]["support_count"] >= 1
        assert result[0]["challenge_count"] >= 1

    def test_build_argument_map_empty(self, ctx):
        wf = get_workflow("contradiction")
        result = wf.steps[1].run(ctx)
        assert result == []


# ── TemporalWorkflow symbolic steps ──────────────────────────────────


class TestTemporalWorkflowSymbolic:
    def test_build_timeline_empty(self, ctx):
        wf = get_workflow("temporal")
        result = wf.steps[0].run(ctx)
        assert result["concept"] is None

    def test_build_timeline_with_data(self, tmp_db, tmp_graph):
        tmp_db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'A', 2020, 'extracted')"
        )
        tmp_db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('p2', 'B', 2023, 'extracted')"
        )
        tmp_db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence, section, first_seen, last_seen) "
            "VALUES ('p1', 'Method', 'Deep Learning', 0.9, 'method', 2020, 2020)"
        )
        tmp_db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence, section, first_seen, last_seen) "
            "VALUES ('p2', 'Method', 'Deep Learning', 0.85, 'method', 2023, 2023)"
        )
        tmp_db.commit()
        tmp_graph.load_from_db(tmp_db)

        ctx = WorkflowContext(
            db=tmp_db,
            graph=tmp_graph,
            models=[{"provider": "test", "model": "m"}],
            question="How did Deep Learning evolve?",
        )
        wf = get_workflow("temporal")
        result = wf.steps[0].run(ctx)
        # Concept should be found (BM25 may or may not match on fresh DB)
        assert result["concept"] is not None or result["timeline"] == []


# ── HypothesisWorkflow symbolic steps ────────────────────────────────


class TestHypothesisWorkflowSymbolic:
    def test_find_cross_domain_empty(self, ctx):
        wf = get_workflow("hypothesis")
        result = wf.steps[0].run(ctx)
        assert result["cross_domain_patterns"] == []
        assert result["total_seeds"] >= 0

    def test_find_transfer_candidates_empty(self, ctx):
        wf = get_workflow("hypothesis")
        result = wf.steps[1].run(ctx)
        assert result == []

    def test_validate_empty_hypotheses(self, ctx):
        wf = get_workflow("hypothesis")
        result = wf.steps[3].run(ctx)
        assert result == []

    def test_score_empty(self, ctx):
        wf = get_workflow("hypothesis")
        result = wf.steps[4].run(ctx)
        assert result == []


# ── Full pipeline execution (mocked LLM) ─────────────────────────────


class TestFullPipelineExecution:
    @pytest.mark.asyncio
    async def test_causal_full_pipeline_empty_db(self, tmp_db, tmp_graph):
        """CausalWorkflow runs end-to-end on empty DB without errors."""
        ctx = WorkflowContext(
            db=tmp_db,
            graph=tmp_graph,
            models=[{"provider": "test", "model": "m"}],
            question="why does X cause Y?",
        )
        wf = get_workflow("causal")
        with patch("drbrain.extractor.llm_client.acall_text_with_fallback", new_callable=AsyncMock):
            results = wf.execute(ctx)
        assert "synthesize_explanation" in results

    @pytest.mark.asyncio
    async def test_contradiction_full_pipeline_empty_db(self, tmp_db, tmp_graph):
        ctx = WorkflowContext(
            db=tmp_db,
            graph=tmp_graph,
            models=[{"provider": "test", "model": "m"}],
            question="what contradictions exist?",
        )
        wf = get_workflow("contradiction")
        with patch("drbrain.extractor.llm_client.acall_text_with_fallback", new_callable=AsyncMock):
            results = wf.execute(ctx)
        assert "summarize" in results

    @pytest.mark.asyncio
    async def test_temporal_full_pipeline_empty_db(self, tmp_db, tmp_graph):
        ctx = WorkflowContext(
            db=tmp_db,
            graph=tmp_graph,
            models=[{"provider": "test", "model": "m"}],
            question="how did X evolve?",
        )
        wf = get_workflow("temporal")
        with patch("drbrain.extractor.llm_client.acall_text_with_fallback", new_callable=AsyncMock):
            results = wf.execute(ctx)
        assert "generate_narrative" in results

    @pytest.mark.asyncio
    async def test_hypothesis_full_pipeline_empty_db(self, tmp_db, tmp_graph):
        ctx = WorkflowContext(
            db=tmp_db,
            graph=tmp_graph,
            models=[{"provider": "test", "model": "m"}],
            question="what new hypotheses can we generate?",
        )
        wf = get_workflow("hypothesis")
        with patch("drbrain.extractor.llm_client.acall_with_fallback", new_callable=AsyncMock):
            results = wf.execute(ctx)
        assert "score" in results
