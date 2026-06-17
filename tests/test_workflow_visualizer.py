"""Tests for WorkflowVisualizer — pipeline diagrams and result summaries."""

from unittest.mock import AsyncMock, patch

import pytest

from drbrain.graph.engine import GraphEngine
from drbrain.reasoning import WorkflowContext, get_workflow
from drbrain.reasoning.visualizer import WorkflowVisualizer
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


class TestMermaidFlowchart:
    def test_contains_flowchart_header(self):
        wf = get_workflow("causal")
        vz = WorkflowVisualizer(wf)
        mermaid = vz.mermaid_flowchart()
        assert "```mermaid" in mermaid
        assert "flowchart TD" in mermaid

    def test_all_steps_present(self):
        wf = get_workflow("review")
        vz = WorkflowVisualizer(wf)
        mermaid = vz.mermaid_flowchart()
        for step in wf.steps:
            assert step.name.replace("_", " ") in mermaid

    def test_llm_steps_styled_orange(self):
        wf = get_workflow("causal")
        vz = WorkflowVisualizer(wf)
        mermaid = vz.mermaid_flowchart()
        # The synthesize step requires LLM — should have orange style
        assert "#fff3e0" in mermaid or "#ff9800" in mermaid

    def test_symbolic_steps_styled_blue(self):
        wf = get_workflow("causal")
        vz = WorkflowVisualizer(wf)
        mermaid = vz.mermaid_flowchart()
        # extract_entities is symbolic — should have blue style
        assert "#e3f2fd" in mermaid or "#2196f3" in mermaid

    def test_has_arrows_between_steps(self):
        wf = get_workflow("temporal")
        vz = WorkflowVisualizer(wf)
        mermaid = vz.mermaid_flowchart()
        assert "-->" in mermaid

    def test_has_start_and_end_markers(self):
        wf = get_workflow("impact")
        vz = WorkflowVisualizer(wf)
        mermaid = vz.mermaid_flowchart()
        assert "start" in mermaid.lower()
        assert "output" in mermaid.lower() or "Answer" in mermaid


class TestTextFlowchart:
    def test_contains_workflow_name(self):
        wf = get_workflow("causal")
        vz = WorkflowVisualizer(wf)
        text = vz.text_flowchart()
        assert "causal" in text

    def test_contains_step_names(self):
        wf = get_workflow("contradiction")
        vz = WorkflowVisualizer(wf)
        text = vz.text_flowchart()
        for step in wf.steps:
            assert step.name.replace("_", " ") in text

    def test_has_arrow_indicators(self):
        wf = get_workflow("hypothesis")
        vz = WorkflowVisualizer(wf)
        text = vz.text_flowchart()
        assert "▼" in text

    def test_has_llm_and_sym_markers(self):
        wf = get_workflow("review")
        vz = WorkflowVisualizer(wf)
        text = vz.text_flowchart()
        assert "LLM" in text
        assert "SYM" in text


class TestSummarizeResults:
    def test_shows_all_steps(self, tmp_db, tmp_graph):
        ctx = WorkflowContext(
            db=tmp_db,
            graph=tmp_graph,
            models=[{"provider": "test", "model": "m"}],
            question="test",
        )
        wf = get_workflow("causal")
        vz = WorkflowVisualizer(wf)

        # Run with mocked LLM
        with patch("drbrain.extractor.llm_client.acall_text_with_fallback", new_callable=AsyncMock):
            wf.execute(ctx)

        summary = vz.summarize_results(ctx)
        for step in wf.steps:
            assert step.name in summary

    def test_shows_status_icons(self, tmp_db, tmp_graph):
        ctx = WorkflowContext(
            db=tmp_db,
            graph=tmp_graph,
            models=[{"provider": "test", "model": "m"}],
            question="test",
        )
        wf = get_workflow("gap-analysis")
        vz = WorkflowVisualizer(wf)

        with patch("drbrain.extractor.llm_client.acall_text_with_fallback", new_callable=AsyncMock):
            wf.execute(ctx)

        summary = vz.summarize_results(ctx)
        assert "✅" in summary or "❌" in summary

    def test_list_results_show_count(self, tmp_db, tmp_graph):
        ctx = WorkflowContext(
            db=tmp_db,
            graph=tmp_graph,
            models=[{"provider": "test", "model": "m"}],
            question="test",
        )
        wf = get_workflow("contradiction")
        vz = WorkflowVisualizer(wf)

        with patch("drbrain.extractor.llm_client.acall_with_fallback", new_callable=AsyncMock):
            wf.execute(ctx)

        summary = vz.summarize_results(ctx)
        # scan_debates returns a list — should show item count
        assert "item" in summary or "empty" in summary

    def test_empty_db_graceful(self, tmp_db, tmp_graph):
        ctx = WorkflowContext(
            db=tmp_db,
            graph=tmp_graph,
            models=[{"provider": "test", "model": "m"}],
            question="test",
        )
        wf = get_workflow("impact")
        vz = WorkflowVisualizer(wf)

        with patch("drbrain.extractor.llm_client.acall_text_with_fallback", new_callable=AsyncMock):
            wf.execute(ctx)

        summary = vz.summarize_results(ctx)
        assert "impact" in summary.lower() or "Pipeline Results" in summary
