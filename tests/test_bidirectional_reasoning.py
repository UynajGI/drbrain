"""Tests for bidirectional LLM-KG reasoning loop."""

import tempfile
from pathlib import Path
from unittest import mock

from drbrain.extractor.reasoner import ReasonerAgent
from drbrain.graph.engine import GraphEngine

# ---------------------------------------------------------------------------
# _kg_validate tests
# ---------------------------------------------------------------------------


class TestKgValidate:
    """Tests for _kg_validate method."""

    def test_returns_structure(self):
        """_kg_validate returns dict with consistent, violations, patterns keys."""
        g = GraphEngine()
        g.add_edge("transformer", "attention", "addresses", "p1")

        agent = ReasonerAgent(db=None, graph_engine=g, models=[])
        result = agent._kg_validate("transformer addresses attention")
        assert isinstance(result, dict)
        assert "consistent" in result
        assert "violations" in result
        assert "patterns" in result

    def test_no_graph_returns_consistent(self):
        """No graph loaded: returns consistent=True with empty violations."""
        agent = ReasonerAgent(db=None, graph_engine=None, models=[])
        result = agent._kg_validate("transformer addresses attention")
        assert result["consistent"] is True
        assert result["violations"] == []
        assert result["patterns"] == []

    def test_consistent_hypothesis(self):
        """Hypothesis matching valid edges returns consistent=True."""
        g = GraphEngine()
        g.add_edge("transformer", "NLP", "addresses", "p1")

        agent = ReasonerAgent(db=None, graph_engine=g, models=[])
        result = agent._kg_validate("transformer addresses NLP tasks")
        assert result["consistent"] is True
        assert result["violations"] == []

    def test_detects_tbox_violation_in_graph(self):
        """Existing graph edge with TBox-violating relation is flagged."""
        from drbrain.storage.database import Database

        with tempfile.TemporaryDirectory() as td:
            db = Database(Path(td) / "test.db")
            db.insert_paper("p1", "Test", 2024, "uploaded")
            db.insert_concept("p1", "Method", "transformer", 0.9, year=2024)
            db.insert_concept("p1", "Method", "self-attention", 0.8, year=2024)
            db.commit()

            g = GraphEngine()
            # "leaves_open" is NOT in Method's allowed TBox relations
            g.add_edge("transformer", "self-attention", "leaves_open", "p1")

            agent = ReasonerAgent(db=db, graph_engine=g, models=[])
            result = agent._kg_validate("transformer leaves_open self-attention for NLP")
            assert result["consistent"] is False
            assert len(result["violations"]) >= 1
            db.close()

    def test_detects_asymmetric_violation(self):
        """A -> B and B -> A with asymmetric relation is flagged."""
        g = GraphEngine()
        # "extends" is asymmetric
        g.add_edge("A", "B", "extends", "p1")
        g.add_edge("B", "A", "extends", "p1")

        agent = ReasonerAgent(db=None, graph_engine=g, models=[])
        result = agent._kg_validate("A extends B and B extends A")
        assert result["consistent"] is False
        assert len(result["violations"]) >= 1

    def test_finds_graph_patterns(self):
        """_kg_validate detects debate patterns in the subgraph."""
        g = GraphEngine()
        # Both edges must have "challenges" relation for debate pattern detection
        g.add_edge("method_A", "result_X", "challenges", "p1")
        g.add_edge("method_B", "result_X", "challenges", "p2")

        agent = ReasonerAgent(db=None, graph_engine=g, models=[])
        result = agent._kg_validate("method_A and method_B both challenge result_X")
        assert len(result["patterns"]) >= 1


# ---------------------------------------------------------------------------
# reason_bidirectional tests
# ---------------------------------------------------------------------------


class TestReasonBidirectional:
    """Tests for reason_bidirectional method."""

    def test_no_models_returns_error(self):
        """Returns error dict when no LLM models configured."""
        agent = ReasonerAgent(db=None, graph_engine=None, models=[])
        result = agent.reason_bidirectional("test question")
        assert "error" in result

    def test_iterates_rounds_with_mock_llm(self):
        """Mock LLM returns inconsistent hypothesis each round, then loop exits."""
        g = GraphEngine()
        g.add_edge("A", "B", "addresses", "p1")

        agent = ReasonerAgent(
            db=None, graph_engine=g, models=[{"provider": "test", "model": "test"}]
        )

        # Mock _call_llm to return hypotheses; mock _kg_validate to return inconsistent
        with (
            mock.patch.object(agent, "_call_llm") as mock_llm,
            mock.patch.object(agent, "_kg_validate") as mock_validate,
        ):
            mock_llm.side_effect = [
                "Hypothesis round 1: A addresses B",
                "Hypothesis round 2: A extends B",
                "Hypothesis round 3: A replaces B",
            ]
            # Inconsistent every round
            mock_validate.side_effect = [
                {
                    "consistent": False,
                    "violations": [{"type": "tbox", "reason": "TBox: Method cannot extends"}],
                    "patterns": [],
                },
                {
                    "consistent": False,
                    "violations": [{"type": "tbox", "reason": "TBox: Method cannot extends"}],
                    "patterns": [],
                },
                {
                    "consistent": False,
                    "violations": [{"type": "tbox", "reason": "TBox: Method cannot replaces"}],
                    "patterns": [],
                },
            ]

            result = agent.reason_bidirectional("Does A relate to B?", max_rounds=3)

        assert result["rounds"] == 3
        assert len(result["hypotheses"]) == 3
        assert len(result["kg_validations"]) == 3
        assert "answer" in result

    def test_early_exit_when_consistent(self):
        """Loop exits on first consistent hypothesis."""
        g = GraphEngine()
        g.add_edge("transformer", "NLP", "addresses", "p1")

        agent = ReasonerAgent(
            db=None, graph_engine=g, models=[{"provider": "test", "model": "test"}]
        )

        with (
            mock.patch.object(agent, "_call_llm") as mock_llm,
            mock.patch.object(agent, "_kg_validate") as mock_validate,
        ):
            mock_llm.return_value = "Transformer addresses NLP tasks effectively"
            mock_validate.return_value = {"consistent": True, "violations": [], "patterns": []}

            result = agent.reason_bidirectional("Is transformer effective for NLP?", max_rounds=3)

        assert result["rounds"] == 1
        assert len(result["hypotheses"]) == 1
        assert result["kg_validations"][0]["consistent"] is True

    def test_max_rounds_enforced(self):
        """Loop does not exceed max_rounds."""
        g = GraphEngine()
        g.add_edge("A", "B", "addresses", "p1")

        agent = ReasonerAgent(
            db=None, graph_engine=g, models=[{"provider": "test", "model": "test"}]
        )

        with (
            mock.patch.object(agent, "_call_llm") as mock_llm,
            mock.patch.object(agent, "_kg_validate") as mock_validate,
        ):
            mock_llm.return_value = "Always inconsistent hypothesis"
            mock_validate.return_value = {
                "consistent": False,
                "violations": [{"type": "tbox", "reason": "X"}],
                "patterns": [],
            }

            result = agent.reason_bidirectional("test", max_rounds=2)

        assert result["rounds"] == 2
        assert len(result["hypotheses"]) == 2

    def test_revise_prompt_includes_violations(self):
        """Round 2+ LLM prompt includes previous violations."""
        g = GraphEngine()
        g.add_edge("A", "B", "addresses", "p1")

        agent = ReasonerAgent(
            db=None, graph_engine=g, models=[{"provider": "test", "model": "test"}]
        )

        with (
            mock.patch.object(agent, "_call_llm") as mock_llm,
            mock.patch.object(agent, "_kg_validate") as mock_validate,
        ):
            mock_llm.side_effect = ["H1", "H2"]
            mock_validate.side_effect = [
                {
                    "consistent": False,
                    "violations": [
                        {"type": "tbox", "reason": "TBox violation: Method cannot extends"}
                    ],
                    "patterns": [],
                },
                {"consistent": True, "violations": [], "patterns": []},
            ]

            agent.reason_bidirectional("test", max_rounds=2)

            # Second call should have revision prompt
            second_prompt = mock_llm.call_args_list[1][0][0]
            assert "revise" in second_prompt.lower() or "violation" in second_prompt.lower()


# ---------------------------------------------------------------------------
# Additional tool definitions: not affected
# ---------------------------------------------------------------------------


def test_bidirectional_leaves_tool_definitions_unchanged():
    """Existing tool_definitions() still works after adding bidirectional methods."""
    agent = ReasonerAgent(graph_engine=None, models=[])
    tools = agent.tool_definitions()
    tool_names = [t["function"]["name"] for t in tools]
    assert "search_concepts" in tool_names
    assert "get_neighbors" in tool_names
    assert "find_path" in tool_names
