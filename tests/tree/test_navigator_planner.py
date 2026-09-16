"""T40: one request-scoped, model-driven navigation loop (finding 5).

The navigator's next action is chosen by an injectable planner.  These tests
script the planner deterministically — no live model — and verify the executor
contracts: the read set, expanded nodes, revisions, budgets and unresolved
branches stay auditable; original-text evidence only ever comes from actual
read receipts; a summary-only walk reports insufficient evidence instead of
success.
"""

from __future__ import annotations

import hashlib

import pytest

from drbrain.services.chat_model import ChatModel
from drbrain.services.model_roles import ModelRole
from drbrain.storage.database import Database
from drbrain.tree.blocks import BlockPolicy, build_content_blocks
from drbrain.tree.contracts import ChildRef, LeafRef, NodeRecord, leaf_node_id, region_node_id
from drbrain.tree.navigator import (
    ChatActionPlanner,
    ScriptedPlanner,
    TreeNavigator,
)
from drbrain.tree.search import TreeCandidate
from drbrain.tree.tools import ToolBudget

DOC_P1 = "# Methods\n\nalpha beta gamma delta\n\n# Results\n\nepsilon zeta eta theta\n"
DOC_P2 = "# Methods\n\nlambda mu nu xi\n\n# Results\n\nomicron pi rho sigma\n"


def _doc(db: Database, local_id: str, text: str):
    db.insert_paper(local_id, "T", 2024, "uploaded")
    # Fixture docs stay fine-grained: these tests pin navigation semantics,
    # not the paragraph merge the production writer applies.
    blocks = build_content_blocks(
        text,
        local_id=local_id,
        revision=1,
        media_type="md",
        parser="test",
        policy=BlockPolicy(min_chars=0),
    )
    db.upsert_document_revision(
        local_id,
        1,
        source_hash=f"s-{local_id}",
        canonical_hash=hashlib.sha256(text.encode()).hexdigest(),
        media_type="md",
    )
    db.insert_content_blocks(blocks)
    return blocks


def _leaf(block, local_id: str) -> NodeRecord:
    ref = LeafRef(
        local_id=local_id,
        revision=1,
        block_id=block.block_id,
        char_start=0,
        char_end=len(block.text),
    )
    return NodeRecord(
        node_id=leaf_node_id(ref),
        revision=1,
        kind="leaf",
        state="ready",
        layer=0,
        content_hash=block.text_hash,
        leaf=ref,
        heading_path=block.heading_path,
    )


def _region(children, *, layer=1, summary="joint summary") -> NodeRecord:
    refs = tuple(
        ChildRef(child_id=child.node_id, child_revision=1, ordinal=index)
        for index, child in enumerate(children)
    )
    return NodeRecord(
        node_id=region_node_id(refs, {"prompt": "p"}),
        revision=1,
        kind="region",
        state="ready",
        layer=layer,
        content_hash=hashlib.sha256(summary.encode()).hexdigest(),
        summary=summary,
        children=refs,
        contract={"prompt": "p"},
    )


def _candidate(node: NodeRecord, *, score=0.9) -> TreeCandidate:
    return TreeCandidate(
        node_id=node.node_id,
        kind=node.kind,
        layer=node.layer,
        local_id="",
        score=score,
        profile_id="emb-test",
        node_revision=node.revision,
    )


@pytest.fixture()
def layered(tmp_path):
    """layer2 region → two layer1 regions → four real leaves (two papers)."""
    db = Database(tmp_path / "layered.sqlite")
    blocks_a = _doc(db, "p1", DOC_P1)
    blocks_b = _doc(db, "p2", DOC_P2)
    leaves = [
        _leaf(blocks_a[1], "p1"),
        _leaf(blocks_a[3], "p1"),
        _leaf(blocks_b[1], "p2"),
        _leaf(blocks_b[3], "p2"),
    ]
    for leaf in leaves:
        db.insert_tree_node(leaf, publish=True)
    mid_a = _region(leaves[:2], layer=1, summary="p1 joint summary")
    mid_b = _region(leaves[2:], layer=1, summary="p2 joint summary")
    for mid in (mid_a, mid_b):
        db.insert_tree_node(mid, publish=True)
    top = _region([mid_a, mid_b], layer=2, summary="cross-paper summary")
    db.insert_tree_node(top, publish=True)
    return db, leaves, mid_a, mid_b, top


@pytest.fixture()
def soft_multiparent(tmp_path):
    """Soft multi-parent: region A=[L1, LA] and region B=[L1, LB] share L1."""
    db = Database(tmp_path / "soft-multiparent.sqlite")
    blocks_a = _doc(db, "p1", DOC_P1)
    blocks_b = _doc(db, "p2", DOC_P2)
    shared = _leaf(blocks_a[1], "p1")
    only_a = _leaf(blocks_a[3], "p1")
    only_b = _leaf(blocks_b[1], "p2")
    for leaf in (shared, only_a, only_b):
        db.insert_tree_node(leaf, publish=True)
    region_a = _region([shared, only_a], layer=1, summary="A joint summary")
    region_b = _region([shared, only_b], layer=1, summary="B joint summary")
    for region in (region_a, region_b):
        db.insert_tree_node(region, publish=True)
    return db, shared, only_a, only_b, region_a, region_b


def _role(*, suffix="nav", model="deepseek-flash") -> ModelRole:
    return ModelRole(
        role="chat_model",
        endpoint_name=f"nav_{suffix}",
        provider="openai",
        model=model,
        base_url="https://api.deepseek.com/v1",
        api_key="sk-nav-SECRET",
        source="roles",
    )


def _tool_call(name: str, arguments: dict, call_id: str = "call_1") -> dict:
    import json

    return {
        "text": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
        "finish_reason": "tool_calls",
    }


class TestScriptedWalk:
    def test_scripted_planner_drives_the_whole_tool_loop(self, layered):
        db, leaves, mid_a, mid_b, top = layered
        planner = ScriptedPlanner(
            [
                {"action": "read", "node_id": top.node_id},
                {"action": "expand", "node_id": top.node_id},
                {"action": "read", "node_id": mid_a.node_id},
                {"action": "expand", "node_id": mid_a.node_id},
                {"action": "read", "node_id": leaves[0].node_id},
                {"action": "parents", "node_id": leaves[0].node_id},
                {"action": "read_scope", "node_id": mid_a.node_id},
                {"action": "read", "node_id": mid_b.node_id},
                {"action": "expand", "node_id": mid_b.node_id},
                {"action": "read_scope", "node_id": mid_b.node_id},
                {"action": "finish"},
            ]
        )
        result = TreeNavigator(db).navigate("cross-paper topic", [_candidate(top)], planner=planner)
        actions = [step.action for step in result.trace]
        assert actions[:2] == ["read", "expand"]
        assert "parents" in actions and "read_scope" in actions
        leaf_ids = {item["node_id"] for item in result.evidence if item["source"] == "leaf"}
        assert leaf_ids == {leaf.node_id for leaf in leaves}
        assert result.status == "ok"
        assert result.planner == "ScriptedPlanner"
        # Provenance is kept without duplicating the span.
        leaf_items = [item for item in result.evidence if item["source"] == "leaf"]
        assert {tuple(item["via"]) for item in leaf_items} == {
            (mid_a.node_id,),
            (mid_b.node_id,),
        }
        assert len({receipt.span_key for receipt in result.receipts}) == len(result.receipts)
        # The planner saw the executor's state and every outcome.
        assert planner.contexts and planner.contexts[0]["candidates"]
        assert planner.observations and planner.observations[0][1]["ok"] is True
        parents_outcome = next(
            outcome for action, outcome in planner.observations if action.action == "parents"
        )
        assert parents_outcome["parents"][0]["node_id"] == mid_a.node_id

    def test_fabricated_ids_are_not_evidence(self, layered):
        db, leaves, mid_a, mid_b, top = layered
        planner = ScriptedPlanner(
            [
                {"action": "read", "node_id": "nl-fabricated-by-the-model"},
                {"action": "read", "node_id": leaves[0].node_id},
                {"action": "finish"},
            ]
        )
        result = TreeNavigator(db).navigate("q", [_candidate(leaves[0])], planner=planner)
        assert [step.action for step in result.trace] == ["read_failed", "read"]
        assert {item["node_id"] for item in result.evidence} == {leaves[0].node_id}
        assert any(item["node_id"] == "nl-fabricated-by-the-model" for item in result.unresolved)

    def test_summary_only_walk_reports_insufficient_evidence(self, layered):
        db, leaves, mid_a, mid_b, top = layered
        planner = ScriptedPlanner(
            [
                {"action": "read", "node_id": top.node_id},
                {"action": "finish", "reason": "done"},
            ]
        )
        result = TreeNavigator(db).navigate("q", [_candidate(top)], planner=planner)
        assert result.status == "partial"
        assert result.reason == "no_leaf_evidence"
        assert result.evidence_counts() == {"leaf": 0, "summary": 1}

    def test_no_expansion_ablation_refuses_expansion_actions(self, layered):
        db, leaves, mid_a, mid_b, top = layered
        planner = ScriptedPlanner(
            [
                {"action": "read", "node_id": top.node_id},
                {"action": "expand", "node_id": top.node_id},
                {"action": "finish"},
            ]
        )
        result = TreeNavigator(db).navigate(
            "q", [_candidate(top)], planner=planner, expand_regions=False
        )
        assert [step.action for step in result.trace] == ["read", "expand_refused"]
        assert result.status == "partial" and result.reason == "no_leaf_evidence"
        assert any(item["reason"] == "expansion_disabled" for item in result.unresolved)

    def test_expansion_cap_is_auditable(self, layered):
        db, leaves, mid_a, mid_b, top = layered
        planner = ScriptedPlanner(
            [
                {"action": "read", "node_id": top.node_id},
                {"action": "expand", "node_id": top.node_id},
                {"action": "expand", "node_id": mid_a.node_id},
                {"action": "finish"},
            ]
        )
        result = TreeNavigator(db).navigate(
            "q", [_candidate(top)], planner=planner, max_expansions=1
        )
        assert "expand_refused" in [step.action for step in result.trace]
        assert any(item["reason"] == "expansion_limit" for item in result.unresolved)

    def test_budget_exhaustion_keeps_the_audit_trail(self, layered):
        db, leaves, mid_a, mid_b, top = layered
        from drbrain.services.tokens import count_tokens
        from drbrain.tree.tools import TreeTools

        one_read = count_tokens(TreeTools(db).node_text(leaves[0].node_id))
        planner = ScriptedPlanner(
            [
                {"action": "read", "node_id": leaves[0].node_id},
                {"action": "read", "node_id": leaves[1].node_id},
                {"action": "finish"},
            ]
        )
        navigator = TreeNavigator(db, budget=ToolBudget(max_tokens=one_read + 1, max_calls=100))
        result = navigator.navigate(
            "q", [_candidate(leaves[0]), _candidate(leaves[1])], planner=planner
        )
        assert result.status == "partial" and result.reason == "budget_exhausted"
        assert result.budget["truncated"] is True
        assert [step.action for step in result.trace][:2] == ["read", "read_failed"]
        assert any(item["node_id"] == leaves[1].node_id for item in result.unresolved)

    def test_search_nodes_action_reenters_with_a_new_query(self, layered):
        db, leaves, mid_a, mid_b, top = layered
        calls: list[str] = []

        def search_nodes(text: str):
            calls.append(text)
            return [_candidate(leaves[2], score=0.8)]

        planner = ScriptedPlanner(
            [
                {"action": "search_nodes", "query": "another part of the question"},
                {"action": "read", "node_id": leaves[2].node_id},
                {"action": "finish"},
            ]
        )
        navigator = TreeNavigator(db, search_nodes=search_nodes)
        result = navigator.navigate("q", [_candidate(mid_b)], planner=planner)
        assert calls == ["another part of the question"]
        assert [step.action for step in result.trace][:2] == ["search_nodes", "read"]
        assert {item["node_id"] for item in result.evidence} == {leaves[2].node_id}
        assert result.status == "ok"


class TestMultiParentOrigins:
    def test_a_leaf_reached_through_two_parents_keeps_both_origins(self, soft_multiparent):
        db, shared, only_a, only_b, region_a, region_b = soft_multiparent
        planner = ScriptedPlanner(
            [
                {"action": "read", "node_id": region_a.node_id},
                {"action": "expand", "node_id": region_a.node_id},
                {"action": "read", "node_id": shared.node_id},
                {"action": "read", "node_id": region_b.node_id},
                {"action": "expand", "node_id": region_b.node_id},
                {"action": "read", "node_id": shared.node_id},
                {"action": "finish"},
            ]
        )
        result = TreeNavigator(db).navigate(
            "q", [_candidate(region_a), _candidate(region_b, score=0.8)], planner=planner
        )
        shared_rows = [item for item in result.evidence if item["node_id"] == shared.node_id]
        assert len(shared_rows) == 1, "one span must not become two evidence rows"
        assert shared_rows[0]["via"] == [region_a.node_id, region_b.node_id]
        # The second read re-confirms the origin without duplicating the span.
        assert len({receipt.span_key for receipt in result.receipts}) == len(result.receipts)


class TestChatPlanner:
    def _model(self, transport, *, suffix="a") -> ChatModel:
        return ChatModel(role=_role(suffix=suffix), transport=transport, max_attempts=1)

    def test_tool_calls_drive_the_loop_and_results_are_fed_back(self, layered):
        db, leaves, mid_a, mid_b, top = layered
        requests: list[dict] = []
        script = [
            _tool_call("read", {"node_id": top.node_id}, "call-1"),
            _tool_call("expand", {"node_id": top.node_id}, "call-2"),
            _tool_call("read", {"node_id": leaves[0].node_id}, "call-3"),
            {"text": "done", "tool_calls": [], "finish_reason": "stop"},
        ]

        def transport(request):
            requests.append(dict(request))
            return script[len(requests) - 1]

        model = self._model(transport)
        result = TreeNavigator(db).navigate(
            "q", [_candidate(top)], planner=ChatActionPlanner(model)
        )
        assert result.status == "ok"
        assert [item["node_id"] for item in result.evidence] == [top.node_id, leaves[0].node_id]
        # The model got the tool surface, and the executor fed the expansion
        # result back as a tool message before the next decision.
        assert any(tool["function"]["name"] == "read" for tool in requests[0]["tools"])
        assert any(message.get("role") == "tool" for message in requests[1]["messages"]), (
            "the expand observation must travel back to the model"
        )
        assert result.planner == "ChatActionPlanner"

    def test_every_tool_call_gets_a_response_before_the_next_request(self, layered):
        """A multi-call answer is fully answered; strict endpoints 400 otherwise.

        DeepSeek rejects a request whose assistant message carries tool_calls
        that lack tool responses ("insufficient tool messages following
        tool_calls"), so every id must be answered — the executed action's
        result for the first, the one-action-per-round refusal for the rest.
        """
        import json as _json

        db, leaves, mid_a, mid_b, top = layered
        requests: list[dict] = []
        multi = {
            "text": "",
            "tool_calls": [
                {
                    "id": "call-a",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": _json.dumps({"node_id": top.node_id}),
                    },
                },
                {
                    "id": "call-b",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": _json.dumps({"node_id": mid_a.node_id}),
                    },
                },
            ],
            "finish_reason": "tool_calls",
        }
        script = [
            multi,
            _tool_call("read", {"node_id": leaves[0].node_id}, "call-c"),
            {"text": "done", "tool_calls": [], "finish_reason": "stop"},
        ]

        def transport(request):
            requests.append(dict(request))
            return script[len(requests) - 1]

        result = TreeNavigator(db).navigate(
            "q", [_candidate(top)], planner=ChatActionPlanner(self._model(transport))
        )

        assert result.status == "ok"
        second = [message for message in requests[1]["messages"] if message.get("role") == "tool"]
        answered = {message.get("tool_call_id") for message in second}
        assert {"call-a", "call-b"} <= answered, second
        refusal = next(message for message in second if message.get("tool_call_id") == "call-b")
        assert "one action per round" in refusal["content"]
        # Only the first call's action ran; the refused sibling was not read.
        assert mid_a.node_id not in {item["node_id"] for item in result.evidence}
        assert [item["node_id"] for item in result.evidence] == [top.node_id, leaves[0].node_id]

    def test_text_only_answer_ends_the_walk_without_leaf_evidence(self, layered):
        db, leaves, mid_a, mid_b, top = layered
        script = [
            _tool_call("read", {"node_id": top.node_id}, "call-1"),
            {"text": "I am done", "tool_calls": [], "finish_reason": "stop"},
        ]
        calls = {"n": 0}

        def transport(request):
            response = script[calls["n"]]
            calls["n"] += 1
            return response

        result = TreeNavigator(db).navigate(
            "q", [_candidate(top)], planner=ChatActionPlanner(self._model(transport))
        )
        assert result.status == "partial" and result.reason == "no_leaf_evidence"

    def test_invalid_tool_arguments_are_reported_and_the_walk_continues(self, layered):
        db, leaves, mid_a, mid_b, top = layered
        script = [
            _tool_call("read", {"node_id": ""}, "call-1"),
            _tool_call("read", {"node_id": leaves[0].node_id}, "call-2"),
            {"text": "done", "tool_calls": [], "finish_reason": "stop"},
        ]
        calls = {"n": 0}

        def transport(request):
            response = script[calls["n"]]
            calls["n"] += 1
            return response

        result = TreeNavigator(db).navigate(
            "q", [_candidate(leaves[0])], planner=ChatActionPlanner(self._model(transport))
        )
        assert "invalid_action" in [step.action for step in result.trace]
        assert {item["node_id"] for item in result.evidence} == {leaves[0].node_id}
        assert result.status == "ok"

    def test_chat_failure_is_a_reported_planner_state(self, layered):
        db, leaves, mid_a, mid_b, top = layered

        def transport(request):
            raise ConnectionError("endpoint down")

        result = TreeNavigator(db).navigate(
            "q", [_candidate(top)], planner=ChatActionPlanner(self._model(transport))
        )
        assert [step.action for step in result.trace] == ["planner_failed"]
        assert result.status == "empty" and result.reason == "planner_failed"
