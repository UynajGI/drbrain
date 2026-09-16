"""T26: unified summary service, cache identity and failure handling."""

from __future__ import annotations

import pytest

from drbrain.storage.database import Database
from drbrain.tree.summary import (
    SummaryContract,
    SummaryMember,
    SummaryResponse,
    SummaryService,
    build_prompt,
    cache_key,
)


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[tuple[str, int]] = []

    def complete(self, prompt: str, *, max_tokens: int) -> SummaryResponse:
        self.calls.append((prompt, max_tokens))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _members(count: int = 2):
    return [
        SummaryMember(
            node_id=f"nl-{index}",
            node_revision=1,
            content_hash=f"hash-{index}",
            text=f"text {index}",
            local_id="p1",
            heading_path=("Intro",),
        )
        for index in range(count)
    ]


def _contract(**overrides) -> SummaryContract:
    base = {
        "model": "spark-x25-4b",
        "tokenizer": "o200k_base",
        "max_output_tokens": 64,
        "input_budget": 1000,
    }
    base.update(overrides)
    return SummaryContract(**base)


def _count(text: str) -> int:
    return max(1, len(text.split()))


@pytest.fixture()
def db(tmp_path):
    return Database(tmp_path / "db.sqlite")


class TestReuse:
    def test_same_members_and_contract_call_once(self, db):
        service = SummaryService(db, count_tokens=_count)
        model = FakeModel([SummaryResponse("one two three")])
        members = _members()
        first = service.summarize(members, _contract(), model)
        assert first.ok and not first.from_cache
        second = service.summarize(members, _contract(), model)
        assert second.ok and second.from_cache and second.summary == "one two three"
        assert len(model.calls) == 1 and service.model_calls == 1

    def test_different_order_or_members_miss(self, db):
        service = SummaryService(db, count_tokens=_count)
        model = FakeModel([SummaryResponse("a"), SummaryResponse("b")])
        ordered = _contract(member_order="given")
        service.summarize(_members(2), ordered, model)
        service.summarize(list(reversed(_members(2))), ordered, model)
        assert len(model.calls) == 2
        # Sorted order makes the same set hit regardless of input order.
        sorted_contract = _contract(member_order="sorted")
        model2 = FakeModel([SummaryResponse("sorted")])
        service.summarize(_members(2), sorted_contract, model2)
        service.summarize(list(reversed(_members(2))), sorted_contract, model2)
        assert len(model2.calls) == 1

    def test_prompt_model_and_budget_participate_in_identity(self, db):
        service = SummaryService(db, count_tokens=_count)
        model = FakeModel([SummaryResponse("x"), SummaryResponse("y"), SummaryResponse("z")])
        members = _members(2)
        service.summarize(members, _contract(), model)
        service.summarize(members, _contract(model="other-model"), model)
        service.summarize(members, _contract(prompt_id="tree-summarize-v2"), model)
        assert len(model.calls) == 3
        assert cache_key(members, _contract()) == cache_key(members, _contract())

    def test_scope_change_forces_new_summary(self, db):
        service = SummaryService(db, count_tokens=_count)
        model = FakeModel([SummaryResponse("two members"), SummaryResponse("three members")])
        service.summarize(_members(2), _contract(), model)
        service.summarize(_members(3), _contract(), model)
        assert len(model.calls) == 2


class TestFailures:
    def test_empty_response_is_recorded_as_failure_not_cached_success(self, db):
        service = SummaryService(db, count_tokens=_count)
        model = FakeModel([SummaryResponse("   ")])
        outcome = service.summarize(_members(), _contract(), model)
        assert not outcome.ok and outcome.reason == "empty_summary"
        entry = db.get_summary_cache(cache_key(_members(), _contract()))
        assert entry["state"] == "failed" and entry["summary"] == ""

    def test_truncated_response_rejected(self, db):
        service = SummaryService(db, count_tokens=_count)
        model = FakeModel([SummaryResponse("cut off", finish_reason="length")])
        outcome = service.summarize(_members(), _contract(), model)
        assert not outcome.ok and outcome.reason == "summary_truncated"

    def test_over_budget_summary_rejected(self, db):
        service = SummaryService(db, count_tokens=_count)
        long_summary = " ".join(["word"] * 100)
        model = FakeModel([SummaryResponse(long_summary)])
        outcome = service.summarize(_members(), _contract(max_output_tokens=16), model)
        assert not outcome.ok and outcome.reason == "summary_over_budget"

    def test_model_exception_is_reported_not_swallowed(self, db):
        service = SummaryService(db, count_tokens=_count)
        model = FakeModel([RuntimeError("endpoint down")])
        outcome = service.summarize(_members(), _contract(), model)
        assert not outcome.ok and outcome.reason.startswith("model_error")
        assert db.get_summary_cache(cache_key(_members(), _contract()))["state"] == "failed"

    def test_failed_entry_is_retried_not_reused(self, db):
        service = SummaryService(db, count_tokens=_count)
        failing = FakeModel([SummaryResponse("")])
        service.summarize(_members(), _contract(), failing)
        recovering = FakeModel([SummaryResponse("good summary")])
        outcome = service.summarize(_members(), _contract(), recovering)
        assert outcome.ok and not outcome.from_cache
        assert len(recovering.calls) == 1
        assert db.get_summary_cache(cache_key(_members(), _contract()))["state"] == "ready"

    def test_input_over_budget_skips_model_call(self, db):
        service = SummaryService(db, count_tokens=_count)
        model = FakeModel([SummaryResponse("never")])
        outcome = service.summarize(_members(40), _contract(input_budget=5), model)
        assert not outcome.ok and outcome.reason == "input_over_budget"
        assert model.calls == []


class TestPromptContract:
    def test_prompt_carries_member_identity_and_boundaries(self):
        prompt = build_prompt(_members(2), _contract())
        assert "id=nl-0" in prompt and "id=nl-1" in prompt
        assert "paper=p1" in prompt and "heading=Intro" in prompt
        assert prompt.count("[/member]") == 2

    def test_contract_validation(self):
        with pytest.raises(ValueError, match="members"):
            SummaryContract(template="no slot here")
        with pytest.raises(ValueError, match="member_order"):
            SummaryContract(member_order="random")
