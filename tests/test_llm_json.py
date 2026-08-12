"""Tests for the lenient LLM JSON parser (drbrain.utils.llm_json)."""

import json

import pytest

from drbrain.utils.llm_json import parse_llm_json


class TestDirectParse:
    """Plain valid JSON passes straight through."""

    def test_plain_object(self):
        assert parse_llm_json('{"a": 1}') == {"a": 1}

    def test_nested_object(self):
        text = '{"outer": {"inner": {"deep": [1, 2, {"k": "v"}]}}}'
        assert parse_llm_json(text) == {"outer": {"inner": {"deep": [1, 2, {"k": "v"}]}}}

    def test_non_object_json_passthrough(self):
        """Direct json.loads handles arrays/scalars; only extraction is object-focused."""
        assert parse_llm_json("[1, 2, 3]") == [1, 2, 3]
        assert parse_llm_json('"hello"') == "hello"
        assert parse_llm_json("42") == 42


class TestFenceStripping:
    """```json ... ``` fences around the payload."""

    def test_json_fence(self):
        text = '```json\n{"a": 1, "b": "x"}\n```'
        assert parse_llm_json(text) == {"a": 1, "b": "x"}

    def test_bare_fence(self):
        """Fence without a language tag."""
        assert parse_llm_json('```\n{"ok": true}\n```') == {"ok": True}

    def test_fence_same_line(self):
        assert parse_llm_json('```json {"a": 1} ```') == {"a": 1}

    def test_fence_with_prose_around(self):
        """Fence embedded in surrounding text falls back to block extraction."""
        text = 'Here:\n```json\n{"a": 1}\n```\nThat\'s all.'
        assert parse_llm_json(text) == {"a": 1}


class TestProseAroundJson:
    """JSON embedded in explanatory prose."""

    def test_prefix_and_suffix_prose(self):
        text = 'The result is {"answer": 42}. Hope this helps!'
        assert parse_llm_json(text) == {"answer": 42}

    def test_brace_inside_string(self):
        """A '}' inside a string value must not break extraction."""
        assert parse_llm_json('{"s": "text with } brace"}') == {"s": "text with } brace"}


class TestTruncationTolerance:
    """Recover the longest valid JSON prefix when the tail is garbage/truncated."""

    def test_stray_trailing_braces(self):
        assert parse_llm_json('{"a": 1, "b": 2}}}') == {"a": 1, "b": 2}

    def test_second_object_after_first(self):
        """Over-greedy match: keep only the first complete object."""
        assert parse_llm_json('{"a": 1}{"b": 2}') == {"a": 1}

    def test_truncated_tail_after_complete_prefix(self):
        """Cut off mid-value after the closing brace of a nested fragment."""
        assert parse_llm_json('{"a": 1, "b": {"c": 2}, "d": 3} and more, "e": 4}') == {
            "a": 1,
            "b": {"c": 2},
            "d": 3,
        }


class TestInvalidInput:
    """Inputs that cannot be rescued raise json.JSONDecodeError."""

    def test_no_brace_block(self):
        with pytest.raises(json.JSONDecodeError):
            parse_llm_json("not json at all")

    def test_incomplete_object_no_closing_brace(self):
        with pytest.raises(json.JSONDecodeError):
            parse_llm_json('{"a": 1, "b": [1, 2, 3]')

    def test_empty_string(self):
        with pytest.raises(json.JSONDecodeError):
            parse_llm_json("")

    def test_whitespace_only(self):
        with pytest.raises(json.JSONDecodeError):
            parse_llm_json("   \n  ")
