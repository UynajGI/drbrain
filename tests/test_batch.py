"""Tests for drbrain.extractor.batch.BatchLLMProcessor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from drbrain.extractor.batch import BatchLLMProcessor
from drbrain.utils.llm_json import parse_llm_json


class _FakeProcessor(BatchLLMProcessor):
    """Deterministic fake: todo ("a","b","c"), fixed JSON LLM responses.

    ``crash_on`` items raise ``KeyboardInterrupt`` (a BaseException) inside
    ``_call_llm`` — aborts the whole run like an operator Ctrl-C.
    ``fail_on`` items raise a normal exception — counted as failures.
    """

    def __init__(
        self,
        output_path: str | Path,
        checkpoint_path: str | Path,
        done_flag_path: str | Path,
        *,
        todo: tuple[str, ...] = ("a", "b", "c"),
        batch_size: int = 64,
        crash_on: tuple[str, ...] = (),
        fail_on: tuple[str, ...] = (),
    ) -> None:
        super().__init__(output_path, checkpoint_path, done_flag_path, batch_size=batch_size)
        self.todo = list(todo)
        self.crash_on = set(crash_on)
        self.fail_on = set(fail_on)
        self.prompts: list[str] = []

    def build_todo(self) -> list[str]:
        return list(self.todo)

    def build_prompt(self, item: str) -> str:
        return f"EXTRACT {item}"

    def _call_llm(self, prompt: str) -> str:
        self.prompts.append(prompt)
        item = prompt.split()[-1]
        if item in self.crash_on:
            raise KeyboardInterrupt
        if item in self.fail_on:
            raise RuntimeError(f"boom: {item}")
        return json.dumps({"label": item, "ok": True})

    def process_one(self, item: str, raw_response: str) -> dict:
        parsed = parse_llm_json(raw_response)
        parsed["item"] = item
        return parsed


class _NoCallLLM(BatchLLMProcessor):
    """Implements the three abstracts but not the _call_llm hook."""

    def build_todo(self) -> list[str]:
        return ["x"]

    def build_prompt(self, item: str) -> str:
        return item

    def process_one(self, item: str, raw_response: str) -> dict:
        return {"item": item}


def _paths(tmp_path: Path, name: str = "run") -> tuple[Path, Path, Path]:
    return (
        tmp_path / f"{name}.jsonl",
        tmp_path / f"{name}_offset.json",
        tmp_path / f"{name}_done.flag",
    )


def _records(path: Path) -> list[dict]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def test_full_run(tmp_path):
    out, ckpt, flag = _paths(tmp_path)
    proc = _FakeProcessor(out, ckpt, flag, batch_size=2)
    stats = proc.run(concurrency=2)
    assert stats.ok == 3
    assert stats.fail == 0
    assert stats.total == 3
    assert stats.offset == 3
    records = _records(out)
    assert len(records) == 3
    assert {r["item"] for r in records} == {"a", "b", "c"}
    assert ckpt.read_text(encoding="utf-8").strip() == "3"


def test_resume_continues_from_offset_without_duplicates(tmp_path):
    out, ckpt, flag = _paths(tmp_path)
    # First run: batch [a,b] completes, item c crashes mid-run.
    first = _FakeProcessor(out, ckpt, flag, batch_size=2, crash_on={"c"})
    with pytest.raises(KeyboardInterrupt):
        first.run(concurrency=1)
    assert [r["item"] for r in _records(out)] == ["a", "b"]
    assert ckpt.read_text(encoding="utf-8").strip() == "2"
    assert not flag.exists()

    # Second run resumes from offset 2 and only processes c.
    second = _FakeProcessor(out, ckpt, flag, batch_size=2)
    stats = second.run(concurrency=1, resume=True)
    assert stats.ok == 1
    assert [r["item"] for r in _records(out)] == ["a", "b", "c"]
    assert ckpt.read_text(encoding="utf-8").strip() == "3"
    assert flag.exists()


def test_stale_offset_out_of_bounds_resets(tmp_path):
    out, ckpt, flag = _paths(tmp_path)
    ckpt.write_text("10\n", encoding="utf-8")  # stale offset > len(todo) == 3
    proc = _FakeProcessor(out, ckpt, flag)
    stats = proc.run(concurrency=1, resume=True)
    assert stats.ok == 3  # reset to 0, nothing skipped
    assert {r["item"] for r in _records(out)} == {"a", "b", "c"}


def test_done_flag_written_on_completion(tmp_path):
    out, ckpt, flag = _paths(tmp_path)
    proc = _FakeProcessor(out, ckpt, flag)
    proc.run()
    assert flag.exists()
    content = flag.read_text(encoding="utf-8").strip()
    assert "ok=3" in content


def test_jsonl_output_format(tmp_path):
    out, ckpt, flag = _paths(tmp_path)
    proc = _FakeProcessor(out, ckpt, flag)
    proc.run(concurrency=2)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for line in lines:
        record = json.loads(line)  # every line is standalone JSON
        assert set(record) == {"item", "result", "ts"}
        assert record["result"] == {"label": record["item"], "ok": True, "item": record["item"]}


def test_item_failures_counted_not_written(tmp_path):
    out, ckpt, flag = _paths(tmp_path)
    proc = _FakeProcessor(out, ckpt, flag, fail_on={"b"})
    stats = proc.run(concurrency=2)
    assert stats.ok == 2
    assert stats.fail == 1
    assert {r["item"] for r in _records(out)} == {"a", "c"}
    assert flag.exists()  # done flag still written despite failures


def test_resume_false_truncates_output_and_restarts(tmp_path):
    out, ckpt, flag = _paths(tmp_path)
    proc = _FakeProcessor(out, ckpt, flag)
    proc.run()
    ckpt.write_text("3\n", encoding="utf-8")
    out.write_text("stale line\n", encoding="utf-8")
    stats = proc.run(resume=False)
    assert stats.ok == 3
    assert len(_records(out)) == 3  # stale line gone, no duplicates


def test_empty_todo_completes_immediately(tmp_path):
    out, ckpt, flag = _paths(tmp_path)
    proc = _FakeProcessor(out, ckpt, flag, todo=())
    stats = proc.run()
    assert stats.ok == 0
    assert stats.total == 0
    assert flag.exists()


def test_invalid_concurrency_rejected(tmp_path):
    out, ckpt, flag = _paths(tmp_path)
    proc = _FakeProcessor(out, ckpt, flag)
    with pytest.raises(ValueError):
        proc.run(concurrency=0)


def test_default_call_llm_raises_not_implemented(tmp_path):
    out, ckpt, flag = _paths(tmp_path)
    proc = _NoCallLLM(out, ckpt, flag)
    with pytest.raises(NotImplementedError):
        proc.run()
