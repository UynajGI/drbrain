"""P-E1 async-job lifecycle contract tests (review 2026-09-03 §4.1 / §7.2).

Covers: :class:`JobMethods` capability declaration with ``NotImplementedError``
defaults, the registry ``submit_job``/``poll_job``/``cancel_job`` surface,
``PluginResult.job_id``/``artifacts`` plumbing, and the on-disk ``jobs/``
directory contract being documented in the package docstring.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from drbrain.plugins import (
    Artifact,
    JobMethods,
    JobStatus,
    Plugin,
    PluginRegistry,
    PluginResult,
    ResultStatus,
)

PLUGINS_PKG_DIR = Path(__file__).resolve().parents[1] / "src" / "drbrain" / "plugins"


class _MemoryJobs:
    """Minimal in-memory submit/poll/cancel triple for contract tests."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._counter = 0

    def submit(self, arguments: dict) -> str:
        self._counter += 1
        job_id = f"job-{self._counter}"
        self._jobs[job_id] = {"status": JobStatus.DONE, "result": {"echo": arguments}}
        return job_id

    def poll(self, job_id: str) -> dict:
        job = self._jobs[job_id]
        return {"status": job["status"], "result": job["result"]}

    def cancel(self, job_id: str) -> bool:
        return job_id in self._jobs


def _plugin(**overrides) -> Plugin:
    return Plugin(name="run_dft", description="hour-scale compute", input_schema={}, **overrides)


def test_job_status_enum_covers_lifecycle():
    assert {s.value for s in JobStatus} == {
        "pending",
        "running",
        "done",
        "failed",
        "cancelled",
    }


def test_default_job_methods_raise_not_implemented():
    """未注册的方法保持协议默认：显式 NotImplementedError，而不是静默出错。"""
    jobs = JobMethods()
    with pytest.raises(NotImplementedError):
        jobs.submit({})
    with pytest.raises(NotImplementedError):
        jobs.poll("job-1")
    with pytest.raises(NotImplementedError):
        jobs.cancel("job-1")


def test_submit_poll_cancel_round_trip():
    registry = PluginRegistry()
    memory = _MemoryJobs()
    registry.register(
        _plugin(),
        lambda args: {"submitted": True},
        jobs=JobMethods(submit=memory.submit, poll=memory.poll, cancel=memory.cancel),
    )
    assert registry.supports_jobs("run_dft")

    job_id = registry.submit_job("run_dft", {"structure": "Fe3O4"})
    assert job_id == "job-1"

    polled = registry.poll_job("run_dft", job_id)
    assert polled["status"] is JobStatus.DONE
    assert polled["result"] == {"echo": {"structure": "Fe3O4"}}

    assert registry.cancel_job("run_dft", job_id) is True


def test_plugins_without_job_methods_raise_not_implemented():
    """向后兼容：未声明作业能力的现有插件照常工作，作业调用显式失败。"""
    registry = PluginRegistry()
    registry.register(_plugin(), lambda args: {"ok": True})
    assert not registry.supports_jobs("run_dft")
    # 同步调用不受影响
    assert registry.call("run_dft", {}).data == {"ok": True}
    with pytest.raises(NotImplementedError):
        registry.submit_job("run_dft", {})
    with pytest.raises(NotImplementedError):
        registry.poll_job("run_dft", "job-1")
    with pytest.raises(NotImplementedError):
        registry.cancel_job("run_dft", "job-1")


def test_plugin_result_carries_job_id_and_artifacts():
    """Handler 返回的 job_id / artifacts 必须原样穿过 registry.call。"""
    registry = PluginRegistry()
    artifact = Artifact(path="jobs/job-9.json", sha256="deadbeef")

    def handler(args):
        return PluginResult(
            ResultStatus.OK,
            data={"job_id": "job-9"},
            job_id="job-9",
            artifacts=[artifact],
        )

    registry.register(_plugin(), handler)
    result = registry.call("run_dft", {})

    assert result.ok
    assert result.job_id == "job-9"
    assert result.artifacts == [artifact]
    message = result.to_llm_message()
    assert "job_id=job-9" in message
    assert "jobs/job-9.json" in message


def test_registry_default_result_has_no_job_fields():
    registry = PluginRegistry()
    registry.register(_plugin(), lambda args: {"ok": True})
    result = registry.call("run_dft", {})
    assert result.job_id is None
    assert result.artifacts == []
    assert result.truncated is False


def test_jobs_dir_contract_is_documented():
    """jobs/ 目录契约必须写进包文档（这是 T4 门唯一信任的证据）。"""
    package_doc = (PLUGINS_PKG_DIR / "__init__.py").read_text(encoding="utf-8")
    assert "jobs/<job_id>.json" in package_doc
    assert "jobs/<job_id>.log" in package_doc
    assert "T4" in package_doc
