"""P-E2 process-isolation tests: timed-out plugin calls are genuinely reclaimable.

By default :meth:`PluginRegistry.call` runs each handler in a disposable
spawn process and SIGKILLs it on timeout, so stuck compute can never wedge the
registry. Handlers/arguments that cannot cross a process boundary (lambdas,
unimportable closures) fall back to the legacy shared-thread path with
identical semantics — ``ResultStatus.TIMEOUT`` and friends are unchanged.
"""

from __future__ import annotations

import os
import threading
import time

from drbrain.plugins import Plugin, PluginRegistry, ResultStatus


def _echo_pid(arguments: dict) -> dict:
    """Module-level handler: reports the pid it actually ran in."""
    return {"pid": os.getpid(), "echo": arguments.get("x")}


def _sleep_60(arguments: dict) -> dict:
    time.sleep(60)
    return {}


def _return_lock(arguments):
    return threading.Lock()  # 不可 pickle 的返回值


def _register(registry: PluginRegistry, name: str, handler, **overrides):
    registry.register(Plugin(name=name, description="d", input_schema={}, **overrides), handler)


def test_picklable_handler_runs_in_child_process():
    registry = PluginRegistry()
    _register(registry, "iso_echo", _echo_pid, timeout_s=30.0)
    result = registry.call("iso_echo", {"x": 42})
    assert result.ok, result.error
    assert result.data["echo"] == 42
    assert result.data["pid"] != os.getpid()  # 确实在子进程里跑


def test_timeout_kills_worker_and_returns_timeout_status():
    """60s 睡眠的 worker 在 1s 超时后被 SIGKILL 真正回收，registry 不被卡死。"""
    registry = PluginRegistry()
    _register(registry, "iso_echo", _echo_pid, timeout_s=30.0)
    _register(registry, "iso_slow", _sleep_60, timeout_s=1.0)

    started = time.monotonic()
    result = registry.call("iso_slow", {})
    elapsed = time.monotonic() - started
    assert result.status is ResultStatus.TIMEOUT  # 超时异常类型保持兼容
    assert "调用超时" in (result.error or "")
    assert elapsed < 15.0, elapsed

    # 超时后 registry 仍可用：快速调用照常成功
    follow = registry.call("iso_echo", {"x": 1})
    assert follow.ok, follow.error
    assert follow.data["echo"] == 1


def test_unpicklable_handler_falls_back_to_thread_path():
    """lambda handler 不可 pickle → 自动回退线程路径（旧行为）。"""
    registry = PluginRegistry()
    _register(registry, "lambda_fn", lambda args: {"via": "thread"}, timeout_s=10.0)
    result = registry.call("lambda_fn", {})
    assert result.ok, result.error
    assert result.data == {"via": "thread"}


def test_unpicklable_handler_timeout_still_reports_timeout():
    """线程回退路径的超时语义与旧实现一致。"""
    registry = PluginRegistry()
    _register(registry, "lambda_slow", lambda args: time.sleep(5), timeout_s=0.2)
    result = registry.call("lambda_slow", {})
    assert result.status is ResultStatus.TIMEOUT


def test_unpicklable_result_yields_model_unavailable():
    """返回值无法跨进程时报 PLUGIN_ERROR（P-I5），而不是崩溃或假装成功。"""
    registry = PluginRegistry()
    _register(registry, "iso_lock", _return_lock, timeout_s=10.0)
    result = registry.call("iso_lock", {})
    assert result.status is ResultStatus.PLUGIN_ERROR
    assert result.error


def test_discovered_file_plugin_runs_isolated(tmp_path):
    """discover() 以 spec 方式加载的插件（模块名不可 import）也走进程隔离。"""
    (tmp_path / "iso_plugin.py").write_text(
        "import os\n"
        "def _run(arguments):\n"
        "    return {'pid': os.getpid()}\n"
        "def register(registry):\n"
        "    from drbrain.plugins import Plugin\n"
        "    registry.register(\n"
        "        Plugin(name='iso', description='d', input_schema={}), _run\n"
        "    )\n",
        encoding="utf-8",
    )
    registry = PluginRegistry()
    assert registry.discover(tmp_path) == 1
    result = registry.call("iso", {})
    assert result.ok, result.error
    assert result.data["pid"] != os.getpid()


def test_process_isolation_can_be_disabled():
    """显式关闭进程隔离 → 走线程路径，与调用方同进程（旧行为）。"""
    registry = PluginRegistry(process_isolation=False)
    _register(registry, "iso_echo", _echo_pid, timeout_s=30.0)
    result = registry.call("iso_echo", {"x": 1})
    assert result.ok, result.error
    assert result.data["pid"] == os.getpid()
