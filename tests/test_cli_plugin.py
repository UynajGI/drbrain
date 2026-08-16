"""CLI plugin tests: ``shell_command`` exposes external command execution to the agent."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from drbrain.plugins import PluginRegistry

PLUGIN_PATH = Path(__file__).resolve().parents[1] / "research" / "plugins" / "shell_command.py"

pytestmark = pytest.mark.skipif(
    not PLUGIN_PATH.exists(),
    reason="external CLI plugin not present (research/plugins/)",
)


def _load():
    spec = importlib.util.spec_from_file_location("drbrain_plugin_shell_command", PLUGIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shell_command_registers_as_software():
    registry = PluginRegistry()
    _load().register(registry)
    plugin = registry.get("shell_command")
    assert plugin.plugin_type == "software"
    assert plugin.backend == "subprocess"


def test_shell_command_runs_echo():
    result = _load()._run({"command": "echo", "args": ["hello"]})
    assert result["returncode"] == 0
    assert "hello" in result["stdout"]


def test_shell_command_rejects_empty_command():
    with pytest.raises(ValueError):
        _load()._run({"command": ""})
