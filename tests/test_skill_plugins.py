"""Skill plugin tests: list_skills / read_skill expose the skills/ directory to the agent."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from drbrain.plugins import PluginRegistry

PLUGINS_DIR = Path(__file__).resolve().parents[1] / "research" / "plugins"

pytestmark = pytest.mark.skipif(
    not (PLUGINS_DIR / "list_skills.py").exists(),
    reason="external skill plugins not present (research/plugins/)",
)


def _load(name: str):
    path = PLUGINS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"drbrain_plugin_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_skill_plugins_register_as_data():
    registry = PluginRegistry()
    _load("list_skills").register(registry)
    _load("read_skill").register(registry)
    names = {p.name for p in registry.list_plugins()}
    assert names == {"list_skills", "read_skill"}
    for p in registry.list_plugins():
        assert p.plugin_type == "data"


def test_list_skills_returns_skills():
    result = _load("list_skills")._run({})
    assert result["count"] > 0
    names = {s["name"] for s in result["skills"]}
    assert "paper-ingest" in names


def test_read_skill_returns_content():
    result = _load("read_skill")._run({"name": "paper-ingest"})
    assert result["name"] == "paper-ingest"
    assert "Paper Ingest" in result["content"]


def test_read_skill_missing_raises():
    with pytest.raises(ValueError):
        _load("read_skill")._run({"name": "nonexistent-skill"})
