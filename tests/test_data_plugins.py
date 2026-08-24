"""Data-source plugin tests: Sciverse / arXiv / S2 → agent tools.

Proves the architecture-level "global data surface": external ``data`` plugins
discoverable via :class:`PluginRegistry` and bridged to LlamaIndex tools.
HTTP is mocked, so these tests never touch the live APIs. They skip in CI
(where ``research/plugins/`` is not checked in).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from drbrain.plugins import PluginRegistry

PLUGINS_DIR = Path(__file__).resolve().parents[1] / "research" / "plugins"
DATA_PLUGINS = ("sciverse_search", "arxiv_search", "s2_search")

pytestmark = pytest.mark.skipif(
    not (PLUGINS_DIR / "sciverse_search.py").exists(),
    reason="external data plugins not present (research/plugins/)",
)


def _load_module(name: str):
    path = PLUGINS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"drbrain_plugin_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _register_all(registry: PluginRegistry) -> None:
    for name in DATA_PLUGINS:
        _load_module(name).register(registry)


class _FakeResp:
    def __init__(self, json_data=None, text="", status_code=200):
        self._json = json_data or {}
        self.text = text
        self.status_code = status_code

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_data_plugins_register_as_data_type():
    registry = PluginRegistry()
    _register_all(registry)
    names = {p.name for p in registry.list_plugins()}
    assert names == {"sciverse_search", "arxiv_search", "s2_search"}
    for p in registry.list_plugins():
        assert p.plugin_type == "data"


def test_sciverse_search_returns_normalized_papers(monkeypatch):
    mod = _load_module("sciverse_search")
    monkeypatch.setattr(mod, "_token", lambda: "test-token")
    monkeypatch.setattr(
        mod.requests,
        "post",
        lambda *a, **kw: _FakeResp(
            json_data={
                "results": [
                    {
                        "unique_id": "u1",
                        "title": "Flat band in moire",
                        "publication_published_year": 2024,
                        "doi": "10.1/abc",
                    }
                ]
            }
        ),
    )
    result = mod._run({"query": "flat band"})
    assert result["count"] == 1
    assert result["papers"][0]["source"] == "sciverse"
    assert result["papers"][0]["doi"] == "10.1/abc"


def test_arxiv_search_parses_atom(monkeypatch):
    mod = _load_module("arxiv_search")
    atom = (
        '<?xml version="1.0"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        "<entry>"
        "<title>Flat bands</title>"
        "<id>http://arxiv.org/abs/2401.00001v1</id>"
        "<summary>Abstract text</summary>"
        "<published>2024-01-01T00:00:00Z</published>"
        "</entry>"
        "</feed>"
    )
    monkeypatch.setattr(mod.requests, "get", lambda *a, **kw: _FakeResp(text=atom))
    result = mod._run({"query": "flat band"})
    assert result["count"] == 1
    assert result["papers"][0]["arxiv_id"] == "2401.00001"
    assert result["papers"][0]["source"] == "arxiv"


def test_s2_search_returns_normalized_papers(monkeypatch):
    mod = _load_module("s2_search")
    monkeypatch.setattr(mod, "_token", lambda: "test-key")
    monkeypatch.setattr(
        mod.requests,
        "get",
        lambda *a, **kw: _FakeResp(
            json_data={"data": [{"paperId": "p1", "title": "T", "externalIds": {"DOI": "10.1/2"}}]}
        ),
    )
    result = mod._run({"query": "superconductor"})
    assert result["count"] == 1
    assert result["papers"][0]["doi"] == "10.1/2"
    assert result["papers"][0]["source"] == "s2"


def test_data_plugins_bridge_to_llamaindex_tools():
    registry = PluginRegistry()
    _register_all(registry)
    tools = registry.to_llamaindex_tools()
    assert len(tools) == 3
