"""Manifest-style (v2) plugin declaration — discovery, defaults, backward compat.

The manifest style separates metadata from code: a module declares
``PLUGIN_MANIFEST`` (+ ``HANDLER``, optional ``JOB_METHODS``) instead of an
inline ``register(registry)``. These tests pin the discovery contract:
manifest registration, required-field skipping, fail-closed ABI negotiation,
and the inline style working unchanged beside it.
"""

from __future__ import annotations

import logging

from drbrain.plugins import PluginRegistry, ResultStatus

REGISTRY_LOGGER = "drbrain.plugins.registry"

MANIFEST_MODULE = """
from types import SimpleNamespace

PLUGIN_MANIFEST = {
    "name": "manifest_score",
    "description": "manifest-style model plugin",
    "input_schema": {
        "type": "object",
        "properties": {"composition": {"type": "object"}},
        "required": ["composition"],
    },
    "plugin_type": "model",
    "version": "ms_v1",
    "abi_version": 1,
    "side_effect": "read",
    "timeout_s": 30.0,
    "summary_fields": ["score"],
    "metadata": {"family": "gbdt"},
}


def HANDLER(arguments):
    return {"score": 0.42, "composition": arguments.get("composition")}


JOB_METHODS = SimpleNamespace(
    submit=lambda arguments: "job-1",
    poll=lambda job_id: {"status": "done", "result": {"score": 0.42}},
    cancel=lambda job_id: True,
)
"""

INLINE_MODULE = """
from drbrain.plugins import Plugin


def register(registry):
    registry.register(
        Plugin(name="inline_ping", description="inline-style plugin", input_schema={"type": "object"}),
        lambda arguments: {"pong": True},
    )
"""


def _discover(tmp_path, sources: dict[str, str]) -> tuple[PluginRegistry, int]:
    for filename, source in sources.items():
        (tmp_path / filename).write_text(source, encoding="utf-8")
    registry = PluginRegistry(process_isolation=False)
    return registry, registry.discover(tmp_path)


def test_manifest_discovery_registers_plugin(tmp_path):
    registry, count = _discover(tmp_path, {"score_plugin.py": MANIFEST_MODULE})
    assert count == 1
    plugin = registry.get("manifest_score")
    assert plugin.description == "manifest-style model plugin"
    assert plugin.plugin_type == "model"
    assert plugin.version == "ms_v1"
    assert plugin.abi_version == 1
    assert plugin.side_effect == "read"
    assert plugin.timeout_s == 30.0
    assert list(plugin.summary_fields) == ["score"]
    assert plugin.metadata == {"family": "gbdt"}

    result = registry.call("manifest_score", {"composition": {"Fe": 3}})
    assert result.status is ResultStatus.OK
    assert result.data["score"] == 0.42


def test_manifest_jobs_are_wired(tmp_path):
    registry, _ = _discover(tmp_path, {"score_plugin.py": MANIFEST_MODULE})
    assert registry.supports_jobs("manifest_score")
    assert registry.submit_job("manifest_score", {}) == "job-1"
    assert registry.poll_job("manifest_score", "job-1")["status"] == "done"
    assert registry.cancel_job("manifest_score", "job-1") is True


def test_manifest_minimal_uses_dataclass_defaults(tmp_path):
    source = (
        "PLUGIN_MANIFEST = {\n"
        "    'name': 'minimal',\n"
        "    'description': 'only required fields',\n"
        "    'input_schema': {'type': 'object', 'properties': {}},\n"
        "}\n"
        "def HANDLER(arguments):\n"
        "    return None\n"
    )
    registry, count = _discover(tmp_path, {"minimal_plugin.py": source})
    assert count == 1
    plugin = registry.get("minimal")
    assert plugin.plugin_type == "other"
    assert plugin.abi_version == 1
    assert plugin.side_effect == "unspecified"
    assert not registry.supports_jobs("minimal")


def test_manifest_unknown_keys_are_dropped(tmp_path):
    """Forward compat: the loader tolerates unknown keys (conformance flags them)."""
    source = (
        "PLUGIN_MANIFEST = {\n"
        "    'name': 'legacy', 'description': 'd',\n"
        "    'input_schema': {'type': 'object', 'properties': {}},\n"
        "    'brand_new_future_field': 1,\n"
        "}\n"
        "def HANDLER(arguments):\n"
        "    return 1\n"
    )
    registry, count = _discover(tmp_path, {"legacy_plugin.py": source})
    assert count == 1
    assert not hasattr(registry.get("legacy"), "brand_new_future_field")  # dropped, not stored


def test_manifest_missing_required_field_is_skipped_with_warning(tmp_path, caplog):
    source = (
        "PLUGIN_MANIFEST = {'name': 'incomplete', 'description': 'no schema here'}\n"
        "def HANDLER(arguments):\n"
        "    return None\n"
    )
    with caplog.at_level(logging.WARNING, logger=REGISTRY_LOGGER):
        registry, count = _discover(tmp_path, {"incomplete_plugin.py": source})
    assert count == 0
    assert "missing required fields" in caplog.text
    assert "input_schema" in caplog.text


def test_manifest_requires_callable_handler(tmp_path, caplog):
    source = (
        "PLUGIN_MANIFEST = {\n"
        "    'name': 'noh', 'description': 'd',\n"
        "    'input_schema': {'type': 'object', 'properties': {}},\n"
        "}\n"
    )
    with caplog.at_level(logging.WARNING, logger=REGISTRY_LOGGER):
        registry, count = _discover(tmp_path, {"nohandler_plugin.py": source})
    assert count == 0
    assert "HANDLER" in caplog.text


def test_manifest_abi_negotiation_fail_closed(tmp_path, caplog):
    """A manifest written for an unsupported ABI is skipped, never half-loaded."""
    future = MANIFEST_MODULE.replace('"abi_version": 1', '"abi_version": 2').replace(
        '"name": "manifest_score"', '"name": "future_score"'
    )
    sources = {"good_plugin.py": MANIFEST_MODULE, "future_plugin.py": future}
    with caplog.at_level(logging.WARNING, logger=REGISTRY_LOGGER):
        registry, count = _discover(tmp_path, sources)
    assert count == 1
    assert "future_score" not in {p.name for p in registry.list_plugins()}
    assert "future_score" in caplog.text


def test_inline_style_still_works_beside_manifest(tmp_path):
    """v1 inline register() style is untouched and coexists with manifests."""
    registry, count = _discover(
        tmp_path,
        {"score_plugin.py": MANIFEST_MODULE, "inline_plugin.py": INLINE_MODULE},
    )
    assert count == 2
    assert {p.name for p in registry.list_plugins()} == {"manifest_score", "inline_ping"}
    assert registry.call("inline_ping", {}).data == {"pong": True}
    assert not registry.supports_jobs("inline_ping")


def test_manifest_takes_precedence_over_register(tmp_path):
    """A module declaring both is registered via the manifest only."""
    source = (
        "PLUGIN_MANIFEST = {\n"
        "    'name': 'from_manifest', 'description': 'd',\n"
        "    'input_schema': {'type': 'object', 'properties': {}},\n"
        "}\n"
        "def HANDLER(arguments):\n"
        "    return 'manifest'\n"
        "def register(registry):\n"
        "    raise AssertionError('inline register() must not be consulted')\n"
    )
    registry, count = _discover(tmp_path, {"both_plugin.py": source})
    assert count == 1
    assert registry.call("from_manifest", {}).data == "manifest"
