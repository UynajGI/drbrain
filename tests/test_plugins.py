"""Tests for the generic plugin interface abstraction (capability-agnostic)."""

from __future__ import annotations

import time

from drbrain.plugins import Plugin, PluginRegistry, PluginResult, ResultStatus


def _flatband_plugin(**overrides) -> Plugin:
    """A representative descriptor — the protocol never hardcodes it."""
    return Plugin(
        name="predict_flatband_score",
        description="给定成分与空间群，预测平带度 S_bandwidth（0~1）",
        input_schema={
            "type": "object",
            "properties": {
                "composition": {"type": "object"},
                "space_group": {"type": "string"},
            },
            "required": ["composition"],
        },
        plugin_type="model",
        version="flatness_prod_v2",
        resource="models/flatness_prod_v2.joblib",
        backend="inprocess",
        summary_fields=("S_bandwidth",),
        metadata={"arch": "gbdt"},
        **overrides,
    )


def test_plugin_fields():
    """plugin_type / resource / version / metadata are first-class fields."""
    p = _flatband_plugin()
    assert p.plugin_type == "model"
    assert p.version == "flatness_prod_v2"
    assert p.resource == "models/flatness_prod_v2.joblib"
    assert p.metadata == {"arch": "gbdt"}

    # defaults on a minimal plugin
    d = Plugin(name="x", description="y", input_schema={})
    assert d.plugin_type == "other"
    assert d.version == ""
    assert d.resource is None
    assert d.metadata == {}
    assert d.side_effect == "unspecified"
    assert d.required_capabilities == ()
    assert not d.supports_idempotency


def test_plugin_durable_metadata_is_optional_and_additive():
    plugin = _flatband_plugin(
        side_effect="read",
        required_capabilities=("plugin:predict_flatband_score",),
        code_digest="sha256:abc",
        resource_scope={"datasets": ["materials"]},
        supports_idempotency=True,
        supports_reconcile=True,
    )

    assert plugin.side_effect == "read"
    assert plugin.required_capabilities == ("plugin:predict_flatband_score",)
    assert plugin.code_digest == "sha256:abc"
    assert plugin.resource_scope == {"datasets": ["materials"]}
    assert plugin.supports_idempotency
    assert plugin.supports_reconcile


def test_register_and_list():
    reg = PluginRegistry()
    plugin = _flatband_plugin()
    reg.register(plugin, lambda args: {"S_bandwidth": 0.99})
    assert [p.name for p in reg.list_plugins()] == ["predict_flatband_score"]
    assert reg.get("predict_flatband_score").plugin_type == "model"


def test_call_ok_with_evidence_and_summary():
    reg = PluginRegistry()
    plugin = _flatband_plugin()
    reg.register(plugin, lambda args: {"S_bandwidth": 0.99, "composition": "CrF3"})

    result = reg.call("predict_flatband_score", {"composition": {"Cr": 1, "F": 3}})

    assert result.ok
    assert result.data["S_bandwidth"] == 0.99
    assert result.evidence["plugin"] == "predict_flatband_score"
    assert result.evidence["version"] == "flatness_prod_v2"
    # raw JSON + key-field summary
    msg = result.to_llm_message(plugin)
    assert "结果(JSON)" in msg
    assert "S_bandwidth=0.99" in msg


def test_call_preserves_additive_resource_usage_from_a_plugin_result():
    reg = PluginRegistry()
    reg.register(
        _flatband_plugin(),
        lambda _args: PluginResult(
            ResultStatus.OK,
            data={"S_bandwidth": 0.99},
            resource_usage={"gpu_seconds": 1.25},
        ),
    )

    result = reg.call("predict_flatband_score", {"composition": {"Cr": 1}})

    assert result.ok
    assert result.data == {"S_bandwidth": 0.99}
    assert result.resource_usage == {"gpu_seconds": 1.25}


def test_call_preserves_handler_result_failure_status():
    reg = PluginRegistry()
    reg.register(
        _flatband_plugin(),
        lambda _args: PluginResult(ResultStatus.MODEL_UNAVAILABLE, error="offline"),
    )

    result = reg.call("predict_flatband_score", {"composition": {"Cr": 1}})

    assert result.status is ResultStatus.MODEL_UNAVAILABLE
    assert result.error == "offline"


def test_call_no_result():
    reg = PluginRegistry()
    reg.register(_flatband_plugin(), lambda args: None)
    result = reg.call("predict_flatband_score", {"composition": {}})
    assert result.status is ResultStatus.NO_RESULT
    assert not result.ok


def test_call_timeout():
    reg = PluginRegistry()
    reg.register(_flatband_plugin(timeout_s=0.2), lambda args: time.sleep(5))
    result = reg.call("predict_flatband_score", {"composition": {}})
    assert result.status is ResultStatus.TIMEOUT


def test_call_plugin_error_on_load_error():
    reg = PluginRegistry()

    def handler(args):
        raise FileNotFoundError("models/flatness_prod_v2.joblib not found")

    reg.register(_flatband_plugin(), handler)
    result = reg.call("predict_flatband_score", {"composition": {}})
    # P-I5: a crashing plugin is PLUGIN_ERROR — distinct from MODEL_UNAVAILABLE
    # (an offline model) so operators can tell the failures apart.
    assert result.status is ResultStatus.PLUGIN_ERROR
    assert "not found" in result.error


def test_call_unknown_plugin_never_raises():
    """P-I5: call("unknown") must honor the never-raise contract."""
    reg = PluginRegistry()
    result = reg.call("no_such_plugin", {})
    assert result.status is ResultStatus.INVALID_INPUT
    assert "no_such_plugin" in (result.error or "")


def test_call_invalid_input():
    reg = PluginRegistry()

    def handler(args):
        raise KeyError("composition")  # missing required field

    reg.register(_flatband_plugin(), handler)
    result = reg.call("predict_flatband_score", {})
    assert result.status is ResultStatus.INVALID_INPUT


def test_to_llamaindex_tools_does_not_crash():
    """Bridge returns a list (empty when llama-index absent), never raises."""
    reg = PluginRegistry()
    reg.register(_flatband_plugin(), lambda args: {"S_bandwidth": 0.5})
    tools = reg.to_llamaindex_tools()
    assert isinstance(tools, list)


def test_discover_loads_plugins(tmp_path):
    """discover imports a module whose register() adds a Plugin; returns count."""
    (tmp_path / "foo_plugin.py").write_text(
        "from drbrain.plugins import Plugin\n"
        "def register(registry):\n"
        "    registry.register(\n"
        "        Plugin(name='foo', description='a foo plugin', input_schema={}),\n"
        "        lambda args: {'ok': True},\n"
        "    )\n",
        encoding="utf-8",
    )
    reg = PluginRegistry()
    assert reg.discover(tmp_path) == 1
    assert [p.name for p in reg.list_plugins()] == ["foo"]
    assert reg.call("foo", {}) is not None and reg.call("foo", {}).data == {"ok": True}


def test_discover_skips_broken_module(tmp_path):
    """A module that raises on import is skipped without aborting discovery."""
    (tmp_path / "_private.py").write_text(
        "def register(r):\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "bad_plugin.py").write_text(
        "raise RuntimeError('boom')\n",
        encoding="utf-8",
    )
    (tmp_path / "good_plugin.py").write_text(
        "from drbrain.plugins import Plugin\n"
        "def register(registry):\n"
        "    registry.register(\n"
        "        Plugin(name='good', description='ok', input_schema={}),\n"
        "        lambda args: {},\n"
        "    )\n",
        encoding="utf-8",
    )
    reg = PluginRegistry()
    assert reg.discover(tmp_path) == 1
    assert [p.name for p in reg.list_plugins()] == ["good"]
