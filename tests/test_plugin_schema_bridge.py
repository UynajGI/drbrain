"""P-I1 schema-bridge tests: description/enum/default/nested/additionalProperties.

The tool-definition bridge must hand small models real parameter constraints:
every generated pydantic model keeps per-parameter ``description`` / ``enum``
/ ``default``, recurses into nested objects (and object array items), and
forbids extra keys (``additionalProperties: false``).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from drbrain.plugins import Plugin, PluginRegistry, json_schema_to_model
from drbrain.rag.mcp_tools import _schema_to_model

RICH_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "要执行的 python 源码"},
        "mode": {
            "type": "string",
            "enum": ["sync", "async"],
            "default": "sync",
            "description": "执行模式",
        },
        "retries": {"type": "integer", "default": 3},
        "geometry": {
            "type": "object",
            "description": "晶胞结构",
            "properties": {"a": {"type": "number", "description": "晶格常数 (Å)"}},
            "required": ["a"],
        },
        "sites": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"el": {"type": "string"}},
                "required": ["el"],
            },
        },
    },
    "required": ["code"],
}


def test_bridge_preserves_description_enum_default():
    model = json_schema_to_model("run_python", RICH_SCHEMA)
    assert model is not None
    props = model.model_json_schema()["properties"]
    assert props["code"]["description"] == "要执行的 python 源码"
    assert props["mode"]["enum"] == ["sync", "async"]
    assert props["mode"]["default"] == "sync"
    assert props["mode"]["description"] == "执行模式"
    assert props["retries"]["default"] == 3


def test_bridge_forbids_additional_properties():
    model = json_schema_to_model("run_python", RICH_SCHEMA)
    assert model is not None
    schema = model.model_json_schema()
    assert schema["additionalProperties"] is False
    with pytest.raises(ValidationError):
        model(code="print(1)", script="禁止的别名")  # 4B 模型爱犯的别名错误必须被拒


def test_bridge_enforces_required_and_enum():
    model = json_schema_to_model("run_python", RICH_SCHEMA)
    assert model is not None
    instance = model(code="x")
    assert instance.mode == "sync"  # default 被应用
    with pytest.raises(ValidationError):
        model(code="x", mode="threading")  # enum 之外的取值必须被拒
    with pytest.raises(ValidationError):
        model()  # 缺 required 的 code


def _resolve_subschema(schema: dict, prop: dict) -> dict:
    """Resolve a property spec to its sub-model schema ($ref direct or in anyOf)."""
    if "$ref" in prop:
        return schema["$defs"][prop["$ref"].split("/")[-1]]
    for branch in prop.get("anyOf", []):
        if "$ref" in branch:
            return schema["$defs"][branch["$ref"].split("/")[-1]]
    raise AssertionError(f"no sub-model reference in {prop!r}")


def test_bridge_recurses_nested_objects():
    model = json_schema_to_model("run_python", RICH_SCHEMA)
    assert model is not None
    schema = model.model_json_schema()

    # 嵌套 object 递归成子模型（$defs + $ref），约束逐层保留
    sub = _resolve_subschema(schema, schema["properties"]["geometry"])
    assert sub["properties"]["a"]["description"] == "晶格常数 (Å)"
    assert sub["required"] == ["a"]
    assert sub["additionalProperties"] is False

    # object 数组的 items 同样递归
    items_prop = schema["properties"]["sites"]["anyOf"][0]
    item_sub = schema["$defs"][items_prop["items"]["$ref"].split("/")[-1]]
    assert item_sub["required"] == ["el"]

    # 实例化走子模型校验
    instance = model(code="x", geometry={"a": 3.9}, sites=[{"el": "Fe"}])
    assert instance.geometry.a == 3.9
    with pytest.raises(ValidationError):
        model(code="x", geometry={"b": 1.0})  # 子模型同样 forbid extra + required


def test_bridge_compat_with_flat_legacy_schema():
    """旧扁平 schema 行为保持：类型映射不变、无 description 也能建模。"""
    schema = {
        "type": "object",
        "properties": {
            "composition": {"type": "object"},
            "space_group": {"type": "string"},
        },
        "required": ["composition"],
    }
    model = json_schema_to_model("predict_flatband_score", schema)
    assert model is not None
    instance = model(composition={"Cr": 1})
    assert instance.space_group is None
    assert model.model_json_schema()["required"] == ["composition"]


def test_bridge_returns_none_for_trivial_schemas():
    assert json_schema_to_model("x", {}) is None
    assert json_schema_to_model("x", {"type": "string"}) is None
    assert json_schema_to_model("x", {"type": "object", "properties": {}}) is None


def test_registry_tool_bridge_flows_constraints_to_the_model():
    """llama-index 工具定义里能看到 description/enum/default（4B 模型直接受益）。"""
    registry = PluginRegistry()
    registry.register(
        Plugin(name="run_python", description="d", input_schema=RICH_SCHEMA),
        lambda args: {"ok": True},
    )
    tools = registry.to_llamaindex_tools()
    if not tools:  # llama-index 未安装时桥接为空（既有契约）
        pytest.skip("llama-index not installed")
    params = tools[0].metadata.get_parameters_dict()
    assert params["properties"]["code"]["description"] == "要执行的 python 源码"
    assert params["properties"]["mode"]["enum"] == ["sync", "async"]
    assert params["properties"]["mode"]["default"] == "sync"
    assert params["required"] == ["code"]


def test_mcp_bridge_shares_the_same_constraints():
    model = _schema_to_model("search", RICH_SCHEMA)
    assert model is not None
    schema = model.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["properties"]["mode"]["enum"] == ["sync", "async"]
    with pytest.raises(ValidationError):
        model(code="x", bogus=1)
    assert _schema_to_model("empty", {}) is None
