"""Tests for the generic Model-as-Tool protocol (model-agnostic)."""

from __future__ import annotations

import time

from drbrain.rag.plugins import ModelTool, ModelToolRegistry, ResultStatus


def _flatband_tool(**overrides) -> ModelTool:
    """A representative descriptor — the protocol never hardcodes it."""
    return ModelTool(
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
        model_type="gbdt",
        model_version="flatness_prod_v2",
        weights="models/flatness_prod_v2.joblib",
        backend="inprocess",
        summary_fields=("S_bandwidth",),
        **overrides,
    )


def test_register_and_list():
    reg = ModelToolRegistry()
    tool = _flatband_tool()
    reg.register(tool, lambda args: {"S_bandwidth": 0.99})
    assert [t.name for t in reg.list_tools()] == ["predict_flatband_score"]
    assert reg.get("predict_flatband_score").model_type == "gbdt"


def test_call_ok_with_evidence_and_summary():
    reg = ModelToolRegistry()
    tool = _flatband_tool()
    reg.register(tool, lambda args: {"S_bandwidth": 0.99, "composition": "CrF3"})

    result = reg.call("predict_flatband_score", {"composition": {"Cr": 1, "F": 3}})

    assert result.ok
    assert result.data["S_bandwidth"] == 0.99
    assert result.evidence["tool"] == "predict_flatband_score"
    assert result.evidence["model_version"] == "flatness_prod_v2"
    # raw JSON + key-field summary
    msg = result.to_llm_message(tool)
    assert "结果(JSON)" in msg
    assert "S_bandwidth=0.99" in msg


def test_call_no_result():
    reg = ModelToolRegistry()
    reg.register(_flatband_tool(), lambda args: None)
    result = reg.call("predict_flatband_score", {"composition": {}})
    assert result.status is ResultStatus.NO_RESULT
    assert not result.ok


def test_call_timeout():
    reg = ModelToolRegistry()
    reg.register(_flatband_tool(timeout_s=0.2), lambda args: time.sleep(5))
    result = reg.call("predict_flatband_score", {"composition": {}})
    assert result.status is ResultStatus.TIMEOUT


def test_call_model_unavailable_on_load_error():
    reg = ModelToolRegistry()

    def handler(args):
        raise FileNotFoundError("models/flatness_prod_v2.joblib not found")

    reg.register(_flatband_tool(), handler)
    result = reg.call("predict_flatband_score", {"composition": {}})
    assert result.status is ResultStatus.MODEL_UNAVAILABLE
    assert "not found" in result.error


def test_call_invalid_input():
    reg = ModelToolRegistry()

    def handler(args):
        raise KeyError("composition")  # missing required field

    reg.register(_flatband_tool(), handler)
    result = reg.call("predict_flatband_score", {})
    assert result.status is ResultStatus.INVALID_INPUT


def test_to_llamaindex_tools_does_not_crash():
    """Bridge returns a list (empty when llama-index absent), never raises."""
    reg = ModelToolRegistry()
    reg.register(_flatband_tool(), lambda args: {"S_bandwidth": 0.5})
    tools = reg.to_llamaindex_tools()
    assert isinstance(tools, list)
