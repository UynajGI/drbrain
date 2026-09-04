"""P-I3 output-cap tests: registry.call clips oversized outputs by default.

A 0.8M-corpus ``search_papers`` must not be able to blow up the model context
in one call. The hard cap defaults to :data:`DEFAULT_MAX_OUTPUT_BYTES`
(256_000), is overridable by the plugin declaration or the per-call argument,
and a clipped output becomes a ``{"truncated", "bytes", "sha256", "preview"}``
digest flagged via ``PluginResult.truncated``.
"""

from __future__ import annotations

import hashlib
import json
import time

from drbrain.plugins import (
    DEFAULT_MAX_OUTPUT_BYTES,
    Plugin,
    PluginRegistry,
    PluginResult,
    ResultStatus,
)

BIG = {"papers": [{"title": "x" * 100} for _ in range(4000)]}  # JSON ≈ 0.5MB
SMALL = {"count": 1, "papers": [{"title": "t"}]}


def _search_plugin(**overrides) -> Plugin:
    return Plugin(name="search_papers", description="d", input_schema={}, **overrides)


def test_default_cap_is_256k():
    assert DEFAULT_MAX_OUTPUT_BYTES == 256_000


def test_oversized_output_is_truncated_by_default():
    registry = PluginRegistry()
    registry.register(_search_plugin(), lambda args: BIG)
    result = registry.call("search_papers", {"query": "flat band"})

    assert result.ok
    assert result.truncated is True
    assert result.data["truncated"] is True
    full = json.dumps(BIG, ensure_ascii=False).encode("utf-8")
    assert result.data["bytes"] == len(full)
    assert result.data["sha256"] == hashlib.sha256(full).hexdigest()
    assert len(result.data["preview"]) == DEFAULT_MAX_OUTPUT_BYTES
    # evidence 里的 output 也必须是裁剪后的 digest，而不是整段原文
    assert result.evidence["output"] == result.data
    # LLM 消息明示截断
    assert "截断" in result.to_llm_message()


def test_small_output_passes_through_untouched():
    registry = PluginRegistry()
    registry.register(_search_plugin(), lambda args: SMALL)
    result = registry.call("search_papers", {})
    assert result.truncated is False
    assert result.data == SMALL
    assert result.evidence["output"] == SMALL


def test_plugin_declaration_overrides_default():
    registry = PluginRegistry()
    registry.register(
        _search_plugin(max_output_bytes=100), lambda args: {"blob": "y" * 500}
    )
    result = registry.call("search_papers", {})
    assert result.truncated is True
    assert result.data["bytes"] > 100
    assert len(result.data["preview"]) == 100


def test_call_argument_overrides_plugin_declaration():
    registry = PluginRegistry()
    registry.register(
        _search_plugin(max_output_bytes=10_000), lambda args: {"blob": "y" * 500}
    )
    result = registry.call("search_papers", {}, max_output_bytes=50)
    assert result.truncated is True
    assert len(result.data["preview"]) == 50
    # 不传时走插件声明（500 字节 < 10_000 → 不截断）
    assert registry.call("search_papers", {}).truncated is False


def test_zero_cap_disables_truncation_and_none_falls_back():
    registry = PluginRegistry()
    registry.register(_search_plugin(), lambda args: BIG)
    # 显式 0 = 不设上限（与 loop 侧 _bounded 语义一致）
    result = registry.call("search_papers", {}, max_output_bytes=0)
    assert result.truncated is False
    assert result.data == BIG
    # 不传 → 回落到默认硬上限
    assert registry.call("search_papers", {}).truncated is True


def test_plugin_zero_declaration_disables_cap():
    registry = PluginRegistry()
    registry.register(_search_plugin(max_output_bytes=0), lambda args: BIG)
    result = registry.call("search_papers", {})
    assert result.truncated is False
    assert result.data == BIG


def test_envelope_result_data_is_capped_too():
    """handler 直接返回 PluginResult 信封时，data 同样受上限约束。"""
    registry = PluginRegistry()
    registry.register(
        _search_plugin(),
        lambda args: PluginResult(ResultStatus.OK, data=BIG, resource_usage={"tokens": 1}),
    )
    result = registry.call("search_papers", {})
    assert result.ok
    assert result.truncated is True
    assert result.data["truncated"] is True
    assert result.resource_usage == {"tokens": 1}
    assert result.evidence["output"]["truncated"] is True


def test_failed_results_have_no_truncation_flag():
    registry = PluginRegistry()
    registry.register(_search_plugin(timeout_s=0.2), lambda args: time.sleep(5))
    result = registry.call("search_papers", {"q": "timeout"}, max_output_bytes=10)
    # 超时路径根本没有输出可截断，truncated 保持 False
    assert result.status is ResultStatus.TIMEOUT
    assert result.truncated is False
