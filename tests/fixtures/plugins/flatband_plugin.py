"""External plugin fixture: flat-band score prediction (``model`` type).

This module lives *outside* ``src/drbrain`` — drbrain never imports it. It is
loaded at runtime by :meth:`PluginRegistry.discover`, which calls this module's
``register(registry)``. The handler delegates to a fake CLI command (no real
research/ assets) so the test exercises the full ``subprocess`` backend path
without any heavyweight model dependency.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from drbrain.plugins import Plugin, run_subprocess_json


def _predict(arguments: dict[str, Any]) -> dict[str, Any]:
    """Fake flat-band model: echo the inputs and emit a deterministic score."""
    script = (
        "import json, sys;"
        "args = json.loads(sys.argv[1]);"
        "print(json.dumps({"
        "'S_bandwidth': 0.42,"
        "'flatness_ratio': 0.87,"
        "'composition': args['composition'],"
        "'space_group': args['space_group']"
        "}))"
    )
    return run_subprocess_json([sys.executable, "-c", script, json.dumps(arguments)])


def register(registry: Any) -> None:
    """Register the ``predict_flatband_score`` plugin on ``registry``."""
    plugin = Plugin(
        name="predict_flatband_score",
        description=(
            "给定成分与空间群，预测该体系的平带度评分 (S_bandwidth)：数值越高表示越可能出现近平带。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "composition": {
                    "type": "object",
                    "description": '元素→原子数的化学计量比，如 {"Fe": 3, "O": 4}',
                },
                "space_group": {
                    "type": "string",
                    "description": "空间群符号，如 Fd-3m",
                },
            },
            "required": ["composition", "space_group"],
        },
        plugin_type="model",
        backend="subprocess",
        version="flatness_prod_v2",
        summary_fields=("S_bandwidth",),
        metadata={"family": "gbdt", "trained_on": "materials-flatband-corpus"},
    )
    registry.register(plugin, _predict)
