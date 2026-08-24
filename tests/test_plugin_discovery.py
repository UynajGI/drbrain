"""External plugin discovery tests — prove the externalization architecture.

The protocol layer (``src/drbrain/plugins/``) is generic and never imports
a concrete plugin. Concrete plugins live in an external directory
(``tests/fixtures/plugins/``) and are loaded at runtime by
:meth:`PluginRegistry.discover`, which calls each module's ``register(registry)``.

These tests:
  * load the two fixture plugins (a ``model`` and a ``software`` plugin),
  * assert their descriptors and ``plugin_type``,
  * invoke both through ``registry.call`` and assert ``OK`` results,
  * assert the protocol layer source never names a concrete plugin.
"""

from __future__ import annotations

from pathlib import Path

from drbrain.plugins import PluginRegistry, ResultStatus

PLUGIN_DIR = Path(__file__).parent / "fixtures" / "plugins"
PROTOCOL_DIR = Path(__file__).resolve().parents[1] / "src" / "drbrain" / "plugins"


def _loaded_registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.discover(PLUGIN_DIR)
    return registry


def test_discover_loads_external_plugins():
    registry = _loaded_registry()
    names = {p.name for p in registry.list_plugins()}
    assert names == {"predict_flatband_score", "run_dft_calculation"}


def test_plugin_types_are_correct():
    registry = _loaded_registry()
    plugins = {p.name: p for p in registry.list_plugins()}
    assert plugins["predict_flatband_score"].plugin_type == "model"
    assert plugins["predict_flatband_score"].backend == "subprocess"
    assert plugins["run_dft_calculation"].plugin_type == "software"
    assert plugins["run_dft_calculation"].backend == "subprocess"


def test_predict_flatband_score_callable():
    registry = _loaded_registry()
    result = registry.call(
        "predict_flatband_score",
        {"composition": {"Fe": 3, "O": 4}, "space_group": "Fd-3m"},
    )
    assert result.status is ResultStatus.OK
    assert result.data["S_bandwidth"] == 0.42
    # summary_fields surfaces the key field to the LLM path.
    plugin = registry.get("predict_flatband_score")
    message = result.to_llm_message(plugin)
    assert "S_bandwidth=0.42" in message


def test_run_dft_calculation_callable():
    registry = _loaded_registry()
    result = registry.call(
        "run_dft_calculation",
        {
            "structure": {
                "formula": "Fe3O4",
                "sites": [{"Fe": 3}, {"O": 4}],
            },
            "parameters": {"encut": 520, "kpoints": [4, 4, 4]},
        },
    )
    assert result.status is ResultStatus.OK
    assert result.data["total_energy"] == -8.123
    assert result.data["band_gap"] == 1.34
    assert result.data["n_sites"] == 2


def test_protocol_layer_never_names_concrete_plugins():
    forbidden = (
        "predict_flatband_score",
        "run_dft_calculation",
        "flatband_plugin",
        "dft_plugin",
    )
    for py_file in sorted(PROTOCOL_DIR.glob("*.py")):
        text = py_file.read_text()
        for name in forbidden:
            assert name not in text, f"{py_file.name} references concrete plugin {name!r}"
