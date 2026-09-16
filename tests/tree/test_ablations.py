"""T57: one-mechanism ablations and thresholds frozen before the holdout."""

from __future__ import annotations

import dataclasses
import json

import pytest

from drbrain.rag.ablations import (
    ABLATIONS,
    ablation,
    freeze_thresholds,
    read_overrides,
    variant_builder_config,
)
from drbrain.tree.builder import BuilderConfig


class TestRegistry:
    def test_every_ablation_changes_exactly_one_declared_mechanism(self):
        for name, row in ABLATIONS.items():
            assert row.mechanism and row.rationale, name
            if row.kind == "reference":
                assert not row.builder and not row.read, name
            elif row.kind == "build":
                assert row.builder and not row.read, name
            elif row.kind == "read":
                assert row.read and not row.builder, name
            else:  # pragma: no cover - registry discipline
                raise AssertionError(f"{name}: unknown kind {row.kind!r}")

    def test_build_overrides_map_onto_builder_fields(self):
        fields = {field.name for field in dataclasses.fields(BuilderConfig)}
        for name, row in ABLATIONS.items():
            for key in row.builder:
                assert key in fields, f"{name}: {key} is not a BuilderConfig field"

    def test_unknown_names_are_refused(self):
        with pytest.raises(ValueError, match="unknown ablation"):
            ablation("magic")


class TestVariantConfigs:
    def test_lambda_variants_stay_inside_the_protocol_range(self):
        assert variant_builder_config("lambda0").lam == 0.0
        assert variant_builder_config("lambda12").lam == 12.0
        assert variant_builder_config("default").lam == BuilderConfig().lam

    def test_cost_gate_variant_merges_into_cost_params_only(self):
        base = BuilderConfig()
        variant = variant_builder_config("cost_gate_open", base)
        assert variant.cost.summary_output_budget == 1_000_000
        assert variant.cost.summary_input_budget == 1_000_000
        # The protocol's membership floor is not an ablation knob.
        assert variant.cost.min_members == base.cost.min_members == 2
        assert variant.cost.tool_overhead_tokens == base.cost.tool_overhead_tokens
        assert variant.lam == base.lam and variant.contract == base.contract

    def test_structure_hints_variant_toggles_only_that_flag(self):
        variant = variant_builder_config("no_structure_hints")
        assert variant.use_structure_hints is False
        assert variant.lam == BuilderConfig().lam

    def test_default_variant_is_identity(self):
        base = BuilderConfig()
        assert variant_builder_config("default", base) is base

    def test_read_overrides(self):
        assert read_overrides("leaf_only_entry") == {"view": "leaf"}
        assert read_overrides("no_expansion") == {"expand_regions": False}
        assert read_overrides("tight_budget")["max_expansions"] == 1
        assert read_overrides("lambda0") == {}


class TestFrozenThresholds:
    def test_thresholds_are_written_exactly_once(self, tmp_path):
        path = tmp_path / "ablation-thresholds.json"
        assert freeze_thresholds(path, {"hit_rate_paper": 0.7, "tier": "dev"}) is True
        first = path.read_text(encoding="utf-8")
        assert json.loads(first)["hit_rate_paper"] == 0.7
        # A later call cannot move the bar after results were seen.
        assert freeze_thresholds(path, {"hit_rate_paper": 0.1}) is False
        assert path.read_text(encoding="utf-8") == first
