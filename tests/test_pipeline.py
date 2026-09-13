"""Tests for pipeline step resolution and presets.

TDD: tests written before implementation.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest import mock

import pytest
import typer

# ── Pipeline step definitions (will move to services/pipeline.py) ──

# Test the step data structure first — implementation follows
EXPECTED_STEPS = {
    "ingest": "inbox",
    "build": "papers",
    "embed": "global",
    "closure": "global",
}

EXPECTED_PRESETS = {
    "full": ["ingest", "build", "embed", "closure"],
    "quick": ["build", "embed", "closure"],
    "embed": ["embed", "closure"],
}


class TestPipelineStepDefinitions:
    """Test that step/preset definitions are well-formed."""

    def test_all_expected_steps_exist(self):
        """Every expected step should be in STEPS dict."""
        from drbrain.services.pipeline import STEPS

        for name in EXPECTED_STEPS:
            assert name in STEPS, f"Missing step: {name}"

    def test_steps_have_valid_scopes(self):
        """Each step should have a known scope."""
        from drbrain.services.pipeline import STEPS

        valid_scopes = {"inbox", "papers", "global"}
        for name, step in STEPS.items():
            assert step.scope in valid_scopes, f"Step {name} has unknown scope: {step.scope}"
            assert isinstance(step.desc, str) and len(step.desc) > 0

    def test_all_expected_presets_exist(self):
        """Every expected preset should be in PRESETS dict."""
        from drbrain.services.pipeline import PRESETS

        for name, steps in EXPECTED_PRESETS.items():
            assert name in PRESETS, f"Missing preset: {name}"
            assert PRESETS[name] == steps


class TestResolveSteps:
    """Test step resolution from presets and --steps."""

    def test_full_preset_resolves(self):
        from drbrain.services.pipeline import resolve_steps

        steps = resolve_steps(preset="full")
        assert steps == EXPECTED_PRESETS["full"]

    def test_quick_preset_resolves(self):
        from drbrain.services.pipeline import resolve_steps

        steps = resolve_steps(preset="quick")
        assert steps == EXPECTED_PRESETS["quick"]

    def test_embed_preset_resolves(self):
        from drbrain.services.pipeline import resolve_steps

        steps = resolve_steps(preset="embed")
        assert steps == EXPECTED_PRESETS["embed"]

    def test_custom_steps_resolves(self):
        from drbrain.services.pipeline import resolve_steps

        steps = resolve_steps(steps_str="build,embed")
        assert steps == ["build", "embed"]

    def test_custom_steps_trims_whitespace(self):
        from drbrain.services.pipeline import resolve_steps

        steps = resolve_steps(steps_str=" build ,  embed , closure ")
        assert steps == ["build", "embed", "closure"]

    def test_unknown_preset_raises(self):
        from drbrain.services.pipeline import resolve_steps

        with pytest.raises(ValueError, match="Unknown preset"):
            resolve_steps(preset="nonexistent")

    def test_unknown_step_raises(self):
        from drbrain.services.pipeline import resolve_steps

        with pytest.raises(ValueError, match="Unknown step"):
            resolve_steps(steps_str="ingest,fake_step")

    def test_neither_preset_nor_steps_raises(self):
        from drbrain.services.pipeline import resolve_steps

        with pytest.raises(ValueError, match="Specify --preset or --steps"):
            resolve_steps()

    def test_duplicate_steps_deduplicated(self):
        from drbrain.services.pipeline import resolve_steps

        steps = resolve_steps(steps_str="build,build,embed")
        assert steps == ["build", "embed"]

    def test_empty_custom_steps_raises(self):
        from drbrain.services.pipeline import resolve_steps

        with pytest.raises(ValueError, match="At least one pipeline step"):
            resolve_steps(steps_str=" , ")


class TestListSteps:
    """Test that list_steps returns structured data."""

    def test_list_steps_returns_all(self):
        from drbrain.services.pipeline import list_steps_info

        steps, presets = list_steps_info()
        assert len(steps) >= 4
        assert len(presets) >= 3

    def test_each_step_has_required_keys(self):
        from drbrain.services.pipeline import list_steps_info

        steps, _ = list_steps_info()
        for s in steps:
            assert "name" in s
            assert "scope" in s
            assert "description" in s


def _pipeline_context(*, config_path: str | None = None, root: str | None = None):
    """Build the smallest context accepted by ``pipeline_cmd``.

    The command normally receives a runtime object from the root CLI callback;
    using a plain namespace here also keeps this test independent of Click's
    context construction details.
    """
    runtime = SimpleNamespace(config_path=config_path, root=root)
    return SimpleNamespace(obj={"runtime": runtime, "config": {}})


def _invoke_pipeline(ctx, **kwargs):
    from drbrain.cli.ingest_commands import pipeline_cmd

    defaults = {
        "preset": None,
        "steps": "build,embed",
        "list_steps_flag": False,
        "dry_run": False,
        "full": False,
    }
    defaults.update(kwargs)
    return pipeline_cmd(ctx, **defaults)


def test_pipeline_stops_on_failed_step_and_returns_exit_code(capsys):
    """A failed child must stop the chain and never print completion."""
    ctx = _pipeline_context()
    calls: list[tuple[list[str], dict]] = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 7)

    with mock.patch("subprocess.run", side_effect=fake_run):
        with pytest.raises(typer.Exit) as exc_info:
            _invoke_pipeline(ctx)

    captured = capsys.readouterr()
    assert exc_info.value.exit_code == 7
    assert len(calls) == 1
    assert "Pipeline failed at step 'build'" in captured.err
    assert "Pipeline complete" not in captured.out
    assert calls[0][1]["check"] is True


def test_pipeline_propagates_runtime_config_and_root(tmp_path):
    """Child commands must use the same config and repository root."""
    runtime_root = str(tmp_path)
    ctx = _pipeline_context(config_path="config.local.yaml", root=runtime_root)
    calls: list[tuple[list[str], dict]] = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0)

    with mock.patch("subprocess.run", side_effect=fake_run):
        _invoke_pipeline(ctx, steps="build,embed")

    assert len(calls) == 2
    for args, kwargs in calls:
        assert args[:3] == [args[0], "-m", "drbrain.cli.main"]
        assert args[3:5] == ["--config", "config.local.yaml"]
        assert args[5:7] == ["--root", runtime_root]
        assert kwargs["check"] is True
        assert kwargs["cwd"] == runtime_root
        assert kwargs["env"]["DRBRAIN_ROOT"] == runtime_root
