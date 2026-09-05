"""Runtime context and path-isolation contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from drbrain.runtime import RuntimeContext, runtime_root


def test_runtime_context_resolves_relative_paths_without_mutating_config(tmp_path):
    cfg = {
        "db": {"path": "data/library.sqlite"},
        "dirs": {
            "inbox": "data/spool/inbox",
            "papers": "data/papers",
            "logs": "data/logs",
        },
        "llamaindex": {"storage_dir": "data/llamaindex"},
        "autoresearch": {"run_dir": "workspace/runs"},
    }
    context = RuntimeContext.create(tmp_path, run_id="test-run")

    normalized = context.apply_config(cfg)

    assert normalized["db"]["path"] == str(tmp_path / "data/library.sqlite")
    assert normalized["dirs"]["papers"] == str(tmp_path / "data/papers")
    assert normalized["llamaindex"]["storage_dir"] == str(tmp_path / "data/llamaindex")
    assert normalized["autoresearch"]["run_dir"] == str(tmp_path / "workspace/runs")
    assert cfg["db"]["path"] == "data/library.sqlite"


def test_runtime_context_preserves_special_and_explicit_paths(tmp_path):
    external = tmp_path.parent / "external.sqlite"
    cfg = {
        "db": {"path": ":memory:"},
        "dirs": {"papers": str(tmp_path / "papers")},
    }
    context = RuntimeContext.create(tmp_path, run_id="test-run")

    normalized = context.apply_config(cfg)

    assert normalized["db"]["path"] == ":memory:"
    assert normalized["dirs"]["papers"] == str(tmp_path / "papers")
    assert context.resolve_path(str(external)) == external.resolve()


def test_runtime_context_child_environment_is_namespaced(tmp_path):
    context = RuntimeContext.create(tmp_path, run_id="test-run")

    env = context.child_env()

    assert env["DRBRAIN_ROOT"] == str(tmp_path.resolve())
    assert env["DRBRAIN_RUNTIME_ROOT"] == str(tmp_path.resolve())
    assert env["DRBRAIN_RUN_ID"] == "test-run"
    assert env["DRBRAIN_TEMP_ROOT"] == str(tmp_path / "data" / ".runtime" / "test-run")
    assert env["DRBRAIN_CONFIG"] == str(tmp_path / "config.yaml")


def test_runtime_context_honors_inherited_temp_root(tmp_path, monkeypatch):
    inherited = tmp_path / "scratch" / "run"
    monkeypatch.setenv("DRBRAIN_ROOT", str(tmp_path))
    monkeypatch.setenv("DRBRAIN_TEMP_ROOT", str(inherited))

    context = RuntimeContext.create(run_id="inherited")

    assert context.temp_root == inherited.resolve()


def test_runtime_context_explicit_root_does_not_reuse_inherited_temp(tmp_path, monkeypatch):
    inherited = tmp_path / "scratch" / "run"
    monkeypatch.setenv("DRBRAIN_TEMP_ROOT", str(inherited))

    context = RuntimeContext.create(tmp_path, run_id="explicit")

    assert context.temp_root == tmp_path / "data" / ".runtime" / "explicit"


def test_runtime_context_honors_runtime_root_alias(tmp_path, monkeypatch):
    monkeypatch.delenv("DRBRAIN_ROOT", raising=False)
    monkeypatch.setenv("DRBRAIN_RUNTIME_ROOT", str(tmp_path))

    context = RuntimeContext.create(run_id="alias")

    assert context.root == tmp_path.resolve()


def test_runtime_root_matches_context_default_after_chdir(tmp_path, monkeypatch):
    """Standalone scripts and library callers must select one implicit root."""
    monkeypatch.delenv("DRBRAIN_ROOT", raising=False)
    monkeypatch.delenv("DRBRAIN_RUNTIME_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)

    assert runtime_root() == RuntimeContext.create().root == tmp_path.resolve()


@pytest.mark.parametrize("variable", ["DRBRAIN_ROOT", "DRBRAIN_RUNTIME_ROOT"])
def test_runtime_context_rejects_empty_inherited_root(tmp_path, monkeypatch, variable):
    """An explicitly empty selector must not fall back to the process CWD."""
    monkeypatch.delenv("DRBRAIN_ROOT", raising=False)
    monkeypatch.delenv("DRBRAIN_RUNTIME_ROOT", raising=False)
    monkeypatch.setenv(variable, "")

    with pytest.raises(ValueError, match="must not be empty"):
        RuntimeContext.create()
    with pytest.raises(ValueError, match="must not be empty"):
        runtime_root()


def test_runtime_context_does_not_fall_through_empty_primary_selector(tmp_path, monkeypatch):
    """A blank primary selector cannot be bypassed by the legacy alias."""
    monkeypatch.setenv("DRBRAIN_ROOT", "")
    monkeypatch.setenv("DRBRAIN_RUNTIME_ROOT", str(tmp_path))

    with pytest.raises(ValueError, match="DRBRAIN_ROOT.*empty"):
        RuntimeContext.create()


def test_runtime_context_does_not_inherit_another_config_overlay(tmp_path, monkeypatch):
    """A context without an overlay publishes its own base config anchor."""
    monkeypatch.setenv("DRBRAIN_CONFIG", "/other-worktree/config.local.yaml")
    context = RuntimeContext.create(tmp_path, run_id="isolated")

    assert context.child_env()["DRBRAIN_CONFIG"] == str(tmp_path / "config.yaml")


def test_runtime_context_rejects_non_directory_root(tmp_path):
    root_file = tmp_path / "root.txt"
    root_file.write_text("x", encoding="utf-8")

    try:
        RuntimeContext.create(root_file)
    except ValueError as exc:
        assert "directory" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("a file must not be accepted as a runtime root")


def test_runtime_context_setup_can_create_a_fresh_root(tmp_path):
    fresh = tmp_path / "new-runtime"

    with pytest.raises(ValueError, match="does not exist"):
        RuntimeContext.create(fresh)

    context = RuntimeContext.create(fresh, create_root=True, run_id="setup")

    assert fresh.is_dir()
    assert context.root == fresh.resolve()


@pytest.mark.parametrize("kwargs", [{"root": ""}, {"temp_root": ""}, {"overlay_path": ""}])
def test_runtime_context_rejects_explicit_empty_namespace_paths(tmp_path, kwargs):
    with pytest.raises(ValueError, match="must not be empty"):
        if "root" in kwargs:
            RuntimeContext.create(**kwargs)
        else:
            RuntimeContext.create(tmp_path, **kwargs)


@pytest.mark.parametrize("run_id", ["", "...", "---"])
def test_runtime_context_rejects_explicit_empty_run_id(tmp_path, run_id):
    with pytest.raises(ValueError, match="DRBRAIN_RUN_ID"):
        RuntimeContext.create(tmp_path, run_id=run_id)


def test_runtime_context_long_run_ids_keep_distinct_temp_namespaces(tmp_path):
    first = RuntimeContext.create(tmp_path, run_id="x" * 100 + "-a")
    second = RuntimeContext.create(tmp_path, run_id="x" * 100 + "-b")

    assert first.run_id != second.run_id
    assert first.temp_root != second.temp_root
    assert len(first.run_id) <= 80
    assert len(second.run_id) <= 80


def test_runtime_context_rejects_symlinked_root(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "runtime"
    alias.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="Runtime root.*symlink"):
        RuntimeContext.create(alias)


def test_runtime_context_temp_root_is_stable_and_not_created_eagerly(tmp_path):
    context = RuntimeContext.create(tmp_path, run_id="stable")

    assert context.temp_root == Path(tmp_path) / "data" / ".runtime" / "stable"
    assert not context.temp_root.exists()
    context.ensure_temp_root()
    assert context.temp_root.is_dir()


@pytest.mark.parametrize("value", ["../outside", "/tmp/drbrain-outside"])
def test_runtime_context_rejects_external_temp_root(tmp_path, value):
    with pytest.raises(ValueError, match="temp root escapes"):
        RuntimeContext.create(tmp_path, temp_root=value)


def test_runtime_context_rejects_in_root_temp_symlink(tmp_path):
    target = tmp_path / "scratch"
    target.mkdir()
    alias = tmp_path / "temp-link"
    alias.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="temp root.*symlink"):
        RuntimeContext.create(tmp_path, temp_root=alias)


def test_runtime_context_rejects_symlinked_default_temp_parent(tmp_path):
    """The implicit scratch namespace must not follow a pre-existing alias."""
    runtime_data = tmp_path / "data"
    runtime_data.mkdir()
    target = tmp_path / "outside-scratch"
    target.mkdir()
    (runtime_data / ".runtime").symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="temp root.*symlink"):
        RuntimeContext.create(tmp_path, run_id="implicit")


def test_runtime_context_rejects_external_or_symlinked_overlay(tmp_path):
    external = tmp_path.parent / "outside.yaml"
    external.write_text("db: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Config overlay escapes"):
        RuntimeContext.create(tmp_path, overlay_path=external)

    link = tmp_path / "linked.yaml"
    link.symlink_to(external)
    with pytest.raises(ValueError, match="Config overlay escapes"):
        RuntimeContext.create(tmp_path, overlay_path=link)


@pytest.mark.parametrize("uri", ["file:/tmp/config.yaml", "https://example.invalid/config.yaml"])
def test_runtime_context_rejects_uri_overlay(tmp_path, uri):
    with pytest.raises(ValueError, match="Config overlay must be a local path"):
        RuntimeContext.create(tmp_path, overlay_path=uri)


def test_runtime_context_rejects_path_uri_overlay(tmp_path):
    """Path-typed URI values must not bypass the overlay boundary."""
    with pytest.raises(ValueError, match="Config overlay must be a local path"):
        RuntimeContext.create(tmp_path, overlay_path=Path("file:///tmp/config.yaml"))


def test_runtime_context_rejects_symlinked_config_layer(tmp_path):
    target = tmp_path / "real.yaml"
    target.write_text("db: {}\n", encoding="utf-8")
    link = tmp_path / "config.local.yaml"
    link.symlink_to(target)

    context = RuntimeContext.create(tmp_path)

    with pytest.raises(ValueError, match="must not be a symlink"):
        context.validate_config_file(link, label="local config", required=True)


def test_runtime_context_rejects_relative_escape(tmp_path):
    context = RuntimeContext.create(tmp_path, run_id="escape")
    cfg = {"db": {"path": "../shared/library.sqlite"}}

    try:
        context.validate_config(cfg)
    except ValueError as exc:
        assert "escapes runtime root" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("relative paths must not escape the runtime root")

    try:
        context.apply_config(cfg)
    except ValueError as exc:
        assert "escapes runtime root" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("apply_config must reject relative traversal")


def test_runtime_context_rejects_in_root_config_symlink_alias(tmp_path):
    """Resolved containment is not enough when a config path uses an alias."""
    target = tmp_path / "data"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    context = RuntimeContext.create(tmp_path)

    with pytest.raises(ValueError, match="symlink"):
        context.validate_config({"dirs": {"papers": "alias/papers"}})


@pytest.mark.parametrize(
    "path_keys",
    [("db", "path"), ("dirs", "papers"), ("llamaindex", "storage_dir")],
)
def test_runtime_context_rejects_empty_mutable_path(tmp_path, path_keys):
    context = RuntimeContext.create(tmp_path)
    config = {path_keys[0]: {path_keys[1]: ""}}

    with pytest.raises(ValueError, match="must not be empty|valid path"):
        context.validate_config(config)
    with pytest.raises(ValueError, match="must not be empty"):
        context.apply_config(config)


def test_runtime_context_preserves_documented_optional_empty_paths(tmp_path):
    context = RuntimeContext.create(tmp_path)
    config = {"autoresearch": {"plugins_dir": ""}, "embed": {"cache_dir": ""}}

    normalized = context.apply_config(config)

    assert normalized == config


def test_runtime_context_rejects_malformed_path_values(tmp_path):
    context = RuntimeContext.create(tmp_path)
    with pytest.raises(ValueError, match="valid path"):
        context.validate_config({"db": {"path": ["not", "a", "path"]}})


@pytest.mark.parametrize("config", [None, [], "not-a-config", 1])
def test_runtime_context_rejects_malformed_top_level_config(tmp_path, config):
    context = RuntimeContext.create(tmp_path)

    with pytest.raises(ValueError, match="mapping or typed dataclass"):
        context.validate_config(config)
    with pytest.raises(ValueError, match="mapping or typed dataclass"):
        context.apply_config(config)


@pytest.mark.parametrize(
    "path_keys",
    [("db", "path"), ("dirs", "papers"), ("llamaindex", "storage_dir")],
)
def test_runtime_context_rejects_explicit_null_mutable_path(tmp_path, path_keys):
    context = RuntimeContext.create(tmp_path)
    config = {path_keys[0]: {path_keys[1]: None}}

    with pytest.raises(ValueError, match="must be a path, not null"):
        context.validate_config(config)
    with pytest.raises(ValueError, match="must be a path, not null"):
        context.apply_config(config)


@pytest.mark.parametrize(
    "config",
    [
        {"db": "not-a-section"},
        {"dirs": None},
        {"llamaindex": {"eval": "not-a-section"}},
    ],
)
def test_runtime_context_rejects_present_non_mapping_sections(tmp_path, config):
    """Malformed present sections must not be silently skipped by path checks."""
    context = RuntimeContext.create(tmp_path)

    with pytest.raises(ValueError, match="section .*mapping"):
        context.validate_config(config)
    with pytest.raises(ValueError, match="section .*mapping"):
        context.apply_config(config)


@pytest.mark.parametrize("uri", ["file:/tmp/shared.sqlite", "FILE:/tmp/shared.sqlite?mode=rwc"])
def test_runtime_context_rejects_file_uri_for_isolated_paths(tmp_path, uri):
    """SQLite URI syntax must not bypass the selected worktree root."""
    context = RuntimeContext.create(tmp_path)
    config = {"db": {"path": uri}}

    with pytest.raises(ValueError, match="file: URIs are not allowed"):
        context.validate_config(config)
    with pytest.raises(ValueError, match="file: URIs are not allowed"):
        context.apply_config(config)


@pytest.mark.parametrize("uri", ["https://example.invalid/db", "sqlite:///tmp/db"])
def test_runtime_context_rejects_non_file_uri_for_isolated_paths(tmp_path, uri):
    context = RuntimeContext.create(tmp_path)

    with pytest.raises(ValueError, match="URI schemes are not allowed"):
        context.validate_config({"db": {"path": uri}})
    with pytest.raises(ValueError, match="URI schemes are not allowed"):
        context.apply_config({"db": {"path": uri}})


def test_runtime_context_rejects_memory_sentinel_for_non_db_paths(tmp_path):
    context = RuntimeContext.create(tmp_path)

    with pytest.raises(ValueError, match="does not accept the :memory: sentinel"):
        context.validate_config({"dirs": {"papers": ":memory:"}})
    with pytest.raises(ValueError, match="does not accept the :memory: sentinel"):
        context.apply_config({"dirs": {"papers": ":memory:"}})
