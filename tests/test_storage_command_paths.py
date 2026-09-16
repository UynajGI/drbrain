"""``drbrain storage`` reads configured paths from typed configs, not only dicts.

Regression: the module guarded on ``isinstance(cfg, dict)`` while the CLI
context carries a typed ``Config`` (``ctx.obj["config"]``), so a configured
``db.path``/``dirs.papers`` was silently ignored and the defaults won.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from drbrain.cli.storage_commands import _db_path, _papers_root
from drbrain.config import load_config
from drbrain.runtime import RuntimeContext


def _ctx(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(obj={"runtime": RuntimeContext.create(tmp_path)})


def _write_config(tmp_path: Path) -> Path:
    payload = {
        "db": {"path": "data/library.sqlite"},
        "dirs": {
            "inbox": "data/spool/inbox",
            "papers": "data/library-papers",
            "logs": "data/logs",
        },
        "llm": {"models": []},
        "embed": {"provider": "none"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_typed_config_paths_are_honoured(tmp_path: Path) -> None:
    cfg = load_config(_write_config(tmp_path))
    ctx = _ctx(tmp_path)

    assert not isinstance(cfg, dict)
    assert _db_path(ctx, cfg) == tmp_path / "data/library.sqlite"
    assert _papers_root(ctx, cfg, "") == tmp_path / "data/library-papers"


def test_plain_mappings_still_resolve_the_same_paths(tmp_path: Path) -> None:
    cfg = {
        "db": {"path": "data/library.sqlite"},
        "dirs": {"papers": "data/library-papers"},
    }
    ctx = _ctx(tmp_path)

    assert _db_path(ctx, cfg) == tmp_path / "data/library.sqlite"
    assert _papers_root(ctx, cfg, "") == tmp_path / "data/library-papers"


def test_explicit_flags_and_defaults_win_in_order(tmp_path: Path) -> None:
    cfg = load_config(_write_config(tmp_path))
    ctx = _ctx(tmp_path)

    assert _papers_root(ctx, cfg, "data/explicit") == tmp_path / "data/explicit"
    assert _db_path(ctx, {}) == tmp_path / "data/drbrain.db"
