"""YAML config loader with local overlay support."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge: override wins for leaf values, base keys preserved."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = merge_dicts(result[key], val)
        else:
            result[key] = val
    return result


def load_config(
    base_path: str | Path = "config.yaml",
    local_path: str | Path = "config.local.yaml",
) -> dict[str, Any]:
    """Load base config and optionally merge local overlay."""
    base = Path(base_path)
    if not base.exists():
        raise FileNotFoundError(f"Config not found: {base}")

    with open(base) as f:
        cfg = yaml.safe_load(f) or {}

    local = Path(local_path)
    if local.exists():
        with open(local) as f:
            overlay = yaml.safe_load(f) or {}
        cfg = merge_dicts(cfg, overlay)

    return cfg
