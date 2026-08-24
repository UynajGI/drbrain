#!/usr/bin/env python
"""全量增强管线共享模块：配置加载。

设计要点（吸取旧版教训）：
- load_cfg 始终以 config.local.yaml 为基础（含 llm.models 多引擎/ox-alpha-free 顺序），
  --config 只做增量覆盖（如 config.embed1.yaml 只覆盖 embed 部分）。
"""

from __future__ import annotations

from pathlib import Path

import yaml

from drbrain.config import merge_dicts

ROOT = Path("/home/jiangyuan/drbrain")


def load_cfg(config_path: str | None = None) -> dict:
    base = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
    local = yaml.safe_load((ROOT / "config.local.yaml").read_text(encoding="utf-8")) or {}
    if config_path:
        extra = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        local = merge_dicts(local, extra)
    return merge_dicts(base, local)
