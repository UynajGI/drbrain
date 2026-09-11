"""RAG configuration helpers: read the ``llamaindex:`` config section.

The typed dataclasses live in :mod:`drbrain.config` (alongside ``EmbedConfig``
etc.) so the whole config surface stays in one place; this module only wires
the section into the RAG layer. Ticket: T1 (infrastructure).
"""

from __future__ import annotations

from typing import Any
from dataclasses import fields, is_dataclass

from drbrain.config import Config, LlamaIndexConfig, load_config

__all__ = ["LlamaIndexConfig", "get_llamaindex_config"]


def coerce_config(cfg: Config | dict[str, Any]) -> Config:
    """Normalize mapping inputs once at an API boundary without dropping sections."""
    if not isinstance(cfg, dict):
        return cfg
    result = Config()
    known = {item.name for item in fields(result)}
    unknown = set(cfg) - known
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    for key, value in cfg.items():
        default = getattr(result, key)
        if is_dataclass(default) and isinstance(value, dict):
            constructor = type(default)
            convert = getattr(constructor, "from_dict", None)
            value = convert(value) if convert is not None else constructor(**value)
        setattr(result, key, value)
    return result


def get_llamaindex_config(cfg: Config | dict[str, Any] | None = None) -> LlamaIndexConfig:
    """Return the ``llamaindex`` section as a typed :class:`LlamaIndexConfig`.

    Accepts a loaded :class:`~drbrain.config.Config`, a raw dict, or ``None``
    (falls back to ``load_config()`` on the default ``config.yaml``). A missing
    section yields the dataclass defaults, so callers never see ``None``.
    """
    if cfg is None:
        cfg = load_config()
    li = getattr(cfg, "llamaindex", None)
    if isinstance(li, LlamaIndexConfig):
        return li
    if isinstance(li, dict):
        return LlamaIndexConfig.from_dict(li)
    if isinstance(cfg, dict):
        return LlamaIndexConfig.from_dict(cfg.get("llamaindex", {}))
    return LlamaIndexConfig()
