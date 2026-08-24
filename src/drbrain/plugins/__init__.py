"""Plugin interface abstraction: protocol + registry + discovery + backend helpers."""

from drbrain.plugins.backends import load_joblib, run_subprocess, run_subprocess_json
from drbrain.plugins.protocol import (
    Backend,
    OnFailure,
    Plugin,
    PluginResult,
    PluginType,
    ResultStatus,
    make_evidence,
)
from drbrain.plugins.registry import PluginRegistry

__all__ = [
    "Backend",
    "OnFailure",
    "Plugin",
    "PluginRegistry",
    "PluginResult",
    "PluginType",
    "ResultStatus",
    "load_joblib",
    "make_evidence",
    "run_subprocess",
    "run_subprocess_json",
]
