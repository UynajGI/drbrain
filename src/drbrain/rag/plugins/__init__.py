"""Plugin interface abstraction: protocol + registry + discovery + backend helpers."""

from drbrain.rag.plugins.backends import load_joblib, run_subprocess, run_subprocess_json
from drbrain.rag.plugins.protocol import (
    Backend,
    OnFailure,
    Plugin,
    PluginResult,
    PluginType,
    ResultStatus,
    make_evidence,
)
from drbrain.rag.plugins.registry import PluginRegistry

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
