"""Model-as-Tool plugins: expose trained models as agent tools (generic protocol)."""

from drbrain.rag.plugins.backends import load_joblib, run_subprocess_json
from drbrain.rag.plugins.protocol import (
    Backend,
    ModelResult,
    ModelTool,
    ModelType,
    OnFailure,
    ResultStatus,
    make_evidence,
)
from drbrain.rag.plugins.registry import ModelToolRegistry

__all__ = [
    "Backend",
    "ModelResult",
    "ModelTool",
    "ModelToolRegistry",
    "ModelType",
    "OnFailure",
    "ResultStatus",
    "load_joblib",
    "make_evidence",
    "run_subprocess_json",
]
