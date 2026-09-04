"""Plugin interface abstraction: protocol + registry + discovery + backend helpers.

Asynchronous job contract — the ``jobs/`` directory (review 2026-09-03
P-E1 / §7.2)
---------------------------------------------------------------------
Hour-scale compute (e.g. DFT) must not block a loop turn. A plugin exposes it
by registering :class:`JobMethods` next to its synchronous handler:

* ``submit(arguments) -> job_id`` — enqueue the job, return quickly;
* ``poll(job_id) -> {"status": JobStatus | str, "result"?: Any, "error"?: str}``;
* ``cancel(job_id) -> bool`` — best-effort cancellation request.

Directory contract — the **only evidence the T4 gate trusts**:

* the host owns a per-run ``jobs/`` working directory and hands it to the
  plugin (via an input argument such as ``jobs_dir`` or an environment
  variable); plugins never invent their own job directory;
* when a job finishes, the plugin MUST write its durable result JSON to
  ``jobs/<job_id>.json`` and its full log to ``jobs/<job_id>.log``;
* result claims are trusted only from those on-disk files — never from text
  pasted into the conversation or from ``poll()`` payloads alone.

Results and jobs may additionally report :class:`Artifact` entries
(``path`` + ``sha256``) so evidence consumers can re-verify produced files
byte-for-byte instead of trusting the transcript.
"""

from drbrain.plugins.backends import load_joblib, run_subprocess, run_subprocess_json
from drbrain.plugins.protocol import (
    Artifact,
    Backend,
    JobMethods,
    JobStatus,
    OnFailure,
    Plugin,
    PluginResult,
    PluginType,
    ResultStatus,
    make_evidence,
)
from drbrain.plugins.registry import DEFAULT_MAX_OUTPUT_BYTES, PluginRegistry, json_schema_to_model

__all__ = [
    "Artifact",
    "Backend",
    "DEFAULT_MAX_OUTPUT_BYTES",
    "JobMethods",
    "JobStatus",
    "OnFailure",
    "Plugin",
    "PluginRegistry",
    "PluginResult",
    "PluginType",
    "ResultStatus",
    "json_schema_to_model",
    "load_joblib",
    "make_evidence",
    "run_subprocess",
    "run_subprocess_json",
]
