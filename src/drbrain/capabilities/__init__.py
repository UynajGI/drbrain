"""Neutral capability contracts shared by plugins, MCP tools, and Skills.

The package deliberately has no dependency on LlamaIndex or an MCP transport.
Adapters translate their native descriptors into :class:`CapabilityDescriptor`
and return :class:`InvocationResult` so discovery and audit code can remain
host agnostic.
"""

from drbrain.capabilities.adapters import APIAdapter, CLIAdapter, ModelAdapter
from drbrain.capabilities.catalog import CapabilityCatalog, CapabilityEntry
from drbrain.capabilities.protocol import (
    CapabilityAdapter,
    CapabilityAnnotations,
    CapabilityDescriptor,
    CapabilityExecution,
    CapabilityJobMethods,
    CapabilityKind,
    CapabilityProvenance,
    InvocationResult,
    InvocationStatus,
    canonical_json,
    dependency_lock_digest,
    descriptor_id,
    file_digest,
    input_digest,
    runtime_fingerprint,
    valid_job_id,
)
from drbrain.capabilities.skills import SkillFormatError, discover_skills, parse_skill
from drbrain.capabilities.validation import (
    validate_instance,
    validate_schema,
)

__all__ = [
    "CapabilityAnnotations",
    "CapabilityAdapter",
    "CapabilityDescriptor",
    "CapabilityExecution",
    "CapabilityJobMethods",
    "CapabilityKind",
    "CapabilityProvenance",
    "InvocationResult",
    "InvocationStatus",
    "canonical_json",
    "descriptor_id",
    "dependency_lock_digest",
    "file_digest",
    "input_digest",
    "runtime_fingerprint",
    "valid_job_id",
    "validate_instance",
    "validate_schema",
    "SkillFormatError",
    "discover_skills",
    "parse_skill",
    "CapabilityCatalog",
    "CapabilityEntry",
    "APIAdapter",
    "CLIAdapter",
    "ModelAdapter",
]
