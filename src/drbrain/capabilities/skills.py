"""Agent Skills adapter: parse instruction packages without executing code."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from drbrain.capabilities.protocol import (
    CapabilityAnnotations,
    CapabilityDescriptor,
    CapabilityExecution,
    CapabilityProvenance,
    dependency_lock_digest,
    descriptor_id,
    file_digest,
)

_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_MAX_NAME = 64
_MAX_DESCRIPTION = 1024


class SkillFormatError(ValueError):
    """Raised when a SKILL.md does not satisfy the portable Skill contract."""


def _frontmatter(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillFormatError("SKILL.md must start with YAML frontmatter")
    try:
        end = next(index for index, line in enumerate(lines[1:], 1) if line.strip() == "---")
    except StopIteration as exc:
        raise SkillFormatError("SKILL.md frontmatter is not terminated") from exc
    try:
        value = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError as exc:
        raise SkillFormatError(f"SKILL.md frontmatter is not valid YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise SkillFormatError("SKILL.md frontmatter must be a mapping")
    return value


def parse_skill(skill_dir: str | Path) -> CapabilityDescriptor:
    """Parse one skill directory into a neutral descriptor.

    The body and optional resources are retained as metadata.  This adapter
    never imports or executes files under the skill directory.
    """
    directory = Path(skill_dir).resolve()
    document = directory / "SKILL.md"
    if not document.is_file():
        raise SkillFormatError(f"missing SKILL.md in {directory}")
    fields = _frontmatter(document.read_text(encoding="utf-8"))
    name = fields.get("name")
    description = fields.get("description")
    if (
        not isinstance(name, str)
        or not name
        or len(name) > _MAX_NAME
        or not _NAME_RE.fullmatch(name)
    ):
        raise SkillFormatError("name must be 1-64 lowercase characters, digits, or hyphens")
    if (
        not isinstance(description, str)
        or not description.strip()
        or len(description) > _MAX_DESCRIPTION
    ):
        raise SkillFormatError("description must be a non-empty string of at most 1024 characters")
    if directory.name != name:
        raise SkillFormatError(f"skill directory {directory.name!r} must match name {name!r}")
    allowed_tools = fields.get("allowed-tools", fields.get("allowed_tools", []))
    if isinstance(allowed_tools, str):
        allowed_tools = [allowed_tools]
    if not isinstance(allowed_tools, list) or not all(
        isinstance(item, str) for item in allowed_tools
    ):
        raise SkillFormatError("allowed-tools must be a list of strings")
    resource_paths = tuple(
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != document and path.resolve().is_relative_to(directory)
    )
    resources = tuple(str(path.relative_to(directory)) for path in resource_paths)
    metadata = {
        key: value
        for key, value in fields.items()
        if key not in {"name", "description", "allowed-tools", "allowed_tools"}
    }
    metadata.update({"allowed_tools": list(allowed_tools), "resources": list(resources)})
    return CapabilityDescriptor(
        id=descriptor_id("skill", name),
        name=name,
        description=description.strip(),
        kind="skill",
        version=str(fields.get("version") or ""),
        input_schema={},
        annotations=CapabilityAnnotations(read_only=True, destructive=False),
        execution=CapabilityExecution(mode="sync"),
        permissions=tuple(allowed_tools),
        metadata=metadata,
        provenance=CapabilityProvenance(
            source=str(document),
            version=str(fields.get("version") or ""),
            resource_digests=tuple(filter(None, (file_digest(path) for path in resource_paths))),
            dependency_digest=dependency_lock_digest(directory),
        ),
        resource_scope={"root": str(directory), "entrypoint": str(document)},
    )


def discover_skills(root: str | Path, *, strict: bool = False) -> list[CapabilityDescriptor]:
    """Discover direct child skill packages, skipping malformed ones by default."""
    directory = Path(root).resolve()
    if not directory.is_dir():
        return []
    descriptors: list[CapabilityDescriptor] = []
    for child in sorted(directory.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        try:
            descriptors.append(parse_skill(child))
        except (OSError, SkillFormatError):
            if strict:
                raise
    return descriptors
