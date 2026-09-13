"""Project scope — the durable identity behind the WebUI project switcher.

A *project* is the platform-level namespace that owns a corpus reference, a
set of conversation sessions, and the research runs started from them.  It is
deliberately independent from the human-readable workspace *name*: the
``project_id`` never changes when a project is renamed, and legacy data
(created before projects existed) belongs to :data:`DEFAULT_PROJECT_ID`.

The module is dependency-free so both the storage layer, the research loop and
the WebUI can share the constants without importing each other.
"""

from __future__ import annotations

import re
import uuid

#: Stable identity of the implicit project that owns all legacy rows.
DEFAULT_PROJECT_ID = "prj-default"
#: Display name of the implicit default project.
DEFAULT_PROJECT_NAME = "默认项目"
#: Label used for runs that were started outside a conversation session.
UNBOUND_SESSION_LABEL = "未绑定会话"

_PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def new_project_id() -> str:
    """Return a fresh, stable project identifier."""
    return f"prj-{uuid.uuid4().hex[:12]}"


def is_valid_project_id(value: object) -> bool:
    """Return whether *value* is a well-formed project identifier."""
    return bool(_PROJECT_ID_RE.match(str(value or "")))


def normalize_project_id(value: object | None) -> str:
    """Map ``None``/blank selectors to the default project, validating the rest.

    Callers at HTTP boundaries receive user input; an unknown or malformed id
    must fail closed instead of silently falling back to the default project.
    """
    if value is None:
        return DEFAULT_PROJECT_ID
    text = str(value).strip()
    if not text:
        return DEFAULT_PROJECT_ID
    if not is_valid_project_id(text):
        raise ValueError(f"invalid project id: {value!r}")
    return text


__all__ = [
    "DEFAULT_PROJECT_ID",
    "DEFAULT_PROJECT_NAME",
    "UNBOUND_SESSION_LABEL",
    "is_valid_project_id",
    "new_project_id",
    "normalize_project_id",
]
