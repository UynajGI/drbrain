"""Unified model-role resolution (T17).

Design: ``docs/unified-tree-rag-design.md`` §7 — four roles, each bound to a
named endpoint: ``index_model`` (all build-time judgments, title/region and
group summaries), ``chat_model`` (online tree navigation and the final answer),
``embedding_model`` (raw text + new summary vectors) and ``rerank_model``
(query–evidence cross-encoder scores).

This module is the *only* place that assembles an endpoint's ``base_url`` and
``api_key``.  PageIndex/RAPTOR/agent call sites resolve their role here instead
of each building an address from ``llm.models``, and no global environment
variable switches a role between endpoints.

Precedence per role (first hit wins; every later hit must agree):

1. ``llm.roles[<alias>]`` → an entry in ``llm.endpoints`` (canonical alias first,
   then the legacy aliases this repo already ships, e.g. ``pageindex_index``).
   For ``embedding_model``/``rerank_model`` the named endpoint may also live in
   ``retrieval.endpoints``.
2. ``llm.index`` / ``llm.chat`` — the explicit per-role chains; for the
   retrieval roles, ``retrieval.embed`` / ``retrieval.rerank`` naming an entry
   in ``retrieval.endpoints``.
3. ``llm.models`` — legacy default, **chat-ish roles only**.  The index role
   never falls back to it: an unconfigured ``index_model`` raises instead, so a
   build can never silently run on the generic answer model.

Compatibility/conflict rule: every hit for one role must land on the same
endpoint (same ``base_url`` + ``model``); a disagreement, an unknown endpoint
name, or a role that no path configures raises :class:`ModelRoleError` with an
actionable message rather than picking one silently.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from drbrain.extractor.llm_client import resolve_base_url
from drbrain.security import REDACTED, redact_sensitive_text

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a config import cycle
    from drbrain.config import Config

ROLE_INDEX = "index_model"
ROLE_CHAT = "chat_model"
ROLE_EMBEDDING = "embedding_model"
ROLE_RERANK = "rerank_model"

#: The unified roles, in display order for ``drbrain check``.
MODEL_ROLES: tuple[str, ...] = (ROLE_INDEX, ROLE_CHAT, ROLE_EMBEDDING, ROLE_RERANK)

#: Roles whose "native" registry is ``retrieval.endpoints`` rather than an
#: OpenAI-compatible chat endpoint.
RETRIEVAL_ROLES: frozenset[str] = frozenset({ROLE_EMBEDDING, ROLE_RERANK})

#: Accepted ``llm.roles`` keys per unified role.  The canonical name wins; the
#: legacy names are the routing table this repository already ships
#: (``config.local.yaml``: ``pageindex_index``/``pageindex_chat``/``rag_chat``).
ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    ROLE_INDEX: ("index_model", "pageindex_index", "index"),
    ROLE_CHAT: ("chat_model", "rag_chat", "pageindex_chat", "chat"),
    ROLE_EMBEDDING: ("embedding_model", "embed"),
    ROLE_RERANK: ("rerank_model", "rerank"),
}

#: Providers that run in-process (sentence-transformers) or are pure markers:
#: they have no OpenAI-compatible base_url and no credential.
LOCAL_PROVIDERS: frozenset[str] = frozenset({"local", "sentence-transformers", "none", ""})

#: Providers whose deployment ignores the API key (self-hosted OpenAI-compatible
#: servers).  A loopback ``base_url`` is treated the same way.
KEYLESS_PROVIDERS: frozenset[str] = frozenset(
    {
        "local",
        "sentence-transformers",
        "none",
        "",
        "ollama",
        "llama.cpp",
        "llamacpp",
        "vllm",
        "lmstudio",
        "text-generation-inference",
        "tgi",
    }
)

_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})

_DEFAULT_MAX_CONCURRENT = 4


class ModelRoleError(RuntimeError):
    """A role could not be resolved to exactly one usable endpoint.

    Carries the role name and an ``action`` hint so CLI/audit output can tell
    the operator what to fix instead of only what failed.
    """

    def __init__(self, role: str, message: str, *, action: str = "") -> None:
        self.role = role
        self.action = action
        text = f"model role {role}: {message}"
        if action:
            text = f"{text} — {action}"
        super().__init__(text)


@dataclass(frozen=True)
class ModelRole:
    """One resolved role → concrete endpoint binding.

    ``api_key`` is used by the clients but never printed: :meth:`redacted` and
    ``__repr__`` both drop the credential so the role can be logged safely.
    """

    role: str
    endpoint_name: str
    provider: str
    model: str
    base_url: str
    api_key: str
    max_concurrent: int = _DEFAULT_MAX_CONCURRENT
    source: str = "endpoint"
    timeout_secs: float = 60.0

    @property
    def host(self) -> str:
        """Host[:port] of ``base_url`` (no userinfo, path or query)."""
        if not self.base_url:
            return ""
        return urlparse(self.base_url).netloc

    @property
    def identity(self) -> str:
        """Stable per-endpoint identity for concurrency gates and caches.

        Same model name at two different URLs yields two identities; two roles
        pointing at one endpoint yield one identity (and therefore one shared
        concurrency gate).
        """
        return f"{self.endpoint_name}|{self.base_url}|{self.model}"

    @property
    def is_local(self) -> bool:
        return self.provider.strip().lower() in LOCAL_PROVIDERS

    @property
    def requires_api_key(self) -> bool:
        """Whether an empty ``api_key`` is a hard error for this endpoint.

        Loopback and self-hosted providers are keyless by convention; a remote
        provider with no key fails closed in the clients.
        """
        provider = self.provider.strip().lower()
        if provider in KEYLESS_PROVIDERS:
            return False
        return urlparse(self.base_url).hostname not in _LOOPBACK_HOSTS

    @property
    def missing_credential_reason(self) -> str:
        """Actionable reason when this endpoint cannot authenticate.

        Returns ``""`` when the endpoint needs no key or has a usable one.  An
        unresolved ``${ENV_VAR}`` placeholder counts as missing: sending the
        literal marker as a bearer token would fail with an opaque 401.
        """
        if not self.requires_api_key:
            return ""
        label = self.endpoint_name or self.model or "<inline>"
        if not self.api_key:
            return (
                f"endpoint {label} has no api_key; set `api_key` on the endpoint entry "
                "or via an exported ${ENV_VAR}"
            )
        if self.api_key.startswith("${") and self.api_key.endswith("}"):
            variable = self.api_key[2:-1].strip()
            return (
                f"endpoint {label} api_key is the unresolved placeholder {self.api_key}; "
                f"export {variable} (or put the literal key in config.local.yaml)"
            )
        return ""

    @property
    def usable_api_key(self) -> bool:
        """True when a credential is present and not an unresolved placeholder."""
        return not self.missing_credential_reason

    def same_target(self, other: ModelRole) -> bool:
        """True when both roles resolve to the same concrete endpoint."""
        return (
            self.base_url.rstrip("/") == other.base_url.rstrip("/")
            and self.model == other.model
            and self.provider.strip().lower() == other.provider.strip().lower()
        )

    def redacted(self) -> dict[str, Any]:
        """Serializable projection without the credential value."""
        return {
            "role": self.role,
            "endpoint_name": self.endpoint_name,
            "provider": self.provider,
            "model": self.model,
            "base_url": redact_sensitive_text(self.base_url) or "",
            "host": self.host,
            "max_concurrent": self.max_concurrent,
            "source": self.source,
            "timeout_secs": self.timeout_secs,
            "api_key": REDACTED if self.api_key else "not set",
        }

    def __repr__(self) -> str:
        fields = ", ".join(f"{key}={value!r}" for key, value in self.redacted().items())
        return f"ModelRole({fields})"


# ── config access helpers (typed Config *and* plain dicts are accepted) ───────


def _get(source: Any, key: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, Mapping):
        value = source.get(key, default)
    else:
        value = getattr(source, key, default)
    return default if value is None else value


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _as_positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _endpoint_base_url(provider: str, base_url: str, model: str) -> str:
    """Base URL for one endpoint, reusing the shared client's provider defaults.

    Fails closed (never silently routes an unknown provider to api.openai.com).
    """
    if provider.strip().lower() in LOCAL_PROVIDERS:
        return str(base_url or "").rstrip("/")
    return resolve_base_url({"provider": provider, "base_url": base_url, "model": model})


def _role_from_entry(
    role: str,
    *,
    endpoint_name: str,
    entry: Mapping[str, Any],
    source: str,
) -> ModelRole:
    label = endpoint_name or "<inline>"
    provider = str(entry.get("provider") or "").strip()
    model = str(entry.get("model") or "").strip()
    if not model:
        raise ModelRoleError(
            role,
            f"endpoint {label} has no model name",
            action="set `model` on the endpoint entry",
        )
    api_key = str(entry.get("api_key") or "").strip()
    if not api_key and entry.get("api_keys"):
        raise ModelRoleError(
            role,
            f"endpoint {label} only declares `api_keys` (a rotating pool), "
            "which the role clients do not support",
            action="add an `api_key` for this endpoint",
        )
    try:
        resolved_base_url = _endpoint_base_url(provider, str(entry.get("base_url") or ""), model)
    except ValueError as exc:
        raise ModelRoleError(
            role,
            f"endpoint {label} has no usable base_url ({exc})",
            action=f"set `base_url` on {label if endpoint_name else 'the endpoint entry'}",
        ) from exc
    return ModelRole(
        role=role,
        endpoint_name=endpoint_name,
        provider=provider,
        model=model,
        base_url=resolved_base_url,
        api_key=api_key,
        max_concurrent=_as_positive_int(entry.get("max_concurrent"), _DEFAULT_MAX_CONCURRENT),
        source=source,
        timeout_secs=_as_positive_float(entry.get("timeout"), 60.0),
    )


def _named_endpoint(
    role: str,
    alias: str,
    name: str,
    llm_endpoints: Mapping[str, Any],
    retrieval_endpoints: Mapping[str, Any],
) -> ModelRole:
    """Resolve ``llm.roles[alias] = name`` to a concrete endpoint."""
    entry = _as_mapping(llm_endpoints.get(name))
    if not entry and role in RETRIEVAL_ROLES:
        entry = _as_mapping(retrieval_endpoints.get(name))
    if not entry:
        known = sorted(
            set(llm_endpoints) | (set(retrieval_endpoints) if role in RETRIEVAL_ROLES else set())
        )
        registries = (
            "llm.endpoints" if role not in RETRIEVAL_ROLES else "llm.endpoints/retrieval.endpoints"
        )
        raise ModelRoleError(
            role,
            f"llm.roles.{alias} names endpoint {name!r}, which is not defined in {registries}",
            action=f"register it there (known: {', '.join(known) or 'none'})",
        )
    return _role_from_entry(role, endpoint_name=name, entry=entry, source="roles")


def _first_chain_entry(chain: Any) -> Mapping[str, Any]:
    if isinstance(chain, (list, tuple)) and chain:
        first = chain[0]
        return first if isinstance(first, Mapping) else {}
    return {}


def resolve_model_role(cfg: Config | Mapping[str, Any], role: str) -> ModelRole:
    """Resolve one unified role to a single concrete endpoint.

    The returned :class:`ModelRole` is the only carrier of ``base_url`` and
    ``api_key`` for that role.  Raises :class:`ModelRoleError` on a missing
    role, an unknown endpoint name, or two declaration paths that disagree.
    """
    if role not in MODEL_ROLES:
        raise ModelRoleError(
            str(role),
            "unknown model role",
            action=f"use one of: {', '.join(MODEL_ROLES)}",
        )

    llm = _get(cfg, "llm") or {}
    llm_endpoints = _as_mapping(_get(llm, "endpoints"))
    role_map = _as_mapping(_get(llm, "roles"))
    retrieval = _get(cfg, "retrieval") or {}
    retrieval_endpoints = _as_mapping(_get(retrieval, "endpoints"))

    candidates: list[ModelRole] = []

    # 1. Explicit routing table (canonical alias first, then legacy aliases).
    for alias in ROLE_ALIASES[role]:
        name = str(role_map.get(alias) or "").strip()
        if name:
            candidates.append(
                _named_endpoint(role, alias, name, llm_endpoints, retrieval_endpoints)
            )

    # 2. Role-native chain/registry.
    if role == ROLE_INDEX:
        entry = _first_chain_entry(_get(llm, "index"))
        if entry:
            candidates.append(_role_from_entry(role, endpoint_name="", entry=entry, source="index"))
    elif role == ROLE_CHAT:
        entry = _first_chain_entry(_get(llm, "chat"))
        if entry:
            candidates.append(_role_from_entry(role, endpoint_name="", entry=entry, source="chat"))
    else:
        key = "embed" if role == ROLE_EMBEDDING else "rerank"
        name = str(_get(retrieval, key, "") or "").strip()
        if name:
            entry = _as_mapping(retrieval_endpoints.get(name))
            if not entry:
                known = sorted(retrieval_endpoints)
                raise ModelRoleError(
                    role,
                    f"retrieval.{key} names endpoint {name!r}, which is not defined in retrieval.endpoints",
                    action=f"register it there (known: {', '.join(known) or 'none'})",
                )
            candidates.append(
                _role_from_entry(role, endpoint_name=name, entry=entry, source="endpoint")
            )

    # 3. Legacy generic list — chat-ish role only (never the index role).
    if not candidates and role == ROLE_CHAT:
        entry = _first_chain_entry(_get(llm, "models"))
        if entry:
            candidates.append(
                _role_from_entry(role, endpoint_name="", entry=entry, source="models")
            )

    if not candidates:
        raise ModelRoleError(
            role, "no endpoint is configured for this role", action=_missing_hint(role)
        )

    primary = candidates[0]
    for other in candidates[1:]:
        if not other.same_target(primary):
            raise ModelRoleError(
                role,
                "conflicting definitions resolve to different endpoints: "
                f"{_describe(primary)} (from {primary.source}) vs "
                f"{_describe(other)} (from {other.source})",
                action="remove one definition or point both at the same endpoint",
            )
    return primary


def _describe(role: ModelRole) -> str:
    name = role.endpoint_name or "<inline>"
    return f"{name} {role.provider}/{role.model} @ {role.base_url or '(local)'}"


def _missing_hint(role: str) -> str:
    if role == ROLE_INDEX:
        return (
            "add llm.roles.index_model → an llm.endpoints entry, or an llm.index chain. "
            "The generic llm.models list is not used for the build/index role"
        )
    if role == ROLE_CHAT:
        return "add llm.roles.chat_model → an llm.endpoints entry, or an llm.chat chain"
    key = "embed" if role == ROLE_EMBEDDING else "rerank"
    return f"register retrieval.endpoints.<name> and point retrieval.{key} at it"


def role_summary(cfg: Config | Mapping[str, Any], role: str) -> dict[str, Any]:
    """Redacted resolution summary for CLI/audit display.

    Never raises: an unresolvable role becomes ``{"role": ..., "error": ...}`` so
    a status command can report the problem instead of failing.
    """
    try:
        resolved = resolve_model_role(cfg, role)
    except ModelRoleError as exc:
        return {"role": role, "error": str(exc)}
    return resolved.redacted()
