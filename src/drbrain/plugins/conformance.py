"""Plugin conformance self-check — static, handler-free validation for plugin authors.

Plugin authors run this against their plugin directory *before* shipping::

    python -m drbrain.plugins.conformance <plugin_dir>

Each ``*.py`` module is imported exactly the way
:meth:`PluginRegistry.discover` does, then validated **without ever executing
a handler**:

* imports cleanly under discovery rules;
* declares ``PLUGIN_MANIFEST`` (manifest style) or ``register()`` (inline style);
* descriptor fields carry valid types: non-empty name/description,
  ``timeout_s > 0``, known ``side_effect``, supported ``abi_version``;
* ``input_schema`` is an object JSON Schema with dict ``properties``;
* ``code_digest``, when declared, matches the module file's sha256 (hashed
  with the declared value itself blanked — a hash cannot attest a file
  containing it);
* source contains no hardcoded secrets (API-key-shaped strings);
* ``JOB_METHODS``, when declared, exposes callable ``submit``/``poll``/``cancel``.

For inline-style modules the suite additionally runs ``register()`` against a
throwaway registry (registration is part of the documented contract and never
invokes handlers) so descriptor-level checks apply uniformly to both styles.

Conformance is deliberately *stricter* than the loader: the loader must stay
fail-open for forward compatibility (unknown manifest keys are dropped
silently), while this suite reports them, so a typo'd field name cannot vanish
without a trace.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import importlib.util
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args

from drbrain.plugins.manifest import (
    HANDLER_KEY,
    JOB_METHODS_KEY,
    MANIFEST_KEY,
    REQUIRED_FIELDS,
    has_manifest,
)
from drbrain.plugins.protocol import SUPPORTED_ABI_VERSIONS, Plugin, PluginSideEffect
from drbrain.plugins.registry import PluginRegistry

# Known ``side_effect`` literals, derived from the protocol alias so the check
# cannot drift from the contract it validates.
_SIDE_EFFECTS = frozenset(get_args(PluginSideEffect))

# API-key-shaped strings that must never appear in plugin source; secrets are
# referenced via ``Plugin.secret_refs``, never inlined.
_SECRET_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"sk-[A-Za-z0-9]{16,}",
        r"AKIA[0-9A-Z]{16}",
        r"ghp_[A-Za-z0-9]{30,}",
        r"xox[bap]-[A-Za-z0-9-]{10,}",
    )
)

_KNOWN_FIELDS = frozenset(f.name for f in dataclasses.fields(Plugin))


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one conformance check (``name`` is dot-namespaced per module)."""

    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class ConformanceReport:
    """Aggregated result over every module in a plugin directory."""

    checks: list[CheckResult] = field(default_factory=list)
    passed: bool = True


def run_conformance(target: str | Path) -> ConformanceReport:
    """Run the isolated probe conformance suite over a plugin directory.

    Never executes plugin handlers; import and inline-style ``register()``
    run exactly once per module, mirroring discovery.
    """
    directory = Path(target)
    checks: list[CheckResult] = []
    if not directory.is_dir():
        checks.append(CheckResult("directory", False, f"not a directory: {directory}"))
        return ConformanceReport(checks=checks, passed=False)
    checks.append(CheckResult("directory", True, str(directory)))
    modules = sorted(p for p in directory.glob("*.py") if not p.name.startswith("_"))
    if not modules:
        checks.append(CheckResult("modules", False, "no plugin modules (*.py) found"))
        return ConformanceReport(checks=checks, passed=False)
    for path in modules:
        checks.extend(_module_checks(path))
    return ConformanceReport(checks=checks, passed=all(check.passed for check in checks))


def run_probe_conformance(target: str | Path) -> ConformanceReport:
    """Explicit name for the legacy import-and-register probe suite."""
    return run_conformance(target)


def run_lint_conformance(target: str | Path) -> ConformanceReport:
    """Perform AST-only checks without importing or executing plugin code.

    Lint is safe to run in CI on untrusted plugin source.  Inline descriptors
    are intentionally deferred to the isolated probe because their values are
    produced by ``register()``; the lint result still checks syntax, entrypoint
    shape, and source-level secrets.
    """
    directory = Path(target)
    checks: list[CheckResult] = []
    if not directory.is_dir():
        return ConformanceReport(
            checks=[CheckResult("directory", False, f"not a directory: {directory}")],
            passed=False,
        )
    checks.append(CheckResult("directory", True, str(directory)))
    modules = sorted(p for p in directory.glob("*.py") if not p.name.startswith("_"))
    if not modules:
        checks.append(CheckResult("modules", False, "no plugin modules (*.py) found"))
        return ConformanceReport(checks=checks, passed=False)
    for path in modules:
        checks.extend(_lint_module_checks(path))
    return ConformanceReport(checks=checks, passed=all(check.passed for check in checks))


def _lint_module_checks(path: Path) -> list[CheckResult]:
    stem = path.stem
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [CheckResult(f"{stem}.syntax", False, str(exc))]
    checks = [CheckResult(f"{stem}.syntax", True, "parsed without importing")]
    checks.append(_secrets_check(f"{stem}.secrets", path))
    assignments = {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    has_manifest_node = MANIFEST_KEY in assignments
    has_register = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "register"
        for node in tree.body
    )
    if has_manifest_node:
        checks.append(CheckResult(f"{stem}.entrypoint", True, f"declares {MANIFEST_KEY}"))
        try:
            manifest = ast.literal_eval(assignments[MANIFEST_KEY])
        except (ValueError, TypeError, SyntaxError):
            manifest = None
            checks.append(
                CheckResult(f"{stem}.manifest", False, f"{MANIFEST_KEY} must be a literal dict")
            )
        if manifest is not None:
            checks.append(_manifest_shape_check(stem, manifest))
            checks.append(_manifest_fields_check(stem, manifest))
            if all(check.passed for check in checks[-2:]):
                checks.append(_schema_check(f"{stem}.input_schema", manifest["input_schema"]))
                try:
                    plugin = Plugin(**manifest)
                except Exception as exc:  # malformed literal descriptor
                    checks.append(CheckResult(f"{stem}.descriptor", False, str(exc)))
                else:
                    checks.extend(_descriptor_checks(stem, plugin, path))
        handler_ok = HANDLER_KEY in assignments or any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == HANDLER_KEY
            for node in tree.body
        )
        checks.append(
            CheckResult(
                f"{stem}.handler",
                handler_ok,
                "module-level HANDLER declaration found"
                if handler_ok
                else f"missing module-level {HANDLER_KEY}",
            )
        )
    elif has_register:
        checks.append(CheckResult(f"{stem}.entrypoint", True, "declares register()"))
        checks.append(
            CheckResult(
                f"{stem}.probe",
                True,
                "descriptor checks deferred to isolated probe; register() was not executed",
            )
        )
    else:
        checks.append(
            CheckResult(
                f"{stem}.entrypoint",
                False,
                f"declares neither {MANIFEST_KEY} nor register()",
            )
        )
    return checks


def _load_module(path: Path) -> tuple[Any, str | None]:
    """Import a plugin module the way ``discover()`` does; ``(module, error)``."""
    module_name = f"drbrain_conformance_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return None, "cannot build module spec"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 — the failure IS the report entry
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        sys.modules.pop(module_name, None)
    return module, None


def _module_checks(path: Path) -> list[CheckResult]:
    stem = path.stem
    module, error = _load_module(path)
    if module is None:
        return [CheckResult(f"{stem}.imports", False, error or "import failed")]
    checks = [CheckResult(f"{stem}.imports", True, "imported cleanly")]
    checks.append(_secrets_check(f"{stem}.secrets", path))
    manifest = getattr(module, MANIFEST_KEY, None)
    if has_manifest(module):
        if not isinstance(manifest, dict):
            checks.append(
                CheckResult(
                    f"{stem}.manifest",
                    False,
                    f"{MANIFEST_KEY} must be a dict, got {type(manifest).__name__}",
                )
            )
            return checks
        checks.extend(_manifest_checks(stem, path, module, manifest))
        return checks
    register = getattr(module, "register", None)
    if callable(register):
        checks.extend(_inline_checks(stem, path, module))
        return checks
    checks.append(
        CheckResult(
            f"{stem}.entrypoint",
            False,
            f"declares neither {MANIFEST_KEY} nor register()",
        )
    )
    return checks


def _manifest_checks(stem: str, path: Path, module: Any, manifest: Any) -> list[CheckResult]:
    checks = [CheckResult(f"{stem}.entrypoint", True, f"declares {MANIFEST_KEY}")]
    checks.append(_manifest_shape_check(stem, manifest))
    checks.append(_manifest_fields_check(stem, manifest))
    if not all(check.passed for check in checks[1:]):
        return checks  # descriptor checks are meaningless without a usable manifest
    handler = getattr(module, HANDLER_KEY, None)
    checks.append(
        CheckResult(
            f"{stem}.handler",
            callable(handler),
            f"callable module-level {HANDLER_KEY}"
            if callable(handler)
            else f"{HANDLER_KEY} must be a callable module-level function",
        )
    )
    jobs_check = _jobs_check(f"{stem}.job_methods", module)
    if jobs_check is not None:
        checks.append(jobs_check)
    checks.extend(_descriptor_checks(stem, Plugin(**manifest), path))
    return checks


def _manifest_shape_check(stem: str, manifest: Any) -> CheckResult:
    name = f"{stem}.manifest"
    if not isinstance(manifest, dict):
        return CheckResult(name, False, f"{MANIFEST_KEY} must be a dict")
    missing = [key for key in REQUIRED_FIELDS if key not in manifest]
    if missing:
        return CheckResult(name, False, f"missing required fields: {', '.join(missing)}")
    problems = []
    for key in ("name", "description"):
        value = manifest[key]
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{key} must be a non-empty string")
    if not isinstance(manifest["input_schema"], dict):
        problems.append("input_schema must be a dict")
    if problems:
        return CheckResult(name, False, "; ".join(problems))
    return CheckResult(name, True, f"declares {manifest['name']!r}")


def _manifest_fields_check(stem: str, manifest: Any) -> CheckResult:
    name = f"{stem}.manifest_fields"
    if not isinstance(manifest, dict):
        return CheckResult(name, False, f"{MANIFEST_KEY} must be a dict")
    non_str = [key for key in manifest if not isinstance(key, str)]
    if non_str:
        return CheckResult(name, False, f"manifest keys must be strings: {non_str!r}")
    unknown = sorted(set(manifest) - _KNOWN_FIELDS)
    if unknown:
        return CheckResult(
            name,
            False,
            "unknown fields (the loader would silently drop them): " + ", ".join(unknown),
        )
    return CheckResult(name, True, "all fields recognized")


def _inline_checks(stem: str, path: Path, module: Any) -> list[CheckResult]:
    checks = [CheckResult(f"{stem}.entrypoint", True, "declares register()")]
    scratch = PluginRegistry()
    try:
        module.register(scratch)
    except Exception as exc:  # noqa: BLE001 — e.g. fail-closed ABI negotiation
        checks.append(CheckResult(f"{stem}.registration", False, f"register() raised: {exc}"))
        return checks
    registered = scratch.list_plugins()
    if not registered:
        checks.append(
            CheckResult(f"{stem}.registration", False, "register() registered no plugins")
        )
        return checks
    checks.append(
        CheckResult(f"{stem}.registration", True, f"registered {len(registered)} plugin(s)")
    )
    for plugin in registered:
        checks.extend(_descriptor_checks(f"{stem}.{plugin.name}", plugin, path))
    return checks


def _descriptor_checks(prefix: str, plugin: Plugin, path: Path) -> list[CheckResult]:
    checks = [_schema_check(f"{prefix}.input_schema", plugin.input_schema)]
    timeout = plugin.timeout_s
    timeout_ok = isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and timeout > 0
    checks.append(CheckResult(f"{prefix}.timeout_s", timeout_ok, f"timeout_s={timeout!r}"))
    effect_ok = isinstance(plugin.side_effect, str) and plugin.side_effect in _SIDE_EFFECTS
    checks.append(
        CheckResult(
            f"{prefix}.side_effect",
            effect_ok,
            f"side_effect={plugin.side_effect!r}, known={sorted(_SIDE_EFFECTS)}",
        )
    )
    abi_ok = (
        isinstance(plugin.abi_version, int)
        and not isinstance(plugin.abi_version, bool)
        and plugin.abi_version in SUPPORTED_ABI_VERSIONS
    )
    checks.append(
        CheckResult(
            f"{prefix}.abi_version",
            abi_ok,
            f"abi_version={plugin.abi_version!r}, supported={sorted(SUPPORTED_ABI_VERSIONS)}",
        )
    )
    if plugin.code_digest:
        checks.append(_digest_check(f"{prefix}.code_digest", plugin.code_digest, path))
    return checks


def _schema_check(name: str, schema: Any) -> CheckResult:
    if not isinstance(schema, dict):
        return CheckResult(name, False, "input_schema must be a dict (JSON Schema object)")
    if schema.get("type") != "object":
        return CheckResult(
            name, False, f'input_schema.type must be "object", got {schema.get("type")!r}'
        )
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return CheckResult(name, False, "input_schema.properties must be a dict")
    return CheckResult(name, True, f"object schema with {len(properties)} properties")


def _digest_check(name: str, declared: str, path: Path) -> CheckResult:
    if not isinstance(declared, str):
        return CheckResult(
            name, False, f"code_digest must be a string, got {type(declared).__name__}"
        )
    actual = _module_digest(path)
    expected = declared.strip().lower()
    if expected.startswith("sha256:"):
        expected = expected.split(":", 1)[1]
    if expected != actual:
        return CheckResult(name, False, f"declared {declared!r} != actual sha256:{actual}")
    return CheckResult(name, True, "matches module sha256")


# Single-pass blanking for the ``code_digest`` value: the alternation covers
# the dict-literal and keyword forms (any quote style) so the earliest
# declaration in the file is blanked regardless of which style it uses.
_DIGEST_BLANK_RE = re.compile(rb'("code_digest"\s*:\s*|code_digest\s*=\s*)(["\'])[^"\']*\2')


def _module_digest(path: Path) -> str:
    """sha256 of the module file with the ``code_digest`` value blanked.

    The declaration lives in the very file it attests, so hashing the raw
    bytes would be self-referential (a hash cannot contain its own value).
    Blanking is anchored to the ``"code_digest"`` declaration with a single
    combined pattern covering BOTH declaration styles (dict-literal
    ``"code_digest": "..."`` and inline keyword ``code_digest="..."``, any
    quote style): the earliest declaration in the file is blanked in one pass,
    so no per-form first-occurrence skew is possible.  This exactly reverses
    the author's fill-in step (write the digest empty, hash, fill in), works
    whatever form was filled (``sha256:`` prefix, case, whitespace), and never
    touches digest-like strings elsewhere in the file.
    """
    raw = path.read_bytes()
    blanked = _DIGEST_BLANK_RE.sub(rb"\1\2\2", raw, count=1)
    return hashlib.sha256(blanked).hexdigest()


def _jobs_check(name: str, module: Any) -> CheckResult | None:
    raw = getattr(module, JOB_METHODS_KEY, None)
    if raw is None:
        return None
    missing = [
        attr for attr in ("submit", "poll", "cancel") if not callable(getattr(raw, attr, None))
    ]
    if missing:
        return CheckResult(name, False, f"non-callable/missing members: {', '.join(missing)}")
    return CheckResult(name, True, "submit/poll/cancel are callable")


def _secrets_check(name: str, path: Path) -> CheckResult:
    text = path.read_text(encoding="utf-8", errors="replace")
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pattern in _SECRET_PATTERNS:
            if pattern.search(line):
                return CheckResult(
                    name,
                    False,
                    f"hardcoded secret matching {pattern.pattern!r} at line {lineno}"
                    " (reference secrets via secret_refs, never inline them)",
                )
    return CheckResult(name, True, "no hardcoded secrets")


def main(argv: list[str] | None = None) -> int:
    """CLI entry: print the per-check report; exit 0 all-pass / 1 any-fail."""
    parser = argparse.ArgumentParser(
        prog="python -m drbrain.plugins.conformance",
        description=(
            "Static conformance self-check for a drbrain plugin directory"
            " (handlers are never executed)."
        ),
    )
    parser.add_argument("plugin_dir", help="directory containing plugin *.py modules")
    parser.add_argument(
        "--mode",
        choices=("lint", "probe"),
        default="probe",
        help="lint uses AST only; probe imports modules and calls register() in-process",
    )
    args = parser.parse_args(argv)
    report = (
        run_lint_conformance(args.plugin_dir)
        if args.mode == "lint"
        else run_probe_conformance(args.plugin_dir)
    )
    for check in report.checks:
        mark = "PASS" if check.passed else "FAIL"
        detail = f" — {check.detail}" if check.detail else ""
        print(f"[{mark}] {check.name}{detail}")
    failed = sum(1 for check in report.checks if not check.passed)
    total = len(report.checks)
    print(f"\n{total - failed}/{total} checks passed: {'OK' if report.passed else 'FAILED'}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
