"""Conformance suite — clean fixture plugins pass; seeded violations fail.

The suite is static (handlers are never executed): it imports each module the
way ``discover()`` does and validates descriptors, schemas, digests, secret
hygiene and job-method shapes. Violations are seeded here as throwaway modules
in ``tmp_path`` — including a *fake* API-key-shaped string that is assembled
from fragments so no real-looking literal ever lands in this repo's source.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from drbrain.plugins.conformance import CheckResult, ConformanceReport, run_conformance

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "plugins"

MANIFEST_OK = """
PLUGIN_MANIFEST = {
    "name": "conform_ok",
    "description": "clean manifest plugin",
    "input_schema": {
        "type": "object",
        "properties": {"composition": {"type": "object"}},
        "required": ["composition"],
    },
    "abi_version": 1,
    "side_effect": "read",
    "timeout_s": 30.0,
}


def HANDLER(arguments):
    return {"ok": True}
"""


def _write(tmp_path: Path, sources: dict[str, str]) -> Path:
    for filename, source in sources.items():
        (tmp_path / filename).write_text(source, encoding="utf-8")
    return tmp_path


def _failed_names(report: ConformanceReport) -> set[str]:
    return {check.name for check in report.checks if not check.passed}


def test_report_shape():
    report = run_conformance(FIXTURE_DIR)
    assert isinstance(report, ConformanceReport)
    assert report.checks and all(isinstance(check, CheckResult) for check in report.checks)
    assert all(check.name for check in report.checks)
    assert report.passed is all(check.passed for check in report.checks)


def test_clean_fixture_plugins_pass():
    report = run_conformance(FIXTURE_DIR)
    assert report.passed, _failed_names(report)


def test_clean_manifest_module_passes(tmp_path):
    report = run_conformance(_write(tmp_path, {"ok_plugin.py": MANIFEST_OK}))
    assert report.passed, _failed_names(report)


def test_non_string_side_effect_fails_without_crash(tmp_path):
    """A list-vs-string typo must fail the check, not raise TypeError (unhashable)."""
    source = MANIFEST_OK.replace('"side_effect": "read"', '"side_effect": ["read"]')
    report = run_conformance(_write(tmp_path, {"bad_effect.py": source}))
    assert not report.passed
    assert "bad_effect.side_effect" in _failed_names(report)


def test_non_string_code_digest_fails_without_crash(tmp_path):
    source = MANIFEST_OK.replace('"abi_version": 1', '"abi_version": 1, "code_digest": 123')
    report = run_conformance(_write(tmp_path, {"bad_digest.py": source}))
    assert not report.passed
    assert "bad_digest.code_digest" in _failed_names(report)


def test_cli_exit_zero_on_clean_fixtures():
    proc = subprocess.run(
        [sys.executable, "-m", "drbrain.plugins.conformance", str(FIXTURE_DIR)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[PASS]" in proc.stdout and "OK" in proc.stdout


def test_unimportable_module_fails_but_others_still_checked(tmp_path):
    broken = "def broken(:\n"
    report = run_conformance(
        _write(tmp_path, {"broken_plugin.py": broken, "ok_plugin.py": MANIFEST_OK})
    )
    assert not report.passed
    assert "broken_plugin.imports" in _failed_names(report)
    assert all(check.passed for check in report.checks if check.name.startswith("ok_plugin."))


def test_missing_entrypoint_fails(tmp_path):
    report = run_conformance(_write(tmp_path, {"plain_plugin.py": "X = 1\n"}))
    assert not report.passed
    assert "plain_plugin.entrypoint" in _failed_names(report)


def test_bad_input_schema_fails(tmp_path):
    source = MANIFEST_OK.replace(
        '"input_schema": {\n        "type": "object",\n        "properties": {"composition": {"type": "object"}},\n        "required": ["composition"],\n    },',
        '"input_schema": {"type": "string"},',
    )
    report = run_conformance(_write(tmp_path, {"schema_plugin.py": source}))
    assert not report.passed
    assert "schema_plugin.input_schema" in _failed_names(report)


def test_missing_required_manifest_field_fails(tmp_path):
    source = MANIFEST_OK.replace('    "description": "clean manifest plugin",\n', "")
    report = run_conformance(_write(tmp_path, {"incomplete_plugin.py": source}))
    assert not report.passed
    assert "incomplete_plugin.manifest" in _failed_names(report)


def test_unknown_manifest_field_fails(tmp_path):
    source = MANIFEST_OK.replace(
        '    "abi_version": 1,\n',
        '    "abi_version": 1,\n    "timout_s": 5,\n',  # typo'd on purpose
    )
    report = run_conformance(_write(tmp_path, {"typo_plugin.py": source}))
    assert not report.passed
    assert "typo_plugin.manifest_fields" in _failed_names(report)


def test_hardcoded_secret_fails(tmp_path):
    # Fake key assembled from fragments: the tmp module's source carries the
    # secret-shaped string; this repo's own source never does.
    fake_key = "sk-" + "TESTFAKEKEY1234567890"
    source = (
        "from drbrain.plugins import Plugin\n\n"
        f"OPENAI_KEY = {fake_key!r}\n\n\n"
        "def register(registry):\n"
        "    registry.register(\n"
        "        Plugin(name='leaky', description='d', input_schema={'type': 'object'}),\n"
        "        lambda arguments: {},\n"
        "    )\n"
    )
    report = run_conformance(_write(tmp_path, {"leaky_plugin.py": source}))
    assert not report.passed
    assert "leaky_plugin.secrets" in _failed_names(report)


def test_code_digest_mismatch_fails(tmp_path):
    source = MANIFEST_OK.replace(
        '    "timeout_s": 30.0,\n',
        f'    "timeout_s": 30.0,\n    "code_digest": "{"0" * 64}",\n',
    )
    report = run_conformance(_write(tmp_path, {"digest_plugin.py": source}))
    assert not report.passed
    assert "digest_plugin.code_digest" in _failed_names(report)


def test_code_digest_match_passes(tmp_path):
    """Attestation workflow: hash the manifest with the digest blanked, then fill in."""
    source = MANIFEST_OK.replace(
        '    "timeout_s": 30.0,\n',
        '    "timeout_s": 30.0,\n    "code_digest": "",\n',
    )
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    source = source.replace('"code_digest": ""', f'"code_digest": "sha256:{digest}"')
    report = run_conformance(_write(tmp_path, {"digest_plugin.py": source}))
    assert report.passed, _failed_names(report)


def test_bad_side_effect_fails(tmp_path):
    source = MANIFEST_OK.replace('"side_effect": "read"', '"side_effect": "destroy"')
    report = run_conformance(_write(tmp_path, {"effect_plugin.py": source}))
    assert not report.passed
    assert "effect_plugin.side_effect" in _failed_names(report)


def test_unsupported_abi_fails(tmp_path):
    source = MANIFEST_OK.replace('"abi_version": 1', '"abi_version": 99')
    report = run_conformance(_write(tmp_path, {"future_plugin.py": source}))
    assert not report.passed
    assert "future_plugin.abi_version" in _failed_names(report)


def test_bad_job_methods_fail(tmp_path):
    source = MANIFEST_OK + '\n\nJOB_METHODS = {"submit": "not-callable"}\n'
    report = run_conformance(_write(tmp_path, {"jobs_plugin.py": source}))
    assert not report.passed
    assert "jobs_plugin.job_methods" in _failed_names(report)


def test_nonpositive_timeout_fails(tmp_path):
    source = MANIFEST_OK.replace('"timeout_s": 30.0', '"timeout_s": 0')
    report = run_conformance(_write(tmp_path, {"timeout_plugin.py": source}))
    assert not report.passed
    assert "timeout_plugin.timeout_s" in _failed_names(report)


def test_missing_directory_and_empty_directory_fail(tmp_path):
    missing = run_conformance(tmp_path / "nope")
    assert not missing.passed
    assert "directory" in _failed_names(missing)

    empty = run_conformance(_write(tmp_path, {"_helper.py": "X = 1\n"}))
    assert not empty.passed
    assert "modules" in _failed_names(empty)
