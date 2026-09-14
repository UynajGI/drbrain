"""T01/T02: vendored sources are pinned, verified, and loaded in isolation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from drbrain.tree import upstream
from drbrain.tree.vendor import (
    PAGEINDEX_SPEC,
    RAPTOR_SPEC,
    VENDOR_SPECS,
    VendorSourceError,
    verify_spec,
    verify_vendor,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _submodules_present() -> bool:
    return all((REPO_ROOT / spec.relative_path / ".git").exists() for spec in VENDOR_SPECS)


requires_submodules = pytest.mark.skipif(
    not _submodules_present(),
    reason="vendor submodules are not checked out (git submodule update --init vendor)",
)


@requires_submodules
def test_vendor_commits_licenses_and_symbols_verified():
    commits = verify_vendor(REPO_ROOT)
    assert commits["pageindex"] == PAGEINDEX_SPEC.commit
    assert commits["raptor"] == RAPTOR_SPEC.commit


class TestVerificationErrors:
    def test_missing_vendor_dir_is_actionable(self, tmp_path):
        problems = verify_spec(PAGEINDEX_SPEC, tmp_path)
        assert problems and "submodule" in problems[0]

    def test_wrong_commit_reports_both_revisions(self, tmp_path):
        fake = tmp_path / RAPTOR_SPEC.relative_path
        fake.mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=fake, check=True, timeout=60)
        subprocess.run(
            [
                "git",
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "commit",
                "--allow-empty",
                "-q",
                "-m",
                "x",
            ],
            cwd=fake,
            check=True,
            timeout=60,
        )
        problems = verify_spec(RAPTOR_SPEC, tmp_path)
        assert any("expected commit" in problem for problem in problems)
        assert any(RAPTOR_SPEC.commit in problem for problem in problems)

    def test_missing_symbol_is_named(self, tmp_path):
        fake = tmp_path / PAGEINDEX_SPEC.relative_path
        fake.mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=fake, check=True, timeout=60)
        subprocess.run(
            [
                "git",
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "commit",
                "--allow-empty",
                "-q",
                "-m",
                "x",
            ],
            cwd=fake,
            check=True,
            timeout=60,
        )
        # Pin the fake checkout to the expected commit id by matching HEAD.
        import drbrain.tree.vendor as vendor_mod

        original = vendor_mod._head_commit(fake)
        spec = PAGEINDEX_SPEC
        if original == spec.commit:
            pytest.skip("cannot force commit equality in this environment")
        # Compare through a spec whose expected commit is the fake HEAD.
        fake_spec = type(spec)(
            name=spec.name,
            relative_path=spec.relative_path,
            commit=original or "",
            license_file=spec.license_file,
            required_symbols={"pageindex/page_index_md.py": ("md_to_tree",)},
        )
        problems = verify_spec(fake_spec, tmp_path)
        assert any("license file" in problem for problem in problems)
        assert any("required source" in problem for problem in problems)


class TestNarrowLoaders:
    def test_allowlist_rejects_unknown_modules(self):
        with pytest.raises(KeyError, match="allowlist"):
            upstream.load_raptor_module("RetrievalAugmentation")
        with pytest.raises(KeyError, match="allowlist"):
            upstream.load_pageindex_module("local_store")

    @requires_submodules
    def test_raptor_cluster_utils_loads_without_heavy_stack(self):
        upstream.reset_caches()
        module = upstream.load_raptor_module("cluster_utils")
        assert hasattr(module, "GMM_cluster") and hasattr(module, "RAPTOR_Clustering")
        for banned in ("faiss", "torch", "sentence_transformers"):
            assert not any(name.startswith(banned) for name in sys.modules), banned

    @requires_submodules
    def test_pageindex_md_and_classic_load_without_sdk_store(self):
        upstream.reset_caches()
        md_module = upstream.load_pageindex_module("page_index_md")
        classic = upstream.load_pageindex_module("page_index_classic")
        assert hasattr(md_module, "md_to_tree") and hasattr(classic, "page_index_main")
        assert not any("local_store" in name or "cloud_api" in name for name in sys.modules)

    @requires_submodules
    def test_missing_dependency_gives_actionable_error(self):
        """A blocked optional dependency must not surface as a bare ImportError."""
        code = (
            "import sys; sys.modules['umap'] = None\n"
            "from drbrain.tree import upstream\n"
            "try:\n"
            "    upstream.load_raptor_module('cluster_utils')\n"
            "except upstream.MissingTreeDependencyError as exc:\n"
            "    assert '--extra tree' in str(exc), str(exc)\n"
            "    print('ACTIONABLE')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "ACTIONABLE" in result.stdout

    def test_cli_import_does_not_require_tree_deps(self):
        """Minimal installs must still be able to run the CLI."""
        code = (
            "import sys\n"
            "for name in ('umap', 'sklearn', 'tiktoken', 'PyPDF2', 'pypdfium2'):\n"
            "    sys.modules[name] = None\n"
            "import drbrain.cli.main  # noqa: F401\n"
            "print('CLI_IMPORT_OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "CLI_IMPORT_OK" in result.stdout


@requires_submodules
def test_vendor_error_message_mentions_repair():
    problems = verify_spec(PAGEINDEX_SPEC, REPO_ROOT)
    assert problems == []
    with pytest.raises(VendorSourceError) as excinfo:
        verify_vendor(REPO_ROOT / "does-not-exist")
    assert "submodule" in str(excinfo.value)
