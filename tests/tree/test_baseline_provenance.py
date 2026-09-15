"""Finding 8: the RAPTOR baseline may only score its own, provenance-verified build."""

from __future__ import annotations

import json

import pytest

from drbrain.rag.baselines import (
    RAPTOR_PROVENANCE_FIELDS,
    BaselineProvenanceError,
    load_raptor_build,
    raptor_provenance_fingerprint,
    verify_raptor_provenance,
)


def _record(**overrides):
    record = {
        "algorithm": "raptor",
        "algorithm_version": "raptor@0.0.1+vendored-pin",
        "build_params": {"umap_dim": 10, "threshold": 0.1, "lambda": 0.0},
        "corpus": {"name": "mixed-9", "digest": "sha256:aaa"},
        "generated_at": "2026-09-15T00:00:00Z",
        "generation": "gen-raptor-1",
        "manifest_fingerprint": "fp-raptor-1",
        "content_hash": "sha256:bbb",
        "members_hash": "sha256:ccc",
    }
    record.update(overrides)
    return record


_MANIFEST = {"schema": 1, "generation": "gen-raptor-1", "fingerprint": "fp-raptor-1"}


class TestVerification:
    def test_a_complete_record_binds_to_the_published_manifest(self):
        audit = verify_raptor_provenance(_record(), generation="gen-raptor-1", manifest=_MANIFEST)
        assert audit["algorithm"] == "raptor"
        assert audit["manifest_checked"] is True
        assert audit["manifest_fingerprint"] == "fp-raptor-1"
        assert audit["provenance_fingerprint"] == raptor_provenance_fingerprint(_record())

    @pytest.mark.parametrize("field", RAPTOR_PROVENANCE_FIELDS)
    def test_every_required_field_is_enforced(self, field):
        record = _record()
        record[field] = ""
        with pytest.raises(BaselineProvenanceError, match=field):
            verify_raptor_provenance(record, generation="gen-raptor-1", manifest=_MANIFEST)

    def test_a_missing_record_fails_closed(self):
        with pytest.raises(BaselineProvenanceError, match="no RAPTOR provenance"):
            verify_raptor_provenance(None, generation="gen-raptor-1", manifest=_MANIFEST)

    def test_another_algorithm_cannot_claim_the_raptor_label(self):
        with pytest.raises(BaselineProvenanceError, match="not by 'raptor'"):
            verify_raptor_provenance(
                _record(algorithm="unified-tree"), generation="gen-raptor-1", manifest=_MANIFEST
            )

    def test_a_generation_mismatch_is_refused(self):
        with pytest.raises(BaselineProvenanceError, match="names generation"):
            verify_raptor_provenance(_record(), generation="gen-other", manifest=_MANIFEST)

    def test_a_manifest_mismatch_is_refused(self):
        with pytest.raises(BaselineProvenanceError, match="manifest fingerprint"):
            verify_raptor_provenance(
                _record(), generation="gen-raptor-1", manifest={"fingerprint": "fp-other"}
            )

    def test_a_generation_without_a_fingerprint_cannot_be_bound(self):
        with pytest.raises(BaselineProvenanceError, match="no manifest fingerprint"):
            verify_raptor_provenance(
                _record(), generation="gen-raptor-1", manifest={"generation": "gen-raptor-1"}
            )

    def test_the_recorded_fingerprint_moves_with_the_build_params(self):
        assert raptor_provenance_fingerprint(_record(build_params={"umap_dim": 20})) != (
            raptor_provenance_fingerprint(_record())
        )


class TestOnDiskBuild:
    def _publish(
        self, root, *, generation="gen-raptor-1", fingerprint="fp-raptor-1", provenance=None
    ):
        generation_dir = root / "generations" / generation
        generation_dir.mkdir(parents=True)
        (generation_dir / "tree.sqlite3").write_bytes(b"")
        (generation_dir / "manifest.json").write_text(
            json.dumps({"schema": 1, "generation": generation, "fingerprint": fingerprint}),
            encoding="utf-8",
        )
        (root / "active.json").write_text(json.dumps({"generation": generation}), encoding="utf-8")
        if provenance is not None:
            (generation_dir / "raptor-provenance.json").write_text(
                json.dumps(provenance), encoding="utf-8"
            )
        return generation_dir

    def test_a_generation_without_a_provenance_record_is_refused(self, tmp_path):
        self._publish(tmp_path)
        with pytest.raises(BaselineProvenanceError, match="raptor-provenance.json"):
            load_raptor_build(None, storage_root=tmp_path)

    def test_a_verified_independent_build_is_loaded_with_its_audit(self, tmp_path):
        self._publish(tmp_path, provenance=_record())
        build = load_raptor_build(None, storage_root=tmp_path)
        assert build.generation == "gen-raptor-1"
        audit = build.verify()
        assert audit["manifest_checked"] is True
        assert audit["provenance_fingerprint"] == raptor_provenance_fingerprint(_record())

    def test_an_empty_raptor_root_is_refused(self, tmp_path):
        with pytest.raises(BaselineProvenanceError, match="no independent RAPTOR generation"):
            load_raptor_build(None, storage_root=tmp_path)

    def test_a_stale_provenance_record_does_not_bind_to_a_newer_manifest(self, tmp_path):
        self._publish(tmp_path, fingerprint="fp-raptor-2", provenance=_record())
        build = load_raptor_build(None, storage_root=tmp_path)
        with pytest.raises(BaselineProvenanceError, match="manifest fingerprint"):
            build.verify()

    def test_the_default_raptor_arm_reads_its_own_root_not_the_tree_root(
        self, tmp_path, monkeypatch
    ):
        from drbrain.rag import baselines

        class _Li:
            tree_storage = str(tmp_path / "tree")
            raptor_storage = str(tmp_path / "raptor")

        import drbrain.rag.config as rag_config

        monkeypatch.setattr(rag_config, "get_llamaindex_config", lambda cfg=None: _Li())
        monkeypatch.setattr(
            baselines, "_runtime_scoped_path", lambda value, *, label: value, raising=True
        )
        (tmp_path / "tree").mkdir()
        with pytest.raises(BaselineProvenanceError, match="raptor"):
            load_raptor_build(object())
