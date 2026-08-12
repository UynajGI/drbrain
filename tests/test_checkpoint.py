"""Tests for drbrain.utils.checkpoint primitives."""

import pytest

from drbrain.utils.checkpoint import Checkpoint, DoneFlag, IdempotentFile


class TestCheckpoint:
    def test_load_missing_returns_default(self, tmp_path):
        ckpt = Checkpoint(tmp_path / "offset.json")
        assert ckpt.load() == 0

    def test_load_missing_custom_default(self, tmp_path):
        ckpt = Checkpoint(tmp_path / "offset.json", default=-1)
        assert ckpt.load() == -1

    def test_save_and_load_int(self, tmp_path):
        ckpt = Checkpoint(tmp_path / "offset.json")
        ckpt.save(42)
        assert ckpt.load() == 42

    def test_save_and_load_dict(self, tmp_path):
        ckpt = Checkpoint(tmp_path / "state.json")
        ckpt.save({"offset": 7, "ok": 12})
        assert ckpt.load() == {"offset": 7, "ok": 12}

    def test_save_overwrites_previous_state(self, tmp_path):
        ckpt = Checkpoint(tmp_path / "offset.json")
        ckpt.save(1)
        ckpt.save(2)
        assert ckpt.load() == 2

    def test_load_corrupt_returns_default(self, tmp_path):
        path = tmp_path / "offset.json"
        path.write_text("{not json", encoding="utf-8")
        ckpt = Checkpoint(path, default=-1)
        assert ckpt.load() == -1

    def test_out_of_bounds_resets_to_default(self, tmp_path):
        ckpt = Checkpoint(tmp_path / "offset.json")
        ckpt.save(10)
        assert ckpt.load(max_value=10) == 0  # 10 >= 10 -> stale, reset
        assert ckpt.load(max_value=11) == 10  # still in bounds

    def test_save_is_valid_json_and_cleans_tmp(self, tmp_path):
        ckpt = Checkpoint(tmp_path / "offset.json")
        ckpt.save(3)
        assert not (tmp_path / "offset.json.tmp").exists()
        assert ckpt.path.read_text(encoding="utf-8").strip() == "3"


class TestIdempotentFile:
    def test_not_done_before_mark(self, tmp_path):
        src = tmp_path / "data.jsonl"
        src.write_text("line1\n", encoding="utf-8")
        idem = IdempotentFile(src)
        assert not idem.is_done()

    def test_unchanged_source_skips_after_mark(self, tmp_path):
        src = tmp_path / "data.jsonl"
        src.write_text("line1\n", encoding="utf-8")
        idem = IdempotentFile(src)
        idem.mark_done()
        assert idem.is_done()  # source untouched -> done

    def test_changed_source_reruns(self, tmp_path):
        src = tmp_path / "data.jsonl"
        src.write_text("line1\n", encoding="utf-8")
        idem = IdempotentFile(src)
        idem.mark_done()
        src.write_text("line1\nline2\n", encoding="utf-8")  # size changed
        assert not idem.is_done()

    def test_default_marker_path(self, tmp_path):
        src = tmp_path / "data.jsonl"
        src.write_text("x", encoding="utf-8")
        idem = IdempotentFile(src)
        idem.mark_done()
        assert (tmp_path / "data.jsonl.idem_marker").exists()

    def test_custom_marker_path(self, tmp_path):
        src = tmp_path / "data.jsonl"
        marker = tmp_path / "markers" / "data.merged_marker"
        src.write_text("x", encoding="utf-8")
        idem = IdempotentFile(src, marker=marker)
        idem.mark_done()
        assert marker.exists()
        assert idem.is_done()

    def test_missing_source_never_done(self, tmp_path):
        idem = IdempotentFile(tmp_path / "nope.jsonl")
        assert not idem.is_done()
        with pytest.raises(FileNotFoundError):
            idem.mark_done()

    def test_marker_mismatch_reruns(self, tmp_path):
        src = tmp_path / "data.jsonl"
        src.write_text("x", encoding="utf-8")
        idem = IdempotentFile(src)
        idem.mark_done()
        idem.marker.write_text("not-a-matching-fingerprint", encoding="utf-8")
        assert not idem.is_done()

    def test_unreadable_marker_reruns(self, tmp_path):
        src = tmp_path / "data.jsonl"
        src.write_text("x", encoding="utf-8")
        idem = IdempotentFile(src)
        idem.mark_done()
        idem.marker.unlink()
        idem.marker.mkdir()  # marker is now a directory -> read fails
        assert not idem.is_done()

    def test_content_hash_mode_detects_same_size_change(self, tmp_path):
        src = tmp_path / "data.bin"
        src.write_bytes(b"aaaa")
        idem = IdempotentFile(src, use_content_hash=True)
        idem.mark_done()
        src.write_bytes(b"aaab")  # same size, different bytes
        assert not idem.is_done()


class TestDoneFlag:
    def test_not_done_initially(self, tmp_path):
        flag = DoneFlag(tmp_path / "done.flag")
        assert not flag.is_done()

    def test_done_writes_flag_file(self, tmp_path):
        flag = DoneFlag(tmp_path / "done.flag")
        flag.done()
        assert flag.is_done()
        assert (tmp_path / "done.flag").exists()

    def test_done_writes_timestamp_by_default(self, tmp_path):
        flag = DoneFlag(tmp_path / "done.flag")
        flag.done()
        content = (tmp_path / "done.flag").read_text(encoding="utf-8").strip()
        assert content  # non-empty timestamp

    def test_done_custom_content(self, tmp_path):
        flag = DoneFlag(tmp_path / "done.flag")
        flag.done("ok=123 fail=0")
        assert (tmp_path / "done.flag").read_text(encoding="utf-8") == "ok=123 fail=0"

    def test_done_overwrites_previous_flag(self, tmp_path):
        flag = DoneFlag(tmp_path / "done.flag")
        flag.done("first")
        flag.done("second")
        assert (tmp_path / "done.flag").read_text(encoding="utf-8") == "second"
