import threading

import pytest

from drbrain.dedup.resolver import PaperIDs
from drbrain.storage.database import Database


@pytest.mark.parametrize("ids", [PaperIDs(doi="10.1/x"), PaperIDs(arxiv="2301.001")])
def test_malformed_external_identifiers_are_not_preserved(ids):
    assert ids.normalized() == PaperIDs()


def test_strict_paper_insert_rejects_existing_identity(tmp_path):
    db = Database(tmp_path / "test.db")
    try:
        db.insert_paper("paper", "Original", 2020, "uploaded")
        with pytest.raises(ValueError):
            db.insert_paper("paper", "Replacement", 2021, "uploaded", strict=True)
        assert db.get_paper("paper")["title"] == "Original"
    finally:
        db.close()


def test_write_lock_serializes_ingest_batches(tmp_path):
    db = Database(tmp_path / "test.db")
    started = threading.Event()
    entered = threading.Event()

    def contender():
        started.set()
        with db.write_lock():
            entered.set()

    worker = threading.Thread(target=contender)
    try:
        with db.write_lock():
            with db.write_lock():
                worker.start()
                assert started.wait(2)
                assert not entered.wait(0.1)
        worker.join(2)
        assert entered.is_set()
        assert not worker.is_alive()
    finally:
        worker.join(2)
        db.close()
