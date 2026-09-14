from __future__ import annotations

import json
from pathlib import Path

from drbrain.rag.pageindex_native import _find_doc_id, chat_document, filesystem_status


def test_pageindex_filesystem_status_and_doc_lookup(tmp_path: Path):
    storage = tmp_path / ".pageindex"
    (storage / "docs" / "pi-test").mkdir(parents=True)
    manifest = {
        "docs": {
            "pi-test": {
            "id": "pi-test",
            "name": "source.pdf",
            "description": "",
            "status": "completed",
            "createdAt": "2026-01-01T00:00:00Z",
            "pageNum": 1,
            "folderId": None,
            "metadata": {"drbrain_paper_id": "p123"},
            "mode": "flash",
            }
        }
    }
    (storage / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (storage / "docs" / "pi-test" / "doc.json").write_text(
        json.dumps(manifest["docs"]["pi-test"]), encoding="utf-8"
    )

    assert _find_doc_id(storage, "p123") == "pi-test"
    assert filesystem_status({}, tmp_path) == {
        "storage_path": str(storage),
        "documents": 1,
        "ready": True,
    }


def test_native_chat_extracts_local_chat_response(monkeypatch, tmp_path: Path):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf")

    class FakeClient:
        def chat_completions(self, *args, **kwargs):
            assert kwargs["doc_id"] == "pi-test"
            return {"choices": [{"message": {"content": "native answer"}}]}

    monkeypatch.setattr("drbrain.rag.pageindex_native.ensure_document", lambda *a: "pi-test")
    monkeypatch.setattr("drbrain.rag.pageindex_native._client", lambda *a: FakeClient())

    result = chat_document({}, "p123", tmp_path, "What is this?")
    assert result["answer"] == "native answer"
    assert result["engine"] == "pageindex_native_chat"
    assert result["native"] is True
