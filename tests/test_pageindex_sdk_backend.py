from types import SimpleNamespace

import fitz

from drbrain.parser.pageindex.sdk_backend import (
    _adapt_nodes,
    _markdown_to_temp_pdf,
    build_tree_with_sdk,
)


def test_adapt_sdk_tree_to_drbrain_nodes():
    """PDF page indices must not masquerade as Markdown line numbers."""
    result = _adapt_nodes(
        [
            {
                "title": "Methods",
                "node_id": "0001",
                "page_index": 3,
                "text": "Methods body",
                "nodes": [],
            }
        ]
    )
    assert result == [{"title": "Methods", "node_id": "0001", "text": "Methods body"}]


def test_markdown_temp_pdf_paginates_long_documents(tmp_path):
    text = "\n".join(f"line {i}: " + "x" * 90 for i in range(400))
    pdf_path = _markdown_to_temp_pdf(text)
    try:
        document = fitz.open(str(pdf_path))
        try:
            assert document.page_count > 1
            rendered = "".join(page.get_text() for page in document)
        finally:
            document.close()
        assert "line 0:" in rendered
        assert "line 399:" in rendered
    finally:
        pdf_path.unlink(missing_ok=True)


def test_sdk_backend_accepts_markdown_without_source_pdf(tmp_path, monkeypatch):
    md = tmp_path / "raw.md"
    md.write_text("# Title\n", encoding="utf-8")

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def submit_document(self, path, wait=True):
            return {"doc_id": "x"}

        def get_tree(self, doc_id, **kwargs):
            return {"result": []}

    monkeypatch.setitem(
        __import__("sys").modules, "pageindex", SimpleNamespace(PageIndexClient=FakeClient)
    )
    assert build_tree_with_sdk(md, SimpleNamespace(sdk_mode="local"))["structure"] == []
