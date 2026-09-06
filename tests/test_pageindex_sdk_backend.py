from types import SimpleNamespace

from drbrain.parser.pageindex.sdk_backend import _adapt_nodes, build_tree_with_sdk


def test_adapt_sdk_tree_to_drbrain_nodes():
    result = _adapt_nodes([{"title": "Methods", "node_id": "0001", "page_index": 3, "nodes": []}])
    assert result == [{"title": "Methods", "node_id": "0001", "line_num": 4}]


def test_sdk_backend_accepts_markdown_without_source_pdf(tmp_path, monkeypatch):
    md = tmp_path / "raw.md"
    md.write_text("# Title\n", encoding="utf-8")
    class FakeClient:
        def __init__(self, **kwargs): pass
        def submit_document(self, path, wait=True): return {"doc_id": "x"}
        def get_tree(self, doc_id, **kwargs): return {"result": []}
    monkeypatch.setitem(__import__('sys').modules, "pageindex", SimpleNamespace(PageIndexClient=FakeClient))
    assert build_tree_with_sdk(md, SimpleNamespace(sdk_mode="local"))["structure"] == []
