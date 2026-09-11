"""Optional adapter for the official PageIndex Python SDK.

The adapter deliberately returns DrBrain's ``DocumentTree`` shape so the rest
of the parser and retrieval stack remains independent of SDK response details.
"""

from __future__ import annotations

import os
import tempfile
import textwrap
from pathlib import Path
from typing import Any


def configure_tree_backend(tree_config: Any, pageindex_config: Any) -> Any:
    """Apply typed ``Config.pageindex`` settings to a ``TreeConfig``."""
    if pageindex_config is None:
        return tree_config
    get = (
        pageindex_config.get
        if isinstance(pageindex_config, dict)
        else lambda k, d=None: getattr(pageindex_config, k, d)
    )
    tree_config.backend = get("backend", "sdk")
    tree_config.sdk_mode = get("mode", "local")
    tree_config.sdk_model = get("model")
    tree_config.sdk_chat_model = get("chat_model")
    tree_config.sdk_storage_path = get("storage_path")
    tree_config.sdk_api_key = get("api_key", "")
    tree_config.sdk_base_url = get("base_url", "")
    tree_config.sdk_index_backend = get("index_backend", {}) or {}
    tree_config.sdk_chat_backend = get("chat_backend", {}) or {}
    return tree_config


def _wrap_line(line: str, width: int = 100) -> list[str]:
    pieces = textwrap.wrap(line, width=width, replace_whitespace=False, drop_whitespace=False)
    wrapped = []
    for piece in pieces or [""]:
        while len(piece) > width:
            wrapped.append(piece[:width])
            piece = piece[width:]
        wrapped.append(piece)
    return wrapped


def _markdown_to_temp_pdf(text: str) -> Path:
    """Render markdown into a paginated disposable PDF for SDK submission.

    Long documents must flow across pages: a single fixed-size textbox would
    truncate or silently drop content and PageIndex would index an incomplete
    document even though the full ``raw.md`` is available.
    """
    import fitz

    fd, name = tempfile.mkstemp(prefix="drbrain-pageindex-", suffix=".pdf")
    os.close(fd)
    document = fitz.open()
    try:
        page = document.new_page()
        y = 48.0
        for raw_line in text.splitlines() or [""]:
            for piece in _wrap_line(raw_line):
                if y > 800.0:
                    page = document.new_page()
                    y = 48.0
                page.insert_text(fitz.Point(36.0, y), piece, fontsize=10)
                y += 12.0
        document.save(name)
    finally:
        document.close()
    return Path(name)


def build_tree_with_sdk(md_path: str | Path, config: Any) -> dict:
    """Build a tree with PageIndex local or cloud indexing.

    The SDK accepts PDFs; ``md_path`` must therefore live beside ``source.pdf``.
    Set ``config.sdk_mode`` to ``local`` or ``cloud`` and install ``pageindex``.
    """
    try:
        from pageindex import PageIndexClient
    except ImportError as exc:
        raise RuntimeError("PageIndex SDK backend requires: pip install pageindex") from exc

    md = Path(md_path)
    pdf = md.with_name("source.pdf")
    temporary_pdf: Path | None = None
    if not pdf.is_file():
        # The SDK accepts PDF input; preserve markdown-only callers by
        # rendering a paginated disposable PDF from the extracted material.
        temporary_pdf = _markdown_to_temp_pdf(md.read_text(encoding="utf-8"))
        pdf = temporary_pdf

    mode = getattr(config, "sdk_mode", None) or getattr(config, "mode", "local")
    model = (
        getattr(config, "sdk_model", None) or getattr(config, "model", None) or "deepseek-v4-flash"
    )
    chat_model = (
        getattr(config, "sdk_chat_model", None)
        or getattr(config, "chat_model", None)
        or "deepseek-v4-pro"
    )
    storage = (
        getattr(config, "sdk_storage_path", None)
        or getattr(config, "storage_path", None)
        or str(md.parent / ".pageindex")
    )
    api_key = (
        (
            getattr(config, "sdk_api_key", None)
            or getattr(config, "api_key", None)
            or os.getenv("PAGEINDEX_API_KEY")
        )
        if mode == "cloud"
        else None
    )
    kwargs: dict[str, Any] = {
        "index": "cloud" if mode == "cloud" else model,
        "chat": chat_model,
    }
    if getattr(config, "sdk_base_url", ""):
        kwargs["index_backend"] = {"base_url": config.sdk_base_url}
        kwargs["chat_backend"] = {"base_url": config.sdk_base_url}
    if getattr(config, "sdk_index_backend", None):
        kwargs["index_backend"] = dict(config.sdk_index_backend)
    if getattr(config, "sdk_chat_backend", None):
        kwargs["chat_backend"] = dict(config.sdk_chat_backend)
    if api_key:
        kwargs["api_key"] = api_key
    if mode == "local":
        kwargs["storage_path"] = storage
    client = PageIndexClient(**kwargs)
    try:
        result = client.submit_document(str(pdf), wait=True)
        tree = client.get_tree(result["doc_id"], node_summary=True, include_text=True)
    finally:
        if temporary_pdf:
            temporary_pdf.unlink(missing_ok=True)
    structure = tree.get("result", tree) if isinstance(tree, dict) else tree
    return {
        "doc_name": md.stem,
        "line_count": len(md.read_text(encoding="utf-8").splitlines()),
        "structure": _adapt_nodes(structure),
    }


def _adapt_nodes(nodes: Any) -> list[dict]:
    if not isinstance(nodes, list):
        return []
    output = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        item = {
            "title": node.get("title", ""),
            "node_id": node.get("node_id", ""),
        }
        for key in ("summary", "prefix_summary", "text"):
            if node.get(key):
                item[key] = node[key]
        children = _adapt_nodes(node.get("nodes", []))
        if children:
            item["nodes"] = children
        output.append(item)
    return output
