"""Optional adapter for the official PageIndex Python SDK.

The adapter deliberately returns DrBrain's ``DocumentTree`` shape so the rest
of the parser and retrieval stack remains independent of SDK response details.
"""

from __future__ import annotations

import os
import tempfile
import textwrap
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
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
    if get("chat_base_url", ""):
        tree_config.sdk_chat_backend = {
            **tree_config.sdk_chat_backend,
            "base_url": get("chat_base_url"),
            "api_key": get("chat_api_key", "") or os.getenv("DEEPSEEK_API_KEY", ""),
        }
    tree_config.sdk_processing_mode = get("processing_mode", "standard")
    tree_config.sdk_timeout = float(get("sdk_timeout", get("timeout", 600.0)))
    tree_config.sdk_allow_fallback = bool(get("allow_fallback", False))
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
    needs_render = not pdf.is_file()
    if pdf.is_file():
        try:
            import fitz

            with fitz.open(str(pdf)) as doc:
                needs_render = not any(page.get_text("text").strip() for page in doc)
        except Exception:
            needs_render = True
    if needs_render:
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
        "index": "cloud" if mode == "cloud" else {"model": model},
        "chat": {"model": chat_model},
    }
    if getattr(config, "sdk_base_url", ""):
        local_backend = {"base_url": config.sdk_base_url, "api_key": "local"}
        kwargs["index"]["backend"] = dict(local_backend)
        kwargs["chat"]["backend"] = dict(local_backend)
    if getattr(config, "sdk_index_backend", None):
        kwargs["index"]["backend"] = dict(config.sdk_index_backend)
    if getattr(config, "sdk_chat_backend", None):
        kwargs["chat"]["backend"] = dict(config.sdk_chat_backend)
    if api_key:
        kwargs["api_key"] = api_key
    if mode == "local":
        # Recent PageIndex SDK rejects ``index=...`` together with the flat
        # ``storage_path`` spelling; pass storage through the index backend
        # namespace instead so local runs remain compatible across versions.
        index_slot = dict(kwargs["index"])
        index_slot["storage_path"] = storage
        kwargs["index"] = index_slot
    client = PageIndexClient(mode="local", **kwargs)
    try:
        # The SDK defaults local indexing to the fast Flash path.  DrBrain's
        # tree contract needs the full model-built hierarchy, so request the
        # standard mode explicitly for local runs.
        submit_mode = (
            getattr(config, "sdk_processing_mode", None) or "flash"
            if mode == "local"
            else None
        )
        try:
            timeout = float(getattr(config, "sdk_timeout", 600.0) or 600.0)
            pool = ThreadPoolExecutor(max_workers=1)
            future = pool.submit(client.submit_document, str(pdf), mode=submit_mode, wait=True)
            try:
                result = future.result(timeout=timeout)
                tree = client.get_tree(result["doc_id"], node_summary=True, include_text=True)
            finally:
                # Do not let executor.__exit__ wait for a stuck SDK request;
                # the caller must receive the configured timeout promptly.
                pool.shutdown(wait=False, cancel_futures=True)
        except FutureTimeout as exc:
            if not getattr(config, "sdk_allow_fallback", False):
                raise TimeoutError(f"PageIndex SDK timed out after {timeout:.0f}s") from exc
            # A timed-out optional index must degrade to the extracted
            # markdown tree; do not propagate into the ingest partial state.
            text = md.read_text(encoding="utf-8")
            sections = []
            current = None
            for line_no, line in enumerate(text.splitlines(), 1):
                if line.startswith("#"):
                    current = {"title": line.lstrip("#").strip(), "node_id": f"fallback-{len(sections)+1}", "line_num": line_no, "text": ""}
                    sections.append(current)
                elif current is not None:
                    current["text"] += line + "\n"
            if not sections:
                sections = [{"title": md.stem, "node_id": "fallback-1", "line_num": 1, "text": text}]
            return {"doc_name": md.stem, "line_count": len(text.splitlines()), "structure": sections,
                    "warnings": [f"pageindex_sdk_timeout: {timeout:.0f}s"]}
        except Exception as exc:
            if not getattr(config, "sdk_allow_fallback", False):
                raise
            # Keep CLI ingestion usable when the optional SDK cannot finish a
            # local tree (for example Spark's slow/invalid TOC repair). The
            # extracted markdown remains authoritative; construct a small
            # deterministic section tree and expose the degradation clearly.
            text = md.read_text(encoding="utf-8")
            fallback = []
            current = None
            for line_no, line in enumerate(text.splitlines(), 1):
                if line.startswith("#"):
                    title = line.lstrip("#").strip()
                    current = {"title": title, "node_id": f"fallback-{len(fallback)+1}", "line_num": line_no, "text": ""}
                    fallback.append(current)
                elif current is not None:
                    current["text"] += line + "\n"
            if not fallback:
                fallback = [{"title": md.stem, "node_id": "fallback-1", "line_num": 1, "text": text}]
            return {
                "doc_name": md.stem,
                "line_count": len(text.splitlines()),
                "structure": fallback,
                "warnings": [f"pageindex_sdk_fallback: {type(exc).__name__}: {exc}"],
            }
    finally:
        if temporary_pdf:
            temporary_pdf.unlink(missing_ok=True)
    structure = tree.get("result", tree) if isinstance(tree, dict) else tree
    return {
        "doc_name": md.stem,
        "line_count": len(md.read_text(encoding="utf-8").splitlines()),
        "structure": _adapt_nodes(structure),
    }


def chat_with_sdk(pdf_path: str | Path, question: str, config: Any) -> str:
    """Ask PageIndex's native chat lane about one local PDF.

    This is intentionally separate from corpus SQL fusion: callers get the
    SDK's document-scoped answer while multi-document queries retain DrBrain
    evidence/provenance semantics.
    """
    from pageindex import PageIndexClient

    model = getattr(config, "sdk_chat_model", None) or getattr(config, "chat_model", None)
    backend = None
    chat_url = getattr(config, "sdk_chat_backend", {}) or {}
    if chat_url:
        backend = dict(chat_url)
    elif getattr(config, "sdk_chat_base_url", ""):
        backend = {"base_url": config.sdk_chat_base_url, "api_key": os.getenv("DEEPSEEK_API_KEY", "")}
    kwargs = {"index": {"model": model}, "chat": {"model": model}}
    if backend:
        kwargs["index"]["backend"] = backend
        kwargs["chat"]["backend"] = backend
    client = PageIndexClient(**kwargs)
    timeout = float(getattr(config, "sdk_timeout", 120.0) or 120.0)
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        result = pool.submit(client.submit_document, str(pdf_path), mode="flash", wait=True).result(timeout=timeout)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    answer = client.chat(question, doc_id=result["doc_id"], stream=False, backend=backend)
    return answer if isinstance(answer, str) else str(answer)


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
            "line_num": node.get("line_num", len(output) + 1),
        }
        for key in ("summary", "prefix_summary", "text"):
            if node.get(key):
                item[key] = node[key]
        children = _adapt_nodes(node.get("nodes", []))
        if children:
            item["nodes"] = children
        output.append(item)
    return output
