"""Native PageIndex local filesystem and chat integration.

PageIndex's local SDK stores one document as ``docs/<doc_id>/{doc,tree,pages}.json``
and runs its document-QA agent through ``chat_completions``.  This adapter keeps
that contract intact while attaching DrBrain paper ids to the filesystem so a
RAG caller can address the native document unambiguously.

The adapter is deliberately separate from the four-leg SQL fusion path:
PageIndex native chat returns an answer, not ranked candidates.  It is used for
document-scoped native QA after a corpus retriever has selected a paper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class PageIndexNativeError(RuntimeError):
    """Raised when the optional PageIndex SDK/local filesystem is unavailable."""


def _value(cfg: Any, key: str, default: Any = None) -> Any:
    pageindex = getattr(cfg, "pageindex", None)
    if pageindex is not None:
        value = getattr(pageindex, key, None)
        if value not in (None, ""):
            return value
    if isinstance(cfg, dict):
        value = (cfg.get("pageindex") or {}).get(key)
        if value not in (None, ""):
            return value
    return default


def _runtime_path(cfg: Any, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    try:
        from drbrain.runtime import runtime_root

        return runtime_root() / path
    except Exception:  # pragma: no cover - runtime is available in CLI
        return Path.cwd() / path


def _client(cfg: Any, storage_path: Path):
    try:
        from pageindex import PageIndexClient
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise PageIndexNativeError("PageIndex SDK is not installed") from exc

    index_model = str(_value(cfg, "model", "spark-x25-4b"))
    chat_model = str(_value(cfg, "chat_model", "deepseek-flash"))
    index_backend = dict(_value(cfg, "index_backend", {}) or {})
    chat_backend = dict(_value(cfg, "chat_backend", {}) or {})
    base_url = str(_value(cfg, "base_url", "") or "")
    chat_base_url = str(_value(cfg, "chat_base_url", "") or "")
    if base_url:
        index_backend.setdefault("base_url", base_url)
        index_backend.setdefault("api_key", "local")
    if chat_base_url:
        chat_backend.setdefault("base_url", chat_base_url)
        chat_backend.setdefault("api_key", str(_value(cfg, "chat_api_key", "") or ""))

    index = {"model": index_model, "storage_path": str(storage_path)}
    if index_backend:
        index["backend"] = index_backend
    chat = {"model": chat_model}
    if chat_backend:
        chat["backend"] = chat_backend
    try:
        # Slot arguments are intentional: recent SDK versions reject mixing
        # ``index=``/``chat=`` with their flat aliases.
        return PageIndexClient(mode="local", index=index, chat=chat)
    except Exception as exc:
        raise PageIndexNativeError(f"PageIndex local client unavailable: {exc}") from exc


def _find_doc_id(storage_path: Path, paper_id: str) -> str | None:
    """Find a paper-tagged or source-named document in a native filesystem."""
    try:
        from pageindex.local_store import DocStore

        metas = DocStore(str(storage_path)).list_metas()
    except Exception:
        return None
    for meta in metas:
        tags = meta.get("metadata") or {}
        if str(tags.get("drbrain_paper_id") or "") == paper_id:
            return str(meta.get("id"))
    source_name = "source.pdf"
    for meta in metas:
        if str(meta.get("name") or "") == source_name:
            return str(meta.get("id"))
    return None


def ensure_document(cfg: Any, paper_id: str, paper_dir: str | Path) -> str:
    """Ensure a paper exists in the PageIndex local filesystem and return doc id.

    Existing SDK-built documents are reused.  If absent, the native SDK indexes
    ``source.pdf`` with the configured PageIndex processing mode.  Non-PDF
    materials are intentionally skipped because the SDK's local contract only
    accepts PDFs; their DrBrain tree remains the compatible evidence route.
    """
    paper_dir = Path(paper_dir)
    source = paper_dir / "source.pdf"
    if not source.is_file():
        raise PageIndexNativeError(f"native PageIndex requires source.pdf: {paper_dir}")
    storage = paper_dir / ".pageindex"
    doc_id = _find_doc_id(storage, paper_id)
    if doc_id:
        return doc_id
    client = _client(cfg, storage)
    mode = str(_value(cfg, "processing_mode", "flash") or "flash")
    try:
        result = client.submit_document(
            str(source),
            mode=mode,
            metadata={"drbrain_paper_id": paper_id},
            wait=True,
        )
    except Exception as exc:
        raise PageIndexNativeError(f"native PageIndex indexing failed: {exc}") from exc
    doc_id = str(result.get("doc_id") or "")
    if not doc_id:
        raise PageIndexNativeError("native PageIndex returned no doc_id")
    return doc_id


def chat_document(
    cfg: Any,
    paper_id: str,
    paper_dir: str | Path,
    question: str,
    *,
    max_turns: int | None = None,
) -> dict[str, Any]:
    """Run PageIndex's native local chat agent over one DrBrain paper."""
    if not str(question).strip():
        raise ValueError("question must be non-empty")
    paper_dir = Path(paper_dir)
    storage = paper_dir / ".pageindex"
    doc_id = ensure_document(cfg, paper_id, paper_dir)
    client = _client(cfg, storage)
    backend = dict(_value(cfg, "chat_backend", {}) or {}) or None
    chat_url = str(_value(cfg, "chat_base_url", "") or "")
    if backend is None and chat_url:
        backend = {
            "base_url": chat_url,
            "api_key": str(_value(cfg, "chat_api_key", "") or ""),
        }
    kwargs: dict[str, Any] = {
        "stream": False,
        "doc_id": doc_id,
        "model": str(_value(cfg, "chat_model", "deepseek-flash")),
        "backend": backend,
    }
    if max_turns is not None:
        kwargs["max_turns"] = int(max_turns)
    try:
        response = client.chat_completions(question, **kwargs)
    except Exception as exc:
        raise PageIndexNativeError(f"native PageIndex chat failed: {exc}") from exc
    answer = response
    if isinstance(response, dict):
        choices = response.get("choices") or []
        message = choices[0].get("message") if choices else None
        answer = (message or {}).get("content") if isinstance(message, dict) else response
    return {
        "answer": str(answer or ""),
        "paper_id": paper_id,
        "pageindex_doc_id": doc_id,
        "engine": "pageindex_native_chat",
        "native": True,
    }


def filesystem_status(cfg: Any, paper_dir: str | Path) -> dict[str, Any]:
    """Return a small, JSON-safe status record for one native filesystem."""
    storage = Path(paper_dir) / ".pageindex"
    manifest = storage / "manifest.json"
    docs = 0
    if manifest.is_file():
        try:
            docs = len((json.loads(manifest.read_text(encoding="utf-8")) or {}).get("docs", {}))
        except Exception:
            docs = 0
    return {"storage_path": str(storage), "documents": docs, "ready": docs > 0}


__all__ = [
    "PageIndexNativeError",
    "chat_document",
    "ensure_document",
    "filesystem_status",
]
