"""Embedding execution and cache reuse for index construction."""

from __future__ import annotations

import gc
import os
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from drbrain.config import Config

try:
    from llama_index.core import load_index_from_storage
    from llama_index.core.schema import TextNode
    from llama_index.core.storage import StorageContext
    from llama_index.core.vector_stores import SimpleVectorStore

    _LLAMA_INDEX_AVAILABLE = True
except ImportError:
    if not TYPE_CHECKING:
        load_index_from_storage = TextNode = StorageContext = SimpleVectorStore = None
    _LLAMA_INDEX_AVAILABLE = False


def preserve_gc_state(operation):
    """Restore the caller's GC setting even when an index build fails early."""

    @wraps(operation)
    def run(*args, **kwargs):
        enabled = gc.isenabled()
        try:
            return operation(*args, **kwargs)
        finally:
            if enabled:
                gc.enable()
            else:
                gc.disable()

    return run


def _load_old_embeddings(vector_dir: Path) -> dict[str, list[float]]:
    """Load node_id → embedding from a previously persisted SimpleVectorStore."""
    if not _LLAMA_INDEX_AVAILABLE or not (vector_dir / "default__vector_store.json").exists():
        return {}
    try:
        store = SimpleVectorStore.from_persist_dir(str(vector_dir))
        return dict(store._data.embedding_dict)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[rag] could not read previous vector store: %s", exc)
        return {}


def _load_old_index_nodes(vector_dir: Path, embed_model: Any) -> dict[str, TextNode]:
    """Load all previously indexed nodes (with embeddings) from a persisted index.

    Used to carry non-target papers through a ``--paper`` subset rebuild so the
    persisted index keeps covering the whole library.
    """
    if not _LLAMA_INDEX_AVAILABLE or not (vector_dir / "docstore.json").exists():
        return {}
    try:
        sc = StorageContext.from_defaults(persist_dir=str(vector_dir))
        idx = load_index_from_storage(sc, embed_model=embed_model)
        out: dict[str, TextNode] = {}
        for node in idx.docstore.docs.values():
            if not isinstance(node, TextNode):
                continue
            # ``.get(node_id)`` is the duck-typed vector-store lookup the
            # concrete store (e.g. SimpleVectorStore) implements; the base
            # ``BasePydanticVectorStore`` type does not declare it.
            embedding = node.embedding or idx._vector_store.get(  # type: ignore[attr-defined]
                node.node_id
            )
            if embedding is None:
                continue
            node.embedding = embedding
            out[node.node_id] = node
        return out
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[rag] could not load previous index nodes: %s", exc)
        return {}


def _default_embed_model(cfg: Config) -> Any:
    """T1 DrbrainEmbedding adapter (lazy; loads the model on first embed call)."""
    from drbrain.rag.llm import DrbrainEmbedding

    return DrbrainEmbedding(cfg)


def _embed_devices(cfg: Config) -> list[int]:
    """GPU ids for parallel chunk embedding: configured cuda:N plus extra_gpus."""
    devices: list[int] = []
    dev = str(getattr(cfg.embed, "device", "") or "")
    if dev.startswith("cuda:"):
        try:
            devices.append(int(dev.split(":", 1)[1]))
        except ValueError:
            pass
    for g in getattr(cfg.embed, "extra_gpus", None) or []:
        if int(g) not in devices:
            devices.append(int(g))
    return devices


def _embed_model_path(cfg: Config) -> str:
    from drbrain.services.embedding import _resolve_model_path

    path = _resolve_model_path(
        cfg.embed.model, os.path.expanduser(cfg.embed.cache_dir), cfg.embed.source
    )
    if not path:
        raise RuntimeError("cannot resolve local model path for parallel embedding")
    return str(path)


_WORKER_MODEL: Any = None


def _embed_chunk_worker(args: tuple[Any, int, list[str], int, int]) -> Any:
    """Spawn-pool worker: pin one GPU, embed a chunk, return vectors."""
    model_path, gpu, texts, ci, total = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    global _WORKER_MODEL
    if _WORKER_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _WORKER_MODEL = SentenceTransformer(model_path, device="cuda")
        # Align with scripts/serve_embedding.py: truncate to 512 tokens so
        # these vectors match the corpus-pipeline embedding semantics.
        try:
            _WORKER_MODEL.max_seq_length = 512
        except Exception:  # noqa: BLE001 - non-fatal optimization
            pass
    logger.info("[rag] embedding chunk {}/{} ({} nodes) on GPU {}", ci, total, len(texts), gpu)
    return _WORKER_MODEL.encode(texts, normalize_embeddings=True, batch_size=64)
