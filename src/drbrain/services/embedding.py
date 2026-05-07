"""Tree node embedding service (ScholarAIO pattern).

Lightweight vectors for semantically-complete tree nodes.
Provider=none disables all vectors; falls back to BM25 + LLM navigation.

Usage:
    from drbrain.services.embedding import build_tree_vectors, search_tree
    build_tree_vectors(db_path, paper_dir, cfg)
    results = search_tree("turbulent drag reduction", db_path, top_k=5)
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from drbrain.config import EmbedConfig


# ── Helpers ──────────────────────────────────────────────────────────────────


def _embed_provider(cfg: EmbedConfig | None = None) -> str:
    """Return normalized embedding backend provider."""
    if cfg is None:
        return "local"
    provider = (cfg.provider or "local").strip().lower()
    return provider or "local"


def _content_hash(text: str) -> str:
    """Stable content hash for incremental update detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _embed_signature(cfg: EmbedConfig | None = None) -> str:
    """Build a stable signature for current embedding backend settings."""
    if cfg is None:
        return "local:Qwen/Qwen3-Embedding-0.6B:auto"
    return f"{cfg.provider}:{cfg.model}:{cfg.device}"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    import numpy as np

    va = np.asarray(a, dtype="float32")
    vb = np.asarray(b, dtype="float32")
    dot = float(np.dot(va, vb))
    return dot  # assumes normalized input


# ── Embedding dispatch ──────────────────────────────────────────────────────


def _embed_batch(texts: list[str], cfg: EmbedConfig | None = None) -> list[list[float]]:
    """Embed a batch of texts. Returns list of float vectors.

    Stub: real implementation loads sentence-transformers on demand.
    For now, returns zero vectors of dim=8 as placeholder.
    """
    provider = _embed_provider(cfg)
    if provider == "none":
        return []
    if provider == "openai-compat":
        raise NotImplementedError("openai-compat embedding not yet implemented")
    # Local: load model on demand
    return _embed_batch_local(texts, cfg)


def _embed_batch_local(texts: list[str], cfg: EmbedConfig | None = None) -> list[list[float]]:
    """Embed texts using local sentence-transformers model."""
    import importlib
    import os

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if cfg is not None:
        model_name = cfg.model
        device_cfg = cfg.device
    else:
        model_name = "Qwen/Qwen3-Embedding-0.6B"
        device_cfg = "auto"

    sentence_transformers_mod = importlib.import_module("sentence_transformers")
    SentenceTransformer = sentence_transformers_mod.SentenceTransformer  # noqa: N806

    if device_cfg == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = device_cfg

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(texts, normalize_embeddings=True)
    return [e.tolist() for e in embeddings]


# ── Build ────────────────────────────────────────────────────────────────────


def _collect_tree_nodes(paper_dir: Path) -> list[dict]:
    """Collect all tree nodes from a paper's tree.json, with their text content."""
    tree_path = paper_dir / "tree.json"
    if not tree_path.exists():
        return []

    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    structure = tree.get("structure", [])

    raw_md_path = paper_dir / "raw.md"
    if raw_md_path.exists():
        raw_text = raw_md_path.read_text(encoding="utf-8")
    else:
        raw_text = ""

    def _walk(nodes: list[dict], path_prefix: str = "") -> list[dict]:
        result = []
        for node in nodes:
            nid = node.get("node_id", "")
            title = node.get("title", "")
            text = f"{title}\n"

            # Try to extract content from raw.md using line ranges
            line_start = node.get("line_start")
            line_end = node.get("line_end")
            if line_start is not None and line_end is not None and raw_text:
                text += "\n".join(raw_text.split("\n")[line_start:line_end])
            elif node.get("content"):
                text += str(node.get("content", ""))

            result.append(
                {
                    "node_id": nid,
                    "title": title,
                    "text": text.strip(),
                }
            )

            child_nodes = node.get("nodes", [])
            if child_nodes:
                result.extend(_walk(child_nodes, f"{path_prefix}{nid}/"))
        return result

    return _walk(structure)


def build_tree_vectors(
    db_path: Path,
    paper_dir: Path,
    cfg: EmbedConfig | None = None,
) -> int:
    """Embed all tree nodes for a paper and store in tree_vectors.

    Args:
        db_path: SQLite database path.
        paper_dir: Paper directory containing tree.json and raw.md.
        cfg: Optional EmbedConfig.

    Returns:
        Number of vectors written.
    """
    import sqlite3

    provider = _embed_provider(cfg)
    if provider == "none":
        logger.info("embed.provider=none; tree vector generation is disabled")
        return 0

    nodes = _collect_tree_nodes(paper_dir)
    if not nodes:
        return 0

    # Check existing hashes for incremental update
    conn = sqlite3.connect(str(db_path))
    try:
        existing_hashes: dict[str, str] = {}
        for row in conn.execute("SELECT node_id, content_hash FROM tree_vectors").fetchall():
            existing_hashes[row[0]] = row[1]

        paper_id = paper_dir.name

        # Collect nodes that need embedding
        to_embed_texts: list[str] = []
        to_embed_nodes: list[dict] = []
        for node in nodes:
            nhash = _content_hash(node["text"])
            if existing_hashes.get(node["node_id"]) == nhash:
                continue  # unchanged, skip
            to_embed_texts.append(node["text"])
            to_embed_nodes.append(node)
            to_embed_nodes[-1]["_hash"] = nhash

        if not to_embed_texts:
            return 0

        # Embed in batch
        vectors = _embed_batch(to_embed_texts, cfg)
        if not vectors:
            return 0

        # Store
        for node, vec in zip(to_embed_nodes, vectors):
            blob = struct.pack(f"{len(vec)}f", *vec)
            conn.execute(
                "INSERT OR REPLACE INTO tree_vectors "
                "(node_id, paper_id, embedding, content_hash, tree_layer) "
                "VALUES (?, ?, ?, ?, ?)",
                (node["node_id"], paper_id, blob, node["_hash"], "pageindex"),
            )

        conn.commit()
        return len(to_embed_nodes)

    finally:
        conn.close()


async def build_paper_tree_vectors(
    paper_dir: Path,
    db_path: Path,
    embed_cfg: EmbedConfig | None = None,
    llm_models: list[dict] | None = None,
) -> int:
    """Build PageIndex tree vectors + RAPTOR recursive summaries for a single paper.

    Combines both layers:
    1. build_tree_vectors — embed PageIndex leaf nodes
    2. build_raptor_tree — recursive GMM clustering + LLM summarization

    Args:
        paper_dir: Paper directory with tree.json and raw.md.
        db_path: SQLite database path.
        embed_cfg: Embedding configuration.
        llm_models: LLM model list for RAPTOR summarization.

    Returns:
        Total number of vectors + summaries created.
    """
    from drbrain.extractor.raptor import build_raptor_tree

    pageindex_count = build_tree_vectors(db_path, paper_dir, embed_cfg)

    raptor_count = 0
    if llm_models:
        try:
            raptor_count = await build_raptor_tree(paper_dir, db_path, embed_cfg, llm_models)
        except Exception:
            logger.warning(
                "RAPTOR tree build failed for %s, PageIndex vectors still created",
                paper_dir.name,
            )

    return pageindex_count + raptor_count


# ── Search ───────────────────────────────────────────────────────────────────


def search_tree(
    query: str,
    db_path: Path,
    top_k: int = 10,
    cfg: EmbedConfig | None = None,
) -> list[dict]:
    """Vector search over tree_vectors using cosine similarity.

    Args:
        query: Natural language query text.
        db_path: SQLite database path.
        top_k: Number of results to return.
        cfg: Optional EmbedConfig.

    Returns:
        List of {node_id, paper_id, score, tree_layer}.
    """
    import sqlite3

    import numpy as np

    provider = _embed_provider(cfg)
    if provider == "none":
        return []

    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    try:
        # Check if tree_vectors has data
        count = conn.execute("SELECT COUNT(*) FROM tree_vectors").fetchone()[0]
        if count == 0:
            return []

        # Embed query
        query_vec = _embed_batch([query], cfg)[0]
        qv = np.asarray(query_vec, dtype="float32")
        query_dim = len(query_vec)

        # Cosine similarity over all stored vectors
        results = []
        for row in conn.execute(
            "SELECT node_id, paper_id, embedding, tree_layer FROM tree_vectors"
        ):
            blob = row[2]
            stored_dim = len(blob) // 4  # float32 = 4 bytes
            if stored_dim != query_dim:
                logger.warning(
                    "Dimension mismatch in tree_vectors node_id={}: stored={} query={}",
                    row[0],
                    stored_dim,
                    query_dim,
                )
                continue
            stored_vec = np.asarray(struct.unpack(f"{stored_dim}f", blob), dtype="float32")
            sim = float(np.dot(qv, stored_vec))
            results.append(
                {
                    "node_id": row[0],
                    "paper_id": row[1],
                    "score": sim,
                    "tree_layer": row[3],
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    finally:
        conn.close()
