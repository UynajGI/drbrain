"""Logical section Documents and exact, bounded physical index fragments."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from drbrain.storage.node_projection import collect_tree_node_records
from drbrain.storage.paths import paper_id_from_dir

try:
    from llama_index.core.schema import Document

    _LLAMA_INDEX_AVAILABLE = True
except ImportError:
    if not TYPE_CHECKING:
        Document = None
    _LLAMA_INDEX_AVAILABLE = False
TREE_LAYER_PAGEINDEX = "pageindex"
DEFAULT_MAX_NODE_TOKENS = 4000
CHARS_PER_TOKEN = 4


def _content_hash(text: str) -> str:
    """Stable content hash for incremental update detection (sha256[:16])."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _node_key(paper_id: str, node_id: str) -> str:
    """Globally-unique node key (tree ``node_id``s are only unique per paper)."""
    return f"{paper_id}:{node_id}"


def _chunk_document(doc: Document, max_node_tokens: int) -> list[Document]:
    """Bound physical index fragments while retaining exact parent-text offsets.

    Offsets are Python character offsets into the logical parent Document
    (title plus body), not raw.md byte offsets or inherited section line ranges.
    """
    max_chars = max(1, int(max_node_tokens)) * CHARS_PER_TOKEN
    if len(doc.text) <= max_chars:
        return [doc]
    spans = []
    start = 0
    while start < len(doc.text):
        end = min(start + max_chars, len(doc.text))
        if end < len(doc.text):
            boundary = doc.text.rfind("\n\n", start, end)
            if boundary > start:
                end = boundary + 2
        spans.append((start, end))
        start = end
    parent_checksum = hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
    out = []
    for i, (start, end) in enumerate(spans):
        md = dict(doc.metadata)
        parent = str(md.get("node_id") or doc.id_)
        md.update(
            {
                "parent_node_id": parent,
                "parent_document_id": doc.id_,
                "parent_checksum": parent_checksum,
                "node_id": f"{parent}#{i}",
                "char_start": start,
                "char_end": end,
                "offset_basis": "parent_document",
                "chunk_index": i,
                "chunk_count": len(spans),
            }
        )
        for key in ("line_start", "line_end"):
            if key in md:
                md["parent_" + key] = md.pop(key)
        out.append(Document(text=doc.text[start:end], id_=f"{doc.id_}#{i}", metadata=md))
    return out


def collect_tree_nodes(
    paper_dir: str | Path,
    tree_json: str | Path | dict | None = None,
    max_node_tokens: int | None = None,
    *,
    paper_id: str | None = None,
) -> list[Document]:
    """Collect one :class:`Document` per PageIndex tree node.

    ``tree_json`` may be a path, a raw parsed dict, or ``None`` (defaults to
    ``<paper_dir>/tree.json``). Each node becomes a Document with
    ``text = "<title>\\n<body>"`` where the body is loaded from ``raw.md`` by
    line range, and metadata::

        {paper_id, node_id, title, line_start, line_end, tree_layer: "pageindex"}

    When ``max_node_tokens`` is given, oversized nodes become physical
    fragments with unique ids, exact parent-text character offsets, and
    ``parent_line_start``/``parent_line_end`` section locators. Their text is
    an exact slice of the title-inclusive parent text; fragments are not
    re-prefixed with a separate title. The character cap is a token estimate,
    not a tokenizer guarantee. Without the parameter the
    node-to-Document mapping remains 1:1.

    Body resolution order (mirrors ``services.embedding._collect_tree_nodes``
    semantics, but also handles the actual tree.json format which carries
    ``line_num`` + inline ``text``):

    1. explicit ``line_start``/``line_end`` → ``raw.md[line_start:line_end]``
    2. ``line_num`` (1-based header line) → flat range up to the next node's
       header line in ``raw.md`` (same computation the PageIndex builder used)
    3. inline node ``text`` → used verbatim (when ``raw.md`` is missing)

    ``raw.md`` is only read when at least one node needs line-based extraction
    (body loaded on demand). Documents whose text is empty are dropped.
    """
    if not _LLAMA_INDEX_AVAILABLE:  # pragma: no cover - envs without llama-index
        raise RuntimeError("llama-index is not installed; cannot collect Documents")

    paper_dir = Path(paper_dir)
    resolved_paper_id = paper_id or paper_id_from_dir(paper_dir)
    records = collect_tree_node_records(paper_dir, tree_json, paper_id=resolved_paper_id)
    docs: list[Document] = [
        Document(
            text=row["text"],
            id_=_node_key(resolved_paper_id, row["node_id"]),
            metadata={
                "paper_id": resolved_paper_id,
                "node_id": row["node_id"],
                "title": row["title"],
                "line_start": row["line_start"],
                "line_end": row["line_end"],
                "tree_layer": TREE_LAYER_PAGEINDEX,
            },
        )
        for row in records
    ]
    if max_node_tokens and max_node_tokens > 0:
        out: list[Document] = []
        for doc in docs:
            out.extend(_chunk_document(doc, int(max_node_tokens)))
        return out
    return docs
