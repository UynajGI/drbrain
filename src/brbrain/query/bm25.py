"""BM25 full-text search over paper titles and concept labels."""
from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric."""
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


class BM25Search:
    """BM25 search index over papers and concepts."""

    def __init__(self):
        self._documents: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None
        self._tokenized: list[list[str]] = []

    def add_document(self, local_id: str, doc_type: str, label: str, text: str = "") -> None:
        """Add a searchable document (paper title or concept label)."""
        search_text = f"{label} {text}".strip()
        self._documents.append({
            "local_id": local_id, "type": doc_type,
            "label": label, "text": text,
        })

    def build(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Build the BM25 index from added documents."""
        self._tokenized = [tokenize(d["label"] + " " + d.get("text", "")) for d in self._documents]
        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized, k1=k1, b=b)

    def search(self, query: str, type_filter: str | None = None, limit: int = 20) -> list[dict]:
        """Search the index. Returns ranked list of matching documents."""
        if not self._bm25 or not self._tokenized:
            return []

        query_tokens = tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        results = []
        for i, score in enumerate(scores):
            doc = self._documents[i]
            if type_filter and doc["type"] != type_filter:
                continue
            if score > 0:
                results.append({
                    **doc, "score": round(float(score), 4),
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]


def build_bm25_index(db, k1: float = 1.5, b: float = 0.75) -> BM25Search:
    """Build a BM25 index from database papers and concepts."""
    index = BM25Search()

    # Add paper titles
    papers = db.get_all_papers()
    for p in papers:
        index.add_document(p["local_id"], "Paper", p["title"], p.get("status", ""))

    # Add concept labels
    rows = db.conn.execute(
        "SELECT local_id, type, label FROM concepts"
    ).fetchall()
    for local_id, ctype, label in rows:
        index.add_document(local_id, ctype, label)

    index.build(k1=k1, b=b)
    return index
