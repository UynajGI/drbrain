"""BM25 full-text search over paper titles, concept labels, and argument claims."""

from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric."""
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


class BM25Search:
    """BM25 search index over papers, concepts, and arguments."""

    def __init__(self):
        """Initialize an empty BM25 index."""
        self._documents: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None
        self._tokenized: list[list[str]] = []

    def add_document(
        self,
        local_id: str,
        doc_type: str,
        label: str,
        text: str = "",
        arg_type: str = "",
        year: int | None = None,
        confidence: float | None = None,
    ) -> None:
        """Add a searchable document (paper title, concept label, or argument claim)."""
        doc: dict[str, Any] = {
            "local_id": local_id,
            "type": doc_type,
            "label": label,
            "text": text,
        }
        if arg_type:
            doc["arg_type"] = arg_type
        if year is not None:
            doc["year"] = year
        if confidence is not None:
            doc["confidence"] = confidence
        self._documents.append(doc)

    def build(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Build the BM25 index from added documents."""
        self._tokenized = [tokenize(d["label"] + " " + d.get("text", "")) for d in self._documents]
        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized, k1=k1, b=b)

    def search(
        self,
        query: str,
        type_filter: str | None = None,
        arg_type_filter: str | None = None,
        limit: int = 20,
        min_confidence: float | None = None,
    ) -> list[dict]:
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
            if arg_type_filter and doc.get("arg_type") != arg_type_filter:
                continue
            if min_confidence is not None:
                conf = doc.get("confidence")
                if conf is None or conf < min_confidence:
                    continue
            results.append(
                {
                    **doc,
                    "score": round(float(score), 4),
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]


def build_bm25_index(db, k1: float = 1.5, b: float = 0.75) -> BM25Search:
    """Build a BM25 index from database papers, concepts, and arguments."""
    index = BM25Search()

    # Add paper titles + abstracts
    papers = db.get_all_papers()
    for p in papers:
        index.add_document(
            p["local_id"],
            "Paper",
            p["title"],
            text=p.get("abstract", "") + " " + p.get("status", ""),
            year=p.get("year"),
        )

    # Add concept labels
    rows = db.conn.execute(
        "SELECT c.local_id, c.type, c.label, c.confidence, p.year "
        "FROM concepts c JOIN papers p ON c.local_id = p.local_id "
        "WHERE p.year IS NOT NULL"
    ).fetchall()
    for local_id, ctype, label, confidence, year in rows:
        index.add_document(local_id, ctype, label, year=year, confidence=confidence)

    # Add argument claims
    args = db.conn.execute(
        "SELECT a.source_paper, a.claim, a.claim_type, a.confidence, p.year "
        "FROM arguments a JOIN papers p ON a.source_paper = p.local_id "
        "WHERE p.year IS NOT NULL"
    ).fetchall()
    for local_id, claim, claim_type, confidence, year in args:
        index.add_document(
            local_id,
            "Argument",
            claim,
            arg_type=claim_type,
            year=year,
            confidence=confidence,
        )

    index.build(k1=k1, b=b)
    return index
