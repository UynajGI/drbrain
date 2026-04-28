"""Concept label normalization and alias table for entity alignment."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

log = logging.getLogger(__name__)

# Load stopwords from file
_STOPWORDS_FILE = Path(__file__).parent.parent / "stopwords.txt"


def _load_stopwords() -> set[str]:
    """Load stopwords from stopwords.txt, filtering to usable tokens."""
    if not _STOPWORDS_FILE.exists():
        # Fallback to minimal set if file missing
        return {"the", "a", "an", "of", "for", "in", "on", "with", "to"}

    stopwords: set[str] = set()
    with open(_STOPWORDS_FILE, encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if not token:
                continue
            # Skip pure symbols, punctuation, single chars (except common Chinese)
            # Keep: alphabetic words, Chinese characters, hyphenated words
            if re.match(r"^[a-zA-Z一-鿿]+$", token):
                stopwords.add(token.lower())
            elif re.match(r"^[a-zA-Z一-鿿]['-][a-zA-Z一-鿿]+$", token):
                stopwords.add(token.lower())
    return stopwords


_STOPWORDS = _load_stopwords()

# BM25 alignment thresholds
BM25_AUTO_ALIGN = 0.8  # score above this → automatic alignment
BM25_PENDING_MIN = 0.3  # score above this → queue for LLM arbitration


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25: lowercase, split on non-alphanumeric."""
    text = text.lower()
    return re.findall(r"[a-z0-9一-鿿]+", text)


def normalize_label(label: str) -> str:
    """Normalize a concept label for canonical matching.

    If all words are filtered out by stopwords, returns the original label
    lowercased to prevent empty strings from breaking alignment.
    """
    t = label.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    words = t.split()
    filtered = [w for w in words if w not in _STOPWORDS]
    # Guard: if stopwords removed everything, keep original
    if not filtered:
        return t
    t = " ".join(filtered)
    # Simple singularization
    if t.endswith("s") and len(t) > 3:
        t = t[:-1]
    t = re.sub(r"\s+", " ", t).strip()
    return t


class AliasTable:
    """Maps label variants to canonical concept IDs."""

    def __init__(self):
        self._canonical: dict[str, str] = {}
        self._aliases: dict[str, str] = {}
        self._counter = 0

    def add_canonical(self, label: str, canonical_id: str) -> str:
        """Register a canonical label. Returns canonical_id."""
        norm = normalize_label(label)
        self._canonical[norm] = canonical_id
        return canonical_id

    def add_alias(self, variant: str, canonical_id: str) -> None:
        """Register an alias pointing to an existing canonical_id."""
        self._aliases[variant.lower().strip()] = canonical_id

    def lookup(self, label: str) -> str | None:
        """Look up canonical_id by label. Returns None if not found."""
        key = label.lower().strip()
        if key in self._aliases:
            return self._aliases[key]
        norm = normalize_label(label)
        return self._canonical.get(norm)

    def get_or_create(self, label: str) -> str:
        """Look up or create a new canonical_id."""
        existing = self.lookup(label)
        if existing is not None:
            return existing
        self._counter += 1
        new_id = f"concept_{self._counter}"
        self._canonical[normalize_label(label)] = new_id
        return new_id


class SmartAligner:
    """BM25 + LLM hybrid concept alignment across papers.

    Pipeline:
    1. normalize_label + exact match → ~30%
    2. BM25 fuzzy search → auto-align if score > 0.8 → ~40%
    3. BM25 score 0.3-0.8 → queue for LLM batch arbitration → ~20%
    4. LLM uncertain → confidence queue for manual review → <5%
    """

    def __init__(self, db, models: list[dict] | None = None, alias_table: AliasTable | None = None):
        self._db = db
        self._alias_table = alias_table or AliasTable()
        self._models = models
        self._pending: list[dict] = []  # labels awaiting LLM arbitration
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[dict] = []  # {label, type, canonical_id}
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Build BM25 index from existing concepts in DB."""
        rows = self._db.conn.execute("SELECT DISTINCT label, type FROM concepts").fetchall()
        if not rows:
            return

        self._bm25_docs = [
            {"label": label, "type": ctype, "canonical_id": f"concept_{label}"}
            for label, ctype in rows
        ]
        tokenized = [_tokenize(d["label"]) for d in self._bm25_docs]
        if tokenized:
            self._bm25 = BM25Okapi(tokenized)

    def align(self, label: str, ctype: str) -> str:
        """Return canonical_id for a new concept label.

        Step 1: Exact match via normalize_label
        Step 2: BM25 fuzzy search
        Step 3: Queue for LLM batch arbitration if ambiguous
        Step 4: Create new canonical_id if no match
        """
        # Step 1: Exact match
        existing = self._alias_table.lookup(label)
        if existing:
            return existing

        # Step 2: BM25 fuzzy search
        bm25_match = self._bm25_search(label, ctype)
        if bm25_match:
            score, candidate = bm25_match
            if score >= BM25_AUTO_ALIGN:
                return candidate["canonical_id"]
            elif score >= BM25_PENDING_MIN:
                self._pending.append(
                    {
                        "label": label,
                        "type": ctype,
                        "candidates": [candidate["label"]],
                        "score": score,
                    }
                )
                return candidate["canonical_id"]

        # Step 4: No match → create new canonical_id
        new_id = self._alias_table.get_or_create(label)
        self._alias_table.add_canonical(label, new_id)
        return new_id

    def _bm25_search(self, label: str, ctype: str) -> tuple[float, dict] | None:
        """Search BM25 index for matching concepts. Returns (score, doc) or None."""
        if not self._bm25:
            return None

        query_tokens = _tokenize(label)
        if not query_tokens:
            return None

        scores = self._bm25.get_scores(query_tokens)
        best_score = 0.0
        best_doc = None

        for i, score in enumerate(scores):
            doc = self._bm25_docs[i]
            if doc["label"].lower() == label.lower():
                continue
            if score > best_score:
                best_score = score
                best_doc = doc

        if best_doc and best_score >= BM25_PENDING_MIN:
            return (best_score, best_doc)
        return None

    def flush_pending(self) -> None:
        """Submit pending labels to LLM for batch arbitration."""
        if not self._pending or not self._models:
            return

        decisions = _llm_arbitrate(self._pending, self._models)
        for decision in decisions:
            label = decision["label"]
            canonical = decision.get("canonical")
            conf = decision.get("confidence", 0.5)

            if canonical and conf >= 0.7:
                cid = self._alias_table.lookup(canonical)
                if cid is None:
                    cid = self._alias_table.get_or_create(canonical)
                self._alias_table.add_alias(label, cid)
                log.info("LLM aligned '%s' → '%s' (conf=%.2f)", label, canonical, conf)
            else:
                log.info("LLM uncertain: '%s' (conf=%.2f), keeping independent", label, conf)

        self._pending.clear()


def _llm_arbitrate(pending: list[dict], models: list[dict]) -> list[dict]:
    """Call LLM once with all pending labels for batch arbitration."""
    from drbrain.extractor.llm_client import call_with_fallback

    prompt = (
        "You are an academic concept alignment assistant. "
        "For each input label, determine which existing canonical concept it belongs to.\n\n"
        "Input:\n"
        f"{json.dumps(pending, ensure_ascii=False, indent=2)}\n\n"
        "Output JSON (no extra text):\n"
        '{\n  "decisions": [\n'
        '    {"label": "input label", "canonical": "best matching canonical or null", "confidence": 0.0-1.0}\n'
        "  ]\n}\n\n"
        "Rules:\n"
        "- canonical should be one of the candidate labels or null if no match\n"
        "- confidence: 0.9+ = clearly same concept, 0.7-0.8 = likely same, <0.7 = uncertain\n"
        "- Cross-language equivalents (e.g., '长程依赖' and 'long-range dependency') are the same concept\n"
        "- Only match if semantically equivalent, not merely related\n"
    )

    data = call_with_fallback(prompt, models, max_tokens=2048)
    if data is None:
        log.warning("LLM arbitration failed, keeping all pending labels independent")
        return []

    return data.get("decisions", [])
