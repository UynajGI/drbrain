"""Tree search entry: ANN candidates over every published layer (T39).

The tree leg starts from the shared ANN index, never from a BM25 paper hit:
with an empty keyword index it still returns origin text.  Candidates are
taken from all published layers (``view="all"``) so navigation can start from
a summary or from a leaf, and the same canonical node is returned once even
when several layers are close to the query.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from drbrain.tree.vector_store import UnifiedVectorStore, VectorHit, VectorStoreError


class TreeSearchError(RuntimeError):
    """The tree search entry could not run."""


@dataclass(frozen=True)
class TreeCandidate:
    node_id: str
    kind: str
    layer: int
    local_id: str
    score: float
    profile_id: str
    node_revision: int

    def to_json(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "layer": self.layer,
            "local_id": self.local_id,
            "score": round(float(self.score), 6),
            "profile_id": self.profile_id,
            "node_revision": self.node_revision,
        }


def _hit_to_candidate(hit: VectorHit) -> TreeCandidate:
    return TreeCandidate(
        node_id=hit.node_id,
        kind=hit.kind,
        layer=int(hit.layer),
        local_id=hit.local_id,
        score=float(hit.score),
        profile_id=hit.profile_id,
        node_revision=int(hit.node_revision),
    )


class TreeSearch:
    """Entry point of the tree leg: query embedding -> published candidates."""

    def __init__(
        self,
        store: UnifiedVectorStore,
        *,
        profile_id: str | None = None,
        top_k: int = 50,
    ) -> None:
        self.store = store
        self.profile_id = profile_id
        self.top_k = max(1, int(top_k))

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int | None = None,
        view: str = "all",
        local_ids: Sequence[str] | None = None,
        ready_only: Sequence[str] | None = None,
    ) -> list[TreeCandidate]:
        """Ranked candidates from the shared index.

        ``ready_only`` is the set of node ids the caller has verified as
        published; candidates outside it are dropped so a stale vector can
        never surface an unpublished node.
        """
        limit = max(1, int(top_k or self.top_k))
        try:
            hits = self.store.query(
                query_vector,
                top_k=limit * 2 if ready_only else limit,
                view=view,
                local_ids=local_ids,
                profile_id=self.profile_id,
            )
        except VectorStoreError as exc:
            raise TreeSearchError(f"tree search unavailable: {exc}") from exc
        seen: set[str] = set()
        out: list[TreeCandidate] = []
        for hit in hits:
            if hit.node_id in seen:
                continue
            seen.add(hit.node_id)
            if ready_only is not None and hit.node_id not in set(ready_only):
                continue
            out.append(_hit_to_candidate(hit))
            if len(out) >= limit:
                break
        return out

    def search_from_text(
        self,
        embed_query,
        query_text: str,
        **kwargs: Any,
    ) -> list[TreeCandidate]:
        """Convenience wrapper: embed one query text and search."""
        vectors = embed_query([query_text])
        if not vectors or not vectors[0]:
            raise TreeSearchError("query embedding failed")
        return self.search(vectors[0], **kwargs)
