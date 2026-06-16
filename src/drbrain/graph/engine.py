"""In-memory graph with NetworkX + rule-based closure."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from drbrain.graph.engine_closure import ClosureMixin
from drbrain.graph.engine_embeddings import EmbeddingsMixin


@dataclass
class TraverseStep:
    src: str
    relation: str
    dst: str
    hop: int


@dataclass
class TraverseResult:
    target: str
    target_type: str
    source: str
    distance: int
    path: list[TraverseStep]


class GraphEngine(ClosureMixin, EmbeddingsMixin):
    """Graph operations and rule-based relationship closure."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self._transE = None  # cached TransE instance from learn_embeddings()

    def add_edge(
        self, src: str, dst: str, relation: str, source_paper: str, weight: float = 1.0
    ) -> None:
        self.graph.add_edge(src, dst, relation=relation, source=source_paper, weight=weight)

    def get_neighbors(self, node: str, hops: int = 2) -> set[str]:
        """Get N-hop neighborhood (includes start node)."""
        visited: set[str] = {node}
        current = {node}
        for _ in range(hops):
            next_layer: set[str] = set()
            for n in current:
                if n in self.graph:
                    next_layer |= set(self.graph.predecessors(n))
                    next_layer |= set(self.graph.successors(n))
            next_layer -= visited
            if not next_layer:
                break
            visited |= next_layer
            current = next_layer
        return visited

    def traverse(
        self,
        start_nodes: set[str],
        hops: int = 2,
        relations: set[str] | None = None,
        direction: str = "both",
    ) -> list[TraverseResult]:
        """BFS from start_nodes with relation filtering and directional control.

        Args:
            start_nodes: Seed node labels to start traversal from.
            hops: Maximum number of hops to traverse.
            relations: Edge types to follow (None = all types).
            direction: "forward" (out-edges), "backward" (in-edges), or "both".

        Returns:
            List of TraverseResult with full path from seed to target.
        """
        results: list[TraverseResult] = []
        visited: set[tuple[str, str]] = set()  # (node, seed) dedup

        for seed in start_nodes:
            if seed not in self.graph:
                continue
            visited.add((seed, seed))
            # Queue: (current_node, seed_origin, path_so_far)
            queue: list[tuple[str, str, list[TraverseStep]]] = [(seed, seed, [])]

            for hop in range(1, hops + 1):
                next_queue: list[tuple[str, str, list[TraverseStep]]] = []
                for current, origin, path_so_far in queue:
                    neighbors: list[tuple[str, str]] = []

                    if direction in ("forward", "both"):
                        for _, dst, data in self.graph.out_edges(current, data=True):
                            if relations is None or data["relation"] in relations:
                                neighbors.append((dst, data["relation"]))

                    if direction in ("backward", "both"):
                        for src, _, data in self.graph.in_edges(current, data=True):
                            if relations is None or data["relation"] in relations:
                                neighbors.append((src, data["relation"]))

                    for neighbor, rel in neighbors:
                        if (neighbor, origin) not in visited:
                            visited.add((neighbor, origin))
                            step = TraverseStep(src=current, relation=rel, dst=neighbor, hop=hop)
                            new_path = path_so_far + [step]
                            results.append(
                                TraverseResult(
                                    target=neighbor,
                                    target_type="unknown",
                                    source=origin,
                                    distance=hop,
                                    path=new_path,
                                )
                            )
                            next_queue.append((neighbor, origin, new_path))
                queue = next_queue

        return results

    def load_from_db(self, db, paper_ids: set[str] | None = None) -> None:
        """Load all edges from database into NetworkX graph.

        Args:
            db: Database instance.
            paper_ids: Optional set of local_ids to filter edges by source_paper.
        """
        if paper_ids:
            placeholders = ",".join("?" for _ in paper_ids)
            rows = db.conn.execute(
                f"SELECT src_id, dst_id, relation, source_paper, weight FROM edges "
                f"WHERE source_paper IN ({placeholders})",
                tuple(paper_ids),
            ).fetchall()
        else:
            rows = db.conn.execute(
                "SELECT src_id, dst_id, relation, source_paper, weight FROM edges"
            ).fetchall()
        for src, dst, rel, src_paper, weight in rows:
            self.graph.add_edge(src, dst, relation=rel, source=src_paper, weight=weight)

    def persist_to_db(self, db) -> None:
        """Write all graph edges to database."""
        for u, v, data in self.graph.edges(data=True):
            db.insert_edge(u, v, data["relation"], data["source"], data.get("weight", 1.0))
        db.commit()

    # ── Layer 5: Tree-aware graph operations ────────────────────────────────

    def get_concepts_by_node(self, conn, node_id: str) -> list[dict]:
        """Get all concepts linked to a specific tree node.

        Args:
            conn: SQLite connection.
            node_id: Tree node ID from tree.json.

        Returns:
            List of {concept_id, label, type, section, confidence}.
        """
        rows = conn.execute(
            "SELECT concept_id, label, type, section, confidence FROM concepts WHERE node_id = ?",
            (node_id,),
        ).fetchall()
        return [
            {
                "concept_id": r[0],
                "label": r[1],
                "type": r[2],
                "section": r[3],
                "confidence": r[4],
            }
            for r in rows
        ]

    def get_section_context(self, conn, concept_label: str) -> dict | None:
        """Get the tree node context for a concept by label.

        Returns {node_id, section, paper_id} or None.
        """
        row = conn.execute(
            "SELECT c.node_id, c.section, c.local_id FROM concepts c WHERE c.label = ? LIMIT 1",
            (concept_label,),
        ).fetchone()
        if not row or not row[0]:
            return None
        return {
            "node_id": row[0],
            "section": row[1] or "",
            "paper_id": row[2] or "",
        }

    def get_section_contexts_batch(self, conn, labels: list[str]) -> dict[str, dict]:
        """Get section contexts for multiple concept labels.

        Returns mapping of label → {node_id, section, paper_id}.
        Only includes concepts that have a node_id.
        """
        if not labels:
            return {}
        placeholders = ",".join("?" for _ in labels)
        rows = conn.execute(
            f"SELECT label, node_id, section, local_id FROM concepts "
            f"WHERE label IN ({placeholders}) AND node_id != ''",
            labels,
        ).fetchall()
        return {
            r[0]: {"node_id": r[1], "section": r[2] or "", "paper_id": r[3] or ""} for r in rows
        }

    def _get_section_by_cid(self, conn, concept_id: str) -> dict | None:
        """Get section context by concept_id (integer as string)."""
        row = conn.execute(
            "SELECT node_id, section, local_id, label FROM concepts WHERE concept_id = ? LIMIT 1",
            (int(concept_id),),
        ).fetchone()
        if not row or not row[0]:
            return None
        return {
            "node_id": row[0],
            "section": row[1] or "",
            "paper_id": row[2] or "",
            "label": row[3] or "",
        }

    def traverse_with_sections(self, conn, start_label: str, max_hops: int = 2) -> list[dict]:
        """Traverse graph from a concept label, enriching with section provenance.

        Each step includes src_section and dst_section when available.
        """
        # Resolve label → concept_id
        row = conn.execute(
            "SELECT concept_id FROM concepts WHERE label = ? LIMIT 1",
            (start_label,),
        ).fetchone()
        if not row:
            return []
        start_id = str(row[0])

        if start_id not in self.graph:
            return []

        contexts: dict[str, dict | None] = {}
        # Lookup label→section using concept_id in DB
        visited: set[str] = {start_id}
        current = {start_id}
        steps: list[dict] = []

        for hop in range(max_hops):
            next_layer: set[str] = set()
            for node in current:
                for _, dst, data in self.graph.out_edges(node, data=True):
                    if node not in contexts:
                        contexts[node] = self._get_section_by_cid(conn, node)
                    if dst not in contexts:
                        contexts[dst] = self._get_section_by_cid(conn, dst)

                    src_ctx = contexts.get(node)
                    dst_ctx = contexts.get(dst)
                    step = {
                        "hop": hop + 1,
                        "src": src_ctx["label"] if src_ctx else node,
                        "dst": dst_ctx["label"] if dst_ctx else dst,
                        "src_id": node,
                        "dst_id": dst,
                        "relation": data.get("relation", ""),
                        "source_paper": data.get("source_paper", ""),
                        "weight": data.get("weight", 1.0),
                    }
                    if src_ctx:
                        step["src_section"] = src_ctx.get("section", "")
                        step["src_node_id"] = src_ctx.get("node_id", "")
                    if dst_ctx:
                        step["dst_section"] = dst_ctx.get("section", "")
                        step["dst_node_id"] = dst_ctx.get("node_id", "")

                    steps.append(step)
                    if dst not in visited:
                        visited.add(dst)
                        next_layer.add(dst)
            current = next_layer
            if not current:
                break

        return steps
