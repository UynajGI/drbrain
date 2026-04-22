"""In-memory graph with NetworkX + rule-based closure."""

from __future__ import annotations

from collections import defaultdict

import networkx as nx


class GraphEngine:
    """Graph operations and rule-based relationship closure."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_edge(self, src: str, dst: str, relation: str, source_paper: str, weight: float = 1.0) -> None:
        self.graph.add_edge(src, dst, relation=relation, source=source_paper, weight=weight)

    def get_neighbors(self, node: str, hops: int = 2) -> set[str]:
        """Get N-hop neighborhood."""
        visited: set[str] = set()
        current = {node}
        for _ in range(hops):
            next_layer: set[str] = set()
            for n in current:
                if n in self.graph:
                    next_layer |= set(self.graph.predecessors(n))
                    next_layer |= set(self.graph.successors(n))
            visited |= current
            current = next_layer - visited
        return visited

    def closure(self) -> list[dict]:
        """Run rule-based closure, return inferred edges.

        Rules:
        - challenges(P, C) & supports(Q, C) => creates_debate(P, Q, C)
        - leaves_open(P, G) & addresses(Q, G) => gap_addressed(G, Q)
        - extends(M1, M2) & replaces(M2, M3) => indirect_evolution(M1, M3)
        """
        inferred: list[dict] = []

        # Build relation indices
        challenges: dict[str, list[str]] = defaultdict(list)
        supports: dict[str, list[str]] = defaultdict(list)
        leaves_open: dict[str, list[str]] = defaultdict(list)
        addresses: dict[str, list[str]] = defaultdict(list)
        extends: dict[str, list[str]] = defaultdict(list)
        replaces: dict[str, list[str]] = defaultdict(list)

        for u, v, data in self.graph.edges(data=True):
            rel = data["relation"]
            if rel == "challenges":
                challenges[v].append(u)
            elif rel == "supports":
                supports[v].append(u)
            elif rel == "leaves_open":
                leaves_open[v].append(u)
            elif rel == "addresses":
                addresses[v].append(u)
            elif rel == "extends":
                extends[u].append(v)
            elif rel == "replaces":
                replaces[u].append(v)

        # Rule 1: creates_debate
        for conclusion in challenges:
            if conclusion in supports:
                for p in challenges[conclusion]:
                    for q in supports[conclusion]:
                        if p != q:
                            inferred.append({
                                "src": p, "dst": q, "relation": "creates_debate",
                                "via": conclusion,
                            })

        # Rule 2: gap_addressed
        for gap in leaves_open:
            if gap in addresses:
                for p in leaves_open[gap]:
                    for q in addresses[gap]:
                        inferred.append({
                            "src": gap, "dst": q, "relation": "gap_addressed",
                            "via": gap,
                        })

        # Rule 3: indirect_evolution
        for m1 in extends:
            for m2 in extends[m1]:
                if m2 in replaces:
                    for m3 in replaces[m2]:
                        inferred.append({
                            "src": m1, "dst": m3, "relation": "indirect_evolution",
                            "via": m2,
                        })

        return inferred

    def detect_research_seeds(self) -> list[dict]:
        """Detect research opportunities via graph patterns."""
        seeds: list[dict] = []

        # Pattern 1: High in-degree Problem with no recent addresses
        problem_in_degree = dict(self.graph.in_degree())
        for node, deg in problem_in_degree.items():
            if deg >= 3:
                seeds.append({
                    "type": "stale_problem",
                    "node": node,
                    "signal": f"High attention ({deg} edges), check for recent solutions",
                })

        # Pattern 2: Gaps with no incoming addresses
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "Gap":
                incoming = list(self.graph.predecessors(node))
                if not incoming:
                    seeds.append({
                        "type": "unaddressed_gap",
                        "node": node,
                        "signal": "Gap identified but no method addresses it",
                    })

        # Pattern 3: Conclusion with both supports and challenges
        in_edges = defaultdict(list)
        for u, v, data in self.graph.edges(data=True):
            if data["relation"] in ("supports", "challenges"):
                in_edges[v].append((u, data["relation"]))

        for conclusion, edges in in_edges.items():
            rels = {r for _, r in edges}
            if len(rels) == 2:
                seeds.append({
                    "type": "debate_zone",
                    "node": conclusion,
                    "signal": f"Active debate: {len(edges)} papers with conflicting views",
                })

        return seeds

    def load_from_db(self, db) -> None:
        """Load all edges from database into NetworkX graph."""
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
