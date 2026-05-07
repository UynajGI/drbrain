"""In-memory graph with NetworkX + rule-based closure."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import networkx as nx

from drbrain.graph.path_reasoning import _apply_path_rules_subgraph, apply_path_rules
from drbrain.validator.schema import detect_asymmetric_violations, enforce_transitive


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


class GraphEngine:
    """Graph operations and rule-based relationship closure."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()

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

    def closure(
        self,
        section_map: dict[str, str] | None = None,
        mode: str = "symbolic",
    ) -> list[dict]:
        """Run rule-based closure, return inferred edges.

        Args:
            section_map: Optional mapping of node label → section title.
                When provided, each inferred edge gets a ``confidence`` field
                computed via section-aware decay.
            mode: ``"symbolic"`` (default) for rule-based inference only,
                or ``"hybrid"`` to additionally weight confidence via TransE
                embedding scores.

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
        points_to: dict[str, list[str]] = defaultdict(list)
        constrains: dict[str, list[str]] = defaultdict(list)

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
            elif rel == "points_to":
                points_to[u].append(v)
            elif rel == "constrains":
                constrains[u].append(v)

        # Rule 1: creates_debate
        for conclusion in challenges:
            if conclusion in supports:
                for p in challenges[conclusion]:
                    for q in supports[conclusion]:
                        if p != q:
                            inferred.append(
                                {
                                    "src": p,
                                    "dst": q,
                                    "relation": "creates_debate",
                                    "via": conclusion,
                                }
                            )

        # Rule 2: gap_addressed
        for gap in leaves_open:
            if gap in addresses:
                for p in leaves_open[gap]:
                    for q in addresses[gap]:
                        inferred.append(
                            {
                                "src": gap,
                                "dst": q,
                                "relation": "gap_addressed",
                                "via": gap,
                            }
                        )

        # Rule 3: indirect_evolution
        for m1 in extends:
            for m2 in extends[m1]:
                if m2 in replaces:
                    for m3 in replaces[m2]:
                        inferred.append(
                            {
                                "src": m1,
                                "dst": m3,
                                "relation": "indirect_evolution",
                                "via": m2,
                            }
                        )

        # Rule 4: gap_to_debate — Gap points_to a target that has challenges/supports
        for gap in points_to:
            for target in points_to[gap]:
                if target in challenges and target in supports:
                    inferred.append(
                        {
                            "src": gap,
                            "dst": target,
                            "relation": "gap_to_debate",
                            "via": target,
                        }
                    )

        # Rule 5: actor_network — Papers sharing an Actor form a research lineage
        actor_papers: dict[str, list[str]] = defaultdict(list)
        for u, v, data in self.graph.edges(data=True):
            if data["relation"] == "affiliated_with":
                actor_papers[v].append(u)
        for actor, papers in actor_papers.items():
            if len(papers) > 1:
                for i, p1 in enumerate(papers):
                    for p2 in papers[i + 1 :]:
                        inferred.append(
                            {
                                "src": p1,
                                "dst": p2,
                                "relation": "shared_actor",
                                "via": actor,
                            }
                        )

        # Rule 6: Transitive closure for RBOX transitive relations
        edge_list = [
            {"src": u, "dst": v, "relation": data["relation"], "source_paper": data["source"]}
            for u, v, data in self.graph.edges(data=True)
        ]
        transitive_inferred = enforce_transitive(edge_list)
        for edge in transitive_inferred:
            inferred.append(
                {
                    "src": edge["src"],
                    "dst": edge["dst"],
                    "relation": edge["relation"],
                    "via": edge["via"],
                }
            )

        # Rule 7: Asymmetric violation detection (logged, not inferred)
        detect_asymmetric_violations(edge_list)

        # Rule 8: Multi-hop path rules
        path_inferred = apply_path_rules(self)
        inferred.extend(path_inferred)

        # Section-aware confidence propagation
        if section_map:
            from drbrain.extractor.confidence_propagation import propagate_confidence_with_section

            for edge in inferred:
                src_section = section_map.get(edge["src"], "")
                edge["confidence"] = propagate_confidence_with_section(
                    confidence=1.0,
                    section=src_section,
                )

        # Hybrid mode: re-weight confidence with TransE embedding scores
        if mode == "hybrid" and inferred:
            from drbrain.graph.embedding import TransE

            t = TransE(dim=128, epochs=50)
            t.train(self.graph)
            for edge in inferred:
                score = t.score(edge["src"], edge["relation"], edge["dst"])
                edge["embedding_score"] = round(float(1.0 / (1.0 + score)), 4)
                existing = edge.get("confidence", 1.0)
                edge["confidence"] = round(0.5 * existing + 0.5 * edge["embedding_score"], 3)

        return inferred

    def ground_rules(self, min_confidence: float = 0.5) -> list[dict]:
        """Ground closure rules as concrete edges via path matching (t-norm style).

        For transitive rules: if A→B and B→C exist, add A→C with
        confidence = min(conf_AB, conf_BC) as a grounded triple.
        """
        grounded: list[dict] = []
        seen: set[tuple[str, str, str]] = set()

        transitive_relations = {"extends", "contains", "proposes", "addresses"}
        for u, v, data in self.graph.edges(data=True):
            rel = data.get("relation", "")
            if rel not in transitive_relations:
                continue
            # Check if v has outgoing edges of the same relation type
            for w, v2, data2 in self.graph.edges(v, data=True):
                if data2.get("relation", "") == rel:
                    key = (u, w, rel)
                    if key not in seen:
                        conf_uv = data.get("weight", 1.0)
                        conf_vw = data2.get("weight", 1.0)
                        confidence = min(conf_uv, conf_vw)
                        if confidence >= min_confidence:
                            seen.add(key)
                            grounded.append(
                                {
                                    "src": u,
                                    "dst": w,
                                    "relation": rel,
                                    "confidence": round(confidence, 3),
                                    "via": [v],
                                    "source": "rule_grounding",
                                }
                            )

        return grounded

    def detect_research_seeds(self, db=None) -> list[dict]:
        """Detect research opportunities via graph patterns + temporal data.

        When db is provided, also detects:
        - technology_cliff: dense extends chain ends, related Gap constrains it
        - cross_domain_isomorphism: disconnected subgraphs share same Problem
        - confidence_collapse: avg_confidence drops > 0.2 between 2-year windows

        Without db, only detects graph-based patterns:
        - stale_problem: Problem with many incoming edges
        - unaddressed_gap: Gap with leaves_open but no solves/addresses
        - debate_zone: Same target has both supports and challenges
        """
        seeds: list[dict] = []

        # Build relation index
        edges_by_rel: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for u, v, data in self.graph.edges(data=True):
            edges_by_rel[data["relation"]].append((u, v))

        # --- Pattern 1: Stale problem ---
        if db:
            seeds.extend(self._detect_stale_problems(db, edges_by_rel))
        else:
            problem_in_degree: dict[str, int] = defaultdict(int)
            for _, dst in edges_by_rel.get("addresses", []):
                problem_in_degree[dst] += 1
            for problem, deg in problem_in_degree.items():
                if deg >= 3:
                    seeds.append(
                        {
                            "type": "stale_problem",
                            "concept": problem,
                            "description": f"High attention ({deg} edges), check for recent solutions",
                            "confidence": 0.6,
                        }
                    )

        # --- Pattern 2: Unaddressed gap ---
        gap_nodes: set[str] = set()
        for _, dst in edges_by_rel.get("leaves_open", []):
            gap_nodes.add(dst)
        addressed_gaps: set[str] = set()
        for rel_name in ("solves", "addresses"):
            for _, dst in edges_by_rel.get(rel_name, []):
                addressed_gaps.add(dst)
        for gap in gap_nodes:
            if gap not in addressed_gaps:
                count = len([v for _, v in edges_by_rel["leaves_open"] if v == gap])
                if db:
                    seeds.append(
                        {
                            "type": "unaddressed_gap",
                            "concept": gap,
                            "description": f"Gap '{gap}' identified by {count} papers but no proposed solution exists",
                            "confidence": 0.8,
                        }
                    )
                else:
                    seeds.append(
                        {
                            "type": "unaddressed_gap",
                            "concept": gap,
                            "description": f"Gap identified but no method addresses it ({count} leaves_open edges)",
                            "confidence": 0.6,
                        }
                    )

        # --- Pattern 3: Debate zone ---
        supports_targets = {v for _, v in edges_by_rel.get("supports", [])}
        challenges_targets = {v for _, v in edges_by_rel.get("challenges", [])}
        debate_targets = supports_targets & challenges_targets
        for target in debate_targets:
            n_support = len([v for _, v in edges_by_rel["supports"] if v == target])
            n_challenge = len([v for _, v in edges_by_rel["challenges"] if v == target])
            if db:
                seeds.append(
                    {
                        "type": "debate_zone",
                        "concept": target,
                        "description": f"{n_support} papers support '{target}', {n_challenge} challenge it — active debate",
                        "confidence": 0.75,
                    }
                )
            else:
                seeds.append(
                    {
                        "type": "debate_zone",
                        "concept": target,
                        "description": f"Active debate: {n_support + n_challenge} papers with conflicting views",
                        "confidence": 0.6,
                    }
                )

        # --- New DB-augmented patterns ---
        if db:
            seeds.extend(self._detect_technology_cliffs(db))
            seeds.extend(self._detect_cross_domain_isomorphism(db))
            seeds.extend(self._detect_confidence_collapse(db))

        return seeds

    # ---------- DB-augmented pattern detectors ----------

    def _detect_stale_problems(self, db, edges_by_rel) -> list[dict]:
        """Problem with >=5 incoming addresses edges but no new ones in last 2 years."""
        current = datetime.now().year
        seeds = []
        address_targets: dict[str, list] = defaultdict(list)
        for src, dst in edges_by_rel.get("addresses", []):
            address_targets[dst].append(src)

        for problem, sources in address_targets.items():
            if len(sources) < 5:
                continue
            recent = db.conn.execute(
                "SELECT COUNT(*) FROM edges e JOIN papers p ON e.source_paper = p.local_id "
                "WHERE e.relation = 'addresses' AND e.dst_id = ? AND p.year >= ?",
                (problem, current - 2),
            ).fetchone()[0]
            if recent == 0:
                seeds.append(
                    {
                        "type": "stale_problem",
                        "concept": problem,
                        "description": f"Problem '{problem}' addressed by {len(sources)} papers but no progress since {current - 3}",
                        "confidence": 0.85,
                    }
                )
        return seeds

    def _detect_technology_cliffs(self, db) -> list[dict]:
        """Method with dense extends chain that ended, and a related Gap constrains it."""
        seeds = []

        # Get all methods that appear in extends edges
        extends_methods: set[str] = set()
        for u, v, data in self.graph.edges(data=True):
            if data["relation"] == "extends":
                extends_methods.add(u)
                extends_methods.add(v)

        if not extends_methods:
            return seeds

        for method in extends_methods:
            # Check for constraining Gap
            constraining_gaps = db.conn.execute(
                "SELECT e.src_id FROM edges e JOIN concepts c ON e.src_id = c.label "
                "WHERE e.relation = 'constrains' AND e.dst_id = ? AND c.type = 'Gap'",
                (method,),
            ).fetchall()

            if constraining_gaps:
                gap_label = constraining_gaps[0][0]
                # Get last active year
                last_active = db.conn.execute(
                    "SELECT MAX(p.year) FROM concepts c JOIN papers p ON c.local_id = p.local_id "
                    "WHERE c.label = ? AND p.year IS NOT NULL",
                    (method,),
                ).fetchone()
                year_str = str(last_active[0]) if last_active and last_active[0] else "unknown"

                seeds.append(
                    {
                        "type": "technology_cliff",
                        "concept": method,
                        "description": (
                            f"Method '{method}' stalled after {year_str} due to constraint "
                            f"'{gap_label}' — current conditions may enable revival"
                        ),
                        "confidence": 0.7,
                    }
                )

        return seeds

    def _detect_cross_domain_isomorphism(self, db) -> list[dict]:
        """Two disconnected subgraphs share the same Problem label, path length > 3."""
        seeds = []

        # Find problems addressed by methods in distinct groups
        problem_methods: dict[str, set[str]] = defaultdict(set)
        for u, v, data in self.graph.edges(data=True):
            if data["relation"] == "addresses":
                # Check if dst is a Problem
                row = db.conn.execute(
                    "SELECT type FROM concepts WHERE label = ? LIMIT 1", (v,)
                ).fetchone()
                if row and row[0] == "Problem":
                    problem_methods[v].add(u)

        for problem, methods in problem_methods.items():
            if len(methods) < 4:
                continue

            # Check for disconnected pairs
            method_list = list(methods)
            disconnected_pairs = 0
            for i in range(len(method_list)):
                for j in range(i + 1, len(method_list)):
                    if not nx.has_path(self.graph, method_list[i], method_list[j]):
                        disconnected_pairs += 1

            if disconnected_pairs > 0:
                seeds.append(
                    {
                        "type": "cross_domain_isomorphism",
                        "concept": problem,
                        "description": (
                            f"Multiple disconnected approaches address '{problem}' "
                            f"({len(methods)} methods, {disconnected_pairs} disconnected pairs) "
                            f"— potential transfer opportunity"
                        ),
                        "confidence": 0.65,
                    }
                )

        return seeds

    def _detect_confidence_collapse(self, db) -> list[dict]:
        """Concept with avg_confidence dropping > 0.2 between consecutive 2-year windows."""
        seeds = []

        rows = db.conn.execute(
            "SELECT c.label, c.type, p.year, AVG(c.confidence) as avg_conf "
            "FROM concepts c JOIN papers p ON c.local_id = p.local_id "
            "WHERE p.year IS NOT NULL GROUP BY c.label, c.type, p.year"
        ).fetchall()

        concept_years: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
        for label, ctype, year, avg_conf in rows:
            concept_years[label].append((year, avg_conf, ctype))

        for label, data in concept_years.items():
            if len(data) < 4:
                continue
            years_data = sorted(data, key=lambda x: x[0])
            ctype = years_data[0][2]

            min_year = min(y[0] for y in years_data)
            max_year = max(y[0] for y in years_data)
            if max_year - min_year < 4:
                continue

            mid_year = min_year + (max_year - min_year) // 2
            early_confs = [y[1] for y in years_data if y[0] <= mid_year]
            late_confs = [y[1] for y in years_data if y[0] > mid_year]

            if not early_confs or not late_confs:
                continue

            early_avg = sum(early_confs) / len(early_confs)
            late_avg = sum(late_confs) / len(late_confs)

            if early_avg - late_avg > 0.2:
                seeds.append(
                    {
                        "type": "confidence_collapse",
                        "concept": label,
                        "description": (
                            f"Concept '{label}' confidence dropped from {early_avg:.2f} to "
                            f"{late_avg:.2f} — paradigm shift detected"
                        ),
                        "confidence": 0.8,
                    }
                )

        return seeds

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

    def closure_incremental(self, seed_nodes: set[str]) -> list[dict]:
        """Run closure rules only for edges touching seed_nodes.

        Instead of scanning the full graph, build a subgraph containing
        seed nodes and their 2-hop neighborhood, then run the same
        closure rules on that subgraph.
        """
        if not seed_nodes:
            return []

        # Build subgraph: seed nodes + 2-hop neighborhood
        relevant_nodes: set[str] = set()
        for node in seed_nodes:
            if node not in self.graph:
                continue
            relevant_nodes.add(node)
            relevant_nodes |= self.get_neighbors(node, hops=2)

        if not relevant_nodes:
            return []

        sub = nx.MultiDiGraph()
        for u, v, data in self.graph.edges(data=True):
            if u in relevant_nodes and v in relevant_nodes:
                sub.add_edge(
                    u,
                    v,
                    relation=data["relation"],
                    source=data["source"],
                    weight=data.get("weight", 1.0),
                )

        if sub.number_of_edges() == 0:
            return []

        # Build relation indices for the subgraph
        challenges: dict[str, list[str]] = defaultdict(list)
        supports: dict[str, list[str]] = defaultdict(list)
        leaves_open: dict[str, list[str]] = defaultdict(list)
        addresses: dict[str, list[str]] = defaultdict(list)
        extends: dict[str, list[str]] = defaultdict(list)
        replaces: dict[str, list[str]] = defaultdict(list)
        points_to: dict[str, list[str]] = defaultdict(list)
        constrains: dict[str, list[str]] = defaultdict(list)

        for u, v, data in sub.edges(data=True):
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
            elif rel == "points_to":
                points_to[u].append(v)
            elif rel == "constrains":
                constrains[u].append(v)

        inferred: list[dict] = []

        # Rule: creates_debate
        for conclusion in challenges:
            if conclusion in supports:
                for p in challenges[conclusion]:
                    for q in supports[conclusion]:
                        if p != q:
                            inferred.append(
                                {
                                    "src": p,
                                    "dst": q,
                                    "relation": "creates_debate",
                                    "via": conclusion,
                                }
                            )

        # Rule: gap_addressed
        for gap in leaves_open:
            if gap in addresses:
                for p in leaves_open[gap]:
                    for q in addresses[gap]:
                        inferred.append(
                            {"src": gap, "dst": q, "relation": "gap_addressed", "via": gap}
                        )

        # Rule: indirect_evolution
        for m1 in extends:
            for m2 in extends[m1]:
                if m2 in replaces:
                    for m3 in replaces[m2]:
                        inferred.append(
                            {"src": m1, "dst": m3, "relation": "indirect_evolution", "via": m2}
                        )

        # Rule: gap_to_debate
        for gap in points_to:
            for target in points_to[gap]:
                if target in challenges and target in supports:
                    inferred.append(
                        {"src": gap, "dst": target, "relation": "gap_to_debate", "via": target}
                    )

        # Rule: shared_actor
        actor_papers: dict[str, list[str]] = defaultdict(list)
        for u, v, data in sub.edges(data=True):
            if data["relation"] == "affiliated_with":
                actor_papers[v].append(u)
        for actor, papers in actor_papers.items():
            if len(papers) > 1:
                for i, p1 in enumerate(papers):
                    for p2 in papers[i + 1 :]:
                        inferred.append(
                            {"src": p1, "dst": p2, "relation": "shared_actor", "via": actor}
                        )

        # Rule: transitive closure
        edge_list = [
            {"src": u, "dst": v, "relation": data["relation"], "source_paper": data["source"]}
            for u, v, data in sub.edges(data=True)
        ]
        from drbrain.validator.schema import enforce_transitive

        for edge in enforce_transitive(edge_list):
            inferred.append(
                {
                    "src": edge["src"],
                    "dst": edge["dst"],
                    "relation": edge["relation"],
                    "via": edge["via"],
                }
            )

        # Rule: path rules
        path_inferred = _apply_path_rules_subgraph(sub)
        inferred.extend(path_inferred)

        return inferred

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

    def closure_with_sections(self, conn) -> tuple[list[dict], dict]:
        """Run closure with section provenance enrichment.

        Reuses existing closure() logic and enriches inferred edges
        with section context from source/destination concepts.

        Returns (enriched_inferred_edges, meta).
        """

        # Run standard closure
        inferred = self.closure()

        # Collect all concept IDs involved (closure uses src/dst = concept IDs)
        cids: set[str] = set()
        for edge in inferred:
            cids.add(str(edge.get("src", "")))
            cids.add(str(edge.get("dst", "")))

        # Fetch section contexts by concept ID (not label)
        section_map: dict[str, dict] = {}
        for cid in cids:
            if not cid:
                continue
            ctx = self._get_section_by_cid(conn, cid)
            if ctx:
                section_map[cid] = ctx

        # Enrich edges
        enriched = []
        for edge in inferred:
            enriched_edge = dict(edge)
            src_cid = str(edge.get("src", ""))
            dst_cid = str(edge.get("dst", ""))
            src_ctx = section_map.get(src_cid)
            dst_ctx = section_map.get(dst_cid)
            if src_ctx:
                enriched_edge["src_section"] = src_ctx.get("section", "")
                enriched_edge["src_node_id"] = src_ctx.get("node_id", "")
            if dst_ctx:
                enriched_edge["dst_section"] = dst_ctx.get("section", "")
                enriched_edge["dst_node_id"] = dst_ctx.get("node_id", "")
            enriched.append(enriched_edge)

        meta = {
            "total_inferred": len(enriched),
            "with_section_provenance": sum(
                1 for e in enriched if "src_section" in e or "dst_section" in e
            ),
        }
        return enriched, meta
