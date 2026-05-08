"""LLM Agent for graph reasoning with tool-calling loop."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


class ReasonerAgent:
    """Agent that explores a knowledge graph using LLM tool-calling."""

    def __init__(self, db=None, graph_engine=None, models=None):
        self.db = db
        self.graph = graph_engine
        self.models = models or []

    def tool_definitions(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_concepts",
                    "description": "Search concepts in the knowledge graph by keyword",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search keyword"},
                            "limit": {"type": "integer", "default": 5},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_neighbors",
                    "description": "Get neighbors of a concept node in the graph",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node": {"type": "string"},
                            "hops": {"type": "integer", "default": 1},
                            "direction": {
                                "type": "string",
                                "enum": ["forward", "backward", "both"],
                            },
                        },
                        "required": ["node"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "find_path",
                    "description": "Find shortest path between two concepts in the graph",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "src": {"type": "string"},
                            "dst": {"type": "string"},
                        },
                        "required": ["src", "dst"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_document_structure",
                    "description": "Get the section tree skeleton for a paper (titles + node_ids only, no content)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "paper_id": {"type": "string", "description": "Paper local_id"},
                        },
                        "required": ["paper_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_section_content",
                    "description": "Get the full text content of a specific section within a paper",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "paper_id": {"type": "string", "description": "Paper local_id"},
                            "node_id": {
                                "type": "string",
                                "description": "Tree node ID from document structure",
                            },
                        },
                        "required": ["paper_id", "node_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_tree",
                    "description": "Search across all paper sections by semantic similarity (collapsed tree retrieval)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for finding relevant sections",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_raptor_summaries",
                    "description": "Get RAPTOR cross-section summaries for a paper. Returns hierarchical summaries that capture themes across multiple sections.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "paper_id": {
                                "type": "string",
                                "description": "Paper local_id",
                            },
                        },
                        "required": ["paper_id"],
                    },
                },
            },
        ]

    def _search_concepts(self, query: str, limit: int = 5) -> list[dict]:
        if not self.db:
            return []
        from drbrain.query.bm25 import build_bm25_index

        bm25 = build_bm25_index(self.db)
        results = bm25.search(query, limit=limit)
        return [{"label": r["label"], "type": r["type"], "score": r["score"]} for r in results]

    def _get_neighbors(self, node: str, hops: int = 1, direction: str = "both") -> list[dict]:
        if not self.graph:
            return []
        results = self.graph.traverse(start_nodes={node}, hops=hops, direction=direction)
        return [
            {
                "target": r.target,
                "source": r.source,
                "distance": r.distance,
                "path": [{"src": s.src, "relation": s.relation, "dst": s.dst} for s in r.path],
            }
            for r in results
        ]

    def _find_path(self, src: str, dst: str) -> dict | None:
        if not self.graph or src not in self.graph.graph or dst not in self.graph.graph:
            return None
        import networkx as nx

        ug = self.graph.graph.to_undirected()
        try:
            node_path = nx.shortest_path(ug, source=src, target=dst)
            return {"path": node_path, "length": len(node_path) - 1}
        except (nx.NetworkXNoPath, nx.NetworkXError):
            return None

    # ── Layer 6: Tree tools ──────────────────────────────────────────────

    def _get_document_structure(self, paper_id: str) -> list[dict]:
        """Return the tree skeleton for a paper (titles + node_ids, no content)."""
        import json

        from drbrain.storage.paths import tree_json_path

        papers_dir = self._papers_dir()
        if not papers_dir:
            return []

        tree_path = tree_json_path(papers_dir / paper_id)
        if not tree_path.exists():
            return []

        tree = json.loads(tree_path.read_text(encoding="utf-8"))
        structure = tree.get("structure", [])

        def _extract(nodes: list[dict]) -> list[dict]:
            result = []
            for n in nodes:
                item = {
                    "node_id": n.get("node_id", ""),
                    "title": n.get("title", ""),
                }
                child_nodes = n.get("nodes", [])
                if child_nodes:
                    item["children"] = _extract(child_nodes)
                result.append(item)
            return result

        return _extract(structure)

    def _get_section_content(self, paper_id: str, node_id: str) -> str:
        """Return the raw text content for a tree node."""
        from drbrain.parser.pageindex_parser import get_node_content
        from drbrain.storage.paths import raw_md_path, tree_json_path

        papers_dir = self._papers_dir()
        if not papers_dir:
            return ""

        paper_dir = papers_dir / paper_id
        tree_path = tree_json_path(paper_dir)
        md_path = raw_md_path(paper_dir)
        if not tree_path.exists() or not md_path.exists():
            return ""

        import json

        tree = json.loads(tree_path.read_text(encoding="utf-8"))
        structure = tree.get("structure", [])
        try:
            return get_node_content(md_path, structure, node_id) or ""
        except Exception:
            return ""

    def _search_tree(self, query: str) -> list[dict]:
        """Cross-paper collapsed tree search (vector + BM25 hybrid)."""
        if not self.db:
            return []
        from drbrain.query.tree_retrieval import query_cross_paper

        results = query_cross_paper(query, self.db.path)
        return results

    def _get_raptor_summaries(self, paper_id: str) -> list[dict]:
        """Return RAPTOR cross-section summaries for a paper.

        Queries tree_summaries table for hierarchical summaries
        produced by the RAPTOR recursive semantic tree builder.
        """
        if not self.db:
            return []

        rows = self.db.conn.execute(
            "SELECT node_id, paper_id, summary_text, source_node_ids, tree_layer "
            "FROM tree_summaries WHERE paper_id = ? ORDER BY tree_layer",
            (paper_id,),
        ).fetchall()

        import json

        return [
            {
                "node_id": r[0],
                "paper_id": r[1],
                "summary_text": r[2],
                "source_node_ids": json.loads(r[3]) if r[3] else [],
                "tree_layer": r[4],
            }
            for r in rows
        ]

    def _papers_dir(self) -> Path | None:
        """Resolve the papers data directory from DB config."""
        if not self.db:
            return None

        db_path = self.db.path
        return db_path.parent / "papers"

    async def reason(self, question: str, max_turns: int = 5) -> str:
        """Run LLM agent loop to reason about a question using graph tools."""
        if not self.models:
            return "No LLM models configured."

        tools = self.tool_definitions()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a knowledge graph reasoning assistant. "
                    "Use the provided tools to explore the graph and answer questions. "
                    "Explain your reasoning step by step."
                ),
            },
            {"role": "user", "content": question},
        ]

        for _ in range(max_turns):
            try:
                import litellm

                model = self.models[0]
                name = f"{model['provider']}/{model['model']}"
                kwargs = {
                    "model": name,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 1024,
                    "timeout": 60,
                    "tools": tools,
                    "extra_body": {"thinking": {"type": "disabled"}},
                }
                if model.get("api_key"):
                    kwargs["api_key"] = model["api_key"]
                if model.get("base_url"):
                    kwargs["api_base"] = model["base_url"]

                resp = litellm.completion(**kwargs)
                msg = resp.choices[0].message

                if msg.tool_calls:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": msg.content or "",
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in msg.tool_calls
                            ],
                        }
                    )
                    for tc in msg.tool_calls:
                        args = json.loads(tc.function.arguments)
                        if tc.function.name == "search_concepts":
                            result = self._search_concepts(**args)
                        elif tc.function.name == "get_neighbors":
                            result = self._get_neighbors(**args)
                        elif tc.function.name == "find_path":
                            result = self._find_path(**args)
                        elif tc.function.name == "get_document_structure":
                            result = self._get_document_structure(**args)
                        elif tc.function.name == "get_section_content":
                            result = self._get_section_content(**args)
                        elif tc.function.name == "search_tree":
                            result = self._search_tree(**args)
                        elif tc.function.name == "get_raptor_summaries":
                            result = self._get_raptor_summaries(**args)
                        else:
                            result = []
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps(result, ensure_ascii=False, default=str),
                            }
                        )
                else:
                    return msg.content or "No answer generated."
            except Exception as e:
                return f"Reasoning error: {e}"

        return "Unable to answer after maximum reasoning turns."

    def _call_llm(self, prompt: str, system: str | None = None) -> str | None:
        """Call LLM with a prompt and return text response (no tool-calling).

        Iterates through all configured models with fallback.

        Args:
            prompt: User message to send.
            system: Optional system prompt override.

        Returns:
            LLM text response or None on failure.
        """
        if not self.models:
            return None

        import litellm

        system_content = system or (
            "You are a knowledge graph reasoning assistant. Answer concisely based on evidence."
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

        for i, model in enumerate(self.models):
            name = f"{model['provider']}/{model['model']}"
            try:
                kwargs = {
                    "model": name,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 1024,
                    "timeout": 60,
                    "extra_body": {"thinking": {"type": "disabled"}},
                }
                if model.get("api_key"):
                    kwargs["api_key"] = model["api_key"]
                if model.get("base_url"):
                    kwargs["api_base"] = model["base_url"]
                resp = litellm.completion(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception:
                log.warning("Model %s failed (attempt %d/%d)", name, i + 1, len(self.models))

        log.error("All %d models failed in _call_llm", len(self.models))
        return None

    def _kg_validate(self, hypothesis: str) -> dict:
        """Check hypothesis against KG for consistency.

        Extracts entity mentions from hypothesis text by matching
        concept labels in the DB, then checks the subgraph of those
        entities for TBox violations, RBox violations, and graph
        patterns (debates, gaps).

        Args:
            hypothesis: Free-text hypothesis from LLM.

        Returns:
            {"consistent": bool, "violations": [...], "patterns": [...]}
        """
        result: dict = {"consistent": True, "violations": [], "patterns": []}

        if not self.graph or self.graph.graph.number_of_nodes() == 0:
            return result

        # 1. Find entity mentions in hypothesis text by matching DB concept labels
        mentioned_labels: list[str] = []
        if self.db:
            rows = self.db.conn.execute(
                "SELECT DISTINCT label FROM concepts ORDER BY length(label) DESC"
            ).fetchall()
            hypothesis_lower = hypothesis.lower()
            for (label,) in rows:
                if label and label.lower() in hypothesis_lower:
                    mentioned_labels.append(label)
        else:
            # No DB: extract potential labels from graph nodes
            for node in self.graph.graph.nodes():
                if node and node.lower() in hypothesis.lower():
                    mentioned_labels.append(node)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_labels = []
        for label in mentioned_labels:
            if label not in seen:
                seen.add(label)
                unique_labels.append(label)
        mentioned_labels = unique_labels[:20]  # cap at 20 entities

        if len(mentioned_labels) < 2:
            return result

        # 2. Gather all edges between mentioned entities
        subgraph_edges: list[dict] = []
        for u, v, data in self.graph.graph.edges(data=True):
            if u in seen and v in seen:
                subgraph_edges.append(
                    {
                        "src": u,
                        "dst": v,
                        "relation": data.get("relation", ""),
                        "source_paper": data.get("source", ""),
                    }
                )

        # 3. TBox validation: check each edge's relation against concept types
        from drbrain.validator.schema import detect_asymmetric_violations, validate_tbox

        if self.db:
            # Build type lookup from DB
            type_rows = self.db.conn.execute(
                "SELECT label, type FROM concepts WHERE label IN ({})".format(
                    ",".join("?" for _ in mentioned_labels)
                ),
                mentioned_labels,
            ).fetchall()
            label_to_type: dict[str, str] = {row[0]: row[1] for row in type_rows}

            for edge in subgraph_edges:
                src_type = label_to_type.get(edge["src"])
                if src_type:
                    tbox_result = validate_tbox(src_type, edge["relation"])
                    if not tbox_result.valid:
                        result["consistent"] = False
                        result["violations"].append(
                            {
                                "type": "tbox",
                                "edge": edge,
                                "reason": tbox_result.reason,
                            }
                        )
        else:
            # Without DB, we can't do TBox checks (don't know concept types)
            pass

        # 4. RBox validation: check for asymmetric violations
        asym_violations = detect_asymmetric_violations(subgraph_edges)
        if asym_violations:
            result["consistent"] = False
            for v in asym_violations:
                result["violations"].append(
                    {
                        "type": "rbox_asymmetric",
                        "edge": v,
                        "reason": f"'{v['relation']}' is asymmetric but reverse edge exists.",
                    }
                )

        # 5. Graph pattern detection
        # Debate: two entities connected by "challenges" to the same target
        debates_found: set[tuple] = set()
        for i, e1 in enumerate(subgraph_edges):
            for e2 in subgraph_edges[i + 1 :]:
                if e1["relation"] == "challenges" and e2["relation"] == "challenges":
                    if e1["dst"] == e2["dst"] and e1["src"] != e2["src"]:
                        debates_found.add((e1["src"], e2["src"], e1["dst"]))

        for a, b, target in debates_found:
            result["patterns"].append(
                {
                    "type": "debate",
                    "description": f"'{a}' and '{b}' both challenge '{target}'",
                    "entities": [a, b, target],
                }
            )

        # Gaps: pairs of entities of same/compatible types with no edge between them
        if len(mentioned_labels) >= 2 and self.db:
            connected_pairs: set[tuple[str, str]] = set()
            for e in subgraph_edges:
                connected_pairs.add((e["src"], e["dst"]))
                connected_pairs.add((e["dst"], e["src"]))

            for i, a in enumerate(mentioned_labels):
                for b in mentioned_labels[i + 1 :]:
                    if (a, b) not in connected_pairs:
                        a_type = label_to_type.get(a)
                        b_type = label_to_type.get(b)
                        if a_type and b_type:
                            result["patterns"].append(
                                {
                                    "type": "gap",
                                    "description": f"No edge between '{a}' ({a_type}) and '{b}' ({b_type})",
                                    "entities": [a, b],
                                }
                            )

        return result

    def reason_bidirectional(self, question: str, max_rounds: int = 3) -> dict:
        """Iterative LLM-KG reasoning loop.

        Each round:
        1. LLM proposes hypothesis based on question + previous KG feedback
        2. KG validates hypothesis via TBox/RBox consistency check
        3. If contradiction: feed contradiction back to LLM, repeat
        4. KG detects graph patterns (gaps, contradictions, debates) -> feeds to LLM

        Args:
            question: The research question to reason about.
            max_rounds: Maximum number of hypothesis-revision rounds.

        Returns:
            {"answer": str, "rounds": int, "hypotheses": [...], "kg_validations": [...]}
        """
        if not self.models:
            return {"error": "No LLM models configured."}

        hypotheses: list[str] = []
        validations: list[dict] = []
        previous_violations: list[str] = []
        previous_patterns: list[str] = []

        for round_num in range(1, max_rounds + 1):
            # Build prompt
            if round_num == 1:
                prompt = (
                    f"Question: {question}\n\n"
                    "Based on your knowledge, propose a clear hypothesis that answers "
                    "this question. Include specific entities and relationships "
                    "(e.g., 'Method X addresses Problem Y', 'Finding A supports Conclusion B')."
                )
            else:
                violation_text = (
                    "\n".join(f"- {v}" for v in previous_violations)
                    if previous_violations
                    else "None"
                )
                pattern_text = (
                    "\n".join(f"- {p}" for p in previous_patterns) if previous_patterns else "None"
                )
                prompt = (
                    f"Question: {question}\n\n"
                    f"Your previous hypothesis was:\n"
                    f'  "{hypotheses[-1]}"\n\n'
                    f"The knowledge graph found these issues:\n"
                    f"  Violations: {violation_text}\n"
                    f"  Patterns: {pattern_text}\n\n"
                    "Please revise your hypothesis to address these issues. "
                    "Propose a new hypothesis that is consistent with the graph constraints. "
                    "If you cannot resolve the contradictions, acknowledge the uncertainty."
                )

            hypothesis = self._call_llm(prompt)
            if hypothesis is None:
                return {
                    "answer": "LLM call failed during bidirectional reasoning.",
                    "rounds": round_num,
                    "hypotheses": hypotheses,
                    "kg_validations": validations,
                }

            hypotheses.append(hypothesis)

            # Validate against KG
            validation = self._kg_validate(hypothesis)
            validations.append(validation)

            # Collect violations and patterns for next round
            previous_violations = [v["reason"] for v in validation["violations"]]
            previous_patterns = [p["description"] for p in validation["patterns"]]

            if validation["consistent"]:
                return {
                    "answer": hypothesis,
                    "rounds": round_num,
                    "hypotheses": hypotheses,
                    "kg_validations": validations,
                }

        # Exhausted all rounds
        best = hypotheses[-1] if hypotheses else "No hypothesis generated."
        return {
            "answer": best,
            "rounds": max_rounds,
            "hypotheses": hypotheses,
            "kg_validations": validations,
        }
