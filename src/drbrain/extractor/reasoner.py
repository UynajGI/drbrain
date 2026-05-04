"""LLM Agent for graph reasoning with tool-calling loop."""
from __future__ import annotations

import json
import asyncio


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
                "target": r.target, "source": r.source, "distance": r.distance,
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

    async def reason(self, question: str, max_turns: int = 5) -> str:
        """Run LLM agent loop to reason about a question using graph tools."""
        if not self.models:
            return "No LLM models configured."

        tools = self.tool_definitions()
        messages = [
            {"role": "system", "content": (
                "You are a knowledge graph reasoning assistant. "
                "Use the provided tools to explore the graph and answer questions. "
                "Explain your reasoning step by step."
            )},
            {"role": "user", "content": question},
        ]

        for _ in range(max_turns):
            try:
                import litellm

                model = self.models[0]
                name = f"{model['provider']}/{model['model']}"
                kwargs = {
                    "model": name, "messages": messages, "temperature": 0.3,
                    "max_tokens": 1024, "timeout": 60, "tools": tools,
                }
                if model.get("api_key"):
                    kwargs["api_key"] = model["api_key"]
                if model.get("base_url"):
                    kwargs["api_base"] = model["base_url"]

                resp = litellm.completion(**kwargs)
                msg = resp.choices[0].message

                if msg.tool_calls:
                    messages.append({"role": "assistant", "content": msg.content or "",
                                     "tool_calls": [{
                                         "id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name,
                                                      "arguments": tc.function.arguments},
                                     } for tc in msg.tool_calls]})
                    for tc in msg.tool_calls:
                        args = json.loads(tc.function.arguments)
                        if tc.function.name == "search_concepts":
                            result = self._search_concepts(**args)
                        elif tc.function.name == "get_neighbors":
                            result = self._get_neighbors(**args)
                        elif tc.function.name == "find_path":
                            result = self._find_path(**args)
                        else:
                            result = []
                        messages.append({
                            "role": "tool", "tool_call_id": tc.id,
                            "content": json.dumps(result, ensure_ascii=False, default=str),
                        })
                else:
                    return msg.content or "No answer generated."
            except Exception as e:
                return f"Reasoning error: {e}"

        return "Unable to answer after maximum reasoning turns."
