---
name: kg-reason
description: >
  Reason over the knowledge graph using LLM agents — natural language Q&A and tool-calling
  iterative reasoning with bidirectional LLM↔KG validation. Use this skill whenever the user
  asks a research question that requires synthesizing information across multiple papers, wants
  to "reason about", "what does the literature say about", "compare approaches", "find the
  relationship between", or needs evidence-backed answers from their library. Also use when the
  user wants to validate hypotheses against the graph structure, explore contradicting findings,
  or needs an AI research assistant that can query concepts, traverse the graph, and read paper
  sections. Trigger proactively when the user asks complex questions that go beyond simple keyword
  search.
---

# KG Reasoning

Query the knowledge graph with natural language questions. Two modes: `ask` for quick KGQA
(retrieve → answer), and `reason` for deep LLM agent reasoning with tool-calling (iterative
graph exploration, section reading, hypothesis validation).

## Prerequisites

The knowledge graph must be built (`kg-build` skill). Best results with embeddings and closure:

```bash
drbrain embed --tree && drbrain closure --mode hybrid
```

## Operations

### ask — Quick KGQA

Natural language question against the knowledge graph. Retrieves top-k relevant concepts and
edges, then generates an answer grounded in the graph:

```bash
drbrain ask "Is attention better than CNN for NLP?"
drbrain ask "what are the main approaches to knowledge distillation" --top 10
drbrain ask "how does BERT handle long sequences" --json
```

Options: `--top` / `-k` (concepts to retrieve, default 5), `--json`.

The answer includes provenance — concepts and edges used as evidence.

### reason — LLM Agent Reasoning

LLM agent with tool access to the knowledge graph. The agent can query concepts, traverse edges,
read paper sections (via PageIndex tree), and iteratively refine its answer:

```bash
drbrain reason "compare the efficiency of sparse vs dense attention mechanisms"
drbrain reason "what open problems remain in graph neural network explainability?"
```

**Bidirectional mode** (`-b`): The agent proposes hypotheses, validates them against graph
constraints (TBox/RBox), and revises. Useful for questions where contradictions or constraints
matter:

```bash
drbrain reason -b "does method A actually solve problem B, or just claim to?"
drbrain reason -b -r 5 "map the causal chain from X to Y across these papers"
```

Options: `-b` / `--bidirectional` (LLM↔KG validation loop), `-r N` / `--max-rounds` (default 3).

## When to use ask vs reason

| Situation | Use |
|-----------|-----|
| Factual lookup ("what is X?") | `ask` |
| Concept comparison ("A vs B") | `ask` (simple) / `reason` (complex) |
| Multi-paper synthesis | `reason` |
| Causal chain tracing | `reason -b` |
| Hypothesis validation | `reason -b` |
| Contradiction resolution | `reason -b` |

## Examples

**Quick concept lookup:**
```bash
drbrain ask "what loss functions are used for contrastive learning?"
```

**Deep multi-paper reasoning:**
```bash
drbrain reason "how do different papers address the over-smoothing problem in GNNs, and which approach is most promising?"
```

**Hypothesis validation with bidirectional loop:**
```bash
drbrain reason -b -r 5 "does the proposed method actually solve the cold-start problem, or does it just shift the burden to feature engineering?"
```

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain ask "<question>"` | KGQA: retrieve → answer |
| `drbrain ask "<q>" --top 10` | More concepts in context |
| `drbrain ask "<q>" --json` | JSON output with provenance |
| `drbrain reason "<question>"` | LLM agent with tool-calling |
| `drbrain reason -b "<q>"` | Bidirectional LLM↔KG validation |
| `drbrain reason -b -r 5 "<q>"` | More hypothesis-revision rounds |
