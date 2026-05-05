# KG Reasoning Enhancement

## Context
Compared against 2202.07412 (Logics+Embeddings) and 2306.08302 (LLMs+KGs).
DrBrain already has TransE, closure rules, TBox/RBox, LLM agent reasoning.

## Requirements

### T1: Complex Query Answering
- Embedding-based ∧,∨,¬ query operators (project/intersect/union/negate)
- Works on existing TransE embeddings
- CLI: `drbrain graph query '{"type":"intersect",...}'`

### T2: Synergized Bidirectional Reasoning
- LLM↔KG iterative feedback loop (max 3 rounds)
- KG validates LLM hypotheses via TBox/RBox
- `ReasonerAgent.reason_bidirectional()`

### T3: Rule Mining from Embeddings
- Learn path rules from relation embedding similarity
- Confidence scoring via embedding composition
- CLI: `drbrain closure --mine-rules`

### T4: Graph-to-Text
- LLM generates NL description of KG subgraph
- CLI: `drbrain graph describe <node>`
- Integrated into drbrain analyze report

## Success Criteria
- All tests pass
- CLI commands work
- No regression
