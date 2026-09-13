# RAG layer

## Current contract

The RAG layer is a retrieval service over stored papers and derived evidence units. It is independent of the CLI and loop orchestration and can be used directly from Python.

## Retrieval path

1. The indexer reads stored paper sections, PageIndex nodes, RAPTOR summaries, concepts, and arguments.
2. Independent retrievers produce ranked candidates: BM25, vector, tree/PageIndex, RAPTOR, and graph-aware retrieval.
3. `FusionRetriever` combines legs with reciprocal-rank fusion or weighted fusion.
4. Optional ACL post-filtering removes candidates without matching scope metadata.
5. Optional reranking refines the fused candidate set.
6. The result preserves source, paper, section, node, and score provenance for synthesis and audit.

Each leg is isolated. A degraded or unavailable leg is recorded in retrieval trace and does not invalidate candidates returned by other legs.

## Evidence semantics

PageIndex sections and RAPTOR summaries are logical evidence units. Long documents may have physical fragments with parent checksums and character offsets. A response must cite the logical unit and retain enough provenance to locate the source artifact.

## Agent boundary

The RAG agent exposes retrieval and knowledge validation tools. MCP tools are not special-cased in the agent; they enter through the shared capability catalog and return the standard invocation envelope.

## Verification

```bash
uv run pytest tests/test_rag* tests/test_retrieval* -q
uv run ruff check src/drbrain/rag
```

For a new retriever, implement the retriever contract, add a source label, emit trace data for success and degradation, and test provenance and ACL behavior.
