# Embeddings

The embedding service builds vectors for PageIndex/tree evidence and supports retrieval-time query encoding.

## Providers

- `local`: sentence-transformers model loaded in-process, with batching and GPU-aware tuning.
- `openai-compat`: calls a configured OpenAI-compatible `/v1/embeddings` endpoint.
- `none`: disables vector generation; lexical and structural retrieval remain available.

`build_tree_vectors()` hashes node content and updates only changed nodes. Vectors are stored with metadata and can be searched by tree retrieval and hybrid retrieval.

## Operations

```bash
uv run drbrain index build            # tree text embeddings are part of the index build
uv run drbrain search "topic"         # evidence retrieval over the enabled legs
```

The historical `embed --tree` (tree text vectors) and `query --engine llamaindex` remain available as hidden compatibility aliases. Configure the provider, model, endpoint, batch size, and dimension in the `embed` section. The standalone `scripts/serve_embedding.py` provides a local OpenAI-compatible service when embedding work should be isolated from the CLI process.

## Failure behavior

Model download, endpoint, and GPU failures are reported at the embedding boundary. Existing vectors remain valid; callers can fall back to BM25/tree retrieval or set `provider: none`.
