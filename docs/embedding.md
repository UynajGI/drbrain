# Embedding Guide

How to choose and configure DrBrain's text embedding backend.

## What Gets Embedded

Only **semantically-complete tree nodes** — PageIndex section leaves and RAPTOR recursive
summaries. Never arbitrary text chunks. Embeddings live in the `tree_vectors` SQLite table
alongside everything else; no separate vector database.

Embeddings are used for:
- Tree node cosine similarity search (`search_tree()`)
- RAPTOR recursive clustering (`build_raptor_tree`)
- Cross-paper collapsed-tree retrieval (`query_cross_paper`)

They are **not** used for: graph reasoning (rule-based), BM25 search, graph embeddings (TransE).

---

## Provider Comparison

| | local | openai-compat | none |
|---|-------|--------------|------|
| **Requires GPU** | Recommended | No | No |
| **Requires network** | First download only | Every call | No |
| **Cost** | Free | API pricing | Free |
| **Model** | Qwen3-Embedding-0.6B (default) | Any `/v1/embeddings` model | N/A |
| **Dimension** | 1024 | Model-dependent | N/A |
| **Setup effort** | Model download (~1.2 GB) | API key + endpoint | None |
| **Best for** | Offline, high-volume, privacy | No local GPU, small batches | BM25-only workflows |

---

## Provider: local

### How It Works

1. On first use, downloads the sentence-transformers model from ModelScope (or HuggingFace fallback).
2. Loads the model into memory (GPU if available, CPU otherwise).
3. Encodes tree node text in batches, with adaptive batch sizing on GPU.

### Setup

```yaml
embed:
  provider: "local"
  model: "Qwen/Qwen3-Embedding-0.6B"
  device: "auto"
  source: "modelscope"
  cache_dir: "~/.cache/modelscope/hub/models"
  batch_size: 64
```

### GPU

Set `device: "auto"` (default). DrBrain detects CUDA and uses GPU if available. On GPU,
it profiles memory once per model+GPU combination and auto-tunes batch size to avoid OOM.
Profile cached at `~/.cache/drbrain/gpu_profile.json`.

| Setting | Effect |
|---------|--------|
| `device: "auto"` | GPU if CUDA available, else CPU (default) |
| `device: "cuda"` | Force GPU. Fails if CUDA unavailable. |
| `device: "cpu"` | Force CPU. No GPU profiling. |

### Model Download Sources

| Source | URL | Notes |
|--------|-----|-------|
| `modelscope` (default) | modelscope.cn | Faster in China |
| `huggingface` | huggingface.co | May need mirror |

For HuggingFace behind a firewall, set `hf_endpoint` to your mirror:

```yaml
embed:
  source: "huggingface"
  hf_endpoint: "https://hf-mirror.com"
```

---

## Provider: openai-compat

### How It Works

Sends POST requests to `{api_base}/embeddings`. Splits large batches into chunks of
`batch_size`. Retries 3 times with exponential backoff on 429/5xx. First-chunk failure
propagates; subsequent chunk failures return partial results.

### Setup

```yaml
embed:
  provider: "openai-compat"
  model: "text-embedding-3-small"
  api_base: "https://api.openai.com/v1"
  api_key: "${OPENAI_API_KEY}"
  batch_size: 64
```

### Compatible Services

Any endpoint implementing the OpenAI `/v1/embeddings` contract:

| Service | api_base | model example |
|---------|----------|---------------|
| OpenAI | `https://api.openai.com/v1` | `text-embedding-3-small` / `text-embedding-3-large` |
| DeepSeek | `https://api.deepseek.com` | *(check DeepSeek docs)* |
| Zhipu | `https://open.bigmodel.cn/api/paas/v4` | `embedding-3` |
| Bailian | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `text-embedding-v4` |
| vLLM self-hosted | `http://localhost:8000/v1` | model name on server |
| Ollama | `http://localhost:11434/v1` | `nomic-embed-text` |

### Error Handling

| Scenario | Behavior |
|----------|----------|
| `api_base` not set | `ValueError` at call time |
| `api_key` not set | `ValueError` at call time |
| First chunk fails (connection/timeout/4xx) | Exception propagates |
| Later chunk fails | Warning logged, partial results returned |
| 429 / 5xx | Automatic retry (3 attempts, exponential backoff) |

---

## Provider: none

```yaml
embed:
  provider: "none"
```

Disables all text embeddings. Search uses BM25 + LLM tree navigation. `drbrain embed --tree`
becomes a no-op. RAPTOR clustering is skipped during `build_paper_tree_vectors`.

Best for: teams relying purely on symbolic reasoning and BM25.

---

## Build Pipeline

Generate embeddings for all ingested papers:

```bash
# PageIndex leaf nodes only
drbrain embed --tree

# Full pipeline: PageIndex + RAPTOR recursive summaries
drbrain build --all   # includes embed step
```

Embedding is incremental: unchanged nodes (by content hash) are skipped on re-runs.

---

## Search

```bash
# Vector search over all tree nodes
drbrain query "turbulent drag reduction" --hybrid

# Tree retrieval on a single paper (LLM navigation + vector auxiliary)
drbrain query "residual connections" --paper p6a321e
```

Vector results include `tree_layer` tags (`pageindex`, `raptor_L1`, `raptor_L2`, etc.)
for provenance.

---

## Troubleshooting

**"Model not found" on first run:**
The model needs to download. Check network and `source`/`hf_endpoint` settings.
First download is ~1.2 GB.

**CUDA out of memory:**
Set `device: "cpu"` or reduce `batch_size`. The GPU profiler auto-tunes on next run.

**openai-compat returns empty:**
Check `api_base` ends with `/v1` (not `/v1/`). Verify the endpoint responds to
`GET {api_base}/models` with your API key.

**Dimension mismatch in search:**
Happens when switching embedding models. Re-run `drbrain embed --tree` to regenerate
all vectors with the new model.
