# Configuration Reference

Every DrBrain setting, its default, and what it controls.

## How Config Works

Three sources, merged in order (later wins):

1. `config.yaml` — base template, checked into git
2. `config.local.yaml` — local overrides and secrets, **gitignored**
3. Environment variables — `${VAR_NAME}` syntax expands at load time

`drbrain setup` generates `config.local.yaml` interactively. `drbrain check` validates
your configuration.

Typed access via `src/drbrain/config.py::Config` dataclass — supports both attribute
access (`cfg.embed.provider`) and dict-style backward compat (`cfg["embed"]["provider"]`).

---

## LLM Models

```yaml
llm:
  models:
    - provider: openai
      model: gpt-4o
      api_key: "${OPENAI_API_KEY}"
      base_url: null
```

First model is primary. On failure (timeout, auth, parse error), DrBrain tries the next
model in the list. All litellm providers work.

| Field | Default | Description |
|-------|---------|-------------|
| `provider` | *(required)* | litellm provider: `openai`, `anthropic`, `ollama`, etc. |
| `model` | *(required)* | Model ID (e.g. `gpt-4o`, `claude-sonnet-4-6`, `deepseek-v4-pro`) |
| `api_key` | `null` | API key. Use `"${ENV_VAR}"` to read from environment. |
| `base_url` | `null` | Custom base URL. `null` = provider default. |

### Provider Templates

**OpenAI** (native):
```yaml
- provider: openai
  model: gpt-4o
  api_key: "${OPENAI_API_KEY}"
```

**Anthropic (Claude)**:
```yaml
- provider: anthropic
  model: claude-sonnet-4-6
  api_key: "${ANTHROPIC_API_KEY}"
```

**OpenAI-compatible providers** (DeepSeek, Zhipu, Bailian, Moonshot, MiniMax, vLLM):
```yaml
- provider: openai
  model: deepseek-v4-pro
  api_key: "${DEEPSEEK_API_KEY}"
  base_url: "https://api.deepseek.com"
```

| Provider | Default base_url | Model examples |
|----------|-----------------|----------------|
| DeepSeek | `https://api.deepseek.com` | `deepseek-v4-pro`, `deepseek-v4-flash` |
| Zhipu (GLM) | `https://open.bigmodel.cn/api/paas/v4` | `glm-4.5`, `glm-4`, `glm-4-flash` |
| Bailian (Qwen) | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen3-235b-a22b`, `qwen-max` |
| Moonshot (Kimi) | `https://api.moonshot.cn/v1` | `kimi-k2`, `moonshot-v1-8k` |
| MiniMax | `https://api.minimax.chat/v1` | `minimax-m2.1` |
| Ollama (local) | `http://localhost:11434` | `qwen2.5:7b` |
| vLLM / SGLang | `http://localhost:8000/v1` | `meta-llama/Llama-4-Scout-17B-16E-Instruct` |

---

## MinerU PDF Parser

```yaml
mineru:
  token: "${MINERU_TOKEN}"
  model: "vlm"
  is_ocr: false
  enable_formula: true
  enable_table: true
  max_pages: 150
```

Primary PDF parser. Falls back to PyMuPDF (pymupdf4llm) when MinerU is unreachable.

| Field | Default | Description |
|-------|---------|-------------|
| `token` | `""` | MinerU API token. Free tier works without token. |
| `model` | `vlm` | Pipeline: `pipeline`, `vlm`, `MinerU-HTML` |
| `is_ocr` | `false` | Force OCR extraction (ignores embedded text) |
| `enable_formula` | `true` | Parse LaTeX formulas |
| `enable_table` | `true` | Parse tables |
| `max_pages` | `150` | Split threshold. PDFs over this are chunked. MinerU limit: 200. |

---

## Database

```yaml
db:
  path: "data/drbrain.db"
```

| Field | Default | Description |
|-------|---------|-------------|
| `path` | `data/drbrain.db` | SQLite database path. WAL mode enabled. |

Metrics tracked separately in `data/metrics.db`.

---

## Data Directories

```yaml
dirs:
  inbox: "data/spool/inbox"
  pending: "data/spool/pending"
  papers: "data/papers"
  reports: "data/reports"
  cache: "data/cache"
  logs: "data/logs"
```

| Field | Default | Purpose |
|-------|---------|---------|
| `inbox` | `data/spool/inbox` | Drop PDFs here to ingest |
| `pending` | `data/spool/pending` | Failed ingests land here |
| `papers` | `data/papers` | Per-paper directories (`papers/<id>/`) |
| `reports` | `data/reports` | JSON analysis reports |
| `cache` | `data/cache` | API response cache (safe to delete) |
| `logs` | `data/logs` | Rotating log files |

---

## External APIs

```yaml
api:
  deepxiv_token: "${DEEPXIV_TOKEN}"
  s2_rate_limit: 100
  s2_api_key: ""
  cache_ttl: 86400
  crossref_email: "${CROSSREF_EMAIL}"
  openalex_token: "${OPENALEX_TOKEN}"
```

| Field | Default | Description |
|-------|---------|-------------|
| `deepxiv_token` | `""` | [DeepXiv](https://data.rag.ac.cn) token for TLDR + keywords |
| `s2_rate_limit` | `100` | Semantic Scholar requests/minute |
| `s2_api_key` | `""` | [S2 API key](https://www.semanticscholar.org/product/api) for higher limits |
| `cache_ttl` | `86400` | API cache TTL in seconds (24 hours) |
| `crossref_email` | `""` | Email for CrossRef polite pool |
| `openalex_token` | `""` | OpenAlex token for higher rate limits |

---

## Search (BM25)

```yaml
bm25:
  k1: 1.5
  b: 0.75
```

| Field | Default | Range | Description |
|-------|---------|--------|-------------|
| `k1` | `1.5` | 0.5–2.0 | Term frequency saturation |
| `b` | `0.75` | 0–1 | Document length normalization |

---

## Extraction

```yaml
extract:
  max_concurrent: 10
```

| Field | Default | Description |
|-------|---------|-------------|
| `max_concurrent` | `10` | Max parallel LLM calls during Stage 2 (entity extraction) |

Higher values increase throughput but also API cost and rate-limit risk.

---

## Embedding

```yaml
embed:
  provider: "local"
  model: "Qwen/Qwen3-Embedding-0.6B"
  device: "auto"
  top_k: 10
  source: "modelscope"
  cache_dir: "~/.cache/modelscope/hub/models"
  hf_endpoint: ""
  api_base: ""
  api_key: ""
  batch_size: 64
```

Text embeddings for tree nodes (PageIndex leaf sections + RAPTOR summaries). Not used for
graph embeddings (TransE).

### Provider: `local` (default)

Downloads and runs a sentence-transformers model locally. No API key needed.

| Field | Default | Description |
|-------|---------|-------------|
| `model` | `Qwen/Qwen3-Embedding-0.6B` | HuggingFace model ID. 1024-dim output. |
| `device` | `auto` | `auto` / `cpu` / `cuda`. GPU used if available. |
| `source` | `modelscope` | Download source: `modelscope` or `huggingface` |
| `cache_dir` | `~/.cache/modelscope/hub/models` | Local model cache directory |
| `hf_endpoint` | `""` | HuggingFace mirror URL (e.g. `https://hf-mirror.com`) |
| `batch_size` | `64` | Texts per encoding call. Auto-tuned downward on GPU OOM. |

### Provider: `openai-compat`

Calls any OpenAI-compatible `/v1/embeddings` endpoint. No local GPU needed.

| Field | Default | Description |
|-------|---------|-------------|
| `api_base` | `""` | API base URL (e.g. `https://api.openai.com/v1`) |
| `api_key` | `""` | API key. Prefer `"${EMBED_API_KEY}"` with env var. |
| `model` | `text-embedding-3-small` | Model name passed to the API |
| `batch_size` | `64` | Texts per API call. Requests split into chunks. |

Retry: 3 retries with exponential backoff on 429/5xx. First-chunk failure raises;
subsequent chunk failures return partial results.

### Provider: `none`

Disables all text embeddings. Search falls back to pure BM25 + LLM tree navigation.

| Field | Description |
|-------|-------------|
| `top_k` | Ignored. BM25 and tree traversal handle retrieval. |

---

## Quality Control

```yaml
queue:
  weak_threshold: 0.7
  auto_accept: 0.9
```

| Field | Default | Description |
|-------|---------|-------------|
| `weak_threshold` | `0.7` | Confidence below this → human review queue |
| `auto_accept` | `0.9` | Confidence above this → auto-accept |

Confidence between `weak_threshold` and `auto_accept` enters the queue for optional review.

---

## Rsync Backup

```yaml
backup:
  ssh_bin: ssh
  rsync_bin: rsync
  targets:
    myserver:
      host: backup.example.com
      user: drbrain
      path: /backups/drbrain/
      port: 22
      identity_file: "~/.ssh/id_ed25519"
      mode: default          # default | append | append-verify
      compress: true
      enabled: true
      exclude: []
```

| Field | Default | Description |
|-------|---------|-------------|
| `ssh_bin` | `ssh` | SSH binary path |
| `rsync_bin` | `rsync` | Rsync binary path |
| `targets.<name>.host` | — | Remote SSH host |
| `targets.<name>.user` | `""` | SSH username |
| `targets.<name>.path` | — | Remote destination path |
| `targets.<name>.port` | `22` | SSH port |
| `targets.<name>.identity_file` | `""` | SSH private key path |
| `targets.<name>.password` | `""` | SSH password (stored in `config.local.yaml`) |
| `targets.<name>.mode` | `default` | Transfer mode: `default`, `append`, `append-verify` |
| `targets.<name>.compress` | `true` | Enable rsync compression |
| `targets.<name>.enabled` | `true` | Whether the target is active |
| `targets.<name>.exclude` | `[]` | Rsync exclude patterns |

Configure targets in `config.local.yaml` (contains secrets). Local tar.gz backups
are always available without configuration.

---

## Fetch

```yaml
fetch:
  max_concurrent: 3
  timeout_per_fetch: 60
  user_agent: "DrBrain/0.1"
  fallback_order: ["openalex", "arxiv", "unpaywall", "doi_direct"]
  unpaywall_email: ""
  institutional_proxy: ""
  proxy_type: ""
```

| Field | Default | Description |
|-------|---------|-------------|
| `max_concurrent` | `3` | Parallel PDF downloads |
| `timeout_per_fetch` | `60` | Seconds per download attempt |
| `user_agent` | `DrBrain/0.1` | HTTP User-Agent header |
| `fallback_order` | `[openalex, arxiv, unpaywall, doi_direct]` | Source priority for PDF acquisition |
| `unpaywall_email` | `""` | Email for Unpaywall API access |
| `institutional_proxy` | `""` | Proxy host for paywalled papers |
| `proxy_type` | `""` | `ezproxy` or `url_prefix` |

---

## File Locations Summary

| What | Where |
|------|-------|
| Base config | `config.yaml` |
| Secrets | `config.local.yaml` |
| Template | `config.example.yaml` |
| Config class | `src/drbrain/config.py` |
| Setup wizard | `drbrain setup` (bilingual EN/ZH) |
| Validate | `drbrain check` |
| Citation styles | `data/citation_styles/` |
