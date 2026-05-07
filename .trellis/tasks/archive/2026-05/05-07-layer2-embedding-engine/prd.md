# Layer 2: Embedding Engine

## Goal

实现轻量向量嵌入引擎。为 PageIndex 树节点生成语义向量，支持本地模型和 OpenAI-compatible API，provider=none 时优雅降级。

## Reference

ScholarAIO `services/vectors.py` — build_vectors, _load_model, _embed_batch_local, FAISS, signature tracking.

## Requirements

### EmbedConfig
- provider: "local" | "openai-compat" | "none"
- model: 默认 "Qwen/Qwen3-Embedding-0.6B"
- device: "auto" (GPU if available)
- cache_dir, top_k, source ("modelscope" | "huggingface")

### Embedding service (`services/embedding.py`)
- `build_tree_vectors(db_path, paper_dir, cfg)` — embed all nodes for a paper
- `embed_text(text, cfg)` — single text → vector
- `embed_batch(texts, cfg)` — batch embedding
- `search_tree(query, db_path, top_k, cfg)` — cosine similarity over tree_vectors
- `provider=none` → skip silently, return empty/0

### Storage
- tree_vectors 表 (Layer 1 已创建)
- content_hash 用于增量更新
- vector_metadata 存储 embed_signature

## Out of Scope
- FAISS 索引（先做 numpy 内积，后续加 FAISS）
- RAPTOR 节点嵌入（Layer 3 做）
- 跨论文索引（Layer 4 做）
