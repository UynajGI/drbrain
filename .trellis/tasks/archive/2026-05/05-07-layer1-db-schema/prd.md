# Layer 1: DB Schema + Provenance

## Goal

在 concepts 和 arguments 表加 `node_id` 列链接回 PageIndex 树节点，新增 `tree_vectors` 和 `tree_summaries` 表支持嵌入和 RAPTOR 摘要。

## Requirements

### Migration v4
- concepts 表加 `node_id TEXT DEFAULT ''`
- arguments 表加 `node_id TEXT DEFAULT ''`
- 向后兼容：现有数据的 node_id 为空字符串

### New tables
- `tree_vectors`: node_id TEXT PK, paper_id TEXT, embedding BLOB, content_hash TEXT, tree_layer TEXT
- `tree_summaries`: node_id TEXT PK, paper_id TEXT, summary_text TEXT, source_node_ids TEXT, tree_layer INTEGER
- `vector_metadata`: key TEXT PK, value TEXT (参考 ScholarAIO)

### Reference patterns
- Migration: idempotent, PRAGMA table_info 检查, commit after each step
- ScholarAIO: content_hash for incremental update, vector_metadata for signature tracking

## Out of Scope
- 实际嵌入生成（Layer 2）
- 摘要生成（Layer 3）
