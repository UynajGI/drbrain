# PageIndex Tree Integration into Graph Engine

## Goal

把 PageIndex 文档结构树从"提取脚手架"升级为图引擎的一级索引结构，让所有图操作（检索、推理、溯源、遍历）都能利用文档层级信息。同时引入 RAPTOR 递归摘要树和轻量向量检索。

## 向量哲学：从"无向量"到"轻向量"

```
拒绝 ❌                              接受 ✅
─────────────────────              ─────────────────────
全文切块 → embed → RAG 知识库      完整语义节点 → embed → 增强检索
100-token 任意切片                   PageIndex 树节点（完整章节/段落）
向量是知识表示                      向量是检索加速层
embedding 替代 BM25                  embedding 补充 BM25（混合检索）
```

**原则**：PageIndex 树节点是语义完整的单元（一个 Section、一个论点），不是被任意切碎的文字碎片。嵌入这样的节点不会丢失上下文和出处。向量只用于"找到相关节点"，不用于"表示知识"——内容理解仍然走 LLM。

## What I Already Know

### 现状

- `pageindex_parser.py` (1027行)：自顶向下树构建，3 模式回退，节点摘要，验证修复
- `tree_retrieval.py` (362行)：自适应树导航，LLM 引导的逐层检索
- `concept.py`：build 阶段用 `get_node_content` 提取叶子节点内容
- `section` 字段只是字符串，不指回 tree node
- `ask` 命令是唯一使用树检索的入口
- ReasonerAgent 只有 3 个图工具（search_concepts, get_neighbors, find_path）

### 参考源

1. **PageIndex 原始库** (`docs/learn/PageIndex/pageindex/`)：结构/内容分离 API，TOC 验证修复，agentic tool-use 模式
2. **RAPTOR** (`docs/learn/knowgraph/2401.18059/full.md`)：递归摘要 + GMM 聚类 + UMAP 降维 + collapsed tree 检索，QuALITY +20%
3. **ScholarAIO 嵌入方案** (`docs/learn/scholaraio/scholaraio/services/vectors.py`)：Qwen3-Embedding-0.6B 本地推理，FAISS，BLOB 存储，增量更新

### 架构决策

三层树合成：

```
Layer 1: PageIndex 结构树（已有）
  └─ 自顶向下，TOC 驱动
  └─ 保留作者意图 + 章节出处
  └─ 叶子节点 = 语义完整的原文段落

Layer 2: RAPTOR 语义树（新建）
  └─ 自底向上，GMM 聚类驱动
  └─ 每层节点 = LLM 摘要
  └─ 每个摘要节点记录来源（哪些 PageIndex 节点）
  └─ 递归直到无法继续聚类

Layer 3: 知识图谱（已有）
  └─ 概念/关系指回 Layer 1 + Layer 2 节点
```

检索策略：

- collapsed tree（摊平所有节点做余弦相似度检索）→ RAPTOR 验证优于逐层导航
- BM25 + 向量混合排序（分数融合）
- `provider=none` 时降级为纯 BM25 + LLM 导航

## Requirements

### R1: 出处链接（DB 层）

- concepts 表增加 `node_id` 列，指向 tree.json 节点
- arguments 表增加 `node_id` 列
- 现有数据的 section 字段保留，node_id 为可选（向后兼容）

### R2: 嵌入引擎（Embedding 层）

- `EmbedConfig` 配置类：provider (local/openai-compat/none), model, device, top_k
- `tree_vectors` 表：node_id, embedding BLOB, content_hash, paper_id, tree_layer
- 嵌入文本 = 完整节点内容（PageIndex 叶子节点原文 or RAPTOR 摘要节点文本）
- 参考 ScholarAIO：sentence-transformers + FAISS IndexFlatIP + 增量更新
- provider=none 时正常降级为 BM25，不影响现有功能

### R3: RAPTOR 递归语义树（Build 层）

- 在 PageIndex 叶子节点之上运行递归聚类+摘要
- GMM 聚类（BIC 自动确定 k）+ UMAP 降维
- 每层 LLM 生成摘要 → 摘要再嵌入 → 再聚类 → 直到收敛
- 每个摘要节点记录 `source_node_ids`（出处链）
- 摘要生成复用现有 `_generate_node_summary` / `_generate_doc_description`

### R4: 树检索 v2（Query 层）

- collapsed tree 模式：embed query → cosine sim over all nodes（PageIndex + RAPTOR）
- 跨论文检索：多个 tree.json 节点统一 FAISS 索引
- BM25 + 向量混合排序（分数融合：weighted sum or RRF）
- 替代/补充当前 LLM 导航模式（保留 LLM 模式作为 fallback）

### R5: 图引擎树集成（Graph 层）

- GraphEngine 支持按 node_id 查询邻居（在哪个章节讨论？）
- closure/descendants/transfers 输出带 section 面包屑
- landscape 按文档结构分组（某领域 → 核心论文 → 关键章节）

### R6: ReasonerAgent 树工具（Reasoning 层）

- `get_document_structure(paper_id)` → 返回树骨架（PageIndex + RAPTOR 摘要层）
- `get_section_content(paper_id, node_id)` → 返回原文
- `search_tree(query)` → collapsed tree 检索

### R7: CLI 增强（CLI 层）

- `descendants --sections` / `--provenance`
- `landscape --sections`
- `paradigm --sections`
- `transfers --sections`
- `analyze` 输出章节级报告

## Acceptance Criteria

- [ ] concepts/arguments 表有 node_id 列，build 时写入
- [ ] `drbrain embed` 命令为 PageIndex + RAPTOR 节点生成向量
- [ ] `drbrain build --raptor` 生成递归语义摘要树
- [ ] `drbrain ask --paper X "query"` 使用 collapsed tree 混合检索
- [ ] `drbrain reason "query"` 的 ReasonerAgent 可调用树工具
- [ ] `drbrain descendants X --sections` 显示章节出处
- [ ] `provider=none` 时所有功能降级为 BM25 + LLM 导航，不报错
- [ ] 现有测试全部通过（无回归）

## Definition of Done

- Tests added/updated
- Lint / typecheck green
- `uv run pytest -m "not integration"` passes
- CHANGELOG 记录变更
- CLI reference 更新新参数
- CLAUDE.md 更新向量哲学说明

## Decision (ADR-lite)

**Context**: 树(PageIndex)和图(graph engine)是两座孤岛，section 只是字符串。检索只靠 BM25。
**Decision**: node_id 作为一等公民贯穿全栈。引入"轻向量"——完整语义节点嵌入用于增强检索，不切块不做向量知识库。吸收 RAPTOR 完整方案（递归摘要 + GMM + UMAP + collapsed tree）。
**Consequences**: DB migration 加列但不破坏现有数据；embedding 对硬件有要求但 provider=none 保证降级；RAPTOR 递归摘要增加 build 时间但可缓存。

## Out of Scope

- 全文任意切块嵌入（anti-goal）
- 纯向量知识库替代 BM25/图（anti-goal）
- 向量作为知识表示层（anti-goal）

## Technical Notes

### 涉及文件

- `storage/database.py` — migration + schema
- `config.py` — EmbedConfig
- `services/embedding.py` — 新建，嵌入服务
- `extractor/raptor.py` — 新建，递归聚类+摘要
- `parser/pageindex_parser.py` — 已有，摘要生成复用
- `query/tree_retrieval.py` — collapsed tree 模式 + 跨论文 + 混合排序
- `graph/engine.py` — 树感知的图操作
- `extractor/reasoner.py` — 树工具
- `cli/commands.py` — 新参数
- `cli/graph_commands.py` — 新参数

### 数据流

```
ingest: PDF → raw.md → tree.json（不变）
build:   tree.json → leaf nodes → extract → concepts (+node_id) → DB
build:   tree.json → leaf nodes → embed → GMM cluster → LLM summarize → RAPTOR tree
embed:   PageIndex nodes + RAPTOR nodes → embed → tree_vectors 表 → FAISS index
query:   用户query → embed → FAISS over tree_vectors → top-k → BM25 分数融合 → LLM
reason:  LLM agent → search_tree tool → 树节点 → 原文 → 推理
```

### 任务拆分（自底向上）

| # | 子任务 | 层 | 核心变更 | 依赖 |
|---|--------|-----|---------|------|
| 1 | DB schema + provenance | Storage | node_id 列, tree_vectors 表, migration | 无 |
| 2 | Embedding engine | Service | EmbedConfig, build_vectors, FAISS, provider=none | 1 |
| 3 | RAPTOR 递归语义树 | Extractor | GMM+UMAP 聚类, 递归 LLM 摘要, source_node_ids | 2 |
| 4 | Tree retrieval v2 | Query | collapsed tree, 跨论文, BM25+向量混合排序 | 2,3 |
| 5 | Graph engine tree integration | Graph | tree-aware traversal, section 面包屑 | 1 |
| 6 | ReasonerAgent tree tools | Extractor | get_document_structure, get_section_content, search_tree | 4,5 |
| 7 | CLI enhancements | CLI | --sections/--provenance, analyze 章节级, embed 命令 | 4,5,6 |

1-5 可并行，3 依赖 2，4 依赖 2+3，6 依赖 4+5，7 收尾。
