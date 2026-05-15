# absorb-remaining-learnings-p1-p2

## Goal

吸收 4 篇 knowgraph 论文 + ScholarAIO 中已识别的高价值未采用模式，补全 DrBrain 知识图谱引擎的最后缺口。全量 7 项。

## Requirements

### P1: GraphEngine.learn_embeddings()（来源 2202.07412）

- [ ] `GraphEngine.learn_embeddings(dim=128, epochs=100, lr=0.01)` — 训练 TransE 并持久化到 `embeddings` 表
- [ ] `GraphEngine.entity_embedding(label)` → `np.ndarray | None`
- [ ] `GraphEngine.predict_link(head, relation, top_k=10)` → `list[(tail, score)]`
- [ ] `GraphEngine.similar_entities(label, top_k=10)` → `list[(label, similarity)]`
- [ ] 混合闭包复用持久化嵌入（不再每次 `closure()` 内联训练 TransE）

### P1: KG 反哺 LLM 上下文（来源 2306.08302 F1）

- [ ] `ask`/`reason` 命令上下文注入 KG 闭包推理边
- [ ] 格式：`--[inferred: evolves]-->` 标注推断关系，与直接提取关系区分
- [ ] 控制注入数量（top-k 按置信度），避免上下文膨胀

### P2: Embedding-based rule mining（来源 2202.07412）

- [ ] CLI `drbrain closure --mine-rules` 接通 `rule_miner.py`
- [ ] 从嵌入空间挖掘新规则，补充手工 8+4 规则

### P2: Tree traversal 查询（来源 2401.18059）

- [ ] 实现 RAPTOR Figure 2 两阶段检索：逐层 top-k 下降 → 结果不足时 fallback collapsed tree
- [ ] Deep tree 场景 token 更省，collapsed tree 兜底保证召回

### ScholarAIO: GPU batch size 自适应（来源 ScholarAIO vectors.py）

- [ ] `_compute_batch_size()` / `_estimate_mem_per_sample()` — GPU 内存 profiling
- [ ] `_embed_batch` 在 GPU 模式下自动选择最优 batch size

### ScholarAIO: 搜索后过滤（来源 ScholarAIO vectors.py）

- [ ] `_post_filter()` — `search_tree()` 返回结果质量控制
- [ ] 过滤标准：有效年份、非空文本、分数阈值

### ScholarAIO: 多源模型下载（来源 ScholarAIO vectors.py）

- [ ] `_resolve_model_path()` — ModelScope 主源 + HuggingFace 回退
- [ ] 双源容错，支持离线/墙内环境

## Acceptance Criteria

### P1
- [ ] `GraphEngine.learn_embeddings(dim, epochs, lr)` 训练并持久化
- [ ] `GraphEngine.entity_embedding(label)` 返回向量
- [ ] `GraphEngine.predict_link(head, relation, top_k)` 预测尾实体
- [ ] `GraphEngine.similar_entities(label, top_k)` 相似实体搜索
- [ ] `ask` 上下文包含闭包推理边（格式：`--[inferred: evolves]-->`）
- [ ] 新增测试覆盖上述方法

### P2
- [ ] `drbrain closure --mine-rules` 从嵌入空间挖掘新规则
- [ ] RAPTOR tree traversal 在 deep tree 场景 token < collapsed tree
- [ ] 新增测试

### ScholarAIO
- [ ] `drbrain embed` GPU 模式下自动 batch size 调优
- [ ] `search_tree()` post_filter 过滤无效结果
- [ ] 模型加载支持 ModelScope + HuggingFace 双源

## Out of Scope

- 高级 KGE 模型（ComplEx/RotatE/DistMult）
- Neural Theorem Proving
- RDF/Turtle 输出
- KG 增强 LLM 训练/微调
- RAPTOR 幻觉分析验证层
- TKRL/HAKE 层级编码注入

## Technical Notes

- `docs/superpowers/specs/2026-05-04-kg-reasoning-design.md:37-41` — learn_embeddings spec
- `src/drbrain/graph/embedding.py` — TransE 类（116行）
- `src/drbrain/graph/engine.py:281-291` — closure() 混合模式
- `src/drbrain/cli/analysis_commands.py:124-164` — ask 上下文构建
- `src/drbrain/extractor/rule_miner.py` — 289行，未被 CLI 使用
- `src/drbrain/query/tree_retrieval.py:444` — collapsed tree retrieval
- `docs/learn/knowgraph/2401.18059/full.md` — RAPTOR Figure 2
- `docs/learn/scholaraio/scholaraio/services/vectors.py` — GPU profiling + post_filter + multi-source

## Open Questions

- Tree traversal：完整两阶段（逐层下降 + collapsed fallback）— ✅ 已确认
