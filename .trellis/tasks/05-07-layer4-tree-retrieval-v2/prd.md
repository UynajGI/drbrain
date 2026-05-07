# Layer 4: Tree Retrieval v2

## Goal

LLM 推理检索为主，向量辅助。单文本内 LLM 导航优先，向量做候选预筛。跨论文用向量加速召回，最终仍由 LLM 评估。

## 哲学

```
LLM 推理检索 (PRIMARY)
    ├── 向量辅助: 快速召回候选节点（缩小 LLM 搜索范围）
    ├── 向量辅助: 跨论文发现相关 node（LLM 单靠自己做不到）
    └── 向量辅助: 语义相似度排序（补充 BM25）

NOT: 向量优先 → LLM 退回
```

## Requirements

### tree_retrieval.py 增强
- `query_by_structure_hybrid(query, paper_dir, db_path, cfg)` — LLM 导航为主，向量预筛 top-k 候选节点给 LLM 评估
- `query_cross_paper(query, db_path, cfg, paper_ids=None)` — 跨论文向量召回，LLM 最终评估
- `_hybrid_score(bm25_results, vector_results, alpha=0.5)` — 分数融合
- `_rrf_score(results_list, k=60)` — Reciprocal Rank Fusion

### 与现有 query_by_structure 的关系
- 原始 `query_by_structure` (LLM 导航) 仍然是 PRIMARY
- 向量作为辅助：预筛候选节点 → LLM 在缩小后的候选集上推理
- provider=none 时完全回退到纯 LLM 导航（不变）
