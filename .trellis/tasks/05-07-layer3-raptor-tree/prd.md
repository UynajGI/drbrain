# Layer 3: RAPTOR Recursive Semantic Tree

## Goal

在 PageIndex 叶子节点之上构建 RAPTOR 递归语义摘要树。GMM 聚类 + UMAP 降维 + LLM 摘要 + 递归到收敛。

## Reference

RAPTOR (2401.18059) Section 3: embedding → UMAP → GMM+BIC → summarize → re-embed → repeat.

## Algorithm

```
PageIndex leaf nodes (tree.json)
    │
    ├── embed (Layer 2 tree_vectors)
    ├── UMAP reduce dimensionality
    ├── GMM clustering with BIC (k=1..min(N,10))
    ├── For each cluster: LLM summarize texts
    ├── Store in tree_summaries (source_node_ids = child ids)
    ├── Re-embed summaries → tree_vectors (tree_layer="raptor_L{n}")
    │
    └── Repeat until: <3 nodes OR BIC k=1 OR max_layers=3
```

## Requirements

### extractor/raptor.py
- `build_raptor_tree(paper_dir, db_path, cfg)` → int (number of summary nodes created)
- `_gmm_cluster(embeddings, n_samples)` → list[list[int]] (cluster indices)
- `_umap_reduce(embeddings, n_components)` → reduced embeddings
- `_summarize_cluster(texts, models)` → str (LLM summary)
- `_bic_gmm(X, k)` → float (BIC score)

### Storage
- tree_summaries rows: node_id (generated), paper_id, summary_text, source_node_ids (JSON list), tree_layer (int)
- tree_vectors rows: raptor nodes embedded (tree_layer="raptor_L{n}")

## Out of Scope
- Cross-paper RAPTOR (single paper only for now)
- Collapsed tree retrieval (Layer 4)
