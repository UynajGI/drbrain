

---

## LlamaIndex RAG 评估基线 — 2026-08-12T19:08:50

### 配置
- golden_set: `/home/jiangyuan/drbrain/data/llamaindex/golden.jsonl`;split 选项: ['dev', 'val', 'test']
- enabled=True · retrievers=['bm25', 'vector'] · fusion_mode=reciprocal_rank · rerank=True · similarity_cutoff=0.7
- embed_model: `Qwen/Qwen3-Embedding-0.6B`

### Retriever eval(HitRate@K / MRR@K)

- status: `ok`;split: `dev`;queries: 30

| level | metric | K=5 | K=10 |
| --- | --- | --- | --- |
| paper | hit_rate | 0.9667 | 0.9667 |
| paper | mrr | 0.9444 | 0.9444 |
| node | hit_rate | 0.9667 | 0.9667 |
| node | mrr | 0.8444 | 0.8444 |

---

## LlamaIndex RAG 评估基线 — 2026-08-12T19:32:01

### 配置
- golden_set: `/home/jiangyuan/drbrain/data/llamaindex/golden.jsonl`;split 选项: ['dev', 'val', 'test']
- enabled=True · retrievers=['bm25', 'vector'] · fusion_mode=reciprocal_rank · rerank=True · similarity_cutoff=0.7
- embed_model: `Qwen/Qwen3-Embedding-0.6B`

### RAGAS-style eval(自写 4 指标 prompt)

- status: `ok`;split: `dev`;queries: 5

| metric | mean | missing |
| --- | --- | --- |
| faithfulness | 0.36 | 0 |
| answer_relevancy | 0.8 | 0 |
| context_precision | 0.24 | 0 |
| answer_correctness | 0.42 | 0 |


---

## LlamaIndex 重排 A/B — 2026-08-12T22:45 (T8 遗留,真实模型 T9 补跑)

### 配置
- 语料:test-run/papers 46 篇 golden 论文;索引 `data/llamaindex`(T9 GPU 重建,chunked:448 节点 / 33 大节点切分)
- split:`dev` 30 条 golden query;candidates/query = 20(`rerank_top_k`)
- reranker:`Qwen/Qwen3-Reranker-0.6B`(用户 2026-08-12 决策,弃用 bge-reranker-v2-m3;modelscope 下载 ~1.2G / 118s;sentence-transformers CrossEncoder 直接加载,本地 modelscope 缓存离线可用)
- embed:`Qwen/Qwen3-Embedding-0.6B`(GPU)

### 结果(mrr 模式,`scripts/rerank_ab.py --query-file data/llamaindex/golden.jsonl --split dev --rerank-top-k 20`)

| metric | rerank off | rerank on | Δ |
| --- | --- | --- | --- |
| MRR@10 | 0.9417 | **0.9667** | **+0.025** |
| HR@10 | 0.9667 | 0.9667 | 0(k=10 饱和) |

- **Top-1 改变:13/30 query(43%)** — 真实重排扰动显著
- 与 T8 mock 词法重排对比(mock 反而伤 MRR 0.9333→0.8500):语义 bge/Qwen reranker 必要性得到真实验证
- 结论:rerank 默认开(Qwen3-Reranker-0.6B)带来 MRR +2.7% 相对提升;离线环境需预缓存模型(modelscope `Qwen/Qwen3-Reranker-0.6B`),模型缺失时 Noop 降级路径已验证
