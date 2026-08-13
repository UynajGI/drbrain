# DrBrain × LlamaIndex 集成设计文档

> 日期:2026-08-12
> 分支:`feat/llamaindex-upgrade-research`
> 决策(用户 2026-08-12 定案):**直接引入 LlamaIndex 依赖,替换同质实现;保留 drbrain 独有设计(树/图),LlamaIndex 已有的(agent/评估/召回/重排)直接用他的**
> 前置调研:`llamaindex-tutorial-survey.md`(教程能力)、`drbrain-rag-current-state.md`(现状基线 10 薄弱点)、`llamaindex-upgrade-feasibility.md`(早期"不引框架"方案,已被本决策取代)
> 环境基线:Python 3.12.3 · pydantic 2.12.5 · numpy 2.2.6 · torch 2.10.0 · sentence-transformers 5.6.1 · umap-learn 0.5.12(已具备,llama-index 零依赖)
> **后续演进**:PR #8 已合并;PR #9(2026-08-13)叠加 Epistemic Layer,SQLite schema 已从 v10 推到 **v13**(concepts provenance/authority/validity、knowledge_snapshots、answer_records)。本文的 "schema v10" 是设计时快照。

---

## 1. 目标与边界

### 目标
把 drbrain 的 RAG 检索/合成/Agent/评估层统一迁移到 LlamaIndex 生态,消除 10 项薄弱点中的可替换项,同时让 LlamaIndex 检索不到 drbrain 独有的树/图资产。

### 终态(用户 2026-08-12 定案:LlamaIndex 全面接管)
- **LlamaIndex 能直接用的功能,一律直接用,不做自研**:检索(BM25/向量/融合/重排)、查询引擎、响应合成、Agent 编排、评估、流式、结构化输出——全部由 LlamaIndex 接管。
- **drbrain 只保留"理念实现"(LlamaIndex 没有或更弱的)**:
  - PageIndex 文档树(tree.json + raw.md 按行号加载)
  - RAPTOR 层级摘要树(tree_summaries L1-L3)
  - 知识图谱引擎(TransE 嵌入 + 规则闭包 + 概念去重 + 8 图工具)
  - SQLite 存储层(schema v10,数据真相源)
  - LLM fallback 链(KeyRotator + ApiCache + metrics,经 litellm 桥接保留)
- **新旧保留 = 迁移期过渡策略,不是长期双轨**:迁移期间旧实现保留供对比/回退;终态(T9 完成后)被 LlamaIndex 替换的旧实现删除或归档,不再维护。新功能只写在 LlamaIndex 侧。

### 替换清单(用 LlamaIndex)
| 现有实现 | 薄弱点 | LlamaIndex 替换 | 终态动作 |
|---|---|---|---|
| `query/bm25.py` BM25Search(每次内存重建) | ① 无持久化 | `BM25Retriever`(持久化倒排) | **保留+deprecated**(T9 结论:`agent_tools.search_concepts` 保留工具与 legacy hybrid 仍依赖;CLI 检索已不调用) |
| `services/embedding.py:757 search_tree`(全表扫描) | ② 无 ANN | `VectorStoreIndex` + ANN 向量库(Chroma 起步) | **保留**(T9 结论:`agent_tools.search_tree` 保留工具与 tree_retrieval 树导航依赖;向量检索已走 LlamaIndex) |
| `query/fusion.py` + `tree_retrieval.py:_rrf_score` 两套 RRF | ③ 融合分散 | `QueryFusionRetriever`(RRF/加权统一) | **删除 `_rrf_score`(无调用方死代码,已删);`query/fusion.py` 保留+deprecated**(hybrid_retrieval 依赖) |
| CrossEncoder 可选 rerank(默认关) | ④ 重排默认关 | `FlagEmbeddingReranker`(bge-reranker-v2-m3)默认开 | **保留+deprecated**(T9 结论:`query/rerank.py` 仅 legacy hybrid 引用,已无 CLI 调用;真实重排走 `rag/rerank.py` Qwen3-Reranker-0.6B) |
| `ReasonerAgent` / `SessionAgent` 手写工具循环 | — | LlamaIndex `AgentWorker`(FunctionCalling),工具=drbrain 8 图工具 | 编排删除(CLI reason 默认/唯一走 LlamaIndex agent);`SessionAgent.reason_bidirectional` 与 `--workflow` 确定性管线保留;工具(agent_tools)保留 |
| ask/query 手工 prompt 拼接合成 | ⑤ 上下文管理粗糙 | `ResponseSynthesizer`(REFINE/TREE_SUMMARIZE)+ `Settings.context_window` 动态 | 删除手工合成(CLI ask legacy 分支已删) |
| 无检索质量评估 | ⑦ 无评估闭环 | `RetrieverEvaluator` + `BatchEvalRunner` + RAGAS | 新增(T7 自写指标实现) |
| 无流式输出 | ⑤ | LlamaIndex `streaming=True` | 新增(T9:DrbrainLLM 真逐 token) |
| 无引文回链 | ⑧ | ResponseSynthesizer 来源标注 + structured output | 新增 |
| `fsearch` 本地 SQL LIKE 腿 | ⑨ | `KeywordTableIndex`/BM25 统一 | 替换(fsearch 命令保留,未迁移) |
| 无结构化输出(已有 parse_llm_json) | — | `PydanticOutputParser`(可并存,保留 llm_json) | 并存 |

> **T9 终态清理汇总(2026-08-12)**:CLI `ask/query/hybrid/reason` 的 `--engine legacy` 分支全部移除,默认/唯一走 LlamaIndex;`llamaindex.enabled=false` 时打印 warning + 提示开启(不再回退 legacy)。删除:CLI legacy 分支(~700 行)、`tree_retrieval._rrf_score` 死代码、legacy 测试文件(test_query/test_hybrid_cmd/test_ask_hybrid)。保留+deprecated:query 层五模块(bm25/fusion/hybrid_retrieval/rerank/query_transform,理由见上表);保留:tree_retrieval(DrbrainTreeRetriever/RAPTOR 复用)、search_tree/search_concepts 工具。全量 pytest 见 T9 记录。

### 保留清单(drbrain 独有,LlamaIndex 没有或更弱)——终态也保留
- **PageIndex 文档树**(`parser/pageindex/` tree.json + raw.md 按行号加载)——LlamaIndex TreeIndex 更简化,保留;包装为自定义 Retriever
- **RAPTOR 树**(`extractor/raptor.py` tree_summaries L1-L3)——保留;包装为自定义 Retriever
- **知识图谱引擎**(TransE `graph/engine_embeddings.py`、规则闭包 closure、`extractor/agent_tools.py` 8 工具)——LlamaIndex GraphIndex 教学级,保留;工具直接喂给 LlamaIndex Agent
- **SQLite 存储层**(schema v10,2303 测试基线)——不动,作为数据真相源
- **LLM fallback 链**(`extractor/llm_client.py` KeyRotator + 多模型 fallback + ApiCache + metrics)——通过 `llama-index-llms-litellm` 桥接保留

### 明确不做的
- 不迁移 tree_vectors 到 LlamaIndex 的 RAPTOR(保留 drbrain RAPTOR 实现,只换底层向量库)
- 不引入多模态 RAG(与学术场景弱相关)
- 不删除现有代码——替换以"新模块接入 + 命令行内部切换"方式渐进,旧实现保留可回退

---

## 2. 架构

```
┌────────────────────────────────────────────────────────────┐
│ CLI 层(不变:ask/query/hybrid/fsearch/reason/...)           │
└──────────────┬─────────────────────────────────────────────┘
               ▼
┌────────────────────────────────────────────────────────────┐
│ RAG 层(新增 src/drbrain/rag/,LlamaIndex 编排)              │
│  QueryEngine 装配:RetrieverQueryEngine / Router / SubQuestion│
│  合成:ResponseSynthesizer(REFINE)+ 来源标注 + streaming     │
│  Agent:AgentWorker(FunctionCalling,工具=drbrain 8 图工具)   │
└──────┬─────────────────┬──────────────────┬────────────────┘
       ▼                 ▼                  ▼
┌──────────────┐  ┌────────────────┐  ┌────────────────────┐
│ Retriever 层 │  │ 自定义 Retriever│  │ 评估层(新增)        │
│ LlamaIndex:  │  │ (drbrain 资产)  │  │ RetrieverEvaluator │
│ BM25Retriever│  │ DrbrainTree    │  │ (MRR/HitRate)       │
│ VectorIndex  │  │ DrbrainRAPTOR  │  │ RAGAS(faithfulness) │
│ Fusion(RRF)  │  │ DrbrainGraph   │  │ golden set 三集      │
│ Reranker     │  │ (agent_tools)  │  │ BatchEvalRunner     │
└──────┬───────┘  └───────┬────────┘  └────────────────────┘
       ▼                  ▼
┌────────────────────────────────────────────────────────────┐
│ 索引层:LlamaIndex StorageContext                          │
│  VectorStoreIndex(Chroma/内存) ← tree_vectors 迁移         │
│  BM25 倒排持久化                                          │
│  ──────────── 保留 SQLite(drbrain.db 真相源)────────────   │
│  tree.json/raw.md · tree_summaries · concepts/arguments/   │
│  edges/embeddings(TransE)                                  │
└──────┬─────────────────────────────────────────────────────┘
       ▼
┌────────────────────────────────────────────────────────────┐
│ LLM 层:llama-index-llms-litellm ← drbrain llm.models 列表  │
│  (多模型 fallback + KeyRotator + ApiCache + metrics 保留)  │
└────────────────────────────────────────────────────────────┘
```

### 模块划分(新增 `src/drbrain/rag/`)
| 文件 | 职责 |
|---|---|
| `rag/indexer.py` | tree.json/raw.md → LlamaIndex Document/Node;VectorStoreIndex + BM25 索引构建/增量/持久化 |
| `rag/retrievers.py` | 统一 `BaseRetriever` 接口;包装 drbrain 树/RAPTOR/图为自定义 Retriever |
| `rag/fusion.py` | `QueryFusionRetriever` 装配(BM25+向量+树/图可选),RRF/加权 |
| `rag/engine.py` | `RetrieverQueryEngine` / `RouterQueryEngine` / `SubQuestionQueryEngine` 装配 + ResponseSynthesizer + streaming |
| `rag/agent.py` | `AgentWorker` 替换 ReasonerAgent 编排(工具=agent_tools 8 工具 + 检索工具),保留 session 持久化 |
| `rag/llm.py` | drbrain `llm.models` → LlamaIndex Settings.llm(litellm 桥接,保留 fallback/缓存/metrics) |
| `rag/rerank.py` | `FlagEmbeddingReranker`(bge-reranker-v2-m3)默认开 + 后处理(阈值/去重) |
| `rag/eval.py` | golden set 加载、`RetrieverEvaluator`(MRR/HitRate@K)、RAGAS 生成质量、三集防泄露 |
| `rag/config.py` | `llamaindex:` 配置段读取 |

---

## 3. 配置(新增 config.yaml 段 + config.local.yaml 覆盖)

```yaml
llamaindex:
  enabled: true            # false = 回退旧实现
  llm: "litellm"           # 桥接 drbrain models 列表
  vector_store: "chroma"   # chroma | memory(起步)
  storage_dir: "data/llamaindex"   # 索引持久化目录
  retrievers: ["bm25", "vector"]    # fusion 腿
  fusion_mode: "reciprocal_rank"    # | relative_score
  rerank: true
  rerank_model: "BAAI/bge-reranker-v2-m3"
  rerank_top_k: 20
  similarity_cutoff: 0.7   # SimilarityPostProcessor 阈值
  streaming: true
  eval:
    golden_set: "data/llamaindex/golden.jsonl"
    split: ["dev", "val", "test"]   # 三集防泄露
```

---

## 4. 关键设计决策

### 4.1 LLM 桥接(保留 drbrain fallback 链)
- 用 `llama-index-llms-litellm` 的 `LiteLLM` 类,`model` 传 drbrain 的 models 列表形态需验证;若不支持多模型 fallback,则自定义 `BaseLLM` 包装 `acall_text_with_fallback`。
- **保留**:ApiCache(sha256 缓存)、metrics 记录、KeyRotator 轮换——在自定义桥接层调用,不丢。
- **验收**:同一 prompt 走 LlamaIndex 与走原 llm_client 结果一致且命中缓存。

### 4.2 树/RAPTOR 的 Node 映射
- PageIndex 节点 → `TextNode(node_id=<tree node_id>, metadata={paper_id, line_start, line_end, title}, text=<raw.md 行范围正文>)`;正文不落 Node(与 `if_add_node_text=False` 一致),查询时按需加载。
- RAPTOR L1-L3 摘要节点 → `IndexNode(embedding=摘要向量, text=摘要)` + 父链接 `source_node_ids` 保留在 metadata,供 `tree_traversal_search` 语义下钻。
- 自定义 `DrbrainTreeRetriever` 直接读 tree.json(与现 `query_by_structure_hybrid` 同逻辑),**不经过** LlamaIndex 索引——树导航本质是 LLM 结构化选择,非向量检索。

### 4.3 向量索引迁移
- `tree_vectors` 表保留为真相源;**迁移构建** `VectorStoreIndex`(Chroma 或内存)时读取该表嵌入,`_content_hash` 幂等增量。
- 查询走 ANN(index.as_retriever(similarity_top_k))替代全表 numpy 矩阵乘;`top_k` 行为对齐。
- **验收**:同一 query,新检索 top-k 与 `search_tree` 旧结果重合率 ≥ 90%(余弦一致)。

### 4.4 Agent 替换
- `ReasonerAgent.reason` / `SessionAgent.ask` 编排换 `AgentWorker`(tools=8 图工具 + search_hybrid),system prompt 保留 closure_context 注入。
- session 持久化(`agent_sessions/agent_messages`)保留:在工具调用回调层写入,不依赖 LlamaIndex chat memory。**T9 补读恢复**:`reason_llamaindex` 续会话时经 `rag.agent.load_session_history` 从表加载历史注入 `chat_history`(压缩逻辑与 SessionAgent._maybe_compress 一致)。
- `--workflow` / `--bidirectional` 分支保留(确定性管线不替换)。
- **T9 kg_validate 决策(2026-08-12):加入为第 8 个工具**。`kg_validate`(KG 一致性校验:TBox/RBox 违例 + debates/gaps 图模式)对 `reason --engine llamaindex` 有语义价值——agent 可在推理中自校验假设(镜像 SessionAgent.reason_bidirectional 的 propose→validate→revise 循环)。实现:`rag/agent._make_validate_tool` 直接包装 `agent_tools.kg_validate` 为 FunctionTool,**不修改** `TOOL_DEFINITIONS`/`TOOL_HANDLERS`(legacy ReasonerAgent/SessionAgent 的 canonical tool spec 保持逐字节不变,T6 不变量);仅当 `graph` 可用时注册(无图时维持 7 工具)。

### 4.5 评估
- golden set 从 test-run 100 篇语料构造(30-50 条 query → 人工/半自动标注相关段落),存 `data/llamaindex/golden.jsonl`。
- `RetrieverEvaluator(metric=hit_rate|mrr)` + `BatchEvalRunner` 出基线;RAGAS 4 指标评估 ask 输出。
- dev/val/test 三集分离,test 集全程只用一次;结果落 `docs/llamaindex-eval-baseline.md`。

### 4.6 渐进迁移与终态清理
- **迁移期(现在→T9 前)**:每个命令(`ask`/`hybrid`/`query`)加 `--engine llamaindex|legacy` 选项(默认 llamaindex),`llamaindex.enabled=false` 全局回退。旧实现保留,供对比与回退,同时保住 2303 测试。
- **终态(T9 完成)**:旧实现删除/归档(按 §1 替换清单"终态动作"列),CLI 不再暴露 `--engine legacy`;新功能只写在 LlamaIndex 侧。删除前先跑全量测试确认 LlamaIndex 路径覆盖等价场景。

---

## 5. 依赖(pyproject 新增)

```toml
# 核心(必装)
llama-index-core          # 核心编排
llama-index-llms-litellm  # LLM 桥接(若验证可行)
# 向量库(起步)
chromadb                  # 嵌入式向量库(或先 memory)
# 重排(可选 extra,但本方案默认开 → 必装)
llama-index-postprocessor-flag-embedding-reranker
# 评估(可选 extra:evals)
ragas
# 检索器
llama-index-retrievers-bm25  # 若 core 未内置
```

> ⚠️ 以 `uv add` 实际解析为准;llama-index 包体积大,装完需验证 import 耗时与 pydantic 2.12 兼容性。

---

## 6. 工单分解(foreman 派发)

| 工单 | 内容 | 依赖 | 验收 |
|---|---|---|---|
| **T1 基础设施** | `uv add` 依赖;config `llamaindex:` 段;`src/drbrain/rag/` 骨架;Settings 初始化;最小 smoke(import + 一个 retriever) | 无 | `uv run python -c "import llama_index"` OK;config 解析 OK;新增 test_rag_smoke |
| **T2 LLM 桥接** | `rag/llm.py`:drbrain models → LlamaIndex LLM(fallback/缓存/metrics 保留);单测 | T1 | ask 场景经 LlamaIndex 调 opencode key 成功且命中缓存 |
| **T3 索引层** | `rag/indexer.py`:tree.json → Nodes;VectorStoreIndex(Chroma/memory)构建+增量;BM25 索引;`drbrain rag index` 子命令 | T1 | 索引构建跑通;query top-k 与旧 search_tree 重合 ≥90% |
| **T4 检索器统一** | `rag/retrievers.py` + `rag/fusion.py`:BaseRetriever 包装(树/RAPTOR/图)+ QueryFusionRetriever;删除两套 RRF 的接入点 | T2,T3 | hybrid 经 Fusion 输出含 metadata 来源标注;单测 |
| **T5 查询引擎** | `rag/engine.py`:RetrieverQueryEngine + ResponseSynthesizer(REFINE)+ streaming + 来源标注;ask/hybrid/query 接 `--engine` | T4 | ask 流式输出+结构化来源;与旧实现结果可比 |
| **T6 Agent 替换** | `rag/agent.py`:AgentWorker 编排,工具=8 图工具;session 持久化保留;reason 接 `--engine` | T2,T4 | reason 问答经 LlamaIndex Agent 成功,工具轨迹可查 |
| **T7 评估体系** | `rag/eval.py`:golden set 构建(30-50 条)+ RetrieverEvaluator + RAGAS + 三集;基线落盘报告 | T3,T5 | 基线数字(MRR/HitRate/faithfulness)落盘 docs/llamaindex-eval-baseline.md |
| **T8 重排+后处理** | `rag/rerank.py`:重排默认开 + SimilarityPostProcessor + 去重;A/B 对比(模型初定 bge-reranker-v2-m3,**T9 用户决策改为 Qwen/Qwen3-Reranker-0.6B**) | T4 | rerank 前后 MRR 对比数字 |
| **T9 回归与终态收尾** | 全量测试(2303)保持绿;新测试覆盖;README/progress-log 登记;报告合并;**按 §1 终态动作列删除/归档被 LlamaIndex 替换的旧实现,移除 `--engine legacy` 路径,只留理念实现** | T2-T8 | `uv run pytest` 全绿;旧实现清理完成;索引登记完成 |

> **T9 完成(2026-08-12)**:见 `llamaindex-integration-progress.md` §T9。验收要点:大节点切分(4000 token 默认)GPU 全语料索引无 OOM(100 论文/947 节点/271s);DrbrainLLM 四流式接口真逐 token(逐 chunk 非空、首 chunk 到达、fallback/缓存保留);session 读恢复+压缩单测;kg_validate 第 8 工具;真实 rerank A/B(MRR@10 0.9417→0.9667,Qwen3-Reranker-0.6B);CLI `--engine legacy` 全移除;全量 pytest 绿(见进度文档)。

### 派发顺序
```
T1(串行,阻塞全部)→ [T2 ∥ T3](并行)→ T4 → [T5 ∥ T6](并行)→ [T7 ∥ T8](并行)→ T9
```

---

## 7. 风险

| 风险 | 等级 | 缓解 |
|---|---|---|
| llama-index 依赖重(pydantic/包体积) | 中 | 装完验证;只引 core + 必要插件 |
| litellm 桥接不支持多模型 fallback | 中 | 自定义 BaseLLM 包装 llm_client(保留全部能力) |
| VectorStoreIndex 与 tree_vectors 结果不一致 | 中 | 余弦重合率 ≥90% 验收门槛 |
| Agent 替换破坏 reason 行为 | 中 | `--engine legacy` 回退;工具轨迹对比测试 |
| 终态删旧实现导致测试失效 | 中 | T9 删除前先确认 LlamaIndex 路径覆盖等价场景;删除后全量 pytest 必须绿;被删模块测试改指新实现或随删 |
| 2303 现有测试破坏 | 低 | 新模块独立;旧实现不动(迁移期) |

---

*待办:用户确认工单拆分与优先级后,T1 先行派发。*
