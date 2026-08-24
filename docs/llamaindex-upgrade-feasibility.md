# DrBrain × LlamaIndex 二次升级可行性报告

> 日期:2026-08-12
> 分支:`feat/llamaindex-upgrade-research`(基于 3dd7bb1)
> 输入:
> - 调研 A:`research/reports/llamaindex-tutorial-survey.md`(pyllm LlamaIndex 教程全章节能力清单)
> - 调研 B:`research/reports/drbrain-rag-current-state.md`(drbrain 现有 RAG 四层基线 + 10 项薄弱点)
> 结论:引入 LlamaIndex **有价值但需克制**——不替换 drbrain 核心,而是用其补三层短板(检索质量层/评估层/可观测层),并以"统一检索器接口"方式渐进接入。

---

## 1. 结论先行

**推荐:有条件引入(渐进式,Phase 1 只做评估与检索增强,不重写现有管线)。**

| 维度 | 判断 | 依据 |
|---|---|---|
| 是否替换 drbrain 图引擎 | ❌ 不替换 | drbrain 图能力(TransE 嵌入 + 规则闭包 + 概念去重)远超 LlamaIndex 教学级 GraphIndex;引入反而降级 |
| 是否替换 PageIndex/RAPTOR | ❌ 不替换 | tree.json + RAPTOR L1-L3 是定制结构,教程 TreeIndex/SummaryIndex 更简化;迁移成本高、收益低 |
| 是否替换 ReasonerAgent/Agent 层 | ❌ 不替换 | 8 工具调用链 + session 持久化不在 LlamaIndex 核心能力内 |
| 是否引入评估体系 | ✅ 强烈建议 | drbrain 最大空白:无 golden set、无 MRR/NDCG、无 RAGAS |
| 是否引入检索增强组件 | ✅ 建议 | 重排序(bge-reranker-v2-m3)、检索后处理(阈值+去重)、融合算法(RRF 统一) |
| 是否引入持久化 BM25/ANN | ✅ 建议(渐进) | 现有 BM25 每次查询内存重建、向量检索全表扫描,规模不可扩展 |
| 是否引入查询编排 | ⚠️ 可选 | Router/SubQuestion 模式可迁移思路,不必引入框架本体 |

**核心原则**:LlamaIndex 是"RAG 深度优化专家"——我们要的是它的**评估工具、重排器、融合算法、检索器抽象**,而不是它的索引体系。

---

## 2. 能力映射(教程能力 vs drbrain 现状)

### 2.1 检索质量层

| 教程能力 | drbrain 现状 | 差距 | 引入方式 | 改造量 |
|---|---|---|---|---|
| QueryFusionRetriever RRF/加权融合 | 有 2 套 RRF 实现(`query/fusion.py` + `tree_retrieval.py:_rrf_score`)+ 1 个无调用的 `_hybrid_score` | 融合逻辑分散、无 CLI 暴露权重、无自适应加权 | 收敛到单一融合器(可移植教程 RRF 公式),CLI 暴露 `rrf_weights` | 低(不引框架,抄算法) |
| 重排序(粗排→精排) | `--rerank` 显式开启 CrossEncoder,默认 Noop | 默认关闭、可选依赖静默降级 | 引入 `FlagEmbeddingReranker(bge-reranker-v2-m3)` 作默认精排,挂 ask/hybrid | 低(独立组件) |
| HyDE / 查询变换 | 已有 `query/query_transform.py` ahyde_transform(ask --hyde) | 雏形存在,未覆盖 Multi-Query/Decompose | 补齐 Multi-Query(学术查询多角度检索) | 低 |
| 检索后处理(阈值/去重) | 无 | BM25+向量混排无去重 | SimilarityPostProcessor 思路移植(阈值+去重) | 低 |
| KeywordTableIndex 关键词倒排 | fsearch 本地腿是 SQL LIKE 扫描(非 BM25) | 无独立关键词倒排 | 材料名/化学式精确检索直接受益 | 中 |

### 2.2 评估与可观测层(最大空白)

| 教程能力 | drbrain 现状 | 引入方式 | 优先级 |
|---|---|---|---|
| RetrieverEvaluator(hit_rate/MRR)+ BatchEvalRunner | 无任何检索质量评估 | 建 golden set(测试集 100 篇可作候选) + MRR/NDCG 脚本 | 🔥 最高 |
| RAGAS 4 指标(faithfulness/relevancy/context_precision/correctness) | 无生成质量评估 | `uv add ragas` 或自实现 4 指标 prompt | 🔥 高 |
| 三套数据集(dev/val/test)防泄露 | 无 | 检索评估纪律,test 集全程只用一次 | 高 |
| Callback 可观测性(query/retrieve/postprocess/synthesize 事件) | `metrics.py` 只记 LLM 用量 | 检索链路耗时埋点(可自实现,不必引 LlamaIndex) | 中 |
| 评估 CI 化 | 无 | 纳入 audit skill / pytest 回归 | 中 |

### 2.3 查询编排层(可选)

| 教程能力 | drbrain 现状 | 判断 |
|---|---|---|
| RetrieverQueryEngine(自定义 BaseRetriever 接口) | 检索器分散在 query//services//extractor/ 三处,无统一接口(基线 §7.10) | 值得借鉴"统一 `retrieve(query)->List[NodeWithScore]` 接口"思想,把 BM25/向量/树检索包装成统一入口,不引框架 |
| RouterQueryEngine | ReasonerAgent 8 工具由 LLM 自由选 | 教程"工具描述+正反示例+路由分布监控"三步法可改进 ReasonerAgent 工具选择 prompt |
| SubQuestionQueryEngine 问题分解 | 无 | 跨论文比较类问题("A 与 B 方法差异")可借鉴,代价 3-5× 延迟 |
| ChatEngine 多轮 | SessionAgent 已有(agent_sessions 持久化 + 压缩) | 无需引入;`len//4` token 估算可换 LlamaIndex 的 token 计数 |
| 流式/异步输出 | **无流式** | 独立补齐(不依赖 LlamaIndex) |

### 2.4 响应合成层

| 教程能力 | drbrain 现状 | 判断 |
|---|---|---|
| 4 种合成模式(REFINE/TREE_SUMMARIZE 等) | ask 是"扁平拼接 context → 一次 LLM";reason 是工具循环 | REFINE(逐 Node 精炼)值得参考,但可用 prompt 工程实现,不必引框架 |
| 自定义 Synthesizer 带来源标注 | 答案纯文本,无结构化引文回链(基线 §5) | **高价值**:补齐"答案→证据段落"结构化绑定(引用 id 列表) |
| PydanticOutputParser 结构化输出 | 已有 `utils/llm_json.py parse_llm_json`(架构升级引入) | 已有对应物,无需引入 |

### 2.5 数据接入/解析层

| 教程能力 | drbrain 现状 | 判断 |
|---|---|---|
| PDF 解析 Marker 引擎 | MinerU(已有,公式/表格/OCR) | Marker 作对照/备用引擎,可选 |
| 层级化解析(标题感知) | PageIndex 文档树(更精细) | 无需引入 |
| SQLite/DB 连接器 | 自有 SQLite 存储 | 无需引入 |

---

## 3. 集成方案(推荐路径)

### Phase 0:基线冻结(1-2 天)
- 在 `feat/llamaindex-upgrade-research` 分支上完成现状基线与教程调研(本文档即产物)
- 用 test-run 100 篇材料学语料跑通现有管线,留一份"升级前检索质量快照"(当前无评估,先人工标注 20-30 条 golden query)

### Phase 1:评估先行(收益最大,风险最低)
- **建 golden set**:从 test-run 100 篇抽 30-50 条 query → 人工/半自动标注相关论文段落(golden 答案)
- **写评估脚本**(不依赖 LlamaIndex):`retrieval_eval.py` 算 MRR/HitRate@K(BM25 / 向量 / hybrid 三基线)
- **生成质量评估**:faithfulness/relevancy prompt 脚本(或引入 ragas 库)
- 产出:检索质量基线数字,量化"升级前 vs 升级后"

### Phase 2:检索增强(独立组件,可逐个落地)
- 重排序默认开启(bge-reranker-v2-m3,本地开源,英文论文语料匹配)
- 融合器收敛:删 `_rrf_score` 重复实现,统一 `query/fusion.py`;CLI 暴露 `rrf_weights`
- 检索后处理:BM25+向量结果去重 + 相似度阈值
- 可选:持久化 BM25(引入 `rank_bm25` 序列化或直接评估后再定)

### Phase 3:引文回链(高用户价值)
- ask 输出带结构化来源:`[{paper_id, node_id, quote, score}]`;`--json` 模式已部分具备,补齐非 JSON 模式展示
- 对齐教程"引用构建器"与 drbrain citation-tracking skill

### Phase 4:评估 CI 化(可选)
- golden set + 评估脚本纳入 pytest/CI,检索改动必跑回归

**不建议做的**:迁移 PageIndex/RAPTOR/图引擎到 LlamaIndex 索引、引入 GraphIndex、多模态 RAG。

---

## 4. 风险与注意

| 风险 | 等级 | 缓解 |
|---|---|---|
| LlamaIndex 依赖膨胀(核心+向量库+评估库) | 中 | 用 `llama-index-core` 最小依赖;大部分能力可"抄算法不引框架" |
| 双索引体系并存(现有 tree_vectors vs LlamaIndex 向量库) | 中 | Phase 1-2 不建新索引,只用独立组件(重排/评估/融合);持久化 BM25/ANN 单独决策 |
| 版本锁定/API 变化 | 低-中 | 评估脚本独立实现则不依赖其 API;重排器用 sentence-transformers 直连 |
| 评估集泄露 | 低 | 遵循 dev/val/test 三集纪律 |
| 与现有 2303 测试冲突 | 低 | 新增能力独立模块 + 独立测试;不动现有检索函数签名(除非收敛融合器) |
| 升级前无检索质量基线,难以证明收益 | 中 | Phase 1 先出基线数字再谈升级 |

---

## 5. 待用户决策

1. **引入深度**:A) 只做 Phase 1 评估(不引 LlamaIndex,自实现);B) Phase 1+2 评估+检索增强(推荐起点);C) 完整 Phase 1-3。
2. **依赖策略**:引 `llama-index-core` 框架,还是"抄算法不引框架"?(本报告倾向后者,框架只用来学思路)
3. **优先级确认**:检索质量评估是否确认为第一优先级?(调研 B 结论:这是 drbrain 最大空白)
4. 与 test-run 测试管线任务的关系:是否在 Phase 1 复用 100 篇语料做 golden set?

---

*关联文档:`research/reports/llamaindex-tutorial-survey.md`、`research/reports/drbrain-rag-current-state.md`*
