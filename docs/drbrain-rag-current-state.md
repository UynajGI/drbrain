# DrBrain 现有 RAG 设计基线(组件清单与数据流)

> 目的:为"是否引入 LlamaIndex 做二次升级"提供现状基线。纯调研,未修改任何代码。
> 基线日期:2026-08-12(HEAD: `3dd7bb1`,分支 `feat/llamaindex-upgrade-research`)
> 范围:`src/drbrain/query/`、`src/drbrain/parser/pageindex/`、`src/drbrain/services/embedding.py`、`src/drbrain/extractor/raptor.py`、`extractor/reasoner.py`、`extractor/session_agent.py`、`extractor/agent_tools.py`、`extractor/llm_client.py`、`cli/query_commands.py`、`cli/analysis_commands.py`、`cli/build_commands.py`、`cli/_helpers/db_ingest.py`、`storage/database.py` 相关表。

---

## 1. 总体架构(四层)

```
                    ┌──────────────────────────────────────────────┐
                    │              CLI 命令层 (query/search/hybrid/│
                    │          fsearch/ask/reason/query --paper)  │
                    └──────────────┬───────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────────┐
        │        检索器层 (query/)                                  │
        │  BM25Search │ hybrid_search │ query_by_structure(_hybrid)│
        │  tree_traversal_search │ query_cross_paper │ agent_tools│
        └───────┬──────────────┬───────────────┬──────────────────┘
                │              │               │
        ┌───────▼─────┐ ┌──────▼──────┐ ┌──────▼─────────────────┐
        │  索引层 A    │ │  索引层 B    │ │  索引层 C(图)           │
        │  tree.json  │ │  tree_vectors│ │  embeddings (TransE)   │
        │  + raw.md   │ │  + tree_summar│ │  + concepts/arguments │
        │  (PageIndex)│ │  ies (RAPTOR)│ │  (知识图谱)             │
        └─────────────┘ └─────────────┘ └────────────────────────┘
                                   │
        ┌──────────────────────────▼──────────────────────────────┐
        │         LLM 合成层 (extractor/llm_client.py)             │
        │  call(_text|_messages)_with_fallback + litellm + 缓存    │
        └──────────────────────────────────────────────────────────┘
```

- **索引层 A(文档树)**:每篇论文磁盘上的 `tree.json`(PageIndex 结构+摘要)与 `raw.md`,正文按需按行号加载。
- **索引层 B(文本向量)**:SQLite `tree_vectors` 表(pageindex 层 + raptor_L1..L3 层)+ `tree_summaries` 表(RAPTOR 摘要)。
- **索引层 C(符号图)**:SQLite `concepts/arguments/edges/embeddings` 表 + NetworkX 图引擎(TransE 向量、图遍历、closure)。
- 注意:没有独立的 BM25 **持久化**索引——BM25 每次查询都在内存里从 SQLite 重建(见 §2)。

---

## 2. 检索器清单

| # | 检索器 | 文件:行号 | 输入 | 输出 | 索引数据来源 | 是否持久索引 |
|---|--------|-----------|------|------|--------------|--------------|
| 1 | `BM25Search.search` / `build_bm25_index` | `query/bm25.py:57` / `:94` | 查询串 + type/arg_type/min_confidence 过滤 | `list[dict]`(local_id/type/label/text/score) | DB `papers`(标题+摘要+status)、`concepts`(label+confidence+year)、`arguments`(claim+confidence+year);文档 key = `local_id` | 否——每次调用内存重建(rank_bm25 BM25Okapi,k1=1.5,b=0.75) |
| 2 | `search_tree`(collapsed 向量检索) | `services/embedding.py:757` | query + db_path + top_k + EmbedConfig + 可选 paper_id | `list[{node_id,paper_id,score,tree_layer}]` | `tree_vectors` 全表余弦(numpy 矩阵乘,向量化) | 是(`tree_vectors` 表,float32 BLOB) |
| 3 | `hybrid_search`(BM25+向量 RRF) | `query/hybrid_retrieval.py:39` | query + db + db_path + embed_cfg + top_k + 可选 rerank | `list[SearchHit]`(论文级,source="fused",metadata 记录各源贡献) | BM25 腿=SQLite 重建;向量腿=`search_tree`;均折叠到论文级 | BM25 否 / 向量是 |
| 4 | `query_by_structure`(PageIndex LLM 树导航) | `query/tree_retrieval.py:219` | question + paper_dir + models + max_rounds | `list[{node_id,title,content}]` 或 None | `tree.json` skeleton + `raw.md` 正文(按行号加载) | 是(tree.json 磁盘) |
| 5 | `query_by_structure_hybrid`(LLM 主 + 向量预过滤) | `query/tree_retrieval.py:762` | question + paper_dir + db_path + models + embed_cfg | `list[{node_id,title,content,source}]`(source∈llm/vector/llm+vector) | tree.json + `search_tree(paper_id 限定)` | 同上 |
| 6 | `tree_traversal_search`(RAPTOR 两阶段) | `query/tree_retrieval.py:495` | query + db_path + top_k + min_results | `list[{node_id,paper_id,score,tree_layer}]` | `tree_vectors` 按层(raptor_L1..L3→pageindex)+ `tree_summaries.source_node_ids` 父子关系 | 是 | 
| 7 | `query_cross_paper`(跨论文 collapsed) | `query/tree_retrieval.py:462` | query + db_path + cfg + 可选 paper_ids | 同 #2 | 同 #2 | 是 |
| 8 | 图工具(agent 检索):`search_concepts` / `get_neighbors` / `find_path` / `get_document_structure` / `get_section_content` / `search_tree` / `get_raptor_summaries` | `extractor/agent_tools.py:141/152/168/182/212/234/243` | 工具参数(LLM 生成) | 结构化 JSON 回填给 LLM | concepts+BM25 / GraphEngine.traverse / nx.shortest_path / tree.json / raw.md / tree_vectors / tree_summaries | 混合 |
| 9 | `search_local`(fsearch 本地腿) | `services/fsearch.py:91` | query | `list[dict]`(title/local_id/year/authors) | **SQL LIKE 扫描**(`%query%` over papers.title/concepts.label/arguments.label,按年份排序)——非 BM25 | 否 |
| 10 | `search_arxiv`(fsearch 外部腿) | `services/fsearch.py:37` | query | arXiv Atom 元数据 | 外部 arXiv API | 否 |
| 11 | 图嵌入查询 `query_embed`(project/intersect/union/negate) | `graph/query_embeddings.py:155` | DSL query dict | `list[{label,score}]` | `embeddings` 表(TransE,实体+`__rel__` 关系) | 是(表中向量) |

补充:同一库内存在**两份 RRF 实现**——`query/fusion.py:26` 的 `reciprocal_rank_fusion`(SearchHit 级,带 metadata)与 `query/tree_retrieval.py:449` 的私有 `_rrf_score`(仅 id 列表;fusion.py 文档串自述此差异)。另有 `_hybrid_score`(`tree_retrieval.py:396`,BM25+向量加权和)目前**无生产调用**。

---

## 3. 索引层:构建、存储、更新

### 3.1 PageIndex 文档树(tree.json + raw.md)

| 项 | 内容 |
|----|------|
| 构建 | ingest 管线 Stage 3:`cli/_helpers/db_ingest.py:217-234` 调 `md_to_tree(raw.md, TreeConfig(...))` 写入 `papers/<id>/tree.json`;`build_cmd` 在 tree.json 缺失时重试(`build_commands.py:231-242`) |
| 构建器 | `parser/pageindex/builder.py:311` `md_to_tree`;流程:Markdown 标题提取 → 节点文本抽取 → 树形化(`_build_tree_from_nodes`,node_id 四位数) → 大节点按段落拆分(`_recursive_split_large_nodes`,max_node_tokens=10000) → LLM 逐节点摘要(`summary.py _generate_summaries_for_structure_md`,阈值 200 token,叶节点存 `summary`、非叶存 `prefix_summary`) → 文档级描述 |
| 配置 | `TreeConfig`(`builder.py:19`):if_thinning=True/min_token_threshold=5000/if_add_node_text=False(正文不落地,按需加载)/max_node_tokens=10000 |
| 存储 | `papers/<local_id>/tree.json`(`storage/paths.py:18 tree_json_path`)+ `raw.md`(`raw_md_path`)。结构与正文分离:正文在 `raw.md`,tree.json 只含 title/node_id/line_num/summary |
| 更新 | 重新 ingest 或 build 时重建;验证器 `parser/pageindex/validation.py:421 md_to_tree_with_fallback` 提供容错 |

### 3.2 文本向量索引(tree_vectors + tree_summaries)

| 项 | 内容 |
|----|------|
| Schema | `storage/database.py:87-100`:`tree_vectors(node_id PK, paper_id, embedding BLOB float32, content_hash, tree_layer)`;`tree_summaries(node_id PK, paper_id, summary_text, source_node_ids JSON, tree_layer int)`;索引 `idx_tree_vectors_paper`、`idx_tree_vectors_layer_paper`(L129-130)、`idx_tree_summaries_paper`(L131) |
| PageIndex 层构建 | `services/embedding.py:645 build_tree_vectors`:读 tree.json 收集全部节点文本(`_collect_tree_nodes`,标题+raw.md 行范围正文)→ `_embed_batch` 批量嵌入 → INSERT OR REPLACE,`tree_layer='pageindex'` |
| RAPTOR 层构建 | `extractor/raptor.py:176 build_raptor_tree`:从 tree_vectors 取 pageindex 向量 → UMAP(10 维)→ GMM+BIC 自动 k 聚类(≤10 簇)→ LLM 每簇 2-4 句摘要 → 写 `tree_summaries`(source_node_ids 记录子节点)+ 摘要向量写回 `tree_vectors(tree_layer='raptor_L{n}')` → 递归,最多 3 层(`_RAPTOR_MAX_LAYERS=3`) |
| 入口 | `drbrain embed --tree`(`build_commands.py:365`)→ `build_paper_tree_vectors`(`embedding.py:716`);`ingest pipeline` 预设含 embed 步骤(`ingest_commands.py:947`) |
| 增量更新 | 内容哈希(`_content_hash` sha256[:16],`embedding.py:50`)比对,未变节点跳过;论文删除时 `database.py:1283-1284` 级联删两表 |
| 查询 | `search_tree`(`embedding.py:757`):**全表扫描 + numpy 矩阵乘** 计算余弦,无 ANN;`tree_traversal_search` 按层扫描(每层整层余弦,再按 `tree_summaries.source_node_ids` 下钻) |
| 模型 | 默认 `Qwen/Qwen3-Embedding-0.6B`(`config.py:144`、`config.yaml:61-69` provider=local,source=modelscope);GPU 自适应 batch(一次性内存剖析缓存 `~/.cache/drbrain/gpu_profile.json`);支持 openai-compat provider |

### 3.3 符号图索引(embeddings 表 + 图)

| 项 | 内容 |
|----|------|
| Schema | `storage/database.py:81-84`:`embeddings(entity PK, vec BLOB, dim)` |
| 构建 | `drbrain embed`(非 --tree 模式,`build_commands.py:423-485`):TransE 训练(dim=128,epochs=100),增量路径用 `t.train_incremental`(dirty 论文的新边),watermark `set_last_run("embed")` |
| 使用 | `graph/engine_embeddings.py` learn_embeddings/entity_embedding/predict_link/similar_entities;`graph/query_embeddings.py` 向量 DSL(project/intersect/union/negate)。注意:文本 RAG 链路(hybrid/ask)不用 TransE,TransE 主要服务图分析命令 |

### 3.4 BM25 索引(无持久化)

- `build_bm25_index`(`query/bm25.py:94`)每次从 SQLite 拉 papers/concepts/arguments 重建。`index_cmd`(`query_commands.py:210`)只做 watermark 记录(`get_last_run/set_last_run("index")` + 论文时间戳比对),**不产生可复用索引文件**。
- 库规模大时该开销线性增长;这是引入 LlamaIndex 最直接的收益点之一(持久化倒排索引)。

---

## 4. 查询链路(逐命令数据流)

### 4.1 `drbrain query <text> --paper <id>`(PageIndex 树检索,`query_commands.py:250`)
```
用户输入 text + paper_id
 → 校验 papers/<id>/tree.json 存在
 → query_by_structure_hybrid(tree_retrieval.py:762):
    ① LLM(acall_with_fallback,ROUND1_PROMPT)读去 text 的 skeleton,选 top_k*2 个候选叶 node_id
    ② 向量预过滤:search_tree(question, db, top_k*2, paper_id=paper_id) → vector_candidates
    ③ 合并:llm_selected 优先 + vector 补充,取 top_k
    ④ 按 node_id 从 raw.md 取正文(get_node_content,行号范围)
 → 输出 [{node_id,title,content,source}];CLI 每段截 500 字符 / --json 全量
```
- 纯 LLM 路径(skeleton ≤8000 字符)与自适应导航(skeleton 过大 → 分层展开,`_build_top_level_structure`/`_expand_branch`)都在 `query_by_structure`(L219);`--paper` 走的是 hybrid 变体。

### 4.2 `drbrain search <query>`(BM25,`query_commands.py:706`)
```
输入 query
 → build_bm25_index(db)(内存重建全库)
 → bm25.search(query, type_filter, limit)
 → 表格输出(label/score/paper/confidence)
```
- 纯关键词检索,无 LLM、无向量、无融合。

### 4.3 `drbrain query <text>`(BM25 + 图扩展,`query_commands.py:250` 非 --paper 路径)
```
输入 text + 过滤器
 → BM25(同上) → 年份/workspace 后过滤
 → 可选 --hybrid:PageRank 百分位乘性 boost[1.0,2.0](手写 100 轮迭代,query_commands.py:382-402)
 → 可选 --neighbors:以最高分结果为种子做 GraphEngine.traverse 图扩展(hops/relation/direction)
 → 输出带 score/_hybrid_boost/_path 的记录
```

### 4.4 `drbrain hybrid <query>`(BM25+向量 RRF,`query_commands.py:754`)
```
输入 query
 → hybrid_search(hybrid_retrieval.py:39):
    _run_bm25(db, limit=50)         → 论文级折叠(每论文取最高分, _bm25_to_hits)
    _run_embedding(db_path, limit=50)→ search_tree 后论文级折叠(保留最佳 node_id 于 payload)
    reciprocal_rank_fusion(k=60, weights=None) → 论文级 RRF 合并
    (可选)CrossEncoderReranker.rerank(query, fused, top_n)  ← --rerank 开关
 → 输出 SearchHit:paper_id / rrf score / sources / contributions
```
- 两腿各自 try/except 容错;embed.provider=none 时纯 BM25 模式。

### 4.5 `drbrain fsearch <query>`(联邦检索,`query_commands.py:599`)
```
输入 query
 → search_local:SQL LIKE %query%(papers/concepts/arguments),按年份降序 ← 非 BM25
 → (可选 --arxiv/--arxiv-only)search_arxiv:arXiv Atom API
 → _merge_with_local_status:DOI/arxiv_id 与 paper_ids 表比对,标注 ingested
 → 合并输出 local + arxiv
```

### 4.6 `drbrain ask <question>`(混合检索 + LLM 合成,`analysis_commands.py:288`)
```
输入 question
 → (可选 --hyde)ahyde_transform:LLM 生成假设文档替换查询(失败回退原文)
 → hybrid_search(top_k=5, 可选 rerank) → SearchHit 论文级
 → 上下文构建:每篇命中论文 → get_paper + get_concepts_by_paper + 1-hop 邻居(前5)
              + _build_closure_context(closure 推断边) → context 文本(截断 ≤50 行)
 → prompt:"Answer this research question using the knowledge graph context...Cite specific concepts and relations"
 → acall_text_with_fallback(max_tokens=300, temp=0.1)
 → 输出答案(附基于 N 篇论文/M 个概念);--json 含完整 context
```
- 答案=纯文本,无结构化引用/回链;上下文是"论文→概念→邻居"扁平拼接,无重排序后的段落级证据。

### 4.7 `drbrain reason <question>`(工具调用代理,`analysis_commands.py:56`)
```
输入 question
 → BM25 seed labels(5) → _build_closure_context(closure 推断边注入 system)
 → 三条路径之一:
   A. ReasonerAgent.reason(stateless,reasoner.py:70):
      litellm acompletion(tools=TOOL_DEFINITIONS[8个], temp=0.3, max_tokens=1024, max_turns=5)
      循环:LLM 返回 tool_calls → execute_tool 执行(agent_tools.py:424)→ 结果 JSON 回填 → 直至无工具调用输出答案
   B. SessionAgent.ask(session 持久化,session_agent.py:189):
      同上 + agent_sessions/agent_messages 表持久化,跨 CLI 调用恢复,
      超出 8000 token 预算触发 _maybe_compress(保留 system+末6条,中间压成纯文本 [Context summary])
   C. --workflow:reasoning/*.py 确定性多步管线(因果/矛盾/时间/假设),LLM 仅做语义判断
 → (可选 -b/--bidirectional)LLM 假设 ↔ kg_validate(TBox/RBox/模式检测,agent_tools.py:269)多轮迭代
```

### 4.8 非文本检索命令(不在 RAG 链路内)
- `evolve`/`landscape`/`frontier`/`paradigm`/`difficulty`(`analysis_commands.py:450/545/876` 等)全部走 `graph/genealogy.py` 的图分析,无 LLM 检索合成;`frontier` 输出知识前沿(活跃 gap/争论/范式转移)。

---

## 5. LLM 合成层(`extractor/llm_client.py`)

| 能力 | 现状 | 位置 |
|------|------|------|
| 调用形态 | 四类 fallback 函数:JSON 类 `call_with_fallback`/`acall_with_fallback`;文本类 `call_text_with_fallback`/`acall_text_with_fallback`;多轮+工具类 `call_with_messages`/`acall_with_messages` | `llm_client.py` |
| Fallback 链 | models 列表逐模型重试;async 路径遇 RateLimitError 先 sleep 10s | `acall_with_fallback` |
| 温度 | JSON/文本调用 0.1(确定性);agent/tool 调用 0.3 | — |
| 响应缓存 | `ApiCache`(extractor/cache.py)按 sha256(model+system+prompt+max_tokens)缓存;树检索/ask/build workflow 均传 `_cache=`;仅 temp=0 时对 messages 缓存 | `llm_client.py` `_cache_key` |
| Prompt 缓存 | Anthropic:≥4000 字符 system prompt 打 `cache_control: ephemeral`(降输入费) | `_build_litellm_kwargs` |
| Token 计量 | `_record_llm` → `metrics.py MetricsStore.record_llm`(data/metrics.db llm_calls 表) | `llm_client.py` |
| **流式输出** | **无**——grep `stream|streaming` 仅命中 PDF 下载(`services/fetch.py:180 stream=True`)与 `--jsonl` 结果流输出;所有 LLM 响应完整等待后一次性返回 | 全局 |
| 引文回链 | 无结构化回链:`ask` 仅在 prompt 中要求"Cite specific concepts and relations",输出是自由文本;`hybrid` 只输出 paper_id;`reason` 有工具调用轨迹但最终答案不强制标注来源 | `analysis_commands.py:410-425` |
| 上下文窗口管理 | `SessionAgent._maybe_compress`:估算 token = `len(content)//4`(粗糙),预算 8000;压缩=纯文本裁剪拼接(非 LLM 摘要),工具结果只留长度 | `session_agent.py:395-420` |
| 多轮记忆 | `agent_sessions`/`agent_messages` 表持久化;`ask --session` 恢复;压缩后旧上下文仅存文本摘要 | `session_agent.py:104-176` |

---

## 6. 评估与调试

| 项 | 现状 | 位置 |
|----|------|------|
| RAG 检索质量评估 | **无**——没有带标签集上的 MRR/NDCG/命中率评测,没有 golden set 机制 | — |
| RRF 参数调优工具 | `compare_fusion_params` + `kendall_tau`:离线扫描 k/权重,测排序扰动(无相关性标签) | `query/fusion.py:103-183` |
| LLM 用量指标 | `metrics.py`:llm_calls 表(model/provider/tokens_in/out/duration_ms)+ events 表;非检索质量 | `metrics.py` |
| 论文质量审计 | `audit_papers`(`services/audit.py:30`):缺 raw.md/tree.json(→L203)等规则,与检索质量无关 | `services/audit.py` |
| 提取置信度队列 | `confidence_queue` 表 + `extractor/queue.py` route_item/check_consensus(提取置信度分流,非检索) | `extractor/queue.py` |
| 组件测试 | `tests/`:test_bm25 / test_fusion_rerank_hybrid / test_hybrid_cmd / test_tree_retrieval / test_layer3_raptor / test_layer4_tree_retrieval_v2 / test_layer2_embedding / test_ask_hybrid / test_session_agent / test_reasoner / test_rerank_validation / test_fsearch / test_rrf_tuning 等 | `tests/` |

---

## 7. 薄弱点(RAG 工程视角)

1. **BM25 索引不持久化**。每次查询(含 `query`/`search`/`hybrid`/`ask`/`reason` 的种子检索)都在内存里重建全库 BM25;`index_cmd` 只是 watermark,无复用索引。库变大时检索延迟线性恶化。
2. **向量检索是 SQLite 全表扫描**。`search_tree` 每查询读出整张 `tree_vectors` 做 numpy 矩阵乘(O(N) IO+算力);`tree_traversal_search` 每层也整层扫描;609k 节点级规模(竞赛语料)下不可扩展。无 ANN/HNSW、无独立向量库。
3. **混合融合粒度与策略简单**。(a) BM25 文档粒度(local_id)==论文级,embedding 是 section 级再折叠到论文级——RRF 只融合论文排名,段落证据只在 payload 里"搭便车";(b) `rrf_weights` 参数存在但 CLI 未暴露,无基于质量的自适应加权;(c) `query_by_structure_hybrid` 的 LLM+向量合并是**集合拼接**(LLM 优先,无分数加权);(d) 存在两套 RRF 实现(`fusion.py` 与 `tree_retrieval.py:_rrf_score`)与未被调用的 `_hybrid_score`,融合逻辑不收敛于单一实现。
4. **重排序默认关闭**。跨编码器 rerank 仅 `--rerank` 显式开启,默认 Noop;依赖 sentence-transformers 可选安装,加载失败静默降级,无回退统计。
5. **无流式输出**。`ask`/`reason` 的 LLM 合成全部整段等待,长回答体验差,也无法做"边生成边显示证据"。
6. **上下文窗口管理粗糙**。`ask` 上下文硬截断 50 行;SessionAgent 压缩用 `len//4` 估 token、纯文本裁剪做"摘要",丢失中间证据细节;PageIndex skeleton 8k 字符阈值硬编码,无按模型窗口动态适配。
7. **无检索质量评估闭环**。没有评测集/指标(golden queries、MRR/NDCG),无法量化各检索器/融合参数优劣——`compare_fusion_params` 只能测相对扰动。
8. **引文/证据回链缺失**。答案与支撑段落之间无结构化绑定(无引用 id 列表、无可校验的"答案→证据"映射),证据溯源靠 prompt 口头要求。
9. **本地联邦检索腿质量低**。`fsearch` 本地腿是 SQL LIKE 子串扫描(非 BM25、无评分排序),与主检索栈不一致。
10. **检索器实现分散、无统一接口**。BM25/向量/LLM 树导航/RAPTOR traversal/图工具分散在 `query/`、`services/embedding.py`、`extractor/` 三处,入口各异;`SearchHit` 统一类型仅覆盖 BM25+embedding 融合路径,树检索与图工具各自为政。

---

## 8. 对 LlamaIndex 升级的观察(供决策参考,非结论)

- **可被 LlamaIndex 直接替代/增强**的部分:持久化 BM25 索引、向量存储与 ANN、混合检索融合(RRF)、跨编码器重排、查询变换(HyDE 已有雏形,`query/query_transform.py` 可映射到 LlamaIndex QueryTransform)。
- **DrBrain 的差异化资产需要保留**:PageIndex 文档树(tree.json)与 RAPTOR 树(tree_summaries 父子链接)是定制结构,若迁移需映射到 LlamaIndex 的 `Document/IndexNode` 图结构;知识图谱工具调用链(reasoner/session_agent + 8 个图工具)不在 LlamaIndex 核心能力内,属于 Agent 层,升级时应保持接口稳定(`TOOL_DEFINITIONS`、`execute_tool`、`SessionAgent` 的 session 持久化)。
- 报告/子库:另有 `research/reports/llamaindex-tutorial-survey.md`(既有调研文档),本报告与之互补(现状基线 vs 方案调研)。
