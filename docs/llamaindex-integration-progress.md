# LlamaIndex 集成进度记录

> 追加式记录:每个工单完成后追加一节。设计文档:`llamaindex-integration-design.md`。
> 分支:`feat/llamaindex-upgrade-research`(用户决策 2026-08-12:直接引入 LlamaIndex 依赖,替换同质实现,保留 drbrain 独有资产)。

---

## T1 基础设施(2026-08-12 完成)

### 依赖版本(pyproject 已更新)
| 包 | 版本 | 说明 |
|---|---|---|
| llama-index-core | **0.14.23** | 核心编排(`llama_index.core`) |
| llama-index-llms-litellm | **0.7.1** | LLM 桥接插件(`llama_index.llms.litellm.LiteLLM` 可用) |
| llama-index-workflows | 2.23.0 | core 传递依赖 |
| llama-index-instrumentation | 0.5.0 | core 传递依赖 |
| litellm | 1.83.10(未变) | 无版本冲突,未被升降级 |

- chromadb **未装**(工单标记"可选起步";T3 建向量索引前再决定 memory/chroma)。config.yaml 的 `vector_store` 起步设为 `memory`。
- 重排(`llama-index-postprocessor-flag-embedding-reranker`)/评估(`ragas`)插件按工单要求**未加**,留到 T8/T7。
- `uv add` 一次成功,无依赖解析冲突。

### 版本记录注意
`import llama_index` 成功,但 **llama_index 0.14.x 顶层包无 `__version__` 属性**(顶层是空命名空间,各插件走 `llama_index.core` / `llama_index.llms.*` 子模块)。验收命令 `print(llama_index.__version__)` 会报 AttributeError——改用 `importlib.metadata.version("llama-index-core")`(0.14.23)或 `llama_index.core.__version__` 记录版本。属版本形态变化,非错误。

### config 改动摘要
- `src/drbrain/config.py`:
  - 新增 `LlamaIndexEvalConfig`(golden_set/split)与 `LlamaIndexConfig`(enabled/llm/vector_store/storage_dir/retrievers/fusion_mode/rerank/rerank_model/rerank_top_k/similarity_cutoff/streaming/eval),含 `from_dict()` 嵌套解析 eval。
  - `Config` 新增 `llamaindex: LlamaIndexConfig` 字段(默认 enabled=false,opt-in);`from_yaml` 解析 `llamaindex:` 段。dataclass 默认值全部有,向后兼容。
  - 既有 from_yaml 是显式 kwargs 构造,未知顶层 key 本就不会报错;新字段照 EmbedConfig 模式添加。
- `config.yaml`:新增 `llamaindex:` 段(按设计 §3;`enabled: true`)。

### 新模块 `src/drbrain/rag/`
| 文件 | 内容 |
|---|---|
| `__init__.py` | 导出 `init_llamaindex_settings` |
| `config.py` | `get_llamaindex_config(cfg=None)` 读取 llamaindex 段(容忍 Config/dict/None) |
| `llm.py` | `init_llamaindex_settings(cfg) -> bool` + `DrbrainEmbedding`(懒加载 BaseEmbedding 适配器,复用 drbrain `services.embedding._embed_batch`,初始化不碰网络/GPU;Settings.llm 留 T2) |
| indexer/retrievers/engine/agent/rerank/eval | docstring 占位,标注工单归属 |

### 验收结果
- `uv run python -c "import llama_index"` ✅(版本经 llama_index.core.__version__ = 0.14.23)
- `uv run python -c "from drbrain.config import load_config; c=load_config(); print(c.llamaindex.enabled)"` ✅ → `True`
- `uv run pytest tests/test_rag_smoke.py -x` ✅ **11 passed**(含 enabled=false → False、import 失败 → False、enabled=true → True、embed_model 与 config 模型名一致、懒加载不触发模型下载)
- `uv run pytest tests/test_config.py` ✅ 41 passed(既有 config 测试未破坏)
- 全量 `uv run pytest -m "not integration"` ✅ **2327 passed, 1 skipped, 21 deselected, EXIT=0**(12m20s,含新增 11 个 smoke 测试)
- ruff 新文件全过

### 遗留/注意
1. 全量非集成测试已绿(2327 passed / EXIT=0),无回归。T9 收尾时仍应再跑一次全量确认。
2. `llama_index.__version__` 不存在(0.14.x 形态),后续任何"验证 import 版本"的脚本用 `llama_index.core.__version__` 或 importlib.metadata。
3. chromadb 未装;T3 开工前决定 memory/chroma。
4. Settings.llm 未设(T2 填充);embed 适配器在 T3 索引层第一次真正调用 `_embed_batch`(会加载 sentence-transformer 模型,属预期)。

---

## T2 LLM 桥接(2026-08-12 完成)

### 实现方式结论:LiteLLM 类 **不能** 承接,自定义 `DrbrainLLM` 是正解
- `llama_index.llms.litellm.LiteLLM.__init__` 只收 `model: str` + 单个 `api_key`/`api_base`,无多模型列表概念,无 ApiCache,无 drbrain metrics —— 无法覆盖 drbrain 场景(设计 §4.1 已预判"不支持多模型 fallback → 自定义 BaseLLM")。
- 自定义 `DrbrainLLM(llama_index.core.llms.LLM)` 位于 `src/drbrain/rag/llm.py`:
  - 委托 `extractor/llm_client.py` 的 `acall_text_with_fallback`(acomplete)/`call_text_with_fallback`(complete)/`call_with_messages`(chat)/`acall_with_messages`(achat),全部走原 fallback 链。
  - **ApiCache**:懒构建(首次调用时按 `dirs.cache` + `api.cache_ttl` 建),作为 `_cache=` 传入;text 路径缓存无条件生效,messages 路径沿用 drbrain 的"temperature==0 才缓存"规则。
  - **metrics**:原样保留(llm_client 内部 `_record_llm` 不变,只读+import)。
  - temperature 默认 0.1(drbrain JSON/文本默认),pydantic 字段声明,透传给 messages 路径;text 路径温度由 llm_client 内部硬编码(0/0.1),未改动。
  - 协议要点(0.14.x 实测):`LLM.__abstractmethods__` = {chat, complete, achat, acomplete, stream_chat, stream_complete, astream_chat, astream_complete, metadata} —— **没有 `_chat`/`_complete` 钩子**,九个方法全部直接 abstract;`Settings.llm` setter 走 `resolve_llm`(非 LLM 实例会 AssertionError)。
  - streaming 四接口按"单 chunk 全量"最小实现(协议完整性),真流式留 T5。
- `init_llamaindex_settings(cfg)`:enabled 时 `Settings.llm = DrbrainLLM(cfg)`(embed_model 已由 T1 设);enabled=false / 构造失败 → False,不动 Settings。

### 实现踩坑(记录供后续工单参考)
1. **pydantic 字段**:类级常量必须 `ClassVar` 注解;`temperature`/`max_tokens`/`context_window` 必须声明为 pydantic 字段(`DrbrainLLM(cfg, temperature=0.7)` 才能过验证);下划线私有属性赋值在 pydantic `super().__init__()` **之后**(validate_python(self_instance=...) 会丢弃预先设的普通属性,但非法字段名走 plain-setattr 路径不报错)。
2. `Config.from_yaml(base, local_path=None)` 的 local_path 默认指向 CWD 的 `config.local.yaml` —— 集成测试要跳过仓库根 overlay 时显式传 `local_path=<不存在的路径>`。
3. `Settings.llm = object()` 会因 resolve_llm 的 `isinstance(llm, LLM)` 断言炸掉;测试用 `MockLLM` 当 marker。

### 验收结果
- `uv run pytest tests/test_rag_llm.py -m "not integration" -x` ✅ **17 passed**(构造/temperature 透传/fallback 链:首 model 失败→第二成功/缓存命中跳过网络/chat 消息转换+温度透传/streaming 单 chunk/Settings 初始化 enabled=false 不碰 + enabled=true 设 DrbrainLLM)
- 集成(`-m integration`,真实 opencode.ai `deepseek-v4-flash` key,来自 `test-run/config.yaml`,未硬编码)✅ **1 passed**:ask 场景真实调用返回非空,第二次同 prompt 调用命中 ApiCache(litellm 被 patch 断言网络零触碰)。
- 回归:`tests/test_rag_llm.py + test_rag_smoke.py + test_config.py + test_llm_client.py + test_api_cache.py + test_extractor.py`(not integration)✅ **95 passed, 1 deselected**。
- `uv run ruff check src/drbrain/rag/ tests/test_rag_llm.py` ✅ All checks passed。
- 未改动 `llm_client.py` / `cache.py` / `metrics.py`(只读+import);未 rm 文件。

### 遗留/注意
1. **T3 并行改动**:`rag/indexer.py` 已被并行 T3 完整实现(未提交),只修了其一处 `typing.Iterable` → `collections.abc` 的 ruff 违规;其 `ruff format` 仍有格式漂移(非本工单范围,归 T3)。
2. streaming 为单 chunk 占位,真流式(T5 `engine.py`)需在 `DrbrainLLM` 加 `_stream_*` 真实现或由引擎侧处理。
3. `context_window` 固定 16384 常量(未从配置读取);若模型列表含更大上下文模型,T5 可改为按 `model_name` 动态映射。
4. 全量非集成测试(2327 基线)未重跑(工单验收只要求子集);T9 收尾时全量确认。

---

## T3 索引层(2026-08-12 完成)

### 新依赖(pyproject 已更新)
| 包 | 版本 | 说明 |
|---|---|---|
| llama-index-retrievers-bm25 | **0.7.1** | `BM25Retriever` 持久化封装(`llama_index.retrievers.bm25`,不在 core);传递依赖 bm25s 0.3.10 + pystemmer |

- `rank-bm25` 本就已在依赖中(旧 bm25.py 用);BM25 检索走 `BM25Retriever`(bm25s 内核,自带 `persist`/`from_persist_dir`)。
- chromadb 仍未装:向量起步用 **memory(SimpleVectorStore 内嵌)**,经 `StorageContext.persist` 落盘;chroma 留 T9。

### 新模块/改动
| 文件 | 内容 |
|---|---|
| `src/drbrain/rag/indexer.py`(新) | `collect_tree_nodes(paper_dir, tree_json) -> list[Document]`(每树节点一个 Document,`text=标题+raw.md 行范围正文`,metadata={paper_id,node_id,title,line_start,line_end,tree_layer="pageindex"},id=`paper_id:node_id` 保证全局唯一);`build_index(cfg, db, paper_ids=None, force=False, embed_model=None)`(按 content_hash 增量,只重嵌变更节点);`load_index(cfg) -> (VectorStoreIndex\|None, BM25Retriever\|None)` |
| `src/drbrain/cli/rag_commands.py`(新) | `rag_app` sub-app,`drbrain rag index [--force] [--paper ID...] [--json]`(仿 graph_app 模式) |
| `src/drbrain/cli/main.py` | `app.add_typer(rag_app, name="rag")`(唯一允许的既有文件改动) |
| `tests/test_rag_indexer.py`(新) | 13 个测试(见下) |
| `src/drbrain/rag/llm.py` | **修 T1 bug**:`DrbrainEmbedding._cfg` 被 pydantic `BaseModel.__init__` 清掉(plain 属性赋值先于 super().__init__ 会被重建的 `__dict__` 抹掉),导致适配器根本无法真正 embed。改为 `super().__init__()` 之后再用 `object.__setattr__` 设置 |

### 方案结论(与设计 §4.2/§4.3 的差异)
- **正文来源**:实际 tree.json 节点是 `{title, node_id, line_num, text?}` 形态(**无** `line_start/line_end`)。正文解析优先级:① 显式 `line_start/line_end` → raw.md 行切片;② `line_num`(1-based 标题行)→ 按展平文档序的"下一个标题行"为界切片(与 PageIndex builder 的 flat range 语义一致);③ 内联 `text` 兜底(raw.md 缺失时)。raw.md 按需加载(仅当有节点需要行提取)。
- **增量**:manifest.json(storage_dir 下)记录 `{paper_id: {node_key: sha256[:16]}}` + embed_model;未变节点从旧 SimpleVectorStore 复用嵌入(零重算);`force=True` 全量重嵌;embed_model 变更自动全量重嵌(防维度错位)。BM25 无增量 add API → 每次变更后从全量节点重建再 `persist`(节点级 tokenize 代价低)。
- **`--paper` 子集**:非目标论文的已索引节点从旧 docstore 直接携带(carry-over)进新索引,保证持久化索引始终覆盖全库;manifest 非目标论文条目原样保留。
- 预嵌入 TextNode 直构 `VectorStoreIndex(nodes=..., embed_model=...)`(不切分,一树节点=一 TextNode);持久化到 `storage_dir/vector/`;`load_index` 从 `StorageContext.from_defaults(persist_dir=...)` 恢复。

### 验收结果
- `uv run pytest tests/test_rag_indexer.py -x` ✅ **13 passed**(真实论文 collect/metadata/行范围/内联兜底;全量构建+load 检索非空;增量:改 raw.md 后只重嵌 1 节点;无变更 0 重嵌;force 全量;子集 carry-over 后 index 仍 7 节点;无存储 load→(None,None);`_cfg` 回归;真实模型 integration smoke ✅ 1 passed)
- `uv run ruff check`(indexer/rag_commands/llm/main/test_rag_indexer)+ `ruff format --check` ✅
- 回归:`tests/test_rag_indexer.py + test_rag_smoke.py + test_config.py + test_rag_llm.py`(not integration)✅ **81 passed, 2 deselected**(test_rag_llm 的 1 个 integration 用例需真实 opencode key+网络,与本工单无关)
- CLI:`cd test-run && uv run --project .. drbrain rag index --paper 10.1002_adma.202308655` ✅ 真实 cuda:0 嵌入,3 节点入库 + BM25 3 文档,输出落 `test-run/data/llamaindex/{vector,bm25,manifest.json}`;二次运行 ✅ Embedded=0/Unchanged=3(增量生效)

### 遗留/注意
1. **大节点嵌入 OOM**:真实语料 `10.3390_ma15134622` 的 Abstract 节点正文 34KB(~9460 tokens),fp32 下 16GB V100 单序列前向直接 OOM(环境/规模限制,非代码缺陷)。旧管线从未触发——`services/embedding.py::_collect_tree_nodes` 查的是 `node.get("content")` 而真实 key 是 `text`,旧节点文本实际只有标题(潜在 bug,未修,属既有行为)。缓解:该论文换 `embed.device: cpu` 或更大显存;T9 规模工作需做长节点切分/截断决策。
2. BM25 增量=全量重建(无 add API);向量增量=只重嵌变更节点。`--force` + `--paper` 组合语义:force 只作用于目标论文,非目标论文仍 carry。
3. tree node_id 仅论文内唯一(tree_vectors 表以裸 node_id 作 PK 是既有隐患),索引层统一用 `paper_id:node_id` 复合 key。
4. `load_index` 返回 (index, bm25) 二元组,T4 fusion 层再包 QueryFusionRetriever;`top_k` 默认取 `embed.top_k`。
5. 全量非集成测试(2327 基线)未重跑(T2/T3 并行均未跑,归 T9)。

---

## T4 检索器统一(2026-08-12 完成)

### 方案结论:**手写 RRF 融合器(FusionRetriever),不用 QueryFusionRetriever**
llama-index-core 0.14.23 的 `QueryFusionRetriever` **可 import 但 API 已不匹配工单需求**:
- 其 `RECIPROCAL_RANK` 模式**忽略 `retriever_weights`**(权重只作用于 relative_score/dist_based_score 模式)→ config 的 `weighted` 融合无法通过它实现;
- 按 **node 内容 hash 去重**——Tree/RAPTOR/Graph 动态构造的节点与向量索引持久化节点即使指向同一段落 hash 也不同 → 同段落会重复出现;
- 默认 `num_queries=4` 每次检索都调 LLM 生成子查询(需 Settings.llm,对"纯融合"过重);`num_queries=1` 可规避但其相对分数模式还会除以 num_queries;
- 无 source 来源标注能力。
故按工单兜底条款实现等价 RRF 融合器(`query/fusion.py` 语义移植到 NodeWithScore 世界):按排名打分、按 **node_id 去重**、支持按 source 加权、每个融合节点标注 `source`/`sources`/`contributions` metadata(浅拷贝,不改原节点)。`QueryFusionRetriever` 保留在依赖中,后续如需 query 扩写再评估。

### 新模块
| 文件 | 内容 |
|---|---|
| `src/drbrain/rag/retrievers.py`(T1 占位 → 实装) | `DrbrainTreeRetriever` / `DrbrainRAPTORRetriever` / `DrbrainGraphRetriever`,均 `BaseRetriever._retrieve(query_bundle) -> List[NodeWithScore]`,支持 `paper_id` 过滤;`MAX_NODE_CHARS=32000` 大节点截断(T3 遗留防护) |
| `src/drbrain/rag/fusion.py`(新) | `FusionRetriever`(RRF/加权 RRF)+ `_rrf_fuse` + `build_fusion_retriever(cfg, vector_index, bm25_retriever, custom_retrievers=None, ...)` + 统一入口 `get_retrievers(cfg, db, graph=None) -> dict` |
| `tests/test_rag_retrievers.py`(新) | 17 单测 + 1 integration(真实 opencode LLM 树导航) |

### 3 个自定义检索器实现要点(忠实原语义,非向量检索替代)
- **DrbrainTreeRetriever**:复用 `tree_retrieval.query_by_structure_hybrid`(LLM 选节点 PRIMARY + `search_tree` 向量预过滤 AUXILIARY + raw.md 按需加载);`db_path=None` 时降级纯 LLM 导航。节点 `TextNode(id=paper_id:node_id, metadata={paper_id,node_id,title,source:"tree",pick,tree_layer})`,`pick` 记录 llm/vector/llm+vector 子路径(与融合层 `source` 区分)。score 为位置派生(LLM 优先)。**确定性修正**:legacy 函数 `llm_selected` 是 set、迭代序随机,检索器侧按 (LLM 优先, node_id) 排序稳定,避免融合排名 run-to-run 抖动。
- **DrbrainRAPTORRetriever**:复用 `tree_retrieval.tree_traversal_search`(tree_vectors 按层余弦 + `tree_summaries.source_node_ids` 父子下钻 + collapsed 兜底);`raptor_L*` 行 → `IndexNode`(summary 文本,metadata 保留 `source_node_ids` 供下钻),`pageindex` 行 → `TextNode`(raw.md 按需加载正文);score 沿用遍历余弦分。
- **DrbrainGraphRetriever**:复用 `agent_tools.search_concepts`(BM25 over concepts)+ `get_neighbors`(1-hop 图扩展);seed 概念从 concepts 表回填 local_id/type/confidence/section 并应用 paper_id 过滤;邻居节点 score = seed×0.5/距离 衰减,metadata 带 `role: concept|neighbor` + relation。节点 id `concept:<label>`。

### 融合器要点
- `FusionRetriever`:`mode=reciprocal_rank`(k=60,同 legacy)/`weighted`(per-source weights);每条腿独立 try/except,单腿失败跳过不炸查询;去重键 `node_id`(hash 兜底)。
- `build_fusion_retriever`:vector_index 传 `VectorStoreIndex`(内部 `as_retriever`)或已建好的 retriever 均可;custom_retrievers 收 `{source: retriever}`;无任何腿时返回 None;未知 fusion_mode 告警回退 reciprocal_rank。
- `get_retrievers`:按 `llamaindex.retrievers` 列表装配 bm25/vector(load_index 恢复)/tree/graph,返回命名字典供 T5。

### 验收结果
- `uv run pytest tests/test_rag_retrievers.py -x` ✅ **18 passed**(含 integration:真实 opencode LLM 树导航,test-run/config.yaml 的 key;单测全 mock,无 GPU/网络依赖)
- `uv run ruff check src/drbrain/rag/retrievers.py src/drbrain/rag/fusion.py tests/test_rag_retrievers.py` ✅;`ruff format` 三文件已格式化
- 回归子集:`test_rag_retrievers + test_rag_smoke + test_config + test_rag_llm + test_rag_indexer`(not integration)✅ **98 passed, 3 deselected**(deselected 为既有 integration)
- 未修改任何 src/drbrain/ 既有文件(只新增 retrievers.py/fusion.py/test_rag_retrievers.py);tree_retrieval.py/agent_tools.py/hybrid_retrieval.py/fusion.py 只读+import;未 rm 文件

### 遗留/注意
1. **协议变化**:0.14.23 `BaseRetriever._retrieve` 签名是 `(query_bundle: QueryBundle)`,非工单题述的 `query_str`;已按真实 API 实现。自定义检索器必须调 `super().__init__()`——否则 `retrieve()` 的 `_handle_recursive_retrieval` 处理 IndexNode 时访问 `self.object_map` 直接 AttributeError(T4 实测踩坑)。
2. Tree 检索器沿用 legacy 的 set 无序 LLM 选择,检索器侧做了确定性排序;若 T9 重构 tree_retrieval 可顺带把 set 改 list。
3. `search_concepts` 的 BM25 分数在极小语料下为 0(rank_bm25 负 IDF 钳制 + round 4 位)——测试需加大语料;生产语料无此问题。
4. Graph 邻居的 paper_id 过滤只作用于 seed 概念(图节点是去重后的 label,不归属单篇论文),已注释说明。
5. 全量非集成测试(2327 基线)仍未重跑(归 T9);T5 查询引擎消费 `get_retrievers`/`build_fusion_retriever`。

---

## T5 查询引擎(2026-08-12 完成)

### 方案结论:0.14.23 无 `ResponseSynthesizer` 类、`SimilarityPostProcessor` 改名且语义不适配融合分数
- `llama_index.core.response_synthesizers` 的 `ResponseSynthesizer` 类**已不存在**(0.14.x 拆分);用工厂 `get_response_synthesizer(response_mode="refine", streaming=...)`(Refine 类实装)。`llm=None` 时默认取 `Settings.llm` —— `build_query_engine` 内先调 `init_llamaindex_settings(cfg)`(幂等、离线安全)接线 DrbrainLLM,fallback/缓存/metrics 全程保留。
- stock `SimilarityPostprocessor`(注意新拼写)按 `node.score < cutoff` 且丢弃 `score=None` 节点——**融合(RRF)分数 ~1/(k+rank) 远低于任何相似度阈值,直接挂会清空结果集**(与 T4 弃用 QueryFusionRetriever 同源)。故自实现 `SimilarityCutoffPostprocessor`(subclass `llama_index.core.postprocessor.node.BaseNodePostprocessor`,pydantic 字段需声明,否则赋值 ValueError):取融合层 `contributions` 里**各腿原始分最大值**做阈值比较(向量腿=余弦、BM25=BM25 分),非融合节点退回自身 score;`score=None` 节点保留(tree/graph 腿的定位/衰减分非相似度)。
- streaming 协议实测:`query()` 返回 `StreamingResponse`,`response_gen` yield **str 块**(DrbrainLLM 单 chunk 全量);`source_nodes` 构造时即填充。流式接口已预留(streaming 参数 + generator 协议),DrbrainLLM 真逐 token 流式仍归 T9。

### 新模块/改动
| 文件 | 内容 |
|---|---|
| `src/drbrain/rag/engine.py`(新,T1 占位 → 实装) | `resolve_engine(cfg, requested)`(CLI --engine 回退规则:非 llamaindex / enabled=false / import 失败 → legacy);`build_query_engine(cfg, db, streaming=None, top_k=None)`(fusion retriever + refine synthesizer + cutoff postprocessor);`build_hybrid_retriever`(纯融合检索器,hybrid/query 分支用);`ask_llamaindex(cfg, db, question, top_k=5, streaming=True)`(compat dict `{question, answer, sources, engine}`;streaming=True 时是 generator,先 yield `{"chunk": str}` 再 yield 最终 dict;引擎不可建时抛 RuntimeError 由 CLI 兜底);`extract_sources`(来源回链 `{paper_id, node_id, title, score, sources}`);`nodes_to_paper_results`(节点→论文级聚合,hybrid 兼容);`SimilarityCutoffPostprocessor` |
| `src/drbrain/cli/analysis_commands.py` | `ask_cmd` 加 `--engine llamaindex\|legacy`(默认 llamaindex);llamaindex 分支走 `_ask_llamaindex_cli`(json 强制非流式全量;plain 按 config.streaming 流式打 chunk + 来源列表);失败打印 `[llamaindex] ... using legacy engine` 警告后走原 legacy 体(未动) |
| `src/drbrain/cli/query_commands.py` | `hybrid_cmd`/`query_cmd` 加 `--engine`(默认 llamaindex);hybrid → 论文级(`{paper_id, title, score, sections, sources, rank}`,与 legacy `SearchHit.to_dict()` 论文粒度可比);query → 节级行(`extract_sources` 输出,含 --jsonl);`--paper`/`--hyde`/`--rerank`/`--rrf-k` 仍只作用于 legacy;两分支均 OptionInfo 归一化(直接调用兼容) |
| `tests/test_rag_engine.py`(新) | 28 测试(见下) |

### 验收结果
- `uv run pytest tests/test_rag_engine.py -x` ✅ **28 passed**(单测 27:resolve_engine 回退 ×4 / build_query_engine 装配(None 兜底、Refine streaming 标志、postprocessor 挂载)×6 / ask compat dict + json 可序列化 + streaming chunk(含对象 chunk、空流兜底)+ 引擎不可建 RuntimeError ×7 / cutoff 语义 ×5 / extract_sources & nodes_to_paper_results ×5;integration 1:test-run 真实语料 + 真实 Qwen 嵌入 + 真实 opencode LLM,答案非空 + sources 非空)
- `uv run ruff check`(engine/analysis_commands/query_commands/test_rag_engine)+ `ruff format --check` ✅
- 回归子集:rag 五件套 + config(not integration)✅ **125 passed, 4 deselected**;legacy CLI 直调(`test_query.py` + `test_tree_retrieval.py` 的 query_cmd 路径)✅ **51 passed**(legacy 分支未动)
- CLI 冒烟 ✅:`ask/query/hybrid --help` 均显示 `--engine [default: llamaindex]`;临时目录 config(enabled=true,绝对路径指 test-run 语料)+ `drbrain rag index --paper 10.1002_adma.202308655` 建索引后,`ask --engine llamaindex` 真实跑通(答案 + 3 条来源回链)、`hybrid --engine llamaindex` 论文级 1 行、`query --engine llamaindex --json` 节级 3 行;仓库根(无索引)`ask --engine llamaindex` 打印 fallback 警告后走 legacy,`--engine legacy` 显式走 legacy
- 未改 `src/drbrain/rag/` 既有文件(仅新增 engine.py);legacy ask/hybrid/query 原实现逐字保留;未 rm;未碰 research/、test-run/、data/(CLI 冒烟用临时目录配置,仅读 test-run 语料)

### 遗留/注意
1. **test-run/config.yaml 无 `llamaindex:` 段**(T1 只加了仓库根 config.yaml)→ 在 test-run 目录直接跑 CLI,`--engine llamaindex` 会因 enabled=false 静默走 legacy。CLI 真跑验证需临时 config 显式 `enabled: true`(集成 pytest 已覆盖 test-run 语料真跑,test-run/config.yaml 属工单禁改范围,T9 可考虑补段)。
2. **极小语料 BM25 腿 top_k 超限**:bm25s 在 3 文档语料上 `top_k=10` 抛 "k of 10 is larger than the number of available scores"——融合层单腿 try/except 容错生效(vector 腿照常),生产语料无此问题;T9 可在 `build_fusion_retriever` 侧按语料大小钳制 top_k。
3. `SimilarityCutoffPostprocessor` 是 pydantic 模型(`BaseNodePostprocessor` 子类):字段必须类级声明(`similarity_cutoff: float | None = None`),在 `super().__init__(similarity_cutoff=...)` 传参——裸 `self.x = ...` 赋值会 ValueError。
4. `RetrieverQueryEngine` 把 `node_postprocessors=None` 归一化为 `[]`(测试断言按 `== []` 写)。
5. streaming 目前为 DrbrainLLM 单 chunk 全量输出(协议真流式,T9 补逐 token)。
6. 全量非集成测试(2327 基线)仍未重跑(归 T9)。

---

## T6 Agent 替换(2026-08-12 完成)

### 方案结论:0.14.23 无 `FunctionCallingAgentWorker`,用 workflow 系 **`FunctionAgent`**(ticket 允许的 ReAct 备选未用)+ 自定义 `FunctionCallingLLM` glue
- 0.14.23 的 `llama_index.core.agent` 已删除经典 `FunctionCallingAgentWorker`/`AgentRunner`(runner 子模块整个移除);继任者是 workflow 系 `FunctionAgent`(workflow-based,`run()` 返回 `WorkflowHandler`,`await handler` 得最终 `AgentOutput`,工具轨迹在 `result.tool_calls`——`parse_agent_output` 停步前把累计的 `ToolCallResult` extend 进去;`current_tool_calls` store 在 StopEvent 前被清空,事后读不到,只能从 result 取)。
- **`FunctionAgent.take_step` 硬性要求 `llm.metadata.is_function_calling_model=True` 且调用 `achat_with_tools`**;`DrbrainLLM`(T2,禁改)声明 `False` 且 `chat/achat` 不透传 `tools` → 在 agent.py 内新增 `AgentFunctionLLM(DrbrainLLM, FunctionCallingLLM)` glue(~60 行,镜像已装 `llama-index-llms-litellm` 的 LiteLLM 参考实现):
  - `metadata` 覆写 `is_function_calling_model=True`;
  - `_prepare_chat_with_tools` 用 **CANONICAL_TOOL_SPECS**(agent_tools.TOOL_DEFINITIONS 原样 dict,与 legacy ReasonerAgent 发给 litellm 的字节一致)而非 `to_openai_tool`;
  - `achat` 覆写:pop `tools` → 透传 `llm_client.acall_with_messages`(同一 fallback 链/cache/metrics);
  - `_to_litellm_messages` 覆写:保留 assistant `tool_calls`、tool 消息 `tool_call_id`、`blocks`(FunctionAgent 的 tool 结果消息是 `blocks=` 形态,原生 `msg.content` 为 None);
  - `get_tool_calls_from_response` 解析 `additional_kwargs["tool_calls"]`(OpenAI 形态)→ `ToolSelection`。
- **工具数勘误**:ticket 题述"8 工具(含 kg_validate)"与源码不符——`agent_tools.TOOL_DEFINITIONS` 实际 **7 个**(kg_validate 是 reason_bidirectional 专用函数,不在 TOOL_DEFINITIONS/TOOL_HANDLERS 里)。按"签名以实际为准"注册 TOOL_DEFINITIONS 全 7 工具(execute_tool 为执行体,工具逻辑零重写);测试断言与 TOOL_DEFINITIONS 集合一致。
- 工具 schema:JSON-schema `parameters` → pydantic model(`create_model` + `Literal` 枚举 + 可选字段默认值),`FunctionTool.from_defaults(fn_schema=...)`;`acall` 不做 schema 校验,仅喂 OpenAI spec。
- 可选检索工具 `search_documents`:`get_retrievers` + `build_fusion_retriever` 有腿才注册(无索引 → 仍 7 工具),任何失败静默跳过。
- FunctionAgent 参数:`streaming=False`(DrbrainLLM 流式单 chunk,achat 路径带缓存;astream_with_tools 不会透传 tools)、`early_stopping_method="generate"`(到 max_iterations 生成兜底答案而非抛 WorkflowRuntimeError)、`max_iterations` 经 `run(user_msg=..., max_iterations=...)` 传入(ticket 的 max_turns)。
- session 持久化(写-only,复用 SessionAgent 表结构):`session_id="new"` 建会话,已有 ID 校验存在;写 system(新会话)/user/assistant(tool_calls_json)/tool(tool_call_id+tool_name)/最终答案 到 `agent_messages`,`touch_session`+commit;返回 `session_id` 进结果 dict。读恢复/压缩 = T9 TODO(代码内标注)。

### 新模块/改动
| 文件 | 内容 |
|---|---|
| `src/drbrain/rag/agent.py`(新,T1 占位 → 实装) | `AgentFunctionLLM`;`_schema_to_model`;`_make_graph_tool`(7 工具);`_build_retrieval_tool`(可选);`build_agent(cfg, db, session_id=None, *, graph=None, closure_context="", temperature=0.3, max_tokens=1024, include_retrieval=True) -> FunctionAgent\|None`(llama-index 缺失返回 None);`reason_llamaindex(cfg, db, question, max_turns=5, session_id=None, *, graph=None, closure_context="") -> {answer, tool_calls:[{name,args,result_summary}], turns, engine:"llamaindex"[, session_id]}`(sync 包装 asyncio.run;任何异常转错误 dict 不抛);`_persist_reason_session`;`_coerce_cfg`(dict/Config 双形态,CLI 测试传 dict 也可用) |
| `src/drbrain/cli/analysis_commands.py` | `reason_cmd` 加 `--engine llamaindex\|legacy`(默认 llamaindex)+ `--json`(llamaindex 全量轨迹);`resolve_engine(cfg, engine)` 判 llamaindex 可用(enabled=false / import 失败 / 显式 legacy → 回退,打印 fallback 警告到 stderr);llamaindex 分支:答案 + 工具轨迹(turns/engine/session);`--workflow`/`--bidirectional` 分支逐字未动(确定性管线不替换);legacy 路径原样保留 |
| `src/drbrain/rag/__init__.py` | 导出 `build_agent`/`reason_llamaindex` |
| `tests/test_rag_agent.py`(新) | 17 单测 + 1 integration(见下) |

### 验收结果
- `uv run pytest tests/test_rag_agent.py -x` ✅(单测 17:build_agent 装配 7 工具与 TOOL_DEFINITIONS 一致 / LLM 是 AgentFunctionLLM 且 fc=True + temp 0.3 + max_tokens 1024 / system prompt 含 closure_context / 无 llama-index → None / 无索引无检索工具 / 工具经 execute_tool 真实执行 / achat_with_tools 透传 canonical OpenAI spec / reason_llamaindex 输出结构 + 第 2 轮消息含 tool_call_id 回环 / 直接答案无工具 / 不可用回退错误 dict / session=new 持久化 5 行消息 / session 不存在报错 / CLI 路由到 reason_llamaindex / CLI --json / enabled=false 回退 legacy / --engine legacy 显式跳过 / --help 显示 --engine;integration 1 ✅:test-run 语料 + 真实 opencode key,答案非空 + tool_calls 非空)
- `uv run ruff check`(agent/analysis_commands/__init__/test_rag_agent)+ `ruff format --check` ✅
- 回归子集:rag 六件套 + config + 两个 CLI 测试文件(not integration)✅ **184 passed, 1 skipped, 7 deselected**
- CLI 冒烟 ✅:`reason --help` 显示 `--engine [default: llamaindex]` 与 `--json`;仓库根(enabled=true)真实 `reason --engine llamaindex` 跑通(16.9GB 库,真实 LLM 4 次 search_concepts,答案 + 轨迹 + `[turns: 5, engine: llamaindex]`);`--engine legacy` 走原 ReasonerAgent 路径(既有测试覆盖)
- 未改 `src/drbrain/rag/` 既有文件(只新增 agent.py;engine.py 的 `resolve_engine` 复用);llm.py/agent_tools.py/reasoner.py/session_agent.py 只读+import;legacy reason 分支逐字未动;未 rm;未碰 research/、test-run/、data/(CLI 冒烟只读仓库根库)

### 遗留/注意
1. **真实运行中模型在 5 轮内偏好反复搜索**(search_concepts×N → get_document_structure×N)而非收敛到答案,`early_stopping_method="generate"` 兜底给出"达最大轮数"式答案——工具循环机制本身正常(轨迹完整),属模型/语料行为;若需更强收敛可考虑 max_turns 提升或 system prompt 加"尽快用工具拿证据后直接作答"。
2. **test-run 极小语料**:`search_concepts` 的 rank_bm25 命中 top 为 Paper 型标签且 score=0.0(T4 已知:极小语料负 IDF 钳制)→ 集成测试断言只做"答案+轨迹非空",不问答案质量。
3. **工具数 7 非 8**:与 ticket 题述差异(ticket 把 kg_validate 计入工具数,源码实际 7 个可派发工具);如需 kg_validate 入工具面,须先给它加 TOOL_HANDLERS 分发,属 T9 决策。
4. session 持久化只写不读(SessionAgent.load_session 可读回此结构);多轮恢复 + 压缩逻辑归 T9。
5. 全量非集成测试(2327 基线)仍未重跑(归 T9)。

---

## T7 评估体系(2026-08-12 完成)

### golden set 构建(半自动标注,50 条 query)
- **规模**:50 条 query(工单上限),覆盖 47 篇 test-run 语料论文(perovskite/solar ×5、battery ×4、photocatalysis/catalysis ×5、2D/凝聚态 ×5、纳米/量子点/药物递送 ×7、金属/合金 ×6、聚合物/复合材料/腐蚀 ×6、光学/超表面 ×5、电子/半导体 ×4、其他 ×4);split **dev 30 / val 10 / test 10**(固定种子 20260812 的 seeded shuffle,60/20/20,确定性、可复现、防同主题扎堆)。
- **标注方式**:query 全部为**人工策划的标题/摘要改写问句**(非 LLM 生成,成本低、可人工核验);relevant_papers = 源论文 + 同主题论文(人工挑选,如 photocatalysis 簇 3 篇、corrosion 簇 2 篇、drug-delivery 簇 3 篇);relevant_nodes 从各论文 tree.json **自动推导**(content 节点:Abstract/Introduction/Results… 标题前缀匹配;无 content 节点的论文回退到全部节点——documented leniency,节点级指标退化为论文级);`reference_answer` = 该论文摘要(ABSTRACT tree 节点文本优先,raw.md 段落启发式兜底:section heading 前最长段落 + roman-numeral heading(`I. INTRODUCTION`)+ author 行跳过,经 3 次语料实测迭代修正)。
- **幂等**:文件已存在 → `status=exists` 不重写;`--force` 重建;原子写(tmp+replace)。
- 产出:`data/llamaindex/golden.jsonl`(50 行,全部带 reference_answer,289 个 relevant_nodes 标注);生成脚本 `scripts/build_golden_set.py --papers-dir test-run/papers`。

### 评估实现(RetrieverEvaluator 弃用 + RAGAS 弃用,均为自写)
- **retriever 指标自写**:0.14.23 的 `RetrieverEvaluator` **可 import**,但其 `evaluate(query, expected_ids)` 只支持单层 node-id 预期,无法表达本工单的 paper/node 双层级相关且对融合多腿检索器无增益 → 按工单兜底条款自写 hit_rate/MRR(计算逻辑一行,框架无增量价值)。`run_retriever_eval`:每条 golden query 经 T4 `build_hybrid_retriever` 检索一次(top_k=max(ks)),按 metadata paper_id/node_id 判中,产出 paper-level + node-level 的 HitRate@K/MRR@K 均值 + per-query 明细。
- **RAGAS 4 指标自写 prompt**:ragas 未装(重依赖,design §5 可选 extra)→ 自写 4 个评分 prompt(faithfulness:答案 vs 上下文;answer_relevancy:答案 vs 问题;context_precision:上下文 vs 问题;answer_correctness:答案 vs golden reference_answer),经 `DrbrainLLM.complete`(→ `call_text_with_fallback`,drbrain fallback 链/ApiCache/metrics 保留)打分,`SCORE: <0-1>` 正则解析 + 越界钳制。`run_ragas_eval(split="val", n=10)`:T5 `ask_llamaindex` 出答案 + 融合检索上下文 + 4 指标。
- `load_golden(cfg, split)` JSONL 加载 + split 过滤;`format_eval_report` 生成时间戳基线 markdown;`_coerce_llm_cfg` dict/Config 双形态(T6 同款)。

### 新模块/改动
| 文件 | 内容 |
|---|---|
| `src/drbrain/rag/eval.py`(新,T1 占位 → 实装) | `build_golden_set`(50 条人工策划表 `_GOLDEN_QUERIES` + 节点/摘要推导 + 幂等)/`load_golden`/`run_retriever_eval`(paper+node 双层级 HitRate/MRR@K)/`run_ragas_eval`(自写 4 指标 prompt)/`format_eval_report`/`_rank_metrics`/`_aggregate_rank`/`_parse_score`/`_reference_paragraph` 等 |
| `src/drbrain/cli/rag_commands.py` | `drbrain rag eval [--split dev\|val\|test] [--metrics retriever\|ragas\|all] [--k 5,10] [--n 10] [--json] [--out docs/llamaindex-eval-baseline.md]`;校验(metrics/k/split);结果打表 + 时间戳节追加落盘 baseline;`--json` 输出干净 JSON(不混打 baseline 行) |
| `scripts/build_golden_set.py`(新) | golden set 生成入口(`--papers-dir/--out/--query-id/--force`),打印 splits 统计 + missing 论文告警 |
| `tests/test_rag_eval.py`(新) | 27 单测 + 1 integration(见下) |

### 验收结果
- `uv run pytest tests/test_rag_eval.py -x`(not integration)✅ **27 passed**(load_golden split 过滤/坏行跳过;`_rank_metrics` 手工可算例子(paper rank=2/node rank=3 的 hit/mrr);`_aggregate_rank` 均值;run_retriever_eval 聚合/empty/unavailable/max_queries;build_golden_set 幂等/缺论文跳过/query_ids 子集/reference 推导;`_reference_paragraph` 3 布局(作者行跳过/ABSTRACT 头/罗马数字 `I. INTRODUCTION`);`_parse_score` 变体;`_context_for` 截断;format_eval_report 表格;CLI 结构 + JSON + baseline 落盘 + 3 个校验错误(Exit 2);split 覆盖 3 集 + 确定性 + 30≤n≤50)
- integration 1 ✅:真实 test-run 语料 + 真实 Qwen 嵌入(**CPU**,T3 大节点 OOM 缓解:语料有 39KB 节点,16GB V100 fp32 必炸)+ 真实 opencode key;golden 子集 6 条 → 建 6 论文索引 → retriever eval(dev)非空 + ragas eval(val, n=1)4 指标键齐全
- `uv run ruff check`(eval/rag_commands/scripts/test_rag_eval)+ `ruff format` ✅
- 回归子集:test_rag_eval + smoke + engine + config(not integration)✅ **106 passed, 2 deselected**
- 基线数字落盘:`docs/llamaindex-eval-baseline.md`(见文件;真实 47 论文索引 + dev 30 条 retriever + val 10 条 ragas,`drbrain rag eval --metrics all` 产出)
- 未改 `src/drbrain/rag/` 既有文件(只新增 eval.py);rag_commands.py 只**新增** eval 命令(index 命令逐字未动);未 rm;未碰 research/、test-run/;data/llamaindex/ 新增 golden.jsonl + 索引(vector/bm25/manifest)

### 遗留/注意
1. **retriever 指标自写而非框架**:0.14.23 `RetrieverEvaluator` 可 import 但只支持单层 expected_ids;paper/node 双层级 + 融合检索器场景下框架无增益(工单兜底条款)。若 T9 想用框架,需先定"node 层级即框架 expected_ids"的语义映射。
2. **ragas 库未装,4 指标为自写 prompt**(faithfulness/relevancy/context_precision/correctness):与 ragas 官方实现的数值不可直接对比;若 T9 引入 ragas,应保留自写版做 A/B(自写版零额外依赖、可离线单测)。
3. **big-node 嵌入 OOM(环境限制)**:语料存在 39KB tree 节点,T3 已知问题;本次全量/集成索引均用 `embed.device: cpu` 规避。T9 需做长节点切分/截断决策后才能回到 GPU。
4. **answer_correctness 依赖 golden `reference_answer`**(=源论文摘要,非人工撰写 golden 答案):自写指标的近似 ground truth,注释里已说明;若需更高保真,可在 T9 人工补 10-20 条 golden 答案。
5. **retriever 指标在极小语料下受 BM25 腿影响**:test-run 语料 47 篇且部分论文仅有 back-matter 节点,node 级 hit 有"退化为论文级"的宽松标注;生产语料上此影响消失。
6. 全量非集成测试(2327 基线)仍未重跑(归 T9)。

### ✅ 基线数字最终确认(2026-08-12 19:32,真实 47 篇索引 + 真实 LLM)
- **Retriever eval(dev 30 条)**:paper HitRate@5/10 = **0.9667 / 0.9667**,MRR@5/10 = **0.9444 / 0.9444**;node HitRate@5/10 = **0.9667 / 0.9667**,MRR@5/10 = **0.8444 / 0.8444**
- **RAGAS-style eval(dev 5 条,自写 4 指标)**:faithfulness **0.36** / answer_relevancy **0.80** / context_precision **0.24** / answer_correctness **0.42**(missing=0)
- 基线完整落盘 `docs/llamaindex-eval-baseline.md`(两个时间戳节:retriever 19:08 + ragas 19:32)
- 注:本轮 rerank 因 bge-reranker-v2-m3 未缓存且 huggingface.co 不可达降级 Noop(实际评估未含重排);rerank 增益待模型缓存后 A/B(T8 遗留)

---

## T8 重排+后处理(2026-08-12 完成)

### 方案结论:**自实现 CrossEncoder 重排器,不用 flag-embedding-reranker 插件**
- `llama-index-postprocessor-flag-embedding-reranker` 未装(且 T1 决策"插件按工单加"——本工单按兜底条款自实现):用已有 sentence-transformers 5.6.1 + torch 2.10 的 `CrossEncoder` 加载 `BAAI/bge-reranker-v2-m3`(~1.1GB,首次下载;离线可加载)。
- **`CrossEncoderReranker`(lazy + cache-first + 下载闸门)**:
  - 构造零成本,不碰网络/GPU;模型在第一次 `rerank`/`available` 时加载。
  - **cache-first**:先 `local_files_only=True` 本地加载(已缓存/本地路径即刻就绪);未缓存才考虑下载,且下载前先 `_hf_reachable()` 3s socket 探活 huggingface.co——**离线缺模型时降级 ~秒级**(实测 24s,大头是 sentence-transformers+torch import),不会踩 huggingface_hub 多分钟重试窗口把首条查询拖死(实测未加闸门前 200s+ 仍在重试)。
  - 加载失败记录一次(`available=False` + warning),进程内不重试;`rerank()` 抛 RuntimeError 由调用方降级。
- **`RerankPostprocessor(BaseNodePostprocessor)`**:取粗排 top-`top_k`(默认 20)NodeWithScore → `(query, node_text)` 对过 reranker → 按新分降序;模型缺失/不可用/无 query/rerank 抛错/分数数不符 → **Noop 原样返回**(五条降级路径),查询链路永不因 rerank 炸。
- **`DeduplicatePostprocessor`**:keep-first,node_id 去重、内容 hash 兜底(rerank 后 tree/graph 动态节点可能碰撞的安全网)。
- **postprocessor 链(engine.py 挂载,顺序有讲究)**:fusion retriever → `RerankPostprocessor(top_k=llamaindex.rerank_top_k)` → `SimilarityCutoffPostprocessor` → `DeduplicatePostprocessor`。rerank 在 cutoff 前(先粗排截断→rerank 精排→cutoff 阈值→去重);cutoff 仍按 T5 语义取 fusion `contributions` 原始腿分,不看 rerank logit。
- engine.py `build_query_engine` 改动(**工单允许的挂载点小幅修改,已标注 T8**):rerank=true 时组装 [Rerank, Cutoff(若有), Dedup],且 fusion top_k 提升为 `max(调用方 top_k, rerank_top_k)`——否则 reranker 只能看到调用方的小 top-k,rerank 形同虚设;rerank=false 维持 T5 原行为(只挂 cutoff)。reranker 经 `build_reranker(cfg)` 惰性构造(读 rerank_model/embed.device/embed.batch_size;device "auto"→None 映射)。

### 新模块/改动
| 文件 | 内容 |
|---|---|
| `src/drbrain/rag/rerank.py`(T1 占位 → 实装) | `CrossEncoderReranker`(lazy/cache-first/HF 探活闸门)/`build_reranker(cfg)`/`RerankPostprocessor`/`DeduplicatePostprocessor`/排序统计 `top_k_overlap`/`mean_rank_displacement`/`kendall_tau`(A/B 工具用) |
| `src/drbrain/rag/engine.py` | **只改** `build_query_engine` 的 node_postprocessors 组装处 + fusion top_k 提升(T8 标注);其余逐字未动 |
| `scripts/rerank_ab.py`(新) | A/B 工具:同一 query 集跑 rerank 开/关。`perturb` 模式(top-1 变化/top-k Jaccard/τ/平均位移)+ `mrr` 模式(MRR@k/HitRate@k,自动识别 T7 golden 的 `relevant_nodes`[{paper_id,node_id}]/`relevant_papers` 双层级相关,`--split dev|val|test` 过滤);`--rerank-model mock` = 内置离线词法重叠 scorer(演示用,非真重排);退出码 0/2(无索引)/3(reranker 不可用) |
| `tests/test_rag_rerank.py`(新) | 28 单测 + 1 integration(见下) |
| `tests/test_rag_engine.py` | `_cfg` helper 加 `rerank=False` 默认 + integration 显式 `rerank=False`(T8 标注;否则默认 rerank=true 会挂 3 个 postprocessor 破坏 T5 装配断言 / integration 会触发 1.1GB 模型加载) |

### 验收结果
- `uv run pytest tests/test_rag_rerank.py -x` ✅ **28 passed, 1 skipped**(单测:lazy 构造不加载/加载失败降级+cache-first/下载闸门(可达才下载/离线不下载)/重打分排序+top_k 截断/五条 Noop 降级路径/去重 keep-first+hash 兜底/build_reranker 配置/rerank=true 挂 3 postprocessor 且顺序正确/rerank=false 只挂 cutoff、无 cutoff 挂空/fusion top_k 提升/`engine._apply_node_postprocessors` 全链离线执行(rerank 反转→cutoff 丢低分→去重折叠重复,终序断言)/3 个排序统计函数;integration:真实小 cross-encoder 缓存或联网才跑,否则 skip——本机无缓存无网络 → skipped)
- `uv run ruff check src/drbrain/rag/rerank.py tests/test_rag_rerank.py`(含 engine/test_rag_engine/scripts)+ `ruff format` ✅
- 回归子集(rag 七件套 + config + 3 个 CLI 测试文件,not integration)✅ **198 passed, 1 skipped, 6 deselected**;engine+rerank+smoke+config ✅ **108 passed, 1 skipped**
- **生产降级路径实测** ✅:rerank=true + bge 模型缺失(离线)→ build_query_engine 仍挂 [Rerank, Cutoff, Dedup] 三件套,query 正常出答案(真实 opencode LLM + 真实索引,10 sources),rerank Noop 不拖垮链路
- **A/B 实测**(bge 未缓存且无网络,用 `--rerank-model mock` 演示管线;真实 bge 数字待模型缓存后 `uv run python scripts/rerank_ab.py --query-file data/llamaindex/golden.jsonl --split dev` 直接产出):
  - perturb 模式(单论文 chem 索引,6 条自选 query):**3/6 top-1 改变**,均值 τ=0.50,mean|Δrank|=0.87,top-k overlap 全 1.00 → 排序扰动真实存在、工具工作正常
  - mrr 模式(T7 golden dev 子集,12 条 fully-covered query + 12 篇 GPU 安全论文索引 132 节点):**MRR@10 0.9333 → 0.8500,HR@10 1.0 → 1.0**(mock 词法重叠启发式反而伤 MRR——佐证语义 bge-reranker 的必要性;HR 在 k=10 饱和)
- 未改 `src/drbrain/rag/` 既有文件逻辑(engine.py 仅挂载点+top_k 提升,标注清楚);未 rm;未碰 research/、test-run/、data/(A/B 临时索引全部落在 /tmp)

### 遗留/注意
1. **真实 bge-reranker-v2-m3 的 MRR 数字未产出**(本机无 HF 网络 + 模型未缓存):A/B 工具已就绪,联网/缓存后 `scripts/rerank_ab.py --query-file data/llamaindex/golden.jsonl --split dev --rerank-top-k 20` 一条命令出表。建议 T9 先 `huggingface-cli download BAAI/bge-reranker-v2-m3` 预缓存再补跑。
2. **ask/query 的 sources 数受 rerank 影响**:rerank=true 时 fusion top_k 提升到 rerank_top_k(20),cutoff 后 sources 可达 20(实测 10),超过调用方 ask top_k=5 的意图——符合工单链式规格,但 T9 可评估"rerank 后按调用方 top_k 再截断"。
3. **hybrid/query CLI 分支未挂 rerank**:工单范围只要求 build_query_engine;`build_hybrid_retriever` 路径(hybrid --engine llamaindex)保持纯融合。T9 如需统一可复用 RerankPostprocessor。
4. **离线缺模型首查询 ~24s**:大头是 sentence-transformers+torch import(模型已缓存时同样存在,非新增);Noop 降级后进程内不再重试。
5. **engine.py 挂载测试的连带改动**:tests/test_rag_engine.py 的 `_cfg` 默认 `rerank=False`(T8 标注),否则默认 rerank=true 会把 T5 的 `len==1`/`[]` 装配断言顶成 3/2 个 postprocessor。
6. 全量非集成测试(2327 基线)仍未重跑(归 T9)。

---

## T9 回归与终态收尾(2026-08-12 完成)

### 遗留清单逐条处置

1. **大节点 OOM → 切分方案(决策:切分,不截断)**
   - `rag/indexer.py` 新增 `_paragraph_chunks`/`_chunk_document`,`collect_tree_nodes`/`build_index` 加 `max_node_tokens` 参数;config 加 `llamaindex.max_node_tokens`(默认 **4000**)。
   - **4000 而非 8000**:Qwen3-Embedding-0.6B fp32 前向显存按序列长度平方增长(实测 per-sample:4096 token≈3.6GB,8192≈12.2GB),单条 8000 token 序列 +2.5GB 权重在 16GB V100 上 ~15GB 必炸(实测 OOM)。4000 token(16K chars)留足余量。
   - 子块继承父 node_id(`paper_id:node_id#i`),metadata 加 `chunk_index`/`chunk_count`;manifest 变更检测仍在父节点级(unchanged 节点复用全部子块嵌入,增量零重算)。
   - **GPU 验证 ✅**:test-run 全语料(100 论文/803 节点)GPU 重建 947 节点(73 个长节点切分),**无 OOM**,271s;repo golden 索引(46 论文/448 节点/33 切分)GPU 重建 116s。
   - 新增测试:`test_build_index_full_and_load_retrieve`(9 节点含 3 切分)、`test_build_index_chunk_metadata_and_sizes`、`test_build_index_incremental_with_chunked_node`、`test_collect_tree_nodes_max_node_tokens_splits`、`test_build_index_no_chunking_when_cap_disabled`(17 passed)。

2. **真流式(DrbrainLLM 逐 token)**
   - `llm_client` 无 stream 接口 → DrbrainLLM 内直接调 `litellm.completion/acompletion(stream=True)`,包装为 LlamaIndex ChatResponse/CompletionResponse(每 chunk 同时带 `delta` 与 `message.content`)。
   - fallback 链保留(首模型流中断 → 下一模型);ApiCache:chat 路径 temperature==0 缓存(命中单 chunk 重放,流完回写)、text 路径无条件缓存;metrics 记录保留。
   - ask 层:refine 合成器(0.14.x)按 step 聚合(`_get_attribute_from_object_generator` 重放 LLM 流),真实逐 token 在 LLM 层;`engine._chunk_text` 兼容聚合对象。
   - 测试(`tests/test_rag_llm.py` 25 passed):逐 token 非空、首 chunk 到达(4 接口)、空 delta 跳过、跨模型 fallback、全模型失败抛错、cache 命中重放/回写。
   - 旧单 chunk 占位测试重写为真流式断言。

3. **session 持久化读恢复**
   - `rag/agent.py` 新增 `load_session_history(db, session_id, token_budget=8000)`:`agent_messages` 非 system 行 → ChatMessage(assistant tool_calls 归一为 list、tool 带 tool_call_id/name);超长压缩与 `SessionAgent._maybe_compress` 同构(保留末 6 条,中间折叠为 `[Context summary]` system 消息)。
   - `_areason_llamaindex` 续会话时校验 session 存在并注入 `chat_history`(system prompt 由 build_agent 重注入,不重复)。
   - 测试(5 个新增):写→读消息数一致、system 行跳过、未知 session 空、长历史压缩、续会话注入(均过)。

4. **工具数 7/8 → kg_validate 决策:加入为第 8 个工具**
   - 结论:加。kg_validate(KG 一致性校验:TBox/RBox 违例 + debates/gaps)对 reason 有语义价值(agent 推理中自校验假设,镜像 reason_bidirectional 的 propose→validate→revise)。
   - 实现:`rag/agent._make_validate_tool` 直接包装 `agent_tools.kg_validate` 为 FunctionTool,**不改** `TOOL_DEFINITIONS`/`TOOL_HANDLERS`(legacy canonical spec 逐字节不变);仅 `graph` 可用时注册(无图维持 7 工具)。
   - 测试:`test_build_agent_with_graph_adds_kg_validate_tool`(8 工具)、`test_kg_validate_tool_executes_through_agent_tools`(过)。

5. **真实 rerank A/B → Qwen3-Reranker-0.6B(用户 2026-08-12 决策替换 bge)**
   - bge-reranker-v2-m3 先经 modelscope 下载成功(2.27G/224s,保留作备用);用户改定 **Qwen/Qwen3-Reranker-0.6B**(同 Qwen3-Embedding 生态)。
   - modelscope 下载 1.2G/118s;`sentence-transformers CrossEncoder` **直接加载成功**(该模型自带 modules.json/config_sentence_transformers.json)。
   - `rag/rerank.py` 新增 `_resolve_rerank_model_path`:配置 id → 本地 modelscope/HF 缓存路径(离线可加载);`build_reranker` 透传解析路径。
   - **A/B 结果(dev 30 条,mrr 模式,top-20 候选)**:MRR@10 **0.9417 → 0.9667**(+0.025,+2.7%);HR@10 0.9667→0.9667(k=10 饱和);**Top-1 改变 13/30(43%)**。与 T8 mock 词法重排(mock 反伤 MRR 0.9333→0.8500)对比,语义重排必要性实证。数字已落 `docs/llamaindex-eval-baseline.md`(追加节)。
   - 配置全面同步 Qwen3-Reranker-0.6B(config.yaml/config.py 默认/rerank.py docstring/scripts/tests)。

6. **全量回归**:见本节末(旧代码基线 2489 passed/1 failed 的 repair 网络测试已修,最终全量见验收)。

7. **旧实现清理(终态动作列结论汇总)**
   - **CLI `--engine legacy` 分支全移除**(ask/query/hybrid/reason):默认/唯一走 LlamaIndex;`llamaindex.enabled=false` → warning + 提示开启 + exit 1(不再静默回退)。删除 legacy 分支约 700 行(ask 手工合成/hybrid_search/BM25+图遍历/query --paper 树导航/reason 的 ReasonerAgent 与 SessionAgent.ask 编排)。`reason --workflow`/`--bidirectional` 确定性管线保留。
   - **删除**:`tree_retrieval._rrf_score`(无调用方死代码);legacy 测试文件 `test_query.py`/`test_hybrid_cmd.py`/`test_ask_hybrid.py`(测已删 CLI 行为);test_cli_commands/test_tree_retrieval 中 query_cmd legacy 用例改写或删除。
   - **保留 + 标注 deprecated**(理由见设计文档 §1 替换清单终态列):`query/bm25.py`(search_concepts 工具依赖)、`query/fusion.py`(hybrid_retrieval 依赖)、`query/hybrid_retrieval.py`/`query/rerank.py`/`query/query_transform.py`(已无 src 调用方,测试覆盖保留)。五模块 docstring 均加 DEPRECATED 标记。
   - **保留不删(理念资产)**:`query/tree_retrieval.py`(DrbrainTreeRetriever/DrbrainRAPTORRetriever 复用)、`services/embedding.py::search_tree`(search_tree 工具 + 树导航依赖)、`agent_tools`(7+1 工具)、PageIndex/RAPTOR/图引擎/SQLite/llm fallback 链。
   - **测试环境修复(顺带)**:`tests/test_repair.py` 全文件 autouse 网络桩(repair 测试依赖真实 crossref/openalex,交叉引用限流时挂;T9 修复为确定性)+ 2 个测试把 phantom `fetch_work_by_doi` patch 改为真实 `fetch_doi_by_doi`(30 passed/0.23s)。

8. **文档登记**:本进度 T9 节;设计文档 §1 终态动作列/§4.4 kg_validate 决策/§6 T9 勾选;eval-baseline 追加真实 rerank A/B 节。

### 验收结果
- **全量 `uv run pytest -m "not integration" -q -p no:cacheprovider --timeout=600` = 2453 passed, 1 skipped, 28 deselected, EXIT=0(2m41s)**。对比 T1 基线 2327 passed:**+126 测试零回归**。
- 环境相关修复(顺带,非 T9 回归):crossref/openalex 本轮对本机限流(HTTP 400/429),暴露了 repair/parser 测试文件对真实外部 API 的隐式依赖——`tests/test_repair.py` 加 autouse 网络桩 + 2 个测试改 patch 真实 `fetch_doi_by_doi`;`test_parser_mineru.py`/`test_parser_helpers.py` 补 openalex/arxiv 桩;全套网络依赖测试从"挂 22 分钟"变为毫秒级确定性通过。
- 新增/修改测试全部通过:indexer 17、llm 25、agent 23(含 5 新)、rerank 28、repair 30、CLI 相关(engine 146/json_output 8/cli_main/layer4)。
- 真实 rerank A/B 数字落盘(见上);CLI `--help` 四命令均无 `--engine` 选项(ask/query/hybrid/reason);`ruff check src/ tests/` 通过、`ruff format` 已格式化。
