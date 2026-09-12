# RAG 层实现与验收记录

本阶段按模块完成度收敛 RAG，不运行真实模型、外部服务、GPU 或大语料集成评测。
代码与局部契约可独立验收；召回效果、答案质量和生产规模性能仍需另行验证。

## 模块边界

| 职责 | 实现 | 约束 |
| --- | --- | --- |
| 请求、结果、范围与逐路状态 | `rag/contracts.py` | 不依赖 agent、会话和 loop |
| 检索服务 | `rag/retrieval.py` | 统一请求；保留历史列表入口 |
| SQL 检索 / 查询引擎适配 | `rag/sql_retrie.py`、`rag/sql_adapter.py` | 明示两阶段召回；遵守快照与过滤约束 |
| 节点与物理片段 | `rag/index_nodes.py` | 章节是逻辑证据单元，片段有精确父文本位置 |
| SQL 工作库重建与发布 | `rag/preparation.py`（`storage/rag_database.py`、`storage/node_projection.py`） | 单命令原子重建派生库并发布固定版本；文本走共享投影 |
| 版本与保留 | `rag/index_generations.py`、`rag/sql_snapshot.py` | 原子发布、固定版本读取、按运行保留 |
| 嵌入执行 | `rag/index_embeddings.py` | 缓存复用与计算执行；失败恢复进程 GC 设置 |
| Agent 适配 | `rag/agent_llm.py`、`rag/agent_tools.py`、`rag/agent_sessions.py` | 模型、工具装配与会话分离；`agent.py` 保留编排和兼容导出 |
| 评估 | `rag/eval_data.py`、`rag/eval_metrics.py`、`rag/eval_judges.py`、`rag/eval_report.py` | 数据、确定性指标、模型裁判和报告分离；`eval.py` 保留运行入口 |

`rag/config.py` 负责映射配置到类型化配置，保留所有已知配置节，拒绝未知配置节。
包级 agent 导出改为惰性加载，核心检索和 SQL 快照操作不因导入 RAG 包而加载 agent SDK。

## PageIndex、向量与 SQL RAG 的真实调用链

这几类数据不是同一个索引：

1. `drbrain ingest` 的解析阶段把 PDF 写成 `papers/<id>/raw.md`，随后调用
   `parser.pageindex.md_to_tree`，把章节层级、`node_id`、行定位和可选摘要写成
   `papers/<id>/tree.json`。标准 CLI 关闭了节点正文和摘要生成，正文仍以 `raw.md`
   为准；`drbrain build` 读取这棵树做知识图谱抽取，并在缺树时重试生成。
2. `drbrain embed --tree` 调用 `services.embedding.build_paper_tree_vectors`：先由
   `_collect_tree_nodes` 根据 `tree.json + raw.md` 重建节点文本，写入
   `tree_vectors(tree_layer='pageindex')`；随后 `extractor.raptor.build_raptor_tree`
   读取这些向量聚类，生成 `tree_summaries` 的父子链接和 `raptor_L*` 摘要向量。
   这条标准 CLI 路径写的是 `cfg.db.path`，通常为 `data/drbrain.db`。
3. SQL 引擎读取的是派生工作库 `data/drbrain_rag.db`。本地/桌面规模下，`drbrain rag prepare`
   （SQL 模式）一条命令完成重建与发布：`storage/node_projection.collect_tree_node_records`
   从磁盘树重建 `node_texts` 与 FTS5，主库 `tree_vectors`/`tree_summaries` 和
   `papers.categories` 一并写入（`storage/rag_database.py` 原子替换），随后发布不可变
   SQL generation。大语料管线仍用 `scripts/pipeline/ragdb_fill.py extract/load`（分片并行、
   哈希对齐主库向量）与 `scripts/pipeline/ragdb_sync.py`（增量同步向量/摘要），之后
   `drbrain rag index` 才把这份工作库复制成带 manifest 的 SQL generation。
4. SQL 查询先在 `node_texts_fts` 做 BM25 候选，再用 `tree_vectors` 的 PageIndex 向量
   和 `raptor_L1` 向量做候选池内重排；结果正文来自 `node_texts`，图谱和 claims 是可选
   的实时腿。固定 generation 会禁止这些实时腿。
5. `llamaindex` 引擎是另一条路径：`rag.indexer.build_index` 直接从磁盘的
   `tree.json + raw.md` 生成持久化 VectorStore/BM25；自定义 tree/RAPTOR retriever
   仍分别读取磁盘树和 SQLite 向量表，再由 FusionRetriever 合并。

因此，兼容性保留的 `pipeline --preset full` 内置顺序仍是
`ingest → build → embed --tree → closure`；需要可服务固定 generation 时，应使用
`pipeline --preset full-rag`，它在向量阶段之后自动执行 `rag prepare`。也可以单独运行
`drbrain rag prepare`（SQL 模式；大语料仍走分片脚本 + `rag index`）。SQL 准备是完整语料
快照，不能用 `--paper` 发布会丢失其他论文的子集；LlamaIndex 后端继续支持按论文增量索引。
树正文重建也已收敛为单一投影：`storage/node_projection.collect_tree_node_records`
是唯一实现，`services.embedding`、`rag/index_nodes`、`rag/preparation` 以及 `build` 的
节点计数都调用它；哈希回归对齐保留为测试，防止树格式演进时主库向量、`node_texts`
与 LlamaIndex 索引发生漂移。

## 统一检索契约

新调用者使用 `RetrievalRequest` 和 `retrieval.retrieve(cfg, db, graph, request)`，获得
`RetrievalResult`：记录、版本、逐路状态与能力说明。结果状态为 `ok`、`empty` 或 `degraded`；
全部检索路径不可用时抛出 `RetrievalError`，索引不可用时抛出 `RetrievalUnavailableError`。
权限拒绝不得作为普通检索腿降级后继续检索。

历史 `agent.retrieve_documents` 和 `sql_retrie.retrieve_documents_sql` 保留列表返回行为；
列表子类上的 `.result` 提供同一份诊断，空结果也不会丢掉降级信息。

- `top_k` 是结果上限；SQL 多路保底在此上限内安排，不再追加超额结果。
- 内容过滤支持 `paper_ids` 和 `categories`；未知条件报错，空允许列表返回空结果。
- 分类按完整名称或点分层前缀匹配；缺少分类元数据不得扩大范围。
- 访问范围单独通过 `acl_filter` 传入。SQL 当前仅支持 `paper_id`；其他访问字段显式拒绝。
- SQL vector 是 BM25 候选池内的向量重排，RAPTOR 限于候选论文；能力信息明确记录这种差异。
- 底层检索异常传到融合层；单路失败保留诊断，全部失败拒绝继续生成。
- 重排分数缺失、数量不符、NaN 或无穷值时保留粗排顺序；SQL 查询引擎不重复执行重排。
- 新结果区分 `score_kind=rrf/rerank`：RRF 排名分数不套用相似度阈值；重排成功后使用重排分数，避免仍按旧贡献分数过滤。
- Agent 的 SQL 搜索工具保留完整 JSON 结构与固定版本，不再按字符截断序列化结果。

## SQL 快照与迁移

设置 `llamaindex.rag_engine: sql` 后，`drbrain rag index` 将现有 `drbrain_rag.db`
通过 SQLite backup 复制成独立快照，包含已提交的 WAL 内容。它不运行嵌入，也不重建语料。
发布会占用额外磁盘并读取整个库，不能视为轻量查询操作。

`drbrain rag prepare` 在发布前先重建工作库（见上节），把 `ragdb_fill`（文本/FTS5/分类）、
`ragdb_sync`（向量/摘要）与 `rag index` 的快照发布收敛为一次显式操作；它同样不运行嵌入，
也不替代大语料管线的分片与校验步骤。

快照沿用 `llamaindex.storage_dir/generations/<generation>/`、`active.json` 和运行保留记录。
发布清单记录全部普通源表内容的指纹（包括向量字节）及嵌入配置身份。
内容指纹在发布时计算，不在每次查询时扫描全库。复制、验证或发布失败保持旧活动版本；
有运行引用的版本不会被正常保留清理删除。

固定版本查询必须找到对应快照并满足嵌入配置；不接受工作库指纹冒充可恢复版本。
固定 SQL 版本不允许混入实时 graph/claims；这些来源可用于显式的非固定版本查询。
历史 `sql-...` 指纹不再被解释为可复现快照，需要重新发布索引并创建新的运行。

SQL 列表入口省略 `generation` 时保留读取工作库的能力，返回 `working-...` 请求标识，
能力信息中的 `snapshot` 为 false。它不承诺跨请求恢复。Agent 的固定运行与查询引擎使用已发布版本。
`drbrain rag health` 对 SQL 检查活动快照与必要表，不调用模型、不生成嵌入、不创建快照。

## 片段与文献定位

逻辑证据单元仍为 PageIndex 章节或 RAPTOR 摘要。超过输入长度限制的章节可以生成物理索引片段：

- 每个片段有独立 `node_id`，同时记录 `parent_node_id` 和 `parent_document_id`。
- `char_start/char_end` 是构造出的父 Document（标题＋正文）中的 Python 字符偏移。
- 片段文本严格等于该父文本切片；保留父文本校验和，不对每个片段重复添加标题。
- 父章节行号移入 `parent_line_start/parent_line_end`，不冒充片段的精确行号。
- 定位信息进入证据标识计算与查询结果；节点级评估使用逻辑父节点匹配标签。

新 LlamaIndex 清单标记 `fragment_format=2` 与片段长度设置。旧格式或长度策略改变时需要全量
索引构建，不能用部分论文构建混合新旧片段格式；旧已发布版本仍保留用于历史读取。
字符数换算是已有输入长度估计，不是对所有 tokenizer 的精确 token 上限保证。

## 局部验收

`tests/test_rag_layer_contracts.py` 提供专门的离线回归：非首节点/向量变更、旧版本重读、
固定版本禁止实时来源、范围过滤、全部/部分故障、发布失败、长文本脱敏、精确片段定位、
异常重排分数、双后端契约、SQL 查询引擎路由、分数类型、工具 JSON 完整性、依赖方向与 GC 状态恢复。

常规验证使用受控模型替身和临时 SQLite。运行相关 pytest 用例时排除 `integration`，
并在整组验证中禁止网络连接。

2026-09-12（Asia/Shanghai）最终验收结果：

| 检查 | 范围 | 结果 |
| --- | --- | --- |
| 离线 pytest | 17 个 RAG / SQL / security 测试文件，加上受影响的 checkpointing、tool broker、durable execution 三个测试文件 | **353 passed，7 deselected**；24.44 秒，272 条 warning |
| Ruff lint | `ruff check src/ tests/` | 通过 |
| Ruff format | `ruff format --check src/ tests/` | 455 个文件符合格式 |
| mypy | `mypy src/drbrain` | 245 个源文件无类型错误 |

pytest 选择排除了 `test_rag_smoke.py`，使用 `-m "not integration" --timeout=8`，
并设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，在执行进程中禁用 socket 连接。
测试没有调用真实模型或外部服务；SQLite 使用临时库。类型检查同时覆盖模块拆分后的兼容导出。

## 可恢复的阶段状态

主库的 `paper_artifacts` 表为每篇论文记录 `raw/tree/kg/pageindex/raptor/rag_text/rag_snapshot`
各阶段的 `pending/running/ready/degraded/failed/skipped` 状态、指纹、错误和尝试次数。
阶段写入遵循“先持久化输入，再推进派生物”：树或图谱失败不会删除已保存的原始材料，
批量 ingest 与 `pipeline --continue-on-error` 会隔离单篇或单阶段失败，并在最后汇总失败项。
PDF 继续使用 MinerU → PyMuPDF fallback；Markdown、纯文本和 LaTeX 直接保留原文进入同一
`raw.md` 投影，避免把可读素材再次送进 PDF 解析器。RAG 文本、向量和快照均从共享投影生成，
工作库采用临时文件 + 原子替换，发布失败时保留上一代 generation。
当没有任何可用 PageIndex 树时，准备步骤只记录 `degraded` 状态并保留旧 generation，
不会用空库覆盖可用索引。

本阶段不启动真实语料发布，不进行检索效果或性能集成验收；这些结果不能从单元测试通过推导。
