# DrBrain 联合 tree 检索的接入审计

日期：2026-09-14。范围：当前代码的静态审计；没有启动新的 ingest、索引、模型或语料迁移任务。

目标由用户明确为三路 `bm25 / vector / tree`，其中 tree 联合 PageIndex 与 RAPTOR，不涉及知识图谱。本文为本地接入证据，算法设计另见上层方案和两个上游源码审计。

## 当前实现与目标之间的差距

| 项目 | 当前代码事实 | 对设计的约束 |
| --- | --- | --- |
| 统一模型角色 | `LLMConfig` 声明 `endpoints`、`roles`、`index`、`chat`；声明角色不等于所有调用点已使用它 | 必须建立唯一角色解析入口，并测试实际传给模型的配置 |
| ingest 建树 | `db_ingest.py:227` 从通用 `llm.models` 取模型，后面再配置 SDK 树选项 | 不能继续由调用点分别拼出建树与摘要的端点 |
| RAPTOR 模型 | `build_commands.py:592–612` 同样传通用 `llm.models` 给 `build_paper_tree_vectors` | 改由共享 `index_model` 接口供 RAPTOR 和 PageIndex 构建使用 |
| 在线回答 | `rag/llm.py:188` 优先 `llm.chat`，缺失则取通用 models | 显式绑定 chat 角色，并验证 tree 推理与最终回答不落入 index 角色 |
| 结构生成 | ingest 关闭节点摘要、文献描述、thinning；`DRBRAIN_OFFLINE=1` 切到 legacy | 已入库的树不能未经检查标成完整的 PageIndex 推理索引 |
| PageIndex SDK | `sdk_backend.py` 可将 Markdown 转临时 PDF 后提交；native 路又维护每篇 `.pageindex/` | 需要正文/结构 provider，直接复用原生 MD 或真实 PDF builder，消除重提交的文献库 |
| SQL PageIndex 路 | `sql_retrie.py:185–215` 为前 320 字符的 LIKE 匹配 | 必须替换为真实结构推理路径，不能只改名 tree |
| SQL RAPTOR 路 | `sql_retrie.py:328–374` 仅在 BM25 候选 paper 内检索 `raptor_L1` | 新 tree 的启动候选不能依赖 BM25 命中，也不能只看一层摘要 |
| RAPTOR 构建 | `extractor/raptor.py:185–267` 按 paper 构建，读取已存在的 `pageindex` 层向量 | 已有底层嵌入可复用；但语料级语义组织属于新能力 |
| 当前聚类 | 本地 `_umap_reduce` 实际使用 PCA，`_gmm_cluster` 使用硬分配并过滤小簇 | 不能用它作为忠实 RAPTOR 基线；需与固定版本上游软聚类对照 |
| 节点投影 | `node_projection.py` 展开全部结构节点，通过行范围或 inline 内容解析正文 | 必须区分逻辑章节与不重复的物理证据片段，并保存原始范围 |
| RAPTOR 成员关系 | `tree_summaries.source_node_ids` 为 JSON；替换 API 以单个 paper 为边界 | 语料级多父关系需要独立的 membership 关系与 revision，不能继续强制单 paper 所有权 |
| SQL 检索库 | `rag/preparation.py` 复制节点正文、向量和摘要至 `drbrain_rag.db`，默认发布快照 | 存储统一涉及正文、索引与固定版本证据三者，不能只删除第二个 DB |
| Zvec | `_zvec_leg` 要求已发布 generation | 切主库默认读取时，必须同时处理 ANN 与正文的版本一致性 |

## 可保留的基础

- `ParsedPaper` 已有 parser backend、OCR 状态、warnings 与 provenance 字段，可作为文档修订的解析记录。
- `storage/paths.py` 的文件标识规范化、路径隔离、symlink 防护及旧目录兼容应保留。
- `paper_artifacts` 已有阶段、状态、fingerprint、metadata 和失败记录，可扩展用于统一构建任务。
- `build_tree_vectors` 已用内容 hash 判断可复用向量；新实现应把模型修订、预处理版本、维度和归一化规则也纳入缓存键。
- `Database.replace_raptor_artifacts` 已有 staging 后原子替换机制；该思路可保留，但需要覆盖集合级语义层以及空结果/失败的明确状态。
- `rag/evidence.py` 已记录原文/摘录 checksum 与父范围信息；联合 tree 必须返回同一证据契约。
- 现有 retrieval status、fusion 与 generation 测试可扩展，不另造一套静默降级协议。

## 最小接入边界

1. **模型解析**：统一解析 `index_model` 和 `chat_model` 到命名 endpoint。旧 `pageindex.model`、`llm.index` 与通用 models 的迁移需显式冲突检查；不按调用顺序覆盖配置。所有索引任务共用一个限流器，任务用途单独记日志。
2. **内容与节点提供者**：管理真实文档 ID、规范化正文、页/行/字符范围、文献修订；能生成 PageIndex 所需视图和 RAPTOR 所需 Node 输入。
3. **构建适配**：PageIndex 构建结构关系，RAPTOR 构建语义成员关系；共享底层节点、嵌入和摘要缓存。只有语义契约及成员集合相同的摘要才复用。
4. **检索适配**：一个 tree retriever 内执行语义与结构联合导航，返回一次排名；外层不再分别注册 pageindex 和 raptor 加票。
5. **发布与迁移**：主库为事实源；ANN/FTS 为可重建索引；索引修订和证据修订必须关联，旧已发布答案的原文引用仍可解析。

上述边界尚未实现，模块命名也不意味着已存在新的 API。

## 先写的契约测试

- 三种材料通过实际 CLI ingest，原输入哈希不变；重复 ingest 不重复创建内容副本。
- PageIndex / RAPTOR / vector 指向同一证据 ID 与同一模型版本的向量；章节父正文不会作为第二份同等叶片段重复进入聚类。
- 构建及重建过程中，所有 LLM 请求只发往 index endpoint；tree 查询和最终回答只发往 chat endpoint。模拟不同 base URL、同名 model 的情况，避免只按 model 字符串判断。
- 语义节点允许多父；每个来源节点仍有可读入口，软聚类未归属节点不会失去可检索性；跨篇摘要不能伪造单篇页码。
- tree 独立运行时，BM25 零结果不影响它检索。只有摘要命中但没有读过原文时，不返回伪装成原文的证据。
- 外层只有三个 leg，tree 内多条导航路径命中同一证据只返回一次；路由轨迹仍保留。
- 摘要错误、聚类退化、工具超预算及部分失败均有状态；旧成功索引可继续使用但需标出版本，不能给新输入写 ready。
- `storage migrate --dry-run` 不写源数据；旧目录与已发布 evidence 引用可读；正文/ANN 修订不匹配时明确拒绝混用。

## 本次审计的限制

没有运行模型或统计 10k 的最终成功量，因此本文不证明端点可达、索引完整或检索质量。后续验收应从实际 CLI 的小样本和固定问题开始，保留已完成的 10k 数据用于迁移审计，不以整批重跑代替阶段验证。
