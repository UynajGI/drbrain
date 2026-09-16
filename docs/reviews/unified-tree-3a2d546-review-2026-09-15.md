**审核结论：当前改动尚不满足统一 tree RAG 的设计与原子化计划，不能按“已完成”验收。** 主要阻断在生产检索接线、结构条件化分配、失败恢复和存储收敛。新增的数据结构与适配器有实际实现，但部分完成勾选超过了代码和验收证据所能支持的范围。

本次依据 [算法设计](/home/jiangyuan/drbrain-phy/docs/unified-tree-rag-design.md) 与 [原子化任务](/home/jiangyuan/drbrain-phy/docs/unified-tree-atomic-plan.md) 审阅 `d5bc40e..3a2d546`：142 个文件，新增 26,264 行、删除 402 行。另检查了工作树中 `db_ingest.py` 尚未提交的内容哈希身份修改。审阅期间另一个开发持续提交，因此本报告固定到 `3a2d546`；下述核心算法、prepare、导航、SQL 检索及 baseline 文件均已核对与该提交一致。使用了项目 `.codegraph`、diff、调用方核查和隔离复现。

1. **[P1] 生产 tree 路没有消费统一树产物，三路配置不能代表融合算法已经接通。**

   位置：[sql_retrie.py:227](/home/jiangyuan/drbrain-phy/src/drbrain/rag/sql_retrie.py:227)、[fusion.py:457](/home/jiangyuan/drbrain-phy/src/drbrain/rag/fusion.py:457)。对应 T39、T43、T45、T46。

   默认 SQL 路径的 `_tree_leg()` 使用 `resolve_sql_vector_index()` 和 `query_zvec_evidence()`；vector 路的 `query_zvec_index()` 也调用同一个函数。这个旧 ANN 的构建查询只取 `tree_vectors.tree_layer = 'pageindex'`（[zvec_index.py:89](/home/jiangyuan/drbrain-phy/src/drbrain/rag/zvec_index.py:89)）。它不读取 `rag prepare --unified` 发布的 `data/tree` generation，也不导航新建的 region/children。同步向量时它仍在检索旧叶向量，无法提供设计要求的主题摘要及多层导航。LlamaIndex 路径则仍实例化旧 `DrbrainTreeRetriever`，遍历逐篇目录并调用 `query_by_structure_hybrid`（[retrievers.py:421](/home/jiangyuan/drbrain-phy/src/drbrain/rag/retrievers.py:421)）。

   修复要求：生产 ask 的 tree 路应实际读取统一 generation，调用统一导航及证据校验；BM25、vector、tree 使用同一内容修订。验收必须从 `rag prepare --unified` 的产物直接完成 tree-only ask，包含命中 region 后读取原文的轨迹。

2. **[P1] 结构相容性计算出来了，但没有应用到后验，核心新增机制实际无效。**

   位置：[assign.py:243](/home/jiangyuan/drbrain-phy/src/drbrain/tree/assign.py:243)。对应 T04、T29、T30、T34、T57。

   `soft_assignment()` 计算 affinity，并构造带 `lam` 的 `PosteriorStage`，随后直接调用 `membership()`。后者只阈值化 `probs`，不会自动执行 `reweighted()`。因此 lambda 改变不影响最终分配。另一个缺口是 builder 先用未经结构修正的 global labels 构造 local subsets，再仅对 local stages 调用分配（[builder.py:195](/home/jiangyuan/drbrain-phy/src/drbrain/tree/builder.py:195)）；全局阶段也未按冻结的两级协议应用先验。

   隔离复现：原后验 `r1=(0.09, 0.91)`，同文档/不同文档来源画像固定。当前 `lambda=0` 与 `lambda=6` 的候选完全相同，r1 仍只属于 g1；按实现已有的 `reweighted()` 正确计算后，r1 应只属于 g0，概率约 `0.972518`。现有“structural boost”测试只检查原本就满足的成员包含关系，没有检查 lambda 引起的实际变化。

   修复要求：将重加权接入两级分配，并以能跨越成员阈值的样本验证 lambda 有效；在此之前，lambda 消融结果不能证明算法作用。

3. **[P1] 摘要失败被标为 complete，并写入水位，正常重试不再调用恢复后的模型。**

   位置：[prepare.py:443](/home/jiangyuan/drbrain-phy/src/drbrain/tree/prepare.py:443)、[builder.py:312](/home/jiangyuan/drbrain-phy/src/drbrain/tree/builder.py:312)。对应 T06、T35、T36、T42、T45。

   摘要服务捕获模型错误，返回 `ok=False`；builder 把它归为 `summary_rejected` 并返回零新增父节点。`_prepare_hierarchy()` 随即将结果写成 `status='complete'`，更新 `HIERARCHY_WATERMARK`。下一次输入未变时直接返回 `signature-unchanged`。聚类异常被转为 `stop_reason='clustering_failed'` 后，也经过相同的成功状态归类。`PrepareOutcome.ok` 只检查 status 是否为 failed，不能识别这些失败。

   隔离复现使用真实内存 SQLite、12 个叶节点、确定性聚类结果和一次故障模型：首次结果 `complete / no_parent / summary_rejected=1 / overall_ok=true`；替换成可用模型后重跑，结果仍为 `complete / signature-unchanged`，恢复模型调用数为 **0**。

   修复要求：区分合理拒绝父节点与执行失败，保留失败原因及待恢复工作；只有达到约定完成条件才推进水位。验收需覆盖“模型故障 → 恢复 → 相同 prepare 命令成功补建”。

4. **[P1] 新入库仍额外写 raw.md 和 tree.json，统一正文目前是追加的一份存储。**

   位置：[db_ingest.py:232](/home/jiangyuan/drbrain-phy/src/drbrain/cli/_helpers/db_ingest.py:232)、[db_ingest.py:327](/home/jiangyuan/drbrain-phy/src/drbrain/cli/_helpers/db_ingest.py:327)、[db_ingest.py:720](/home/jiangyuan/drbrain-phy/src/drbrain/cli/_helpers/db_ingest.py:720)。对应 T08、T13、T20–T22、T45、T48。

   新的 canonical 写入发生在 `_save_paper_artifacts()` 之后；后者仍为新文献写 `raw.md`，后续 Stage 3 仍生成并持久化 `tree.json`。canonical 写入异常只记 warning，旧流水线继续执行。默认 `rag prepare` 还走 `prepare_sql_rag()`，复制正文到 `drbrain_rag.db`；只有显式 `--unified` 才进入新流程（[rag_commands.py:220](/home/jiangyuan/drbrain-phy/src/drbrain/cli/rag_commands.py:220)）。这不符合“新文献只保留原件/附件，正文和统一节点在主库；旧文件仅兼容读取”的要求。

   修复要求：将 canonical 内容设为 ingest 的必需产物，完成新写入与消费者切换；保留旧数据的只读适配。用三类真实材料检查一次完整 CLI 入库的新增文件清单，不能仅检查“读取操作没有生成额外文件”。

5. **[P1] 导航器缺少计划要求的模型驱动遍历，高层摘要命中会停在没有原文证据的状态。**

   位置：[navigator.py:93](/home/jiangyuan/drbrain-phy/src/drbrain/tree/navigator.py:93)、[navigator.py:167](/home/jiangyuan/drbrain-phy/src/drbrain/tree/navigator.py:167)。对应 T19、T38、T40、T41、T42。

   当前导航器顺序读取 ANN 候选，不接收 ChatModel；query 只写入结果，没有用于决定下一步动作。`_expand_for_reads()` 只读取直接叶孩子，遇到 region 孩子直接 continue。它不会继续下降、沿 parents 跨文献或通过文内邻域补证。

   隔离复现构造“两层 region → 四个真实叶”的合法树，仅从二层 root 进入：返回 `status='ok'`，summary evidence 为 **1**，leaf evidence 为 **0**。因此即使把这个导航器接上生产 ask，也无法通过 T40 所列的主题→原文→父节点→另一篇验收。

   修复要求：实现一次请求内的模型驱动工具循环，保留访问历史、预算及未解决分支；高层命中需能读到实际叶，证据不足应明确报告。

6. **[P2] 生产摘要契约没有绑定实际 index 模型，模型或构建条件变化不能可靠失效旧树。**

   位置：[prepare.py:398](/home/jiangyuan/drbrain-phy/src/drbrain/tree/prepare.py:398)、[summary.py:178](/home/jiangyuan/drbrain-phy/src/drbrain/tree/summary.py:178)。对应 T23、T26、T36、T45。

   CLI 创建的默认 `SummaryContract.model`、`tokenizer` 为空；角色解析只创建模型实例，没有把实际身份填入摘要契约。缓存键与 region 身份依赖契约，因此换 endpoint/model 后仍可能复用旧结果。`_hierarchy_signature()` 也没有包含实际角色、嵌入 profile、完整聚类参数；此外，只要 `leaves_missing_parent()` 为空，就先行返回 complete，不会根据新契约重建已有父节点。

   隔离缓存复现：同一默认契约先由 index-A 生成摘要，再传入 index-B，返回 `index-A summary`，`from_cache=true`，index-B 调用数 **0**。生产调用路径没有补上能避免此情况的身份绑定。

   修复要求：将实际模型/修订、tokenizer、prompt、嵌入与聚类身份纳入相应依赖签名，按失效关系调度重建；测试应修改真实角色配置，不仅手工替换测试用 contract。

7. **[P2] prepare 与 builder 重复嵌入叶节点，没有实现共享向量的计算复用。**

   位置：[builder.py:403](/home/jiangyuan/drbrain-phy/src/drbrain/tree/builder.py:403)、[prepare.py:301](/home/jiangyuan/drbrain-phy/src/drbrain/tree/prepare.py:301)。对应 T23–T25、T34、T45。

   `_prepare_vectors()` 已嵌入并持久化叶向量；紧接着 builder 的 `_embeddings(frontier)` 再把这些正文传给 embedder。新增摘要在 `_embed_new_summary()` 嵌入一次后，进入下一轮 frontier 时同样会再次计算。当前 `UnifiedVectorStore` 参数用于写摘要，但没有被用于读取建树所需的已验证向量。

   在上述 12 叶的 prepare→hierarchy 复现中，每个叶正文都被嵌入 **2 次**。现有 `test_leaves_are_never_re_embedded` 从 builder 开始计数，并明确允许第一次重新嵌入，因此没有覆盖真实 prepare 调用链。

   修复要求：按完整嵌入身份读取已计算向量，只有缺失或失效时计算。测试应从整个 prepare 入口统计每个正文和摘要的实际嵌入次数。

8. **[P2] RAPTOR 对照组直接检索统一算法的产物，不能作为独立的原始 RAPTOR baseline。**

   位置：[baselines.py:186](/home/jiangyuan/drbrain-phy/src/drbrain/rag/baselines.py:186)、[baselines.py:295](/home/jiangyuan/drbrain-phy/src/drbrain/rag/baselines.py:295)。对应 T56、T57、T62。

   `raptor_collapsed` 调用 `_default_tree_search()`，后者打开统一树 active generation，在包含结构候选、条件化分组及父节点成本门产物的 ANN 上执行全层搜索。全层搜索是 RAPTOR collapsed 的一个步骤，但这里没有独立的 RAPTOR 构建产物，也没有验证输入 generation 来自 RAPTOR 基准构建。这样测到的是统一树的平面检索变体，无法隔离联合算法相对 RAPTOR 的改进。

   修复要求：为真实 RAPTOR 对照建立或验证可复用的独立算法产物和来源指纹；当前方式应作为统一树的检索消融单列，不能使用原始 RAPTOR 的算法标签宣称对照通过。

**完成状态需要按证据重新核对。** 源码中确实存在固定上游 submodule、统一 content_blocks/nodes/children、正文读取 API、外部内容 FTS、共享向量存储组件、模型角色客户端、迁移审计及非破坏兼容适配。这些是有效进展；生产链路、算法作用和验收完成应分别判断。

| 计划范围 | 本次审核判断 |
| --- | --- |
| 上游源码、节点协议、正文存储及读取 | 已有实现与契约测试；不能据此推断所有调用方完成迁移 |
| T22 / T48 最小存储与真实入库 | 未满足：新写入仍有 raw.md/tree.json，canonical 仍为追加阶段 |
| T29–T30 结构条件化 | 未满足：lambda 对实际软分配无效 |
| T34 / T36 / T42 / T45 构建与恢复 | 未满足：重复嵌入、执行失败被当完成、失败后跳过重试 |
| T40 / T43 / T46 统一 tree 查询 | 未满足：导航能力不完整，生产 ask 未读取统一树产物 |
| T47 实际模型配置验收 | 记录仍为 blocked，Spark 8010 不可达 |
| T48 已勾选的 CLI 验收 | 证据不足：记录写明 `bm25 ok; vector/tree unavailable until T61` |
| T56 原始算法对照 | RAPTOR 标签与所检索的统一构建产物不匹配 |
| 后续规模验收及默认切换 | 本次不判通过；应先完成上述阻断项的定向复核 |

尤其需要纠正 T48 的通过口径：任务要求三类材料产生统一索引及可核验回答，而 `acceptance.jsonl` 的同条 pass 记录明确注明 vector/tree 不可用。BM25 能回答、主库存在正文、统一索引目录存在，分别只能证明各自环节，不能证明统一 tree 已参与回答。T47 仍阻塞时也不能宣称 Spark 目标配置已经验收。

本次验证结果如下：

| 检查 | 实际结果及边界 |
| --- | --- |
| 定向 pytest：`tests/tree`、SQL/RAG、ingest 相关套件 | 执行时工作树结果为 **543 passed，1 failed**。开发仍在并行提交，此数字不代表冻结提交的全量回归 |
| 失败项 `tests/test_rag_engine.py::test_integration_ask_llamaindex_real` | 在 build_index 的 SQL/per-paper 参数校验处失败；对照 diff，该校验早于本批改动，列为现有验收问题，不归为本 diff 新引入的缺陷 |
| 完整 `pytest -m "not integration"` | 工具等待 300 秒超时，未拿到完成结果，不能记为通过 |
| 变更范围内 131 个 Python 文件 Ruff | **通过**；期间发现的测试 `t.skip` 拼写已在 `3a2d546` 修复，不再列为待修问题 |
| 结构先验隔离复现 | lambda=0 与 lambda=6 输出相同；显式正确重加权得到不同成员，确认发现 2 |
| 故障模型→恢复模型复现 | 首次误报 complete，恢复后模型调用为 0，确认发现 3 |
| 整体 prepare→hierarchy 嵌入计数 | 12 个叶各计算 2 次，确认发现 7 |
| 二层 region 导航复现 | ok，1 条 summary，0 条 leaf evidence，确认发现 5 |
| 默认摘要契约换模型复现 | 复用 index-A 摘要，index-B 调用为 0，确认发现 6 |

隔离复现使用真实内存 SQLite、确定性模型/聚类/向量替身，检验具体调用与状态契约；它们不是生产 CLI 或真实模型验收。没有据此宣称召回质量、规模性能或 Spark 连通性通过。

建议先让开发按发现 1–7 补齐代码及定向回归，再用同一隔离运行目录依次完成 `spool → ingest → rag prepare → ask`：三类材料、新增文件清单、tree-only 高层入口、三路同修订、故障后原命令恢复、模型替换失效、嵌入计数均有证据。之后再恢复 T56 的有效对照和规模验收。本次仅新增审核报告，未修改被审实现或原子化计划的完成状态。
