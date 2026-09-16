# WebUI v2 改版 · 01 新架构调研简报

> 面向读者：没看过本仓库的人、下游产品经理（02-product-plan 的输入）。
> 基线：`spooky-pony` = `origin/main` = `8a78f1a`（PR #76「unified-tree RAG — canonical store, three-leg
> retrieval, corpus-scale bounded build」已合入），2026-09-16。
> 本文只做调研与描述，不含界面设计；每一项关键结论都标注来源（文件路径 + 行号/章节），
> 证据索引见 §5。术语保持英文原文（generation / leaf / region / ReadReceipt …），中文首次出现时给释义。

---

## 0. 一句话总览

DrBrain 是一个**符号驱动的学术知识图谱 + 语料级混合检索系统**：PDF/TeX/Markdown 经 `ingest`
登记为 canonical 正文（唯一事实），`index build` 把正文填进三条检索腿（bm25 / vector / tree）
并发布一个不可变的 **generation**（可查询的索引版本），`search` / `ask` 在同一个 generation 上
做三路召回、RRF 融合、重排与（ask 的）答案合成；知识图谱（concepts / arguments / edges /
TransE / closure）与 claims/研究闭环是**可选活体扩展**，不是检索引擎的前置条件。
（来源：`AGENTS.md:3`、`AGENTS.md:78-85`、`CHANGELOG.md:13`）

与旧 WebUI（v1，PR #68）时代的最大差别是三点：

1. **正文与索引的模型换了**：正文只有一份 canonical 存储（`document_revisions` +
   `content_blocks` + 每块一个 published leaf），新论文不再写 `raw.md` / `tree.json`；
   检索对象是「叶/区域节点」而不是「PageIndex tree.json + RAPTOR summaries」。
2. **检索从"两套树 + SQL 工作副本"收敛为"三腿 + 一个发布世代"**：
   `bm25`（canonical FTS）/ `vector`（共享叶 ANN）/ `tree`（统一层次导航）从同一份数据出，
   缺 generation 时 fail-closed，graph/claims 只是显式 extras。
3. **CLI 主行命令改名**：`search` 现在指"三腿证据检索"，历史 BM25 命令改名 `library search`；
   `index build/status/verify` 成为索引主入口，KG 命令收进 `graph` 命名空间。

---

## 1. 新架构总览

### 1.1 系统形态

```
材料（PDF / TeX / MD / URL）
   │  drbrain ingest / ingest-link / import / fetch
   ▼
canonical store（主库 drbrain.db）
   document_revisions ── content_blocks（唯一可检索正文）── leaf tree_nodes
   │                                  data/papers/<id>/ 只留原始材料 + 附件
   ▼  drbrain index build          （可选分支：graph build → graph embed → graph closure）
─────────────  index build stages: FTS → vectors → hierarchy → publish  ─────────────
   canonical FTS       共享 Zvec ANN（每 leaf/region 向量算一次）      统一树 hierarchy（leaf + region）
   └──────────────────────────────┬──────────────────────────────┘
                                  ▼
                     generation（不可变发布物，data/tree/generations/gen-*）
                     tree.sqlite3（快照） + zvec/（ANN 拷贝） + manifest.json（水印/指纹）
                                  │  active.json 指针
                                  ▼
   drbrain search（三腿证据行，无答案） / drbrain ask（同一链路 + REFINE 合成 + 出处）
   bm25 ∥ vector ∥ tree → RRF → rerank → evidence（带 locator / route / generation）
                                  ▲
                     extras（活体）：graph / claims
```

（来源：`CHANGELOG.md:13-25`、`docs/rag-layer-completion.md:7-40`、`docs/rag-layer-completion.md:77-142`、
`src/drbrain/tree/prepare.py:145-260`）

### 1.2 关键数据流（主行：ingest → index build → search / ask）

| 阶段 | 命令 | 做了什么 | 产物 |
| --- | --- | --- | --- |
| 入口 | `ingest PATH…`（`ingest-link URL…`、`import`、`fetch`） | parser chain（pdf-inspector → MinerU → anydoc/OCRmyPDF → pymupdf4llm → 纯文本）+ 5 源元数据交叉验证；**canonical 写入是必需项**，失败整篇回滚并报 failed | `document_revisions` + `content_blocks` + 每块一个 published leaf；`data/papers/<id>/` 原始材料；paper status `uploaded` |
| 索引 | `index build [--force] [--tree-storage PATH]` | 增量填充每条启用腿：lexical BM25 → canonical FTS → shared vectors（多设备 spool→单写 load）→ hierarchy（结构条件化软聚类 + cost gate 摘要，产出 region）→ publish；任一 stage 失败即 exit 1，失败阶段绝不报 ready | 新的不可变 generation（仅在有变更时发布）+ last-build 记录 |
| 查询 | `search "…"` | 与 `ask` 完全相同链路的三腿证据召回，**不合成答案**；行内含 source / text locator / route / index generation；`--paper` 把三条腿都限定到指定论文，`--source arxiv\|all` 追加外部行（无本地 locator） | evidence rows（JSON/文本） |
| 问答 | `ask "…"` | 同一 route + REFINE 合成；检索失败/无结果/证据不足是**显式 abstain 状态**，不交给 LLM 硬答 | `{answer, sources, status, engine, route…}` |
| 编排 | `pipeline [--preset …]` | 顺序跑主行子命令（`ingest` / `graph build` / `graph embed` / `graph closure` / `index build`） | 各步自己的产物 |

（来源：`docs/cli-reference.md:5-16`、`src/drbrain/cli/index_commands.py:217-305`、
`src/drbrain/cli/search_commands.py:1-16`、`docs/rag-layer-completion.md:7-40`、`AGENTS.md:65-85`）

### 1.3 canonical store 与 generation 发布机制

**canonical store（唯一正文事实）**

- 正文只存一次：`document_revisions`（版本 + canonical hash）+ `content_blocks`（顺序、
  精确文本、hash、PDF 页/版面或 MD/TeX 行字符来源、结构路径与标题锚点）。
- 每个 block 对应一个 **leaf 节点**（`tree_nodes kind='leaf'`），block id 由
  `cb-<sha256(CONTRACT_SCHEMA|block|local_id|revision|ordinal)[:24]>` 稳定生成 —— 身份包含出处，
  不靠文本内容（同文不同篇不会互相吞并）。
- 新论文**不再写** `raw.md` / `tree.json`；旧论文的 legacy 文件保持只读可读，由
  `storage/paper_view.py` 的 display provider「canonical 优先、legacy 兜底」消费，且 display 读取
  不产生任何持久化文件。
- **ReadReceipt**：一次 `read` / `read_scope` / `expand` 工具调用的回执（node_id、node_revision、
  block_id、char span、content hash、tokens）。证据只能由回执构造 —— 模型编造的 node_id / 页码
  不会成为证据；只读到摘要而无叶回执的走访被标为 `partial`。

（来源：`docs/unified-tree-rag-design.md:26-58`、`src/drbrain/tree/contracts.py:131-137`、
`src/drbrain/tree/contracts.py:259-293`、`src/drbrain/tree/contracts.py:400-426`、
`src/drbrain/storage/paper_view.py:1-10`、`CHANGELOG.md:31-38`）

**generation（不可变发布物）**

- 目录：`data/tree/generations/gen-<毫秒时间戳>-<pid>/`（可在配置里改 `tree_storage`），
  内含 `tree.sqlite3`（`VACUUM INTO` 一致性快照）、`zvec/`（共享 ANN 的拷贝）、`manifest.json`；
  运行时另一个 `vectors/` 工作目录放未发布的向量。
- 发布顺序固定且可恢复：写 `.staging-<gen>` → 校验（快照内的向量元数据条数必须与拷贝的 ANN
  实际条数一致；水印基于快照自身而非 live 状态）→ `os.replace` 重命名 → **最后**写 `active.json`
  指针。崩溃只会留下「旧 generation 仍可用 + 丢弃的 `.staging-*`」或「完整的新 generation」。
- manifest 内容：`schema / generation / created_at / profile_id / watermarks{content,nodes,vectors} /
  vector_count / fingerprint`；`content` 水印来自 `document_revisions.canonical_hash`，`nodes` 来自
  ready 节点 fingerprint，`vectors` 来自 `node_vectors`（含 profile_id 与 content_hash）。
- readers 只解析 active generation；`staging` / `failed` 节点永不进 manifest。
- 状态机：**ingested / indexed / retrievable** 是三件事；`index status` 分腿给 `ready + reasons +
  versions + backlog`，`index verify` 复查 `search`/`ask` 真正读的东西（tree generation、embedding
  profile、generation freshness、last-build 结果、content FTS、node-vector backlog、叶可达性）。
  一次失败的 build 会一直门控 tree readiness，直到下一次成功 build 覆盖该记录。
- 部署身份：build 的签名绑定 index model + tokenizer + embedding profile + clustering/cost 参数；
  部署变化会 retire 旧契约的 region 并重建。`hierarchy_frontier_limit` / `hierarchy_summary_workers`
  是调度开关，**不进**签名与部署身份（改它们不会 retire 已发布区域）。

（来源：`src/drbrain/tree/publish.py:1-20`、`src/drbrain/tree/publish.py:88-104`、
`src/drbrain/tree/publish.py:168-230`、`src/drbrain/cli/index_commands.py:306-358`、
`src/drbrain/cli/index_commands.py:359-519`、`docs/rag-layer-completion.md:42-64`）

### 1.4 三腿检索：各自输入 / 输出 / 适用场景

外部只注册三条腿（`CANONICAL_LEGS = ("bm25","vector","tree")`）；历史名字 `pageindex` / `raptor`
折叠进 `tree`（一个 RRF 票，不再各投一票）；`graph` / `claims` 是显式 extras（KG 来源，不属于
文档召回腿）。

| 腿 | 输入 | 输出 | 适用场景 | 依赖 |
| --- | --- | --- | --- | --- |
| **bm25** | query 词项；canonical FTS（external-content，正文引用 `content_blocks`） | block 命中，解析为其 published leaf | 精确术语 / 公式 / 行话；`--paper` 时作用域在 FTS 查询内生效 | canonical FTS（`index build` 的 fts stage） |
| **vector** | query embedding（外部 embedding provider，profile 决定身份） | 叶向量 ANN 命中（`view=leaf`），共享 Zvec 集合 | 语义相似、跨措辞召回 | 共享 Zvec ANN + `node_vectors` 元数据（ready 且 profile 一致） |
| **tree** | query embedding → 全层入口（叶/中间 region/高层 region 都可作入口，RAPTOR collapsed 思想） | 有状态导航器（`search_nodes` / `expand` / `read` / `parents` / `read_scope`）走访后，**经 ReadReceipt 校验的叶原文**（附 entry score 沿路径取最优） | 结构性问题、跨篇主题、需要"先理解结构再读原文"的多证据综合 | 与 vector 同一个 generation 与 ANN；planner 为 `chat_model` 角色，不可用时退化为确定性 walk 并如实记录 |

- 三者走同一 RRF 融合 + 可选 BGE reranker；结果保留 source/paper/section/node/score provenance。
- 失败隔离：某条腿退化只进 retrieval trace，不影响其它腿；**缺 generation 是 fail-closed**
  （tree leg 直接抛 `TreeLegUnavailableError`），绝不静默换引擎或换后端。
- 权限/范围：ACL 过滤在 retrieval 层强制注入，不交给 LLM 事后"别泄密"。
- 与 SQL 工作副本的关系：存在已发布 SQL 快照时，SQL 投影赢；没有 SQL 语料（默认统一部署）时，
  三腿全部由同一个统一存储提供 —— 此时只有统一存储答不了的腿（graph/claims）报
  `source_unavailable`。

（来源：`src/drbrain/rag/legs.py:1-20`、`src/drbrain/rag/legs.py:40-90`、
`src/drbrain/tree/leg.py:1-30`、`src/drbrain/tree/leg.py:173-250`、`docs/rag-layer-completion.md:20-40`、
`docs/rag-layer-completion.md:66-76`、`docs/unified-tree-rag-design.md:162-186`）

### 1.5 graph / embeddings / closure 在新体系中的位置

- 知识图谱是**显式可选分支**：`graph build`（5 阶段 LLM 抽取：ontology → entities → relations →
  coref → refine）→ `graph embed`（TransE 实体/关系向量，写 `embeddings` 表）→ `graph closure`
  （8 条符号规则 + 4 条 embedding 规则，默认增量跑变更概念的 2-hop 邻域）。
- **`graph embed` 不写文本向量**，与共享叶/region 向量、Zvec ANN 无关；文本向量由 `index build`
  负责（历史 `embed --tree` 只是 `index build` 的隐藏别名）。
- 图能力是"活体附加"：检索时作为 `graph` / `claims` extras 参与融合；不可用时报
  `source_unavailable`，不影响三腿。
- 图谱分析族（landscape / frontier / paradigm / evolve / transfers / isomorphism / difficulty /
  seed / analyze）全部建立在这套图上，不属于主行检索。

（来源：`docs/cli-reference.md:18-37`、`docs/rag-layer-completion.md:24-28`、`AGENTS.md:73-75`、
`src/drbrain/rag/legs.py:14-20`）

### 1.6 loop / autoresearch（研究闭环）

- 执行模型：每次运行有 **durable run id**，身份是 `(project_id, session_id, topic)`（ledger v9），
  发起时带 `client_request_id` 幂等键；events 是版本化 JSON 信封
  （`seq / event_type / actor / schema_version / payload / idempotency_key`），有顺序与幂等冲突检查。
- workflow 是 13 步有界流程：`plan_task → retrieve → filter → parse_pdf → extract → normalize →
  fuse → identify_gaps → critique → compute → verify → settle → report`（带条件循环）；
  agent-backed 节点使用 4 个角色：analyst / critic / compute / verifier。
- 讨论层：MessageBoard（消息板）+ ResearchQueue（队列，含非作者门 + queue claim）；结论关系
  （Supports / Refutes / Orthogonal）代码化；实算门用 `job_id` 作业文件校验真实产物。
- 结算（settle）只持久化通过验证的 claims，写回 `claims` 表。
- ledger 表（`workspace/autoresearch/ledger.sqlite3`，路径由 `autoresearch.run_dir` 决定）：
  `research_runs / steps / attempts / checkpoints / tool_calls / events / front_half_node_specs /
  proposals / critic_reviews / queue_items / execution_node_specs / experiments / artifacts /
  claim_settlements / champion_versions / approval_decisions / budget_usage`。
- CLI 运维面：`autoresearch run / adaptive-run / status / pause / cancel / trace / audit / evidence`。

（来源：`src/drbrain/loop/store.py:205-520`、`src/drbrain/loop/research_events.py:45-118`、
`src/drbrain/loop/workflow.py:1442-2900`（各 `@step`）、`AGENTS.md:36`、`src/drbrain/cli/autoresearch_commands.py:114-430`、
`docs/loop-current-state.md:1-40`）

### 1.7 Epistemic layer（claims / evidence / authority / status）

- 数据模型：`claims` + `evidence` + `claim_evidence`（schema v20 专为 epistemic layer / claim
  provenance 增设），literature 层的"一句话结论"与运行结算共用一张 claims 表。
- `rag/authority.py`：同一 label 的冲突陈述按**确定性优先序**解析（authority tier → freshness →
  extraction confidence），不交给 LLM 平均或猜；`stale`（存在但过期）与 `no_evidence`（确实没有）
  是两个不同状态。
- `rag/status.py`：一次检索被显式建模为带状态结果 —— `ok / no_results / retrieval_failure /
  permission_denied / timeout / source_unavailable / insufficient_evidence / empty_answer /
  degraded`；只有 `ok` 才继续生成，其余 abstain，避免"检索失败 → 幻觉硬答"。

（来源：`src/drbrain/rag/authority.py:1-40`、`src/drbrain/rag/status.py:1-46`、`AGENTS.md:252`、
`CHANGELOG.md:19`）

### 1.8 plugins / capabilities

- Plugin 协议：`Plugin` / `PluginResult` / `ResultStatus` + `PLUGIN_MANIFEST` 声明式元数据 +
  `abi_version` fail-closed 协商（不兼容直接拒载而非静默降级）+ 作者可自跑的 conformance suite
  （`python -m drbrain.plugins.conformance <dir>`）。
- 仓库**只发接口**，具体插件运行时从外部（`autoresearch.plugins_dir`）加载；MCP servers 也是能力
  来源（`autoresearch.mcp_servers`）。
- `capabilities/`：`CapabilityDescriptor` 的中立描述 + `CapabilityCatalog` 作为唯一发现/推荐/校验/
  调用/job 入口（适配 Python 插件、MCP、Skills、模型 provider、API、CLI 命令）。
- WebUI v1 目前只做「发现 + 单插件符合性探针」，且发现面比新协议窄（见 §3）。

（来源：`AGENTS.md:35`、`CHANGELOG.md:25`、`docs/architecture.md:104-118`、
`src/drbrain/cli/main.py:401-414`）

### 1.9 存储与数据布局（运行时）

```
data/drbrain.db        主库：papers/ids/citations、concepts/arguments/edges、document_revisions、
                       content_blocks、tree_nodes、node_vectors、tree_summaries、claims/evidence、
                       agent_sessions/agent_messages/session_memory、projects、webui_sessions…
data/tree/             统一树存储根（llamaindex.tree_storage，默认 data/tree）
  vectors/             未发布的共享向量工作目录
  generations/gen-*/   tree.sqlite3 + zvec/ + manifest.json（不可变）
  active.json          当前 activation 指针
workspace/autoresearch/ledger.sqlite3  研究运行账本（autoresearch.run_dir）
workspace/<name>/      workspace.yaml + refs/papers.json（project 指向的语料引用）
data/papers/<id>/      原始材料 + 附件（不再有新论文的 raw.md/tree.json）
data/reports/ data/cache/ data/logs/ data/backups/ data/metrics.db …
```

（来源：`AGENTS.md:259-277`、`src/drbrain/tree/publish.py:36-42`、`src/drbrain/tree/leg.py:46`、
`src/drbrain/app/service.py:121-126`、`AGENTS.md:37`）

---

## 2. 能力清单（系统现在能做什么）

标注：【U】= 面向用户（可被产品化/给界面入口）；【I】= 内部机制（系统自用、UI 只需如实展示状态或
完全不展示）。依据：`AGENTS.md` 的命令参考、`docs/cli-reference.md`、`src/drbrain/cli/`（注册见
`main.py:320-414`）、`CHANGELOG.md`。

### 2.1 数据入口与文献库

| # | 能力 | 类型 | 说明 / 依据 |
| --- | --- | --- | --- |
| 1 | 单篇入库 `ingest`（PDF/MD/TeX；默认扫 `data/spool/inbox/`） | 【U】 | canonical 写入必需，失败回滚；`AGENTS.md:65`、`docs/rag-layer-completion.md:9-16` |
| 2 | URL 入库 `ingest-link` | 【U】 | 走同一 canonical 路径；失败 `status=error` 且非零退出 |
| 3 | 批量取文 `fetch` / `batch-fetch`、`patent-search`（USPTO ODP/PPUBS）、`document`（Office 结构摘要） | 【U】 | `AGENTS.md:63-64`、`AGENTS.md:151` |
| 4 | 导入 `import`（Zotero/BibTeX/Endnote）、`translate`、`enrich` / `repair` | 【U】 | 后两项是运维类写操作 |
| 5 | 文献管理 `list / show / stats / report / delete / ws / proceedings / explore` | 【U】 | `docs/cli-reference.md:56-62` |
| 6 | 书目检索 `library search`（papers/concepts/arguments 的 BM25，历史 `search`） | 【U】 | `src/drbrain/cli/library_commands.py:1-18` |
| 7 | 引文关系 `citations`（refs / citing / shared-refs）+ `check-citations`（正文引用 vs 本地库）+ `lineage` | 【U】 | `AGENTS.md:142-144` |
| 8 | 导出：BibTeX/RIS/Markdown（4 内置引用风格 + 自定义）、`export-okf`（OKF v0.1 bundle）、`graph export`（GraphML/JSON-LD/Cypher） | 【U】 | `docs/cli-reference.md:56-59`、`AGENTS.md:128-130` |

### 2.2 索引与检索（新架构核心）

| # | 能力 | 类型 | 说明 / 依据 |
| --- | --- | --- | --- |
| 9 | 索引构建 `index build`（FTS → vectors → hierarchy → publish，增量；`--force` 全量） | 【U】+【I】 | 长任务、依赖 embedding endpoint；`src/drbrain/cli/index_commands.py:217-305` |
| 10 | 索引状态 `index status`（ingested / indexed / retrievable，分腿 ready + reasons + versions + 向量 backlog；last build 记录含各 stage 结果与 hierarchy frontier 余量） | 【U】 | 只读报告，exit 0；`cli-reference.md:12`、`src/drbrain/tree/prepare.py:274-288`、`:669` |
| 11 | 索引自检 `index verify`（generation manifest/profile/freshness、last-build 门、FTS、向量 backlog、叶可达性） | 【U】+【I】 | exit 1 表示有错；`docs/cli-reference.md:13` |
| 12 | 三腿证据检索 `search`（bm25/vector/tree；locator/route/generation；`--paper`、`--source`） | 【U】 | `src/drbrain/cli/search_commands.py:1-16` |
| 13 | 检索问答 `ask`（同一 route + REFINE 合成 + sources；abstain 状态） | 【U】 | 需 `llamaindex.enabled: true` + 已发布索引；`src/drbrain/cli/analysis_commands.py:328-420` |
| 14 | 路数控制 `--legs`（单次查询覆盖 route）、`normalize_legs` 冲突检测 | 【U】/【I】 | `src/drbrain/rag/legs.py:40-90` |
| 15 | 评测 `rag eval`（HitRate/MRR/RAGAS）、`rag baselines`（`unified_tree_flat` 消融 / 来源门控的 `raptor_collapsed`） | 【U】（研究用途） | `docs/cli-reference.md:38-40` |
| 16 | PageIndex 原生 `rag pageindex-index / pageindex-chat` | 【I】/兼容 | 单篇调试路径 |
| 17 | 存储审计 `storage audit`（统一存储 + legacy 产物） | 【I】+【U】 | 与 `index verify` 分开 |
| 18 | 多设备向量 spool / 有界 hierarchy 批处理（`hierarchy_frontier_limit` 未处理 seeds 保持 roots） | 【I】 | 影响 `index status` 的 `frontier_remaining` 语义；`docs/rag-layer-completion.md:42-64` |
| 19 | 共享向量存储与 profile 身份（`node_vectors` + embedding profile；profile 变化使 generation 失效） | 【I】 | UI 只需展示 profile 一致性与 freshness |

### 2.3 知识图谱与推理

| # | 能力 | 类型 | 说明 / 依据 |
| --- | --- | --- | --- |
| 20 | KG 抽取 `graph build`（5 阶段，增量默认 dirty papers） | 【U】（长任务） | `AGENTS.md:73` |
| 21 | 图嵌入 `graph embed`（TransE，增量 warm-start） | 【I】/【U】 | `AGENTS.md:74` |
| 22 | 规则闭包 `graph closure`（8 符号 + 4 embedding 规则，增量 2-hop） | 【I】/【U】 | `AGENTS.md:75` |
| 23 | 图遍历与描述 `graph neighbors/path/related/describe/traverse-from/query`（TransE ∧∨¬） | 【U】 | `docs/cli-reference.md:27-29` |
| 24 | 图谱分析族：`frontier / landscape / paradigm / evolve / descendants / analyze / seed / difficulty / transfers / isomorphism` | 【U】 | `AGENTS.md:104-120` |
| 25 | 概念共现图 `cg ingest/build/extract/embed/neighbors/map/predict/recommend`（UMAP 交互图、年度预测快照） | 【U】 | `CHANGELOG.md:17`、`src/drbrain/cli/concept_graph_commands.py` |
| 26 | 综述 `survey`、`reason`（工具调用推理，`-b` 双向、`--workflow` 7 种结构化工作流） | 【U】 | `AGENTS.md:90`、`docs/workflows.md` |

### 2.4 研究闭环与运行

| # | 能力 | 类型 | 说明 / 依据 |
| --- | --- | --- | --- |
| 27 | 发起/续跑运行 `autoresearch run`（topic 重复 = 恢复 durable run）、`adaptive-run` | 【U】 | `src/drbrain/cli/autoresearch_commands.py:114-270` |
| 28 | 运行运维 `autoresearch status / pause / cancel / trace / audit / evidence` | 【U】 | 有状态机与人工审核（manual review）路径 |
| 29 | 运行账本（runs/steps/attempts/checkpoints/tool_calls/events/proposals/critic_reviews/queue/experiments/artifacts/settlements/approvals/budget） | 【I】→【U】只读视图 | `src/drbrain/loop/store.py:205-520` |
| 30 | 讨论层（MessageBoard + ResearchQueue 非作者门 + queue claim） | 【I】（值得可视化） | `AGENTS.md:36` |
| 31 | 实算门 + 结算（job_id 作业文件校验；只有通过验证的 claims 落库） | 【I】+【U】只读 | `docs/loop-current-state.md`、`src/drbrain/app/service.py:771-830` |
| 32 | 持久会话 `session new/ask/chat/list/delete/export`（DB 支撑，token 预算压缩） | 【U】 | `docs/cli-reference.md:56` |
| 33 | 工作区 `ws create/add/remove/list/show/delete/rename`（project 的语料引用） | 【U】 | `docs/cli-reference.md:56`、`src/drbrain/app/service.py:198-222` |
| 34 | 运行报告导出（run report markdown/json，含裁决与证据引用） | 【U】 | `src/drbrain/app/web/routes/api.py:355-370` |

### 2.5 平台与运维

| # | 能力 | 类型 | 说明 / 依据 |
| --- | --- | --- | --- |
| 35 | 插件发现 + ABI 协商 + conformance suite | 【U】+【I】 | `AGENTS.md:35`、`src/drbrain/app/service.py:1570-1700` |
| 36 | Capability catalog（descriptor/schema/permission/job；MCP/Skills/CLI 统一适配） | 【I】 | `docs/architecture.md:104-118` |
| 37 | 质量审计 `audit`（15 条规则/3 档严重度）、`queue resolve` 置信队列、`enrich`/`repair` | 【U】 | `AGENTS.md:140-151` |
| 38 | 环境与运维 `check`、`setup`、`metrics`（使用行为分析，独立 `data/metrics.db`）、`backup`/`restore`、`clean` | 【U】 | `docs/cli-reference.md:47-55` |
| 39 | WebUI 自身（`drbrain webui`）：token 认证、项目切换、六页、SSE 运行流、报告下载 | 【U】 | `src/drbrain/cli/webui_commands.py:8-49` |
| 40 | 兼容别名（`query`/`hybrid`/`fsearch`/`build`/`embed`/`closure`/`rag prepare`/`rag health`/裸 `index`） | 【I】（迁移期） | `docs/cli-reference.md:64-81` |

---

## 3. 旧 WebUI 落差分析

v1 的设计契约是 `docs/webui-design.md`（2026-09-11，PR #68 合入），实现是 `src/drbrain/app/`。
它是在**重构前**的架构上写的：当时 `search` 还是 BM25、RAG 索引由 `rag prepare` 建、树是
PageIndex tree.json + RAPTOR、检索按"检索腿列表 + SQL 工作副本"组织。逐项落差如下。

| # | v1 的假设 / 现状描述 | 重构后的现实 | 对 WebUI 的影响 | 证据 |
| --- | --- | --- | --- | --- |
| D1 | 文献库页的数据源写作「关键词检索…；当前 `service.search()` 是 BM25，问答另走 RAG」 | 有**两种**检索概念了：`library search`（书目 BM25，UI 现在用的）与 `search`（三腿**证据**检索，带 locator/route/generation）。v1 页面把两者混为一谈，没有证据检索入口 | 文献库页需要明确区分「找文献」与「找证据」；后者是 ask 的检索层，可独立展示 | `docs/webui-design.md:153`、`src/drbrain/cli/library_commands.py:1-18`、`docs/cli-reference.md:14` |
| D2 | 索引由 `rag prepare` / `rag index` 构建（页面与文案都未提）；页面只依赖 DB 与 ledger | 索引主入口是 `index build`，且引入 **generation**（版本化、可验证、可复现）；`index status` / `index verify` 是新的只读诊断面 | UI 完全没有"索引是否就绪 / 是否 stale / 哪条腿不可用"的位置；文献库/会话页无法解释"为什么搜不到" | `docs/cli-reference.md:11-13`、`src/drbrain/cli/index_commands.py:359-519` |
| D3 | 论文正文/树以 PageIndex tree.json + RAPTOR summaries 为单位；证据定位到「章节/节点」 | 正文唯一事实是 canonical `content_blocks`；证据单位是 leaf（block 内精确 span）与 region（摘要）；`ReadReceipt` 是"真的读到了"的凭证；新论文没有 tree.json | 详情页"章节大纲"其实已经切到 canonical（`body_outline`），但**证据定位**的粒度与语义（leaf span vs region summary）没有在 UI 表达；region 摘要是模型产物，不能当原文证据 | `src/drbrain/app/service.py:526-540`、`src/drbrain/storage/paper_view.py:1-10`、`src/drbrain/tree/contracts.py:400-426` |
| D4 | 问答 `service.ask()` 的可用性判断是「`llamaindex.enabled` 且已建索引」，UI 直接返回 `unavailable` 文案 | CLI 的 ask 现在有**引擎感知**的提示（`rag_engine: sql` → `drbrain index build`；`llamaindex` → `drbrain rag index`）与结构化 abstain（`source_unavailable` + hint）；WebUI 的 `ask()` 没有捕获 `AskIndexNotPreparedError`，会被全局异常处理器变成 **500 internal** | 未建索引时 UI 给出 500 而非"去建索引"的指引，且丢失 `status`/`engine`/`hint` 字段；abstain 状态（`no_results`/`degraded`/`insufficient_evidence`）在 UI 无对应展示 | `src/drbrain/app/service.py:416-435`、`src/drbrain/rag/engine.py:123-149`、`src/drbrain/cli/analysis_commands.py:393-420`、`src/drbrain/app/web/__init__.py:190-199` |
| D5 | 检索结果行是 `{paper_id, node_id, title, source, score, text}`（BM25/RAG 混合），没有版本概念 | 证据行新增 generation/route/leg 状态/真实 span（`block_id`、`char_start/char_end`、`parent_checksum`、`offset_basis`） | 结果卡片需要能展示"读的是哪个 generation、哪条腿给的、原文哪一段"，否则可信度与复现性无从谈起 | `src/drbrain/rag/retrieval.py:43-107`、`src/drbrain/cli/search_commands.py:292-303` |
| D6 | 「RAG 双形态 + 项目/会话/运行三级记忆」由 WebUI 自己实现（`session_memory` 表 + promote + run 回写） | 记忆机制仍在（DB v21），但 loop 的 claims 结算/evidence 是**另一套**结构化结论；v1 的 `record_run_memory` 只是把 claims 文本写进 session memory | 两套"结论"语义（memory 条目 vs claims/settlement）在 UI 里没有区分；promote 到项目的语义与 claims 的关系需要产品定义 | `docs/webui-design.md:105-140`、`src/drbrain/app/service.py:1214-1360` |
| D7 | 研究运行页展示 ledger 的 runs/steps/events/claims/experiments + 报告下载 | ledger 表在 v1 之后继续扩张：proposals、critic_reviews、queue_items、champion_versions、approval_decisions、budget_usage；还有 `manual_review` / `pause` / `cancel` 状态 | 运行页缺：讨论板/队列/预算/审批/冠军版本视图；状态显示需要覆盖 pause/cancel/manual review（v1 只把 running 且无 worker 显示为 `interrupted`） | `src/drbrain/loop/store.py:358-520`、`src/drbrain/app/service.py:665-677` |
| D8 | 插件页 = 发现列表 + 单插件符合性报告；发现数据来自 `autoresearch.plugins_dir` 的简易扫描 | 插件体系有 ABI 版本协商、manifest 声明、conformance suite，以及统一的 capability catalog（含 MCP/Skills/CLI 适配） | 插件页无法展示 ABI/manifest/tool 目录；也无法解释"这个能力为什么在研究运行里可见" | `AGENTS.md:35`、`src/drbrain/app/service.py:1699-1727`、`docs/architecture.md:104-118` |
| D9 | 首屏 KPI（papers/concepts/edges/arguments + ledger 计数）作为系统健康度 | 现在还多了一层"索引健康度"（每腿 ready/reasons/backlog、generation freshness、last build failed stages） | 概览页缺最关键的可用性信号；用户会在"有 9932 篇但没法搜"时困惑 | `src/drbrain/app/service.py:287-365`、`src/drbrain/cli/index_commands.py:359-519` |
| D10 | 侧栏项目切换 + 项目→会话→运行作用域（M0 契约） | 仍然成立，且是 v1 最有价值的遗产：`projects` 表 + `sync_workspace_projects`（workspace 名 → 稳定 project_id） | 继续复用，无需改；注意 project 与 workspace 是「引用」关系而不是复制语料 | `src/drbrain/app/service.py:198-222`、`src/drbrain/projects.py:17-30` |
| D11 | 认证/安全模型（本地 token、cookie、CSRF、Bearer） | 未变，仍是单人本地模式；v2 产品化（多用户）在 roadmap 里仍未开工 | 沿用；不要为了新页面绕过 CSRF/作用域校验 | `docs/webui-design.md:83-104`、`src/drbrain/app/auth.py:156-210` |
| D12 | 「六页是一级导航」的信息架构以当时的检索/树模型为依据 | 新架构多了 generation、三腿 readiness、claims/evidence、capability catalog、cg、评测/审计等对象 | 六页不够用；需要重排信息架构（这是 02 的任务），本简报只列对象 | `docs/webui-design.md:148-162` |
| D13 | 页面文案与诊断字段（RRF、表名、内部路径放诊断详情） | 术语变多：leg / generation / leaf / region / ReadReceipt / λ / profile_id / frontier_remaining | 文案需要给出人话解释，同时保留精确诊断信息（可复制的 ID） | `docs/webui-design.md:167-174`、`docs/glossary.md`（Tree & Retrieval 段） |
| D14 | 运行经理把运行状态放在进程内（`_threads`）作为过热信息 | 现状是「ledger 是唯一事实，进程内只做重复启动门控」；重启后 running 会显示 interrupted | UI 不应把 `interrupted` 当作失败；需要"可恢复"的语义与实际操作（CLI 端是重跑同 topic 续跑） | `src/drbrain/app/service.py:1330-1345`、`src/drbrain/app/service.py:665-677` |
| D15 | 默认配置：`retrievers` 的历史写法与树腿 | 数据类默认 `["bm25","vector"]`；仓库内 `config.yaml` 也是两腿，`config.example.yaml` 是三腿；本机 `config.local.yaml` 仍写 `["bm25","vector","pageindex","raptor"]`（会被折叠成 bm25+vector+tree 并附 note） | UI 必须如实显示"实际生效的 route 与每条腿状态"，不能假设默认三腿 | `src/drbrain/config.py:264`、`config.yaml:87`、`config.example.yaml:155`、`config.local.yaml:81`、`src/drbrain/rag/legs.py:48-80` |

**落差小结**：v1 的**作用域/认证/SSE/报告**这些骨架仍然有效；失效的是**检索语义层**
（两种 search、generation、三腿状态、证据 locator）与**运行语汇层**（loop 的新表与状态）。
换句话说，v2 不是推倒重来，而是把"索引与证据"这一层补进去，并把运行视图对齐新的 loop。

---

## 4. 对 WebUI 的含义

> 本节只列**数据对象与能力**（谁该成为 UI 一等公民、哪些能力值得给入口），不做界面/交互设计。

### 4.1 应当成为 UI 一等公民的数据对象

| 对象 | 为什么是一等公民 | 现有数据源（可复用） |
| --- | --- | --- |
| **Index status / leg readiness** | 决定"能不能搜、能不能问"；也是"搜不到"与"没索引"的分界 | `index status --json`（`build_index_status`）、`index verify --json` |
| **Generation**（active id、manifest、profile、freshness、published 时间、vector_count） | 一切检索结果的版本坐标；verify 的 freshness 检查说明 live DB 可能已跑在 generation 前面 | `tree/publish.py`（`get_active_tree_generation` / `resolve_tree_generation` / `verify_tree_generation`） |
| **Evidence row**（source/leg、paper_id、node_id、block_id、span、score、route、generation、abstain status） | `ask` 与 `search` 的共同产物；"结论 → 原文"的锚点 | `search --json` 的 `evidence` / `legs` / `route` / `generations`；`rag/retrieval.py` 行结构 |
| **Tree leaf / region**（kind、标题、摘要、成员、revision） | 证据的两种粒度：leaf=可引原文，region=模型摘要（不可当原文证据）；也是大纲/导航的骨架 | `tree_nodes`、`body_outline`（canonical 优先）、`TreeLegHit.to_json` |
| **ReadReceipt / 读取范围** | 证明"这段证据真的被读过"，区分摘要与原文 | `ReadReceipt`；`TreeLegOutcome.trace` / `unresolved` / `budget` |
| **Claim / Evidence / Settlement**（verdict keep 等） | 研究结论的唯一结构化载体（memory 条目不是） | ledger 的 `research_proposals` + `research_critic_reviews` + `research_claim_settlements`；DB 的 `claims`/`evidence` |
| **Run**（durable run_id、状态机、events seq、budget、approvals、queue） | 长任务的进度、可恢复性、人工介入点 | ledger 的 `research_*` 账本表（runs/steps/attempts/checkpoints/tool_calls/events/proposals/critic_reviews/queue_items/experiments/artifacts/settlements/approvals/budget）；`autoresearch status/trace`；`/api/runs/{id}/events` + SSE |
| **Session**（对话、消息、三层记忆、所属 runs） | 用户的连续工作面；记忆提升是显式动作 | `agent_sessions` / `agent_messages` / `session_memory`；service 的 sessions/chat/memory |
| **Project / workspace** | 作用域根；project_id ↔ workspace 引用 | `projects` 表 + `sync_workspace_projects` |
| **Plugin / Capability**（发现、ABI、conformance、tool 目录、job） | 科研能力的边界与可信度 | `PluginRegistry`、conformance 报告、`CapabilityCatalog` |
| **Citation**（refs / citing / shared-refs / placeholder paper） | 学术工作流的基础关系；文献库页的自然延伸 | `citations`、`paper_citations` 表、`check-citations` |
| **Export artifact**（report / BibTeX / OKF bundle / 图导出） | 交付物 | `export`、`export-okf`、`report`、run report 下载 |
| **Job（长任务句柄）** | `index build`、conformance、run 都是异步的；需要统一"排队/进行中/完成/失败"表达 | 目前只有 conformance 有 `check_id`；index build 无 job 概念（UI 侧需新增或轮询 `index status`） |
| **Metrics**（可选） | 用户行为与语料统计 | `data/metrics.db`、`metrics` 命令 |

### 4.2 值得给 UI 入口的能力（按对象归类，不设计界面）

- **索引（新增，优先级最高）**：`index build`（长任务发起 + 进度 + 失败 stage 展示）、`index status`、
  `index verify`。
- **检索**：`search`（三腿证据行 + 每腿状态 + generation + 外部源行）、`ask`（合成 + abstain 状态 +
  逐条出处 + 使用的 route/legs）、`library search`（书目/facet 过滤）。
- **文献**：paper detail（canonical 大纲 + concepts/arguments + 引文关系 + 导出）、`citations`、
  `explore`/`proceedings`、`export-okf`。
- **图谱（可选视图）**：`graph neighbors/path/related/describe/export`、分析族
  （`frontier`/`landscape`/`evolve`/`paradigm`/`transfers`/`isomorphism`/`difficulty`/`seed`/`analyze`）、
  `cg` 概念共现图/map。
- **研究**：session（对话 + 记忆 + 发起 run）、run（状态/事件流/claims+evidence/experiments/artifacts/
  报告下载）、`autoresearch status/pause/cancel/trace/audit/evidence`、讨论板与队列（只读或带门控动作）。
- **插件/能力**：发现列表、ABI/manifest 详情、conformance 报告与重跑、tool 目录（只读）。
- **运维**：`audit`、`check-citations`、`queue resolve`（需写契约）、`metrics`、`backup`（需写契约）。
- **评测（研究向）**：`rag eval` / `rag baselines` 的结果查看。

### 4.3 边界与不建议在 v2 里做的事

- **训练/重计算类**不建议进 UI 的默认路径：`graph build`/`graph embed`/`graph closure`、
  `index build` 的 heavy 模式、大批量 `ingest`、embedding 服务——它们是长任务/资源密集，
  应以「显式 job + 明确状态」进入，而不是页面按钮直连（也与 v1 已确立的"论文删除、插件启停、
  任意配置编辑放入后续小版本"同一逻辑，见 `docs/webui-design.md:46`）。
- **不要把模型生成内容当证据**：region 摘要、answer 文本都要与 leaf 原文、ReadReceipt 明确区分
  （`docs/unified-tree-rag-design.md:138-145`）。
- **不要假设"有语料就能搜"**：10k 语料的层级构建仍有界分批进行（`frontier_remaining` 可能非零），
  T59（跨批 frontier 合并与质量对照）仍 open，UI 必须能显示"部分覆盖/本轮未处理"的真实状态
  （`docs/unified-tree-atomic-plan.md` 状态段）。
- **不要绕过项目作用域/认证**：v1 的 Scope 校验、CSRF、幂等启动是产品化的前提，直接沿用。

---

## 5. 证据索引（关键结论 → 来源）

| 结论 | 来源 |
| --- | --- |
| 系统定位、主行命令、模块地图 | `AGENTS.md:3`；`AGENTS.md:25-44`；`AGENTS.md:57-226` |
| `index build` 四阶段与发布语义、失败即 exit 1 | `docs/cli-reference.md:11`；`src/drbrain/cli/index_commands.py:217-305` |
| `index status` 三态与 per-leg reasons | `docs/cli-reference.md:12`；`src/drbrain/cli/index_commands.py:306-519` |
| `index verify` 检查项（含 profile/freshness/last-build） | `docs/cli-reference.md:13`；`src/drbrain/cli/index_commands.py:607-795` |
| 三腿检索定义、legs/extras、别名折叠 | `src/drbrain/rag/legs.py:1-20`；`src/drbrain/rag/legs.py:40-90`；`docs/cli-reference.md:14` |
| `search` 输出结构（route / generations / legs / evidence） | `src/drbrain/cli/search_commands.py:292-303` |
| `ask` 的 abstain 与 hint 语义 | `src/drbrain/rag/status.py:1-46`；`src/drbrain/rag/engine.py:123-149`；`src/drbrain/cli/analysis_commands.py:393-420` |
| canonical store：revisions/blocks/leaf、无 raw.md/tree.json | `docs/unified-tree-rag-design.md:26-58`；`src/drbrain/tree/contracts.py:131-137`；`docs/rag-layer-completion.md:7-22` |
| ReadReceipt 与证据约束 | `src/drbrain/tree/contracts.py:400-426`；`src/drbrain/tree/tools.py:137-200` |
| generation 目录/manifest/水印/发布顺序/fail-closed | `src/drbrain/tree/publish.py:1-20`；`src/drbrain/tree/publish.py:88-104`；`src/drbrain/tree/publish.py:168-230`；`src/drbrain/tree/publish.py:303-360` |
| tree leg 的生产实现（resolve→search→navigate→verify→leaf text；缺 generation 失败） | `src/drbrain/tree/leg.py:1-30`；`src/drbrain/tree/leg.py:173-250`；`src/drbrain/tree/leg.py:314-318` |
| 统一正文的显示读取（canonical 优先、legacy 兜底、display 不写文件） | `src/drbrain/storage/paper_view.py:1-10`；`src/drbrain/app/service.py:526-540` |
| 证据行结构（含 span/parent_checksum/offset_basis） | `src/drbrain/rag/retrieval.py:43-107` |
| loop 账本表、事件信封、13 步、角色、实算门 | `src/drbrain/loop/store.py:205-520`；`src/drbrain/loop/research_events.py:45-118`；`src/drbrain/loop/workflow.py:1442-2900`；`AGENTS.md:36` |
| loop CLI 运维面 | `src/drbrain/cli/autoresearch_commands.py:114-430` |
| epistemic layer（authority / status） | `src/drbrain/rag/authority.py:1-40`；`src/drbrain/rag/status.py:1-46`；`CHANGELOG.md:19` |
| plugins（ABI/manifest/conformance/registry） | `AGENTS.md:35`；`CHANGELOG.md:25`；`docs/plugins.md` |
| capability catalog | `docs/architecture.md:104-118`；`src/drbrain/capabilities/` |
| v1 设计契约与六页/API 面 | `docs/webui-design.md:33-47`（目标）、`:105-140`（层级模型）、`:148-162`（信息架构）、`:184-236`（API） |
| v1 实现（service 的 search/ask/papers/runs/sessions/plugins） | `src/drbrain/app/service.py:1-20`；`:390-435`；`:499-540`；`:542-830`；`:1071-1360`；`:1330-1773` |
| v1 路由与异常边界 | `src/drbrain/app/web/routes/api.py`；`src/drbrain/app/web/__init__.py:150-231` |
| 项目作用域与 workspace 引用 | `src/drbrain/app/service.py:198-285`；`src/drbrain/projects.py:1-30` |
| 认证/CSRF/会话 | `src/drbrain/app/auth.py:156-210`；`docs/webui-design.md:83-104` |
| 兼容别名迁移表 | `docs/cli-reference.md:64-81`；`CHANGELOG.md:11` |
| 默认检索腿与配置差异 | `src/drbrain/config.py:229-290`；`config.yaml:87`；`config.example.yaml:155`；`config.local.yaml:81` |
| 10k 构建现状与 T59 未结项 | `docs/unified-tree-atomic-plan.md`（状态段与 Round 2 记录）；`docs/rag-layer-completion.md:42-64` |

---

## 6. 给下游（产品/前端）的交接提示

1. **先建立"索引与证据"的心智模型**：UI 上任何"搜索结果"都应能回答三件事——用的是哪个
   generation、哪条腿给的、原文的哪一段（block + span）。
2. **两种 search 必须区分**：`library search`（找文献）与 `search`（找证据）不是同一件事。
3. **ask 的状态要如实呈现**：abstain 的六种状态（`no_results` / `retrieval_failure` / `timeout` /
   `source_unavailable` / `insufficient_evidence` / `degraded`）不是错误弹窗，而是"系统诚实地告诉你
   它不知道"，与"正常但有部分腿退化"要区分展示。
4. **长任务需要 job 语义**：`index build`、conformance、`autoresearch run` 都是异步，
   目前只有 conformance 有 `check_id`；v2 需要统一的 job 表达（对象层面，不含界面设计）。
5. **遗留疑问（需团队决策）**：
   - WebUI 应直接调用 Python service 层（现状）还是把 CLI `--json` 作为契约面？两者目前字段不完全
     对齐（`index status`/`search` 尚无 service 包装）。
   - 是否把 tree 腿写进默认 `retrievers`（本机 config 与 example 不一致），以及 UI 是否允许改 route。
   - 运行页是否要暴露讨论层/队列/审批（需要写契约与权限模型），还是先只读。
   - `ask` 是否需要流式（v1 只有 run SSE；CLI 已支持 streaming）。
   - memory（session memory）与 claims 的关系是否需要产品上合并为一种"结论"。
