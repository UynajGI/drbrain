# WebUI v2 · 04 架构答复（A1–A6）

> 作者：arch-researcher（团队任务 #4） · 2026-09-16 · 基线：`spooky-pony` @ `8a78f1a`
> 输入：`docs/webui-v2/02-product-plan.md` §7.2（A1–A6）+ 附录 C（leader 决策）
> 范围：**只读调研 + 本文件**；未改代码、未 commit。每项 ≤ 一屏：结论 → 依据（文件:行）→ 实现建议（文件 + 签名）→ 风险/边界。
> 行号均按当前 worktree 实读核对（含 frontend-dev 已落地的 M8/M2 页面壳改动）。
> ⚠️ 本文件按 leader 要求压缩重排过一版：`02-product-plan.md` 中已写入的"04 §…边界 N"引用可能仍是旧编号，对照表见文末 §9。

---

## 0. 速览

| # | 结论（一句话） | 参考的已有实现 | 新写的量 |
| --- | --- | --- | --- |
| **A1** | 复用**已存在但未被使用**的 `tree_build_jobs` + `TreeJobStore` 做索引构建的 durable job，配 service 的 `RunManager` 线程模式；进度 = stage checkpoint + 现有计数；不引新依赖、不引 SSE | `tree/jobs.py:40-76`、`database.py:3161-3243`、`services/storage_migration.py:429-500`（完整用法范式）、`app/service.py:1330-1345`/`:1541-1576`（RunManager） | core 1 模块 + service 3 门面 + 路由 3 条 |
| **A2** | 5 个门面全部有现成实现；先把 `index status/verify` 的 payload builder 从 CLI 下沉到 core（`app/` 不 import `cli/`），其余是提取/薄封装；**字段一律 = CLI `--json` 原样** | `cli/index_commands.py:359-519`、`:607-795`；`cli/search_commands.py:108-303`；`storage/citation_graph.py:74-132`；`storage/export.py:68-180` | core 2 模块 + storage 1 函数 + service 5 门面 + 路由 6 条 |
| **A3** | 检索层 9 种 abstain 状态早已透出，只缺"索引未就绪"这一前置门：service 捕获 `AskIndexNotPreparedError`，附加 `status/hint/engine`，**保留 `unavailable` 键与其测试语义** | `rag/engine.py:123-149`、`:755-782`；`cli/analysis_commands.py:393-425`；测试 `tests/test_app.py:125-127`、`:354` | service 1 处 + 路由 1 处分流 |
| **A4** | 可给：`tree_nodes(local_id,state)` 索引已在、`leaves_missing_parent(local_id)` 已支持单篇；新增 **1 个只读聚合 + 1 条快照 GROUP BY** 即可支撑 FR-I1 差值 | `database.py:292-293`、`:2850-2862`、`:2788-2860`；`tree/reading.py:124-219` | storage 1 方法 + tree 1 方法 + service 1 门面 + 路由 1 条 |
| **A5** | loop/store **已有**足够的只读 API（`RunGovernance` 四读 + `front_half_snapshot` + `budget_snapshot`/`manual_review_step_ids`）；service 只做作用域 + 脱敏 | `loop/governance.py:14-85`、`loop/transitions.py:830-870`、`loop/front_half.py:83-136`、`loop/store.py:656-676`/`:1737-1749` | service 3 门面 + 路由 3 条（**approvals 列表 API 缺失，列为可选**） |
| **A6** | **已由 frontend-dev 在 M8 落地**（`asset_url()` 内容指纹 + `?v=`，`base.html` 三处已切换）；不动 `StaticFiles`、不加配置项 | `app/web/__init__.py:60-85`、`:110-112`；`templates/base.html:7-10` | 0（只固化约定） |

---

## A1 · 统一 job 抽象（index build / conformance / autoresearch run）

### 结论

**不新造框架，也不做"三合一"表**。三种后台任务保持各自的 durable 身份，只统一"UI 怎么看"：`autoresearch run` → `run_id` / ledger（已实现，走 `/api/runs/{id}/…` + SSE）；插件 conformance → `check_id` / `plugin_conformance`（已实现，走 `/api/plugins/{name}/conformance/{check_id}`）；**`index build` → `job_id` / `tree_build_jobs`（表已存在，仅被 migration 用），走 `GET /api/jobs/{job_id}` + 自轮询片段**。

`index build` 的最小可行 job：**创建/复用 → lease 认领（跨进程互斥）→ 每阶段 checkpoint（stage + pending 计数）→ finish(done/failed)**。

### 依据

- 表 + 状态机已存在：`storage/database.py:373-388`（`pending/running/paused/done/failed`、`owner`、`claim_expires_at`、`checkpoint_json`、`metrics_json`、`reason`；索引 `idx_tree_jobs_state/scope`），`:3161-3243`（`insert/claim/checkpoint/finish/get/list_tree_jobs`；`claim_tree_job` 是**单赢家**，pending/paused/lease 过期才可抢）。
- 包装类与恢复语义：`tree/jobs.py:40-76`（docstring 明列三条恢复性质：摘要缓存、节点内容寻址幂等、向量按 revision+hash+profile 去重）。
- **成套用法范式（照抄即可）**：`services/storage_migration.py:429-500`（按 `scope_key` 找未完成 job 复用 → `claim` 失败即 `RuntimeError("already claimed")` → 每项 `save_checkpoint` → `max_items` 时 `pause`）。
- 进程内线程模式（复制到索引构建）：`app/service.py:1330-1345`（RunManager docstring：「ledger 是唯一事实，内存只做重复启动门控」）、`:1541-1576`（`start_run` 先落库再起 worker）；conformance 的 `check_id + thread` 同型（`:1609-1660`）。
- **失败阶段 / 最后一次构建结果 / 语料水位**：`tree/prepare.py:274-288` `_record_last_build` → `vector_metadata[LAST_BUILD_KEY]`（内容 = `PrepareOutcome.to_json()`，含 `failed_stages`），读侧 `cli/index_commands.py:116-137`、`:306-358`；另有 `db.get_last_run("index")`（`database.py:4213-4221`）判"语料新于索引"。
- **进度数字现成**（向量与 region 都是增量写入，`prepare.py:421-437`）：`cli/index_commands.py:76-114`（`_vector_backlog` 的 ready/pending）、`database.py:2788-2800`/`:3046-3060`、`tree/observability.py:76-106`（`tree_snapshot`，含 `tree_build_jobs` 计数）；阶段 payload 形状见 `prepare.py:293-310`（fts）、`:342-430`（vectors）、`:640-670`（hierarchy + `frontier_remaining`）、`tree/embed_parallel.py:255-370`（并行 spool）。
- ⚠️ **`build_stages` / `paper_artifacts` 不是索引构建的锚点**：`build_stages(paper_id, stage)` 是 **KG 抽取**的 per-paper 阶段（`extractor/agent.py:162-192`；表 `database.py:178-186`、写 `:2049-2054`）；`paper_artifacts(paper_id, stage, status, fingerprint, error, attempts)` 是 **ingest/KG** 的 per-paper 产物（表 `database.py:188-200`；写入 `cli/_helpers/db_ingest.py:264-281`、`cli/build_commands.py:250-315`）。两者都没有全局阶段概念——`paper_artifacts` 的形状可作 A4 "per-paper 状态"参照，但别当 build job。
- UI 轮询范式：`templates/fragments/conformance.html:1-8`（`hx-trigger="load delay:1500ms" hx-swap="outerHTML"`）+ `routes/fragments.py:108-127`。

### 实现建议

```python
# 新增 core（CLI 与 service 共用，避免 app/ 依赖 cli/）
# src/drbrain/services/index_build.py
def index_build_scope_key(cfg, *, tree_storage=None) -> str:
    """f"index-build|{db_path}|{profile_id}|{tree_storage}"；同一部署只有一个活跃 job。"""

def active_index_job(db, scope_key: str) -> dict | None:
    """pending/running/paused 的活跃 job（复用 migration 的遍历写法）。"""

def run_index_build(cfg, *, force=False, tree_storage=None, notify=None,
                    job_id="", owner="") -> dict[str, Any]:
    """就地改 cli/index_commands.py:217-305 的函数体：登记/认领 job → 跑四阶段
    (fts→vectors→hierarchy→publication) → 每阶段 save_checkpoint → finish。
    返回值 = 现在的 index_build payload（含 failed_stages），CLI 输出/退出码不变。"""

# 阶段 checkpoint（checkpoint_json，小对象；完整 payload 仍写 LAST_BUILD_KEY）
# {"stage": "vectors", "stages_done": ["fts"], "started_at": 1757…,
#  "pending": {"vectors": 650631, "leaves": 1622131},
#  "counts": {"vectors_ready": 120000, "regions_ready": 42}}

# CLI 行为不变：index_build_cmd 归一化 OptionInfo 后调 run_index_build；已有活跃 job → exit 1
#   stderr：index build already running (job <id>, owner <owner)

# app/service.py
def start_index_build(cfg, *, force=False) -> dict[str, Any]:
    """{job_id, state, started_at, already_running}；同 scope 复用，不新建。"""
def jobs(cfg, *, kind="", states=(), limit=20) -> list[dict[str, Any]]
def job_state(cfg, job_id: str) -> dict[str, Any]          # 未知 → JobNotFoundError(404)

# app/web/routes/api.py
POST /api/index/build     -> 202 {"job_id","state","already_running"}   # CSRF
GET  /api/jobs?kind=&state=&limit=        GET  /api/jobs/{job_id}
```

**`GET /api/jobs/{job_id}`（前端契约，节选）**

```json
{"job_id":"job-…","kind":"index_build","state":"running","stage":"vectors","stages_done":["fts"],
 "pending":{"vectors":650631,"leaves":1622131},
 "live":{"vectors_ready":120000,"vectors_pending":530631,"regions_ready":42,"leaves_missing_parent":1610000},
 "last_build":{"ok":false,"failed_stages":["hierarchy"],"published":""},"created_at":1757…,"updated_at":1757…}
```

`stage/stages_done/pending` ← job checkpoint；`live/*` ← 与 `index status` 同一套只读计数。**不做 ETA**。

### 风险 / 边界

1. **lease 续租缺失**：`claim_tree_job` 默认 900s 且全仓无 `renew_tree_job` → 小时级构建会被"过期可抢"。实现时二选一：新增 `Database.renew_tree_job(job_id, owner, ttl_seconds)`（推荐，参考 `loop/store.py:1625` `renew_lease`）或在每次阶段 checkpoint 时顺带重认领。
2. **互斥必须在 core**：只在 service 的 `_threads` 做门控会漏掉"终端与浏览器同时构建"。CLI 与 service 都要走 `active_index_job` + `claim`。
3. **不承诺取消**：`finish_tree_job` 只接受 done/failed/paused，构建线程无法安全中断 → UI 只做"发起 + 观察"。
4. **不加 `client_request_id` 列**：请求幂等用"同 scope 是否有活跃 job"表达（加列 = schema 迁移，收益低）。
5. **不用 SSE**：v1 SSE 是 run 单用途实现（`routes/stream.py`）；构建进度用 htmx 自轮询（1.5–3s，终态自动停）。

---

## A2 · 新增只读门面（index_status / index_verify / evidence_search / citations / export）

### 结论

5 个门面**全部有现成实现**，工作量在"搬运"而非"新写"：

| 门面 | 封装哪个现有内部函数 | 前置动作 |
| --- | --- | --- |
| `index_status()` | `cli/index_commands.build_index_status(ctx, cfg)`（`:359-519`） | **下沉**到 core（去 ctx）；CLI 改 import |
| `index_verify()` | `cli/index_commands.build_index_verify(ctx, cfg)`（`:607-795`） | 同上 |
| `evidence_search()` | `cli/search_commands` 的 payload 段（`:245-303`） + `_local_evidence`（`:108-245`） | **提取**为 core 函数；`search_cmd` 改调它 |
| `citations()` | `storage/citation_graph.query_citation_graph`（`:74-132`） + `get_citation_counts`/`find_shared_refs` | 薄封装 + 补 `local_id/ingested` + **禁止 auto-expand** |
| `export()` | `storage/export.meta_to_bibtex/meta_to_ris/meta_to_markdown/batch_export`（`:68-180`） | 薄封装 + 把 `cli/_helpers/display.py:111-160` 的 `_export_paper_to_meta` **下沉**为公开 `paper_meta` |

**字段命名铁律**：service 原样返回 CLI payload（仅 `redact_sensitive`），前端与 CLI 看同一份字段。

### 依据

- **分层前提**：`app/` 从不 import `cli/`（全仓 grep 为空）；web 层禁 SQL 与引擎内部（`docs/webui-design.md:62-81`、`03-frontend-baseline.md:267-268`）。
- **ctx-free 等价路径**：`cli/index_commands._tree_storage_root`（`:63-74`）用的是 `runtime_data_path(ctx, …)`；ctx-free 等价物为 `tree/leg.py:147-154` `tree_storage_root(cfg, storage_dir)` + `runtime.py:657-670` `runtime_scoped_path`（读 `DRBRAIN_ROOT`），而 `cli/main.py:203-209` 在解析出 runtime 后**把 `DRBRAIN_ROOT` 写入 `os.environ`** → 同进程内两者一致；无 selector 时都返回原相对路径。
- payload（已核对，直接作为前端契约）：`build_index_status` → `{ok, status: disabled|ready|partial|not_ready, engine, enabled, tree_storage, route{requested,legs,extras,notes}, generation, states{ingested,indexed,retrievable}, legs{lexical,fts,vector,tree}, backend, pending{documents_stale,documents_failed,vectors_pending,vectors_staging,leaves_missing_parent,summaries_failed}, reasons[], tree_state}`（`index_commands.py:359-519`；FR-I1=states、FR-I2=route+legs、FR-I3=generation+腿 reasons、FR-I6=pending）；`build_index_verify` → `{ok, generation, checks[{name,ok,severity?,reason,…}], errors[], warnings[]}`，检查项含 tree_generation/embedding_profile/generation_freshness/last_build/content_fts/node_vectors/leaf_reachability/engine_generation/storage_audit（`:607-795`，FR-I4 逐项渲染即可）。
- evidence 行结构齐备：`rag/retrieval.py:43-107`（`paper_id/node_id/title/source/score/text` + `block_id/char_start/char_end/parent_checksum/offset_basis`），腿级 `legs[{source,status,count,duration_ms,reason}]`、`route`、`generations` 在 payload 顶层（`search_commands.py:245-303`）。
- scope 实现（决定 project 限定性能）：bm25 腿 `storage/content_search.py:42-44` 用 `json_each(?)`（**单参数，万级安全**）；vector/tree 腿 `tree/vector_store.py:98-100` 拼**原生 `local_id IN (…)` 表达式**（`tree/search.py:86-94` 下传）；旧 SQL 投影 `rag/sql_retrie.py:580-600` 参数化；项目语义沿用 `service.py:251-285` `project_paper_ids`（`None`=不过滤）。
- citations：`storage/citation_graph.py:74-132`（`refs/citing` 行只有 `{title,year,doi}`，`:104-124`；`shared_refs` 有 `status: linked|unlinked` 与本地 `shared_with`，`:8-57`）；**写路径必须绕开** `cli/ingest_commands.py:609-630`（cache 空时联网 `expand_citations_multi`）。
- export：`storage/export.py:68/108/152/170`（`meta_to_bibtex/ris/markdown/batch_export`）；meta 构造在 CLI 私有层 `cli/_helpers/display.py:111-160` `_export_paper_to_meta`；OKF 是**目录产物** `storage/okf_export.py:251-…`（需 GraphEngine 载图）。

### 实现建议

```python
# 1) 新增 core：src/drbrain/services/index_report.py（从 cli/index_commands.py 搬运，去 ctx）
def build_index_status(cfg, *, tree_storage: str | Path | None = None) -> dict[str, Any]
def build_index_verify(cfg, *, tree_storage: str | Path | None = None) -> dict[str, Any]
#   tree_storage=None → tree.leg.tree_storage_root(cfg)；CLI 侧继续先算 _tree_storage_root(ctx, cfg) 再传入（输出/退出码不变）

# 2) 新增 core：src/drbrain/services/evidence_search.py
def run_evidence_search(cfg, query: str, *, limit: int = 10,
                        paper_ids: Sequence[str] | None = None,
                        source: str = "local") -> dict[str, Any]:
    """search_cmd 的 payload builder；索引未就绪/检索异常 → status/sources 结构化，不抛给路由。"""

# 3) 新增 storage：paper_meta(db, local_id) -> dict（从 cli/_helpers/display.py 下沉，公开命名）

# 4) app/service.py（薄门面：作用域 + 脱敏 + 错误映射）
def index_status(cfg, project_id=None) -> dict[str, Any]
def index_verify(cfg, project_id=None) -> dict[str, Any]
def evidence_search(cfg, query, *, limit=10, paper_ids=None, source="local", project_id=None) -> dict[str, Any]
def citations(cfg, local_id, *, ctype="all", project_id=None) -> dict[str, Any]
def paper_export(cfg, *, fmt="bib", local_id=None, paper_ids=None, style="apa", project_id=None) -> dict[str, Any]

# 5) app/web/routes/api.py（全 GET；导出走下载响应）
GET  /api/index/status            GET /api/index/verify
GET  /api/search/evidence?q=&limit=&paper=&source=
GET  /api/papers/{local_id}/citations?type=
GET  /api/papers/{local_id}/export?format=bib|ris|md&style=apa
POST /api/export                  # 筛选集（≤500 篇）→ 文本下载
```

### 风险 / 边界

1. **不重命名字段**：`index_status` 若在 service 层改名（`ready`→`is_ready`…），CLI 与 UI 立刻分叉。约定：原样返回。
2. **大项目作用域是 best-effort**：vector/tree 腿把 scope 拼成原生 `IN` 长表达式（`vector_store.py:98-100`）。MVP 用**显式论文多选**（≤数百篇）；项目作用域在 `len(paper_ids) > 1000` 时不要下推，改为在 `legs[].reason`/页面明示"scope=best-effort"。精确下推 = 核心线 follow-up。
3. **`citations` 只读**：绝不能触发 auto-expand；`citation_cache` 为空 → `{"refs": [], "cached": false, "hint": "在终端运行 drbrain citations <id> 展开"}`。refs/citing 的 `local_id/ingested` 需 service 侧按 DOI/规范化标题补 join（FR-L4 的"未入库"标注**不是现成能力**）。
4. **FR-S8 需改写**：证据检索**没有总数/游标**（单页 top-k）。改为"展示前 N 条（默认 20、上限 100）+ 明示仅展示前 N 条"；要翻页属新增后端能力。
5. **OKF 不进 MVP**：目录产物 + 需 GraphEngine 子图，产物契约（临时目录→zip→下载）另立。

---

## A3 · ask 状态透出（D4 必须修复项）

### 结论

**只修一个前置门**：`service.ask` 捕获 `AskIndexNotPreparedError`，返回与 CLI 完全相同的 `source_unavailable` 结构（附加 `status/hint/engine`），**同时保留 `unavailable` 键**；路由按新字段 `unavailable_reason` 分流状态码——**引擎未启用 → 503（保持）**，**索引未就绪 → 200**（TF2 的"非 5xx"）。检索层的 9 种 abstain 状态无需改动（本就在返回值里）。

### 依据

- 现状缺口：`app/service.py:416-435`（`ask` 无 `try`，`AskIndexNotPreparedError` 会落到全局异常边界 → 500，`app/web/__init__.py:190-199`）。
- 异常与 hint：`rag/engine.py:123-149`（`AskIndexNotPreparedError(engine, hint)`；`ask_prepare_hint` 引擎感知：`sql → drbrain index build`、`llamaindex → drbrain rag index`），抛出点 `:510-526`/`:594`。
- CLI 既有形状（照搬）：`cli/analysis_commands.py:393-425` → `{question, answer: "index not prepared … run `hint`", status: "source_unavailable", engine, hint, sources: [], evidence_ids: []}`。
- abstain 已透出：`rag/engine.py:755-782`（`_abstain_answer` 带 `status`）、`:698-719`（成功形状含 `sources/evidence_ids/route/telemetry`）；枚举 `rag/status.py:30-45`（ok/no_results/retrieval_failure/permission_denied/timeout/source_unavailable/insufficient_evidence/empty_answer/degraded）。
- **必须保持的契约**：`tests/test_app.py:125-127`（`test_ask_reports_unavailable_engine`：`unavailable is True` 且 `"llamaindex" in error`）、`:354`（`/api/ask` → 503 + `unavailable`）；路由 `routes/api.py:93-102`；会话 chat 分支 `routes/pages.py:196-213`。
- FR-P1 源头：`app/service.py:366-388`（`availability()` 的 `rag` = `resolve_engine(…) == "llamaindex"`，在 `rag_engine: sql` + 已发布 generation 时会误报"未启用"）；真实就绪度在 `index_commands.py:455-475`（`states.retrievable`）。

### 实现建议

```python
# src/drbrain/app/service.py
def ask(cfg, question, top_k=5):
    if resolve_engine(cfg, "llamaindex") != "llamaindex":
        return {**现有 error 文案（含 "llamaindex"）,
                "unavailable": True, "unavailable_reason": "engine_disabled",
                "status": "source_unavailable", "engine": str(li.rag_engine),
                "hint": ask_prepare_hint(cfg)}
    try:
        result = ask_llamaindex(cfg, db, question, top_k=top_k, streaming=False)
    except AskIndexNotPreparedError as exc:
        return {"question": question, "answer": f"index not prepared for engine {exc.engine!r}: "
                f"run `{exc.hint}`", "status": "source_unavailable", "engine": exc.engine,
                "hint": exc.hint, "sources": [], "evidence_ids": [],
                "unavailable": True, "unavailable_reason": "index_not_prepared"}
    return dict(result)

def availability(cfg) -> dict:
    """拆成 search_ready（= index_status.states.retrievable.ready 且 route.legs 非空）与
       ask_ready（= search_ready + llamaindex.enabled + llm.models），各自带 reasons + hint。"""

# app/web/routes/api.py:  POST /api/ask
reason = result.get("unavailable_reason")
status = 503 if reason == "engine_disabled" else 200   # index_not_prepared / abstain → 200
```

### 风险 / 边界

1. **`unavailable` 不能删**：三处测试 + 两个模板分支依赖它；本项是"附加字段 + 分原因"，不是改语义。
2. **503 只留给引擎未启用**：把 `index_not_prepared` 也返回 503 会直接违反 TF2。
3. **abstain ≠ 错误**：`no_results`/`insufficient_evidence`/`degraded` 都 200；`answer` 在未就绪/abstain 时是**提示语**，前端必须按 `status` 分支渲染（最容易出的 UI 事故）。
4. **`service.ask` 仍要求 `llamaindex.enabled: true`**（与 CLI 一致，`analysis_commands.py:391-401`）——不改引擎门；`availability()` 必须把这条作为 `ask_ready` 的 reason 如实说明。
5. **web 层临时分类要收口**：`routes/pages.py:85-122` 现在自行捕获 `AskIndexNotPreparedError`（注释已写"service 层接管"）。A3 落地后分类归 service，路由不再 import `drbrain.rag.engine`（过渡期共存，异常不再抛出即自然失效）。

---

## A4 · per-paper 索引可见性（FR-I1 / FR-I6 差值）

### 结论

**可给**，且**不改表**：`tree_nodes` 已有 `(local_id, state)` 索引、`leaves_missing_parent(local_id=…)` 已支持单篇。新增"1 个实时聚合 + 1 条快照 GROUP BY"，输出四态（定义写死、如实标注）：

| 状态 | 判据 |
| --- | --- |
| `no_body` | `papers.status == 'placeholder'`（引用占位，永远不会有正文） |
| `ingested_only` | 有 ready 修订但 active generation 里没有该论文的 leaf |
| `partial` | leaf 数 < blocks 数，或存在缺当前 profile 向量的 leaf / 缺父 leaf |
| `stale` | 实时 `document_revisions.revision` > generation 中该论文的 `doc_revision` |
| `ok` | 其余（三腿就绪且进版本） |

### 依据

- 索引支持：`storage/database.py:292-293`（`idx_tree_nodes_kind_state`、**`idx_tree_nodes_doc(local_id, state)`**）。
- 单篇已支持：`database.py:2850-2862` `leaves_missing_parent(local_id: str | None = None)`。
- 通用计数：`:2788-2800` `count_tree_nodes`、`:3046-3060` `count_node_vectors`、`:2444-2471` `get_content_blocks/count_content_blocks`、`:2312-2338` `get_document_revision`。
- 观测快照：`tree/observability.py:76-106`（`tree_snapshot`：documents/blocks/nodes/vectors/summaries/jobs/generation）。
- 快照读取面：`tree/reading.py:124-140`（`ReadOnlyTreeStore`，`mode=ro`+`query_only`）、`:198-219`（`list_tree_nodes(limit=10_000)`，**不能**用于全库聚合）；`tree_nodes.doc_revision` 是"进没进版本"的判据字段（表定义 `database.py:269-291`）。
- **别用**：`tree/prepare.py:313-333` `ready_node_rows` 会物化全部 ready 节点（10k 语料 ≈ 1.6M 行）。
- 论文侧列表/分页现成：`app/service.py:465-497` `papers()` + `database.py:4286-…` `list_papers`；游标工具 `service.py:445-463`。
- 可参照的 per-paper 状态形状：`paper_artifacts(paper_id, stage, status, fingerprint, error, attempts)`（`database.py:188-200`）——**仅作形状参照**，索引构建不写该表。

### 实现建议

```python
# src/drbrain/storage/database.py（只读聚合；一条 GROUP BY）
def index_facts_by_paper(self, *, paper_ids: Sequence[str] | None = None) -> list[dict]:
    """{local_id, revision, blocks, leaves, leaves_vectorized,
        leaves_missing_parent, state}"""

# src/drbrain/tree/reading.py（generation 侧，一条 GROUP BY；禁止全量物化）
def paper_leaf_counts(self) -> dict[str, int]:      # {local_id: ready_leaf_count}
def paper_doc_revisions(self) -> dict[str, int]:    # {local_id: doc_revision}（判 stale）

# src/drbrain/app/service.py
def index_coverage(cfg, *, project_id=None, status="", limit=50, cursor=None) -> dict[str, Any]:
    """{summary:{docs_total, no_body, indexed, retrievable, stale, partial, ingested_only},
        items:[{local_id,title,year,state,revision,blocks,leaves,leaves_vectorized,
                leaves_missing_parent,in_generation,generation_doc_revision}],
        total, next_cursor}"""

# app/web/routes/api.py
GET /api/index/coverage?project_id=&status=&cursor=&limit=       # 非 ok 排最前
```

顶部三数字（FR-I1）：`docs_total`（有正文）→ `indexed`（live 腿就绪）→ `retrievable`（直接取 `index_status.states.retrievable.ready`）。

### 风险 / 边界

1. **必须有界**：`paper_ids=None` 时走游标分页，`summary` 用聚合（别返回全库明细）；`index_coverage` 复用 `encode_cursor/decode_cursor`。
2. **placeholder ≠ 未索引**：单列 `no_body`，否则 FR-I1 会误导。
3. **`frontier_remaining` 与"未索引论文"是两件事**：前者= ready leaf 还没父 region（合法多根），两者都展示、语义分开。
4. **快照 vs 实时漂移是特性**：`in_generation` 读快照、`revision` 读实时；标注 generation id + manifest `created_at`，不要强行对齐。
5. **成本**：一次快照查询取回全部计数（不要每篇开一次连接）；非 `ok` 明细按 `local_id` 索引查。

---

## A5 · 运行只读扩展（讨论板 / 队列 / 预算 / 审批）

### 结论

**loop/store 的现有只读 API 已覆盖 FR-R6 的全部需求**（消息板/队列/预算/待审核），只差"审批历史列表"一项可选能力。service 只做三件事：作用域校验、**用 run_id 调用**（绝不用 topic）、脱敏。

| FR-R6 需求 | 现成读 API | 覆盖 |
| --- | --- | --- |
| 状态机 / 活跃步骤 / 可恢复 / **待人工审核** / 预算 | `RunGovernance.status(identifier)`（`loop/governance.py:27-44`） | ✅ |
| 事件 + 工具调用轨迹 | `RunGovernance.trace`（`:46-65`） | ✅（需 service 截断，见边界 6） |
| "为什么停下来了"（事件/工具聚合 + 是否完整审计） | `RunGovernance.audit_summary`（`:67-85`） | ✅ |
| 结论 → 证据定位 | `RunGovernance.evidence_lineage`（`:87-…`） | ✅ |
| 讨论板（proposals + critic reviews）+ 队列（含认领状态） | `TransitionService.front_half_snapshot(run_id)`（`loop/transitions.py:830-870`）/ `DurableFrontHalf.snapshot()`（`loop/front_half.py:83-136`） | ✅ |
| 预算明细 | `RunLedger.budget_snapshot`（`loop/store.py:656-676`） | ✅ |
| 审批**历史列表** | 只有按键查询 `RunLedger.approval_decision(run_id, idempotency_key)`（`:949-968`） | ⚠️ 缺列表 API（可选加） |

### 依据

- service 可以 import loop（分层只禁 web 层）：`app/service.py:1495-1496`、`:1401`。
- 现有 ledger 读法可直接沿用：`service.py:159-176`（`_ledger`，路径 `ledger_path(cfg)` = `autoresearch.run_dir/ledger.sqlite3`，`:121-126`）、`:542-580`（`runs()`）、`:630-664`（`run_detail`：events/claims/verified/experiments/**pending_approvals**/pending_steps/display_status）、`:771-828`（`run_claims()` 已把 proposals + critic_reviews + settlements 组装成结论行）。
- 作用域：`service.py:582-620`（`_run_row` / `_assert_run_scope` / `run_project`）；**风险点**：`RunGovernance._resolve`（`governance.py:267-272`）同时接受 run_id 与 topic。
- 写方法在同一个类里（`governance.py:210-266` pause/resume/cancel/approve/reject）→ **路由层绝不能暴露成 POST**。
- `MessageBoard` 是内存/文件对象（`loop/discussion.py:125-256`），持久化读的是 `research_*` 表 → 文案写"持久化的提案与队列"，不承诺"实时讨论流"。
- 轮询/SSE 现状：`routes/fragments.py:48-82`、`routes/stream.py:39-…`（继续用，不重写）。

### 实现建议

```python
# src/drbrain/app/service.py
def run_governance(cfg, run_id, project_id=None) -> dict[str, Any]:
    """{status: RunGovernance.status, audit: audit_summary, evidence: evidence_lineage}"""
def run_discussion(cfg, run_id, project_id=None) -> dict[str, Any]:
    """{proposals:[{proposal_id,claim_id,author,status,score,reviews:[…]}],
        queue_items:[{queue_item_id,proposal_id,status,score,claimed_by,created_at}],
        node_specs:{…}}"""
def run_trace(cfg, run_id, project_id=None, *, limit=500) -> dict[str, Any]:
    """RunGovernance.trace 的脱敏 + 截断版（事件 + tool_calls）。"""

# app/web/routes/api.py（全 GET）
GET /api/runs/{run_id}/governance     # 状态机 + 预算 + 活跃/可恢复/待审核 + 审计聚合
GET /api/runs/{run_id}/discussion     # 讨论板 + 队列 + reviews（只读）
GET /api/runs/{run_id}/trace          # 事件 + 工具调用（诊断折叠区）
```

### 风险 / 边界

1. **topic 会串项目**：service 必须先解析出 `run_id` 再调 governance（只传 run_id）。
2. **审批列表缺失**：FR-R6 用 `manual_review_steps`（有）即可满足；要"审批历史"需新增 `RunLedger.approval_history(run_id)`——**可选，不阻塞**。
3. **只读**：pause/cancel/approve/claim 一律不进本期（02 §4.8）。
4. **`interrupted` 是 WebUI 派生态**（不在 loop 状态机里）：继续用 `service.display_run_status`（`service.py:665-677`），不要写回 ledger。
5. **脱敏与折叠**：`trace`/`evidence_lineage` 的 payload 只进"诊断详情"折叠区；沿用 `redact_sensitive`。
6. **性能**：`RunGovernance.trace` 无 limit 会把全部事件读进内存 → service 侧截断（建议 500）并标 `truncated`。

---

## A6 · 静态资源指纹

### 结论

**已在工作区实现，无需再动代码**：frontend-dev 在 M8 加了 `asset_url(name)`（mtime 触发重算 + sha256 前 10 位 → `/static/<name>?v=<fp>`），`base.html` 三处引用全部切换；`/static/*` 仍 `max-age=86400`。**HTML 是 `no-store`**，所以"URL 随内容变 + 长缓存"组合正确，不需要改 `StaticFiles`、不需要配置注入、不引入构建步骤。

### 依据

- 实现：`app/web/__init__.py:60-85`（`_ASSET_FINGERPRINTS` + `asset_url`；文件缺失时退回无指纹 URL）、`:95`（`env.globals["asset_url"]`）。
- 模板已切换：`templates/base.html:7,9,10`（`app.css` / `vendor/htmx.min.js` / `app.js`）。
- 缓存头：`app/web/__init__.py:110-112`（`/static/` → `public, max-age=86400`；其它 → `no-store`；`setdefault` 可被覆盖）；挂载点未变 `:177-178`。
- 测试：`tests/test_app.py:416-421`（只钉可取/路径穿越 404，不断言缓存头）；`tests/test_webui.py:651-671`（静态文件随 wheel 分发）。
- 基线 P1 要求：`docs/webui-v2/03-frontend-baseline.md:291-293`。

### 实现建议

三条约定（写进实现清单，**0 代码**）：

1. 任何新静态资源（CSS/JS/字体/图片）**必须经 `asset_url()`**；模板禁硬编码 `/static/...`（自查：`grep -rn '"/static/' src/drbrain/app/web/templates/` 应为空）。
2. `/static/*` 的 `max-age` **不降级**为 `no-cache`（有指纹时长缓存是收益）。
3. 若 v2 拆多个 CSS/JS，每个都独立过 `asset_url`（不是只给入口）。

### 风险 / 边界

1. `?v=` 不改文件名 → 反代若"忽略 query"会削弱失效机制（本机单人部署无此问题；将来走反代要么尊重 query，要么切 hashed 文件名）。
2. 指纹按 mtime 触发 `git checkout` 会重算（只是多一次 miss，无正确性风险）。
3. 进程内缓存（单 worker 契约下无关）。
4. 不要改 `StaticFiles` 的 `ETag`/`Last-Modified` 行为（与 `max-age` 并存正常）。

---

## 7. 对 02-product-plan 的回填建议

1. **§7.2**：A1–A6 全行改 `✅ 已答复 → 04-arch-decisions.md §A1–A6`；A6 注"已由 frontend-dev 在 M8 落地"。
2. **§4.11**：第 5 条（静态指纹）注"已落地，见 04 §A6"；新增第 7 条"**后台任务统一读法**：run 用 `/api/runs/{id}/…`，`index build` 与 conformance 用 `/api/jobs/{job_id}`"。
3. **§5.1 M2/M3**：注明"页面壳已就绪（`index_corpus.html` / `search.html`），数据接线依赖 A1/A2/A3 的服务层落地"；`M4` 的 L4/L5（C3 stretch）依赖 A2 的 `citations/export` 薄封装。
4. **FR-S8 改写**：证据检索无总数/翻页 → "展示前 N 条（默认 20、上限 100）+ 明示仅展示前 N 条"。
5. **FR-P1 补充**：`找证据` 可用性只看索引就绪；`要答案` = 索引就绪 + `llamaindex.enabled` + LLM 配置（各自给 reasons），不要合并为一个布尔值。
6. **FR-I5 补充**：不提供取消；"重复提交"= 同 scope 复用既有 job（`already_running: true`）。

## 8. 建议开给核心线的 follow-up（不在本任务范围）

| # | 事项 | 影响 | 建议 |
| --- | --- | --- | --- |
| F1 | `claim_tree_job` 无 lease 续租 | 小时级构建互斥可能失效 | 加 `Database.renew_tree_job`（A1 实现时一并做） |
| F2 | ANN 腿的 scope 是原生 `IN` 长表达式 | 万篇项目无法精确限定 | 短期 best-effort + 如实标注；长期做集合过滤下推 |
| F3 | `research_approval_decisions` 无列表 API | 只能显示"待审核"不能显示历史 | 需要时加 `RunLedger.approval_history` |
| F4 | `record_stage/read_stage` 无调用点（`tree/observability.py:109-160`） | 无阶段历史（A1 用 job checkpoint 覆盖） | 需要 stage 历史时再启用 |
| F5 | `citations` 的 auto-expand 是网络写 | UI 不能触发 | 若要做，走显式 job + 幂等键 |
| F6 | `retrievers` 三处不一致 | 用户看到的找法 ≠ 文档 | leader 已决策（附录 C1）：UI 如实展示；核心线统一默认值 |
| F7 | 证据检索无总数语义 | FR-S8 | 见 §7.4 |

## 9. 编号对照（02 引用校准）

本文件压缩重排过一版，`02-product-plan.md` 中已写入的部分"04 §…边界 N"是旧编号。**以本表为准**（或由 PM 按本表回改 02 的引用）：

| 02 中的旧引用 | 本文件当前位置 | 条目内容 |
| --- | --- | --- |
| 04 §A1 边界 5（"重复提交判定"） | **§A1 边界 4** | 不加 `client_request_id`；幂等 = 同 scope 复用在跑 job |
| 04 §A1 边界 6（"共用读法不合并成表"） | **§A1 结论段**（边界只到 5 条） | 三种任务统一读法、各自 durable 身份 |
| 04 §A2 边界 7（单页 top-k / 无 total） | **§A2 边界 4** | FR-S8 需改写为"前 N 条" |
| 04 §A2 实现建议 5（`paper_meta` 下沉） | **§A2 实现建议 3）** | `cli/_helpers/display.py → storage/export.py:paper_meta` |
| 04 §A3 边界 5（FR-P1 / `ask_ready` reasons） | **§A3 边界 4** | `ask` 仍要求 `llamaindex.enabled`；`availability()` 拆分 |
| 04 §A3 边界 7（web 层临时失败分类归位） | **§A3 边界 5** | `pages.py:85-122` 收口到 service |
| 04 §7.3 / §7.4 / §7.5 / §8 F1–F7 | **不变** | M2/M3 排期 / FR-S8 / FR-P1 / follow-up 表 |
