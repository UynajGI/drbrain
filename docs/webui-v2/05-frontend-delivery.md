# WebUI v2 前端交付报告（阶段 B）

> 作者：frontend-dev（团队任务 #3 阶段 B）
> 日期：2026-09-16
> 范围：`src/drbrain/app/**`（WebUI）、`src/drbrain/services/{index_report,evidence_search}.py`（新 core 门面）、`src/drbrain/storage/paper_view.py`（显示层）、最小 service/API 扩展；未 commit（按 leader 要求留工作区统一收口）。
> 输入：`docs/webui-v2/02-product-plan.md`（M1–M8、§4 决策、§1.3 术语表、§6 验收）、`docs/webui-v2/04-arch-decisions.md`（A1–A6 前端契约）、`docs/webui-v2/03-frontend-baseline.md`（§4 硬约束）。

## 0. 交付状态一览

| 块 | 状态 | 说明 |
| --- | --- | --- |
| M8 壳层 | ✅ 完成 | 七项导航、mockup token 统一、`asset_url()` 指纹（04 §A6 已落地） |
| M4 文献库 | ✅ 完成 | 大纲可展开看原文片段（400 字内）+ 稳定 locator + `?node=` 定位高亮；L4/L5（引用/导出）按 leader 决策不入 MVP |
| M5 会话 | ✅ 完成 | 记忆三层显式分区 + 「记忆 ≠ 研究结论」双向声明 |
| M6 研究运行 | ✅ 完成 | 「已中断（可恢复）」+ 一键预填续跑；结论按通过/未通过分组 + S/R/O（有则显示）+ 证据可跳原文；实验卡状态徽标 + 作业信息折叠 |
| A2 门面下沉 | ✅ 完成 | `services/index_report.py`、`services/evidence_search.py`（CLI 行为零变化，见 §3 等价性证据） |
| M2 索引与语料 | ✅ 完成 | 三段状态 + 三条找法（含 pageindex/raptor 折叠说明）+ 索引版本 + 未完成量 + 自检（htmx 片段）+ 诊断详情 |
| M3 检索与证据 | ✅ 完成 | 找证据（四要素结果行、显示上限、原文/摘要区分）+ 要答案（A3 状态） |
| M7 设置 | ✅ 完成 | availability 修正（找证据/要答案分开 + reasons）+ 能力与插件分区入口 |
| M1 概览 | 🟡 部分 | 「能不能搜？」卡 + 检索与问答可用性卡 + 一键进索引页；逐篇差值与索引健康度细项依赖 A4 |
| A1 构建 job | 🟡 部分（并发协作者） | **core 已由并发协作者落地**：`src/drbrain/services/index_build.py`（587 行，未跟踪）提供 `ensure_index_build_job` / `run_index_build` / `IndexBuildBusy` / 确定性 scope slot + lease（`Database.renew_tree_job` 本就存在于 HEAD → F1 已满足）；`cli/index_commands.py` 已改为委派它。**service/API/UI 层归属待 leader 裁决**（见文末「并发写入提示」） |
| A2 citations/export | ⛔ 未做 | FR-L4/L5 stretch，leader 已定：等 M1–M3 落地后再定 |
| A4 index_coverage | ⛔ 未做 | 页面已注明“逐篇差值随覆盖明细一起提供” |
| A5 运行只读扩展 | ⛔ 未做 | FR-R6（后续迭代） |

## 1. 变更清单

新增（未跟踪）：

| 文件 | 行数 | 作用 |
| --- | --- | --- |
| `src/drbrain/services/index_report.py` | 574 | `index status/verify` 的 core builder（从 `cli/index_commands.py` 搬迁） |
| `src/drbrain/services/evidence_search.py` | 280 | `search` 的 payload builder + `run_evidence_search()`（含 `status`/`hint`） |
| `src/drbrain/app/web/templates/index_corpus.html` | ~150 | M2 页面 |
| `src/drbrain/app/web/templates/search.html` | ~193 | M3 页面（找证据 / 要答案） |
| `src/drbrain/app/web/templates/fragments/index_verify.html` | ~25 | 自检片段（htmx） |

修改（19 文件，+1754/−830）：`app/service.py`、`app/web/{__init__,labels}.py`、`app/web/routes/{pages,api,fragments}.py`、`app/web/static/app.css`、`app/web/templates/{base,dashboard,paper_detail,run_detail,runs,session_detail,settings}.html`、`cli/{index_commands,search_commands}.py`、`storage/paper_view.py`、`tests/{test_webui.py,tree/test_paper_view.py}`。

## 2. §6 验收对照

### TF1 · 建库与"能不能搜"体检
- [x] 空库打开首页能一眼看到“能不能搜？”并**一次点击**到达索引页（概览页卡片 + 侧栏常驻入口）。
- [x] 索引页并列 **已入库 N 篇 / 已建索引 X/Y 条腿 / 可检索 就绪|未就绪** 三段（`states.ingested/indexed/retrievable`，不是“数据库里有几篇”）。
- [x] 三条找法状态来自配置实际生效的路由（`route.legs` → bm25/vector/tree），旧写法 `pageindex/raptor` 用中文说明“已合并为按结构导航”，不显示为不可用。
- [x] 索引版本卡：generation / tree_state / 引擎后端 / 配置指纹一致性（`manifest_profile_id != profile_id` → 提示重建）。
- [ ] 点“构建索引”→ job：**未实现（A1）**；当前提示“显式后台任务 + 终端命令”，并说明页面会自动读到结果。
- [ ] 构建失败显示失败阶段 + 可复制原因：同上（A1 的 job checkpoint 才带 stage/pending）。
- [x] 自检逐项可读、失败项带“下一步”，原始 JSON 只在折叠里（`/ui/fragments/index-verify`）。
- [x] `frontier_remaining`/未完成量的语义分开：待补向量 / 暂存 / 待归父主题 / 落后 / 失败 / 摘要失败 + “多根≠故障”的说明。

### TF2 · 主题问答（含"系统不知道"）
- [x] 未建索引提问 → 页面显示「还没有索引/要答案不可用」+ 下一步，**HTTP 200 不返回 5xx**（自动化测试）。
- [x] 正常提问 → 答案 + 逐条出处（保留 v1 渲染；出处可点进文献、`node` 定位）。
- [x] `/api/ask` 503 只留给“引擎未启用”（04 §A3 边界 2）。
- [ ] 六种 abstain 状态可区分：目前可区分 `unavailable`(engine_disabled) / `index_not_prepared` / `search_failed` / `empty_question` / `source_unavailable` ——其余（no_results / insufficient_evidence / degraded / timeout）待 core 侧状态码（A3 已给方向，`ask` 已把 `status` 透出位留好）。

### TF3 · 找证据
- [x] 结果行四要素：原文片段 + 出处（论文/章节/块/字符区间，可跳转定位）+ 找法标记 + 索引版本。
- [x] 每条标注「原文片段 / 位置未标注」，未标注位置的行明确“只作线索、不作为可引用原文证据”。
- [x] 某条来源不可用时在 legs 行标注（`ok/empty/unavailable` + 原因）。
- [x] 按论文限定检索（`paper` 参数，逗号分隔；API 同名参数）。
- [x] 宽泛查询不卡：**显示上限**（默认 20、上限 100）明确标注“展示上限，不是命中总数”；不做翻页（leader 决策 2 / FR-S8 改写）。
- [x] 大项目（>1000 篇）显示“项目限定：尽力而为”（`scope.best_effort`，leader 决策 1 + F2）。

### TF4 · 研究闭环
- [x] 发起运行 → 详情页实时更新（v1 SSE 沿用，未改动）。
- [x] 断线显示“连接中断/重连中”，不改变运行状态（沿用 + 测试）。
- [x] 服务重启中断的运行显示为**已中断（可恢复）**+ 一键预填续跑入口（TF4 的关键项）。
- [x] 结论区：通过验证 / 未通过或尚未验证 分组；支持/反驳/正交计数（结算结果里有才显示）；证据可点进来源文献或计算产物。
- [x] 报告可下载（沿用）+ 计算产物作业信息可展开（产物字节本身仍只在作业目录）。

### TF5 · 文献库与引用
- [x] 新入库（无 legacy 文件）与老论文详情页表现一致：都走 `body_view()`（canonical 优先、legacy tree.json 兜底），都有大纲 + 原文片段 + 正文来源标注。
- [x] 从检索结果/证据跳进详情能定位片段并高亮（`?node=`，命中/未命中各有明确状态）。
- [x] 列表筛选条件进 URL（沿用）。
- [ ] 导出 BibTeX/RIS/Markdown：**未做**（FR-L4/L5，leader 决策未入硬 MVP）。

### 全局
- [x] 空态/加载/失败三态可区分；“没找到”与“出错”分开；表单错误保留输入（沿用 + 新页面遵守）。
- [x] 所有非 GET 交互带 CSRF 且同源（沿用 + 回归测试）。
- [x] `tests/test_webui.py` + `tests/test_app.py` 保持绿。
- [x] 静态资源带内容指纹（`asset_url()`），改版不会返回旧文件。
- [ ] 1440/1024/768/320 px 与键盘可达性：**未做人工目视**（worktree 无 headless browser；响应式沿用 v1 的媒体查询，新增元素使用既有栅格类）。

## 3. 验证证据

**测试**
- `tests/test_webui.py + test_app.py + tree/test_paper_view.py + test_cli_search.py + test_search_cmd.py` → **92 passed**。
- 本批新增 11 个用例：七项导航/资源指纹、/index 三态、/index 报告+自检、legs/states 视图纯函数、/search 两条路、answer 不 500、A3 原因+503 规则、evidence 上限与 scope、证据页四要素与空态、会话记忆分区、中断续跑、结论分组、文献大纲定位/locator。
- `tests/test_cli_index.py`：16 passed / **9 failed — 与本次改动无关**。失败根因是 `vendor/pageindex`、`vendor/raptor` **submodule 未初始化**（`git submodule status` 显示 `-`）；已在**原始代码**上复跑同一用例确认同样失败，故未 init submodule。

**CLI 行为零变化（搬迁等价性）**
- `drbrain index status --json` / `index verify --json`：重构前后 **字节级 diff 为空**，退出码（0 / 1）与 stderr 一致。
- `drbrain search --json`：diff 仅新增 `status`/`hint` 两行（有意为之的契约补充）+ `duration_ms` 抖动；stderr、退出码一致。

**实机冒烟**（`uv run drbrain webui --port 8765`，空语料 worktree）
- 14 个 URL 全 200 且内容命中：`/`、`/index`、`/search`、`/search?q=…&mode=evidence`、`/search?mode=answer`、`/papers`、`/sessions`、`/runs`、`/plugins`、`/settings`、`/api/index/status`、`/api/index/verify`、`/api/search/evidence`、`/ui/fragments/index-verify`。
- 页面全部按真实 payload 渲染（无编造数字）；服务已停止、端口已释放。

**静态检查**：`ruff check` + `ruff format --check` 对我改动/新增的文件全部干净（仓库仅剩 `scripts/serve_transformers_llm.py` 的既有 I001，未触碰）。

## 4. 约束落实（对照 02 §4.11 / 03 §4.1）

1. 分层：`app/web/*` 只做 HTTP→service 翻译；SQL 只在 `storage/` 与 `service.py`；页面与片段读同一批 service 模型（片段不打自家 JSON API）。
2. CSRF/CSP：沿用 `hx-headers` 与同源校验；未引入内联脚本、未引 CDN、未加依赖、无构建步骤。
3. 错误体双轨（`/api/*` 与 `HX-Request` → JSON；浏览器 → 错误页）。
4. 列表游标分页（沿用）；证据检索改为“前 N 条 + 明示上限”（无总数可用，按 leader 决策）。
5. 静态资源指纹（`asset_url`，sha256 前 10 位、按 mtime 缓存）。
6. `tests/test_webui.py` + `test_app.py` 保持绿（见 §3）。
7. 文案单一来源：新增状态/原因/检查项文案全部进 `app/web/labels.py`（`ANSWER_STATUSES`/`EVIDENCE_STATUSES`/`INDEX_REASONS`/`INDEX_CHECKS`/`LEG_LABELS`/`INDEX_STATE_LABELS`/`SEVERITY_TONES`），模板不硬编码颜色词；精确术语只在“诊断详情”折叠区。
8. 三条“不骗人”红线：不把插件说成在线；不把“部分覆盖”说成全部就绪（索引页明示）；不把模型生成内容当原文证据（无位置证据行明确降级）。另外：**不给索引就绪度编数字**（未接入的部分直接标“未接入/未标注”）。

## 5. 交接清单（未完成项，可照做）

1. **A1 构建 job（最高价值）**：按 `04-arch-decisions.md §A1` 实现
   - core `services/index_build.py`：`index_build_scope_key()` / `active_index_job()` / `run_index_build(cfg, *, force, tree_storage, notify, job_id, owner)`（搬 `cli/index_commands.py:index_build_cmd` 主体，每阶段 `save_checkpoint`）；
   - `Database.renew_tree_job()`（§A1 风险 1：lease 续租，或每次 checkpoint 重认领）；
   - service：`start_index_build()`（同 scope 有活跃 job → `already_running: true`，**单飞互斥**）、`jobs()`、`job_state()`；
   - API：`POST /api/index/build`（CSRF）→ 202；`GET /api/jobs`、`GET /api/jobs/{job_id}`（形状见 §A1）；
   - 页面：M2 行动卡把“终端命令”提示换成按钮 + 确认 + htmx 自轮询 1.5–3s（终态停），显示 `stage/stages_done/pending/live`，不承诺取消、不做 ETA。
2. **A2 citations/export**：`storage/export.py:paper_meta`（从 `cli/_helpers/display.py::_export_paper_to_meta` 纯搬迁，CLI 行为不变）+ service `citations/paper_export` + `GET /api/papers/{id}/citations`、`/api/papers/{id}/export`、`POST /api/export`；`citations` 只读、不得触发 auto-expand。
3. **A4 index_coverage**：`Database.index_facts_by_paper()` + `tree/reading.py`（`paper_leaf_counts`/`paper_doc_revisions`，禁止全量物化）+ service `index_coverage()` + `GET /api/index/coverage`；页面在 M2 的三段状态下方加“逐篇差值”表（非 ok 排最前，游标分页）。
4. **A5 运行只读扩展**：`run_governance/run_discussion/run_trace` + 三个 GET；`trace` 侧必须截断（§A5 风险 6）。
5. **小瑕疵**：证据 legs 的失败原因是 core 自由文本（英文），要中文化需 core 给错误码；`/api/search/evidence` 的 `limit` 超 100 是 422（如需静默截断改为 service 层 clamp 即可）。

## 6. 已知风险与边界

### 并发写入提示（重要）
交付过程中发现**同一 worktree 内有另一个写入者**（非本会话）：`src/drbrain/app/service.py` mtime 12:40:08、`src/drbrain/services/index_build.py` 12:40:03、`src/drbrain/cli/index_commands.py` 12:38:39，均晚于本会话对这些文件的最后编辑（~12:05–12:20）。证据：`services/index_build.py` 是 12:40 才出现的**未跟踪**文件，且其 docstring 明确写着 "(04-arch A1)"。
- 影响：A1 的 **core 层**已由该协作者落地（确定性 scope slot `job-<sha256(scope)[:20]>` + `claim_tree_job`/lease + 每阶段 checkpoint + `IndexBuildBusy`）；本会话剩余的 A1 工作（service `start_index_build/jobs/job_state`、`POST /api/index/build`、`GET /api/jobs*`、M2 构建按钮 + 轮询）与之**同属 `app/service.py` + `app/web/routes/api.py` + `index_corpus.html`**，两个写入者同时改会互相覆盖。
- 处置：本会话**主动停止 A1 的剩余实现**（不抢写），等待 leader 明确归属。
- 已验证共存：该协作者的改动 + 本会话改动**同时通过**测试（`test_webui + test_app + tree/test_paper_view + test_display_helpers` = **99 passed**），且 `drbrain index status --json` 与重构前基线**仍字节级一致**。

### 其他
- **未做人工目视**：worktree 无 headless browser（playwright 未装，且不应为验证引入依赖），视觉验收是 HTML/CSS + HTTP 层证据，没有截图；`design/webui-v1-mockup.html` 的像素级一致性需要人眼确认。
- **`vendor/*` submodule 未初始化**：会使 `index build` 相关 CLI 测试失败（与本次改动无关，但会影响任何人跑全量测试）。
- **`availability()` 的 `search_ready` 是廉价探针**（已发布 generation + ready 节点数），与 `/index` 的完整 `index status` 在“部分可用”边缘情况下口径可能不同；权威口径始终是索引页。
- **空语料下的页面**：所有新页面在空库下均给出可读空态（已实测），但“有真实数据”的分支主要靠测试夹具与 mock 验证，未在真实大语料上跑过。
