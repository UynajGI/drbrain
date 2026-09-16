# WebUI v2 前端基线（阶段 A）

> 作者：frontend-dev（团队任务 #3 阶段 A）
> 日期：2026-09-16
> 基线：worktree `/home/jiangyuan/.paseo/worktrees/3fiayvwv/spooky-pony`，commit `8a78f1a`（= origin/main）
> 范围：**只读盘点 + 开发回路实跑**。本阶段未改 `src/`、未装依赖、未改 `config*.yaml` / `.env`、未 commit。
> 关联：`design/webui-v1-mockup.html`（视觉基线）、`docs/webui-design.md`（旧契约，§5 信息架构 / §6.2 SSE）、`docs/webui-v2/02-product-plan.md`（待产出）

---

## 0. 结论摘要（TL;DR）

1. **开发回路可用**：在本 worktree 执行 `uv run drbrain webui`（cwd = worktree）**约 1 秒启动成功**，监听 `127.0.0.1:8765`，令牌写入 `config/webui_token`（0600）。空 worktree 首次启动会**自动建空库** `data/drbrain.db`（约 580 KB，schema 已建）；6 个一级页面全部 `200` 且都有明确空态文案，无 500、无异常日志。
2. **前端形态**：Jinja2 服务端渲染 + **本地 vendored htmx 2.0.7** + 一份 319 行原生 `app.js`；**零 CDN、零图表库、零构建步骤、零前端框架**。CSP `default-src 'self'` 从架构上禁止外链资源。
3. **SSE 已按 `docs/webui-design.md` §6.2 实现**（`id` = `event_seq`、`Last-Event-ID` 续传、心跳、终态关流、登录过期跳转），但走**原生 `EventSource`**（没用 htmx SSE 扩展），端点是 `/api/runs/{run_id}/stream`（契约文里写作 `/stream`）。
4. **对 v2 的三条硬约束**：(a) 页面/片段一律经 `app/service.py` 门面取数，`app/web/` 不许碰 SQL；(b) 任何非 GET 必须带 CSRF（`X-CSRF-Token` 头或 `csrf_token` 表单字段）+ 同源 `Origin` 校验；(c) 静态资源必须自托管（CSP）并加版本指纹（当前 `/static/*` 是 `max-age=86400` 且无指纹）。
5. **回归护栏**：`tests/test_webui.py`（800 行）+ `tests/test_app.py`（421 行）钉住了现有路由与契约语义，v2 改版必须保持全绿。

---

## 1. 前端资产盘点

### 1.1 目录结构与规模

```
src/drbrain/app/
├── __init__.py              33 行
├── auth.py                 235 行   令牌 / 登录会话 / CSRF cookie（framework-agnostic）
├── service.py             1772 行   业务门面（47 个公开函数），唯一数据入口
└── web/
    ├── __init__.py         230 行   create_app()：中间件、异常边界、模板过滤器、路由挂载
    ├── deps.py             185 行   authenticate / require_csrf / resolve_project / render / is_htmx
    ├── labels.py           101 行   状态与错误码 → 中文文案（唯一"文案字典"）
    ├── routes/
    │   ├── pages.py        466 行   17 条页面与表单路由
    │   ├── api.py          413 行   /api/* 共 27 条 JSON 路由
    │   ├── fragments.py    127 行   /ui/fragments/* 共 4 条 htmx 片段
    │   ├── stream.py       122 行   /api/runs/{run_id}/stream（SSE）
    │   └── auth_routes.py  201 行   /login /logout /api/auth/*
    ├── templates/         17 个文件 / 1084 行（见 1.3）
    └── static/
        ├── app.css         337 行 / 16.7 KB   设计 token + 全部样式（单文件）
        ├── app.js          319 行 / 11.0 KB   原生 JS（无依赖）
        └── vendor/         htmx.min.js 51 KB（v2.0.7）+ htmx.LICENSE
```

WebUI 相关的回归测试：`tests/test_webui.py`（800 行）、`tests/test_app.py`（421 行）。

### 1.2 页面清单与路由

页面路由（`web/routes/pages.py`，全部经 `Depends(authenticate) + Depends(require_csrf)`）：

| URL | 模板 | 数据来源（service 门面） | 说明 |
|---|---|---|---|
| `GET /` | `dashboard.html` | `dashboard()` | 概览：KPI + 首次使用 + 最近运行 / 会话 |
| `GET /papers` | `papers.html` | `papers()` | 文献库列表，`q` / `status` / `cursor` 游标分页 |
| `GET /papers/{local_id:path}` | `paper_detail.html` | `paper_detail()` | 元数据 / 章节大纲 / 证据定位 |
| `GET /sessions` | `sessions.html` | `sessions()` | 会话列表 + 新建表单（支持 `?new=1&paper=ID` 预填） |
| `POST /sessions` | — | `create_session()` | 303 重定向到会话详情 |
| `GET /sessions/{session_id}` | `session_detail.html` | `get_session()` | 对话、记忆分层、发起运行 |
| `POST /sessions/{id}/chat` | `session_messages.html`（htmx）或 303 | `chat_in_session()` | htmx 请求返回片段，否则整页回跳 |
| `POST /sessions/{id}/runs` | — | `start_run()` | 303 → `/runs/{id}` |
| `POST /sessions/{id}/memory/promote` | — | `promote_memory()` | 303 |
| `GET /runs` | `runs.html` | `runs()` + `sessions()` | 运行记录 + 发起新运行 |
| `POST /runs` | — | `start_run()` | 303 |
| `GET /runs/{run_id}` | `run_detail.html` | `run_detail()` + `run_claims()` + `experiments()` + `run_events_tail()` | 阶段 stepper + 事件流（SSE）+ claims/证据 + 报告下载 |
| `POST /runs/{id}/memory/sync` | — | `record_run_memory()` | 303（闭环沉淀） |
| `GET /plugins` | `plugins.html` | `plugin_catalog()` | 发现列表 + conformance 轮询 |
| `POST /plugins/{name}/conformance` | — | `start_conformance()` | 303 |
| `GET /settings` | `settings.html` | `settings_view()` | 有效配置 / 可用性 / 登录会话 / 审计 / token 轮换 |
| `POST /settings/token/rotate` | `settings.html` | `auth.rotate_bootstrap_token()` | 轮换后删除 session 与 CSRF cookie |

其他路由面：

| 前缀 | 文件 | 内容 |
|---|---|---|
| `/api` | `routes/api.py` | 27 条 JSON 路由：projects / dashboard / search / ask / papers / sessions(+chat/memory) / runs(+events/claims/report/evidence/artifacts) / run-status / experiments / plugins(+conformance) / settings / assets |
| `/ui/fragments` | `routes/fragments.py` | 4 条 htmx 片段：`paper-rows`、`run-events`（支持 `after`/`before`）、`session-messages`、`conformance` |
| `/api/runs/{run_id}/stream` | `routes/stream.py` | SSE |
| `/login`、`/logout`、`/api/auth/*` | `routes/auth_routes.py` | 登录 / 登出 / verify / rotate |
| `/static` | `web/__init__.py` | `StaticFiles` 挂载 |

> 事实：JSON API 与页面/片段读取**同一批 service 模型**，不是页面反过来打 API。片段层注释明确写了"never call the JSON API over HTTP, so there is one data path per view"。

### 1.3 模板继承与组件

- **单一布局**：`base.html`（76 行）是唯一壳层，10 个模板 `extends "base.html"`；壳层含 sidebar（brand + 项目切换 + 6 项一级导航 + 退出登录）、内容区、页脚，以及 `hx-headers`（CSRF）与 `data-login-url`。
- **例外**：`login.html`（36 行）**不继承 base**，是独立无侧栏页面。
- **宏库** `_macros.html`（75 行，7 个宏）：`csrf_field`、`status_badge`、`cursor_pager`（内含 `hx-boost` 局部翻页）、`empty_state`、`event_row`、`message_block`、`memory_entry`。10 个页面 + 4 个片段都 `import "_macros.html"`。
- **片段复用**：`papers.html` 用 `{% include "fragments/paper_rows.html" %}` 保证"首屏服务端渲染"与"htmx 翻页"共用同一份行模板。
- 模板里 **0 个内联 `<style>`、0 个内联 `<script>`**（配合严格 CSP 的刻意设计）。

### 1.4 前端库与加载方式

| 资产 | 来源 | 加载方式 | 备注 |
|---|---|---|---|
| htmx **2.0.7** | 本地 `static/vendor/htmx.min.js`（Zero-Clause BSD） | `<script src="/static/vendor/htmx.min.js" defer>` | 无 CDN；不使用 htmx SSE 扩展（`hx-ext` 全仓未出现） |
| `app.js` | 本地 | `<script src="/static/app.js" defer>` | 原生 JS，无依赖；项目切换、提交防重、htmx 错误横幅、SSE 生命周期 |
| `app.css` | 本地 | `<link rel="stylesheet" href="/static/app.css">` | 单文件 337 行，`:root` 定义 token（surface/text/border/action/status、radius、shadow、字体栈） |
| 图表 / 图可视化库 | **无** | — | 全仓 grep 无 chart/plotly/d3/echarts/cytoscape；mockup 里也只有 7 个内联 SVG 图标、0 个 canvas |
| 构建步骤 | **无** | — | 无打包器、无 `package.json`、无 node 依赖 |

htmx 使用点（现状）：`_macros.cursor_pager`（`hx-boost` + `hx-target` + `hx-select` + `hx-push-url` 局部翻页）、`plugins.html` / `fragments/conformance.html`（`hx-get` + `hx-trigger` + `hx-swap` 轮询符合性报告）、`session_detail.html`（`hx-post` 提交对话）、`run_detail.html`（`hx-get` 拉取历史事件）。其余交互（项目切换、SSE、表单防重）是 `app.js` 的原生实现，通过 `data-*` 属性绑定（无内联 handler，满足 `script-src 'self'`）。

### 1.5 SSE / 流式现状（对照 `docs/webui-design.md` §6.2）

| §6.2 契约 | 实现现状 | 位置 |
|---|---|---|
| 1. `id` = `event_seq`；数据保留 `seq/type/actor/payload/created_at` | ✅ 完全一致 | `stream.py:_sse()` / `app.js:buildEventRow()` |
| 2. 首连支持 `after`，重连用 `Last-Event-ID`；至少一次 + 前端去重；快照附游标 | ✅ `after` + `Last-Event-ID`；前端按 `seq` 去重（页面本就是单 run 作用域）；`run_detail` 先渲染末 100 条再以 `after` 接续 | `stream.py`、`app.js:seen` |
| 3. 心跳 / 关代理缓冲 / 断线提示不改变任务状态 | ✅ 每 15 次轮询（≈15 s）发 `: keep-alive`；`X-Accel-Buffering: no`、`Cache-Control: no-store`；断线显示"连接中断，重连中…"徽标 | `stream.py`、`app.js:source.onerror` |
| 4. 主动关闭 / 终态结束 / 历史只读 | ✅ `beforeunload` 关流、终态收到 `end` 后 `source.close()`、历史 `/events` 与 `run-events` 片段只读不重算 | `app.js`、`fragments.py` |
| 5. 浏览器 SSE 语义 + 登录过期处理 | ✅ 额外发 `auth-expired` 事件 → 跳 `/login?next=…`；30 次轮询复审一次 cookie/bearer 有效性 | `stream.py` |

实现层面的补充事实（非契约冲突，属于 v2 优化点）：
- 传输是**轮询式 SSE**：服务端每 1 s 查一次 `run_events` + `run_detail`，并且**每秒都发一帧 `status`**（无论是否有变化）。
- 未设置 SSE `retry:` 指令，重连间隔交给浏览器默认（≈3 s）——契约里"按配置管理重试与超时"目前**没有落成配置项**。
- 端点路径是 `/api/runs/{run_id}/stream`，与契约 §6.2 里写的 `/stream` 命名不同（前者更符合现有 `/api` 前缀约定）。

### 1.6 认证与安全头（约束前端实现的部分）

- 令牌文件：`<cwd>/config/webui_token`（0600，43 字符 URL-safe）。root 的判定是**环境变量 `DRBRAIN_ROOT` / `DRBRAIN_RUNTIME_ROOT` 优先，否则 `Path.cwd()`** → 在 worktree 里跑就是 worktree 局部，不会碰主库。
- Cookie：`drbrain_webui`（HttpOnly、SameSite=Strict、`Max-Age=43200`）+ `drbrain_csrf`（JS 可读，用于双提交校验）；登录会话 TTL 12 h，存 `webui_sessions`（只存 hash）。
- 非 GET 请求（页面表单与 htmx）：需要 `X-CSRF-Token` 头（htmx 由 `base.html` 的 `hx-headers` 统一注入）或表单 `csrf_token` 字段，并要求 `Origin` 与当前 scheme+host 一致；失败返回 `403 {"error","code":"csrf"|"origin"}`。
- 响应头（中间件统一加）：非 `/static` 一律 `Cache-Control: no-store`；`/static/*` 为 `public, max-age=86400`；另有 `X-Content-Type-Options`、`Referrer-Policy`、`X-Frame-Options: DENY`、CSP：
  `default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'`
- 错误分流：浏览器导航 → 渲染 `error.html`（2.4 KB 带壳层）；`/api/*` 或带 `HX-Request: true` → JSON `{"error","code"}`。

---

## 2. 开发回路实跑（证据）

### 2.1 命令与启动

```bash
cd /home/jiangyuan/.paseo/worktrees/3fiayvwv/spooky-pony
uv run drbrain webui --port 8765        # 后台启动，日志重定向到 /tmp/drbrain-webui-baseline-3fiayvwv.log
```

stdout（全部 3 行，`--log-level warning`，无请求日志）：

```
DrBrain WebUI → http://127.0.0.1:8765/
访问令牌: Cq_MTpf4MoYzlJEvsehLth5cgQAJVj4DjK9f_Fi4qM8
（同样保存于 config/webui_token，0600；登录后可在设置页重置）
```

- **启动成功**，HTTP 就绪耗时约 1 s（轮询 `/login` 第一次即 200，无重试）。
- 端口 `8765`（默认值，`--port` 可改）；host `127.0.0.1`（默认）。
- **未见任何 stderr / 异常**；只有一个 lifespan 提示不适用（未触发）。

### 2.2 登录回路

| 步骤 | 结果 |
|---|---|
| `GET /`（匿名） | `303 → /login?next=/`（`deps.authenticate` → 401 → 跳转登录） |
| `GET /login` | `200`，1 269 B，`<title>登录 · DrBrain 工作台</title>`；外链仅 `/static/app.css` |
| 表单字段 | `next`(hidden, `/`) + `token`(password) —— 运行时从 HTML 解析得到 |
| `POST /login`（token） | `303 → /`，`Set-Cookie: drbrain_webui=…; HttpOnly; Max-Age=43200` + `drbrain_csrf=…; Max-Age=43200; Path=/` |
| token 来源 | `config/webui_token`，44 B，权限 `0600`（内容 43 字符） |

### 2.3 页面探针（登录后逐个访问）

| 路径 | 状态 | 响应体字符数 | 结果 |
|---|---|---|---|
| `/` | 200 | 4 466 | 面板：首次使用 / 最近运行 / 最近会话；空态文案齐全 |
| `/papers` | 200 | 3 387 | 空态："文献库还是空的 / 先在终端运行 drbrain ingest 导入 PDF，然后回到这里。" |
| `/sessions` | 200 | 3 361 | 面板：新建会话 / 全部会话 |
| `/runs` | 200 | 3 909 | 面板：发起新运行 / 运行记录 |
| `/plugins` | 200 | 2 495 | 空态："没有发现插件 / 在 config.yaml 中设置 autoresearch.plugins_dir 指向插件目录…" |
| `/settings` | 200 | 11 417 | 有实际内容（config / availability / login_sessions / audit） |
| `/api/dashboard` | 200 | 352 | `papers/concepts/edges/arguments/uploaded = 0`，project = `prj-default（默认项目）`，ledger 全 0，`availability = {llm:true, rag:true, autoresearch:true}` |
| `/api/projects` | 200 | 111 | 1 个项目（`prj-default`，`is_default:true`） |
| `/api/assets` | 200 | 594 | `database.path = <worktree>/data/drbrain.db`（580 KB）；`ledger.path = <worktree>/workspace/autoresearch/ledger.sqlite3`（`bytes: null`，**尚未创建**）；`plugins_dir: null` |
| `/api/plugins` | 200 | 2 | `[]` |
| `/ui/fragments/paper-rows?project_id=prj-default` | 200 | 133 | 片段（无 `<html>`）：空态 `div.empty` |
| `/ui/fragments/conformance?plugin_name=&check_id=` | 200 | 80 | `报告不可用（可能已被删除）。` |
| `/ui/fragments/run-events?run_id=nope` | 404 | 51 | JSON `{"error":"unknown research run","code":"not_found"}` |
| `/runs/does-not-exist`（浏览器式） | 404 | 2 387 | HTML `error.html`（带壳层） |
| `/api/runs/does-not-exist/events` | 404 | 55 | JSON `{"error":"unknown research run","code":"run_not_found"}` |
| `/static/app.css` | 200 | 16 567 | `text/css`（文件实际 16 715 B） |

> 表中"响应体字符数"是探针侧对响应文本的字符计数（含中文的响应按 UTF-8 落盘会更大）；量级用于判断页面大小趋势即可。

安全行为抽查：

| 探针 | 结果 |
|---|---|
| `POST /runs` 无 CSRF token | `403`（HTML 错误页，非 hx/api 请求） |
| `POST /runs` 带 CSRF、topic 为空 | `303 → /runs?project_id=prj-default&error_code=empty_topic`（错误码驱动的表单反馈） |

### 2.4 空库 / 空语料下的表现（worktree 特有）

- worktree 里 `data/` 初始**只有 `logs/`**，`workspace/` 不存在 → 这是一个**完全空白的数据根**。
- 首次启动后：`data/drbrain.db` 被**自动创建**（593 920 B ≈ 580 KB，权限 `600`），schema 已建；`config/webui_token` 新建（同时新建了 `config/` 目录）；`data/logs/drbrain.log`（10 KB）。
- 结论：**空数据根不需要预先 `drbrain setup` 也能把 WebUI 跑起来**；空态由页面文案承担，没有报错、没有 500。这对 v2 是好事——任何新页面都应沿用"空态可读、不报错"的标准。
- 注意（对开发者的提醒）：worktree 内的 WebUI 看到的是**空库**，与主库无关。空态不是 bug。

### 2.5 运行期生成物与副作用（worktree 内，均已 gitignore）

| 路径 | 大小 / 权限 | 说明 |
|---|---|---|
| `data/drbrain.db` | 580 KB / 600 | 首次启动自动建库建 schema（`.gitignore:30 data/`） |
| `data/logs/drbrain.log` | 10 KB | loguru 输出 |
| `config/webui_token` | 44 B / 600 | 令牌文件（`.gitignore:36`） |

`git status --short` 全程为空，仓库未受污染；`workspace/` 未被创建（ledger 只在真正发起运行时才落地）。

### 2.6 观察到的问题与风险

| # | 严重度 | 现象 | 证据 | 建议 |
|---|---|---|---|---|
| P1 | 低 | **静态资源无版本指纹**：`/static/*` 带 `public, max-age=86400`，改 CSS/JS 后浏览器可能 24 h 内继续用旧文件 | `web/__init__.py:_SecurityHeadersMiddleware` | v2 给 `<link>`/`<script>` 加 `?v=<hash 或 mtime>`，或改成 `no-cache` + ETag |
| P2 | 低 | **SSE 每秒一帧 `status`**（不管状态是否变化），且服务端每 1 s 查两次库 | `stream.py` 循环 | v2 只在状态/计数变化时发帧；轮询间隔可配；长跑的 run 详情页会长期占用一个连接 + 每秒 2 次查询 |
| P3 | 低 | **重试/超时没有可配置项**（§6.2 第 3 条要求"按配置管理"） | `stream.py` 无 `retry:`，无 config 读取 | 要么补配置项，要么在 v2 文档里明确降级为"浏览器默认 + 前端徽标提示" |
| P4 | 低 | **非 hx/api 请求的 403/404 返回 HTML 整页**，htmx 请求返回 JSON —— 两套错误体 | `web/__init__.py:_render_error` | 前端若新增 htmx 交互，务必带 `HX-Request` 头（htmx 自动带）并处理 JSON 错误体；`app.js` 已有全局兜底横幅 |
| P5 | 信息 | **文案硬编码在模板 / `labels.py`**，无 i18n 层 | `templates/*.html`、`web/labels.py` | 若 v2 要双语，需要先抽文案字典，不建议在改版里顺手做 |
| P6 | 信息 | `login.html` 不继承 `base.html`（无侧栏、独立壳） | `login.html` 头部无 `extends` | 若 v2 统一壳层，需单独处理登录页 |
| P7 | 信息 | 模板热重载**已生效**（Jinja `auto_reload=True`，`FileSystemLoader`，无字节码缓存）→ 改模板不用重启；但改 `.py`（路由/service）**必须重启**（uvicorn 未开 `--reload`） | `build_templates()` 实测输出 `auto_reload: True` | 开发时注意：模板改动即时可见，Python 改动要重启进程 |

### 2.7 复现步骤

```bash
cd /home/jiangyuan/.paseo/worktrees/3fiayvwv/spooky-pony

# 1) 起服务（后台，日志落 /tmp）
uv run drbrain webui --port 8765 > /tmp/drbrain-webui.log 2>&1 &

# 2) 拿令牌（也可从 stdout 复制）
TOKEN=$(cat config/webui_token)

# 3) 登录并访问首页（curl 在本仓库被工具规则拦；等价做法是用 Node fetch 或浏览器）
#    浏览器：打开 http://127.0.0.1:8765/ ，粘贴 $TOKEN 登录

# 4) 停服务：kill 该进程组
kill -TERM -- -<PGID>
```

本次实测的收尾：`kill -TERM -- -3207421` 后 `ss -ltnp | grep 8765` 为空 → **进程已停止，端口释放**。

---

## 3. v1 视觉基线概括（`design/webui-v1-mockup.html`）

1. **一套深蓝学术控制台视觉**：主色 `#1E40AF`、辅助 `#3B82F6`、强调琥珀 `#D97706`、背景 `#F8FAFC`、卡片纯白、正文 `#0F172A`，配 Fira Sans / Fira Code 与 6 px 圆角、极轻阴影；28 个 token（color / space / radius / font / shadow）构成一套可直接抄进 `app.css` 的设计系统。
2. **固定壳层**：左侧栏（DB 徽标 + 项目下拉含"＋新建项目" + 6 项带内联 SVG 图标的导航，其中"研究运行"带计数徽标）+ 顶栏（面包屑 + 全局检索框带 `Ctrl+K` 提示 + "单用户会话 · 127.0.0.1:8420" 状态芯片）+ 主内容区（`page-head` 标题/副标题/右侧操作按钮）。
3. **每个页面=卡片网格**：`g-stats` KPI 卡（大数字 + delta 行，警告态变色）、`g-2` 双栏卡、表格、`详情抽屉` 可展开行（论文详情含"概览/树/概念/引用" tabs）、**9 段 stepper**（检索→抽取→gap→假设→讨论→实算→核验→沉淀→报告，done/current 三态）、事件流（时间 + 节点 + mono 标识）、claims/证据 kv 行、插件网格（密钥打码 + "未检测"态）、设置页 field 行。
4. **纯静态实现**：全文件零外链（无 CDN、无字体、无图表库），柱状/迷你趋势都用 CSS `div.bar` 画，图标是 7 个手写 SVG，仅 1 个 `<script>`。
5. **它是"信息架构 + 视觉密度"的目标稿**，不是可跑的实现：脚本注释明写"设计稿用；实现版为 FastAPI 路由 + htmx"，页面切换只是前端 `.page.active` 切换。

实现版与 mockup 的可见差异（v2 改版的抓手）：

| 维度 | v1 实现（现状） | v1 设计稿（mockup） |
|---|---|---|
| 侧栏 | 纯文字导航，无图标、无运行计数 | 图标导航 + 运行数徽标 |
| 顶栏 | 无 | 面包屑 + 全局检索（Ctrl+K）+ 环境/端口芯片 |
| 配色 | `--action #2f5fd0`、Inter 字体 | `#1E40AF`、Fira Sans（两套 token 未对齐） |
| 概览 | 3 个面板（首次使用/最近运行/最近会话） | 4 张 KPI 卡 + 趋势/分布卡 + 运行表 |
| 论文详情 | 独立详情页 | 列表内"详情抽屉"+ tabs |
| 运行详情 | stepper + 事件流（有） | stepper + 实算门/claims 证据更密 |
| 图表 | 无任何图表 | CSS 柱状/趋势条（仍无图表库） |

---

## 4. 技术约束与建议

### 4.1 现状事实（硬约束，v2 必须遵守）

1. **分层**：`app/web/*` 只做 HTTP → service 翻译；`app/service.py` 是唯一业务门面（47 个公开函数）；`app/auth.py` 是唯一认证边界面；**web 层禁止 SQL**。
2. **CSP 严格**：`script-src 'self'` + `default-src 'self'` → 不能引 CDN、不能加内联 `<script>`；新库必须 vendored 到 `static/vendor/` 并从 `/static` 加载。`img-src` 允许 `data:`。
3. **CSRF**：新增任何非 GET 交互，都要带 `X-CSRF-Token`（htmx 由 `base.html` 的 `hx-headers` 注入）或表单 `csrf_token`，且保持同源；否则 403。
4. **错误体双轨**：`/api/*` 与 `HX-Request` → JSON；浏览器导航 → `error.html`。新页面要复用 `labels` 的码→文案机制（`ERROR_MESSAGES` 7 条重定向码、`ERROR_TITLES` 12 条错误页标题；URL 里只能传码，不能传文案，防止 `?error=` 注入任意文本）。
5. **数据规模前提**：唯一数据源是本机 SQLite（`data/drbrain.db`）+ ledger，无前端状态存储；列表一律游标分页（`cursor`），不能假设全量返回。`papers()` / `runs()` / `sessions()` 都是游标/limit 语义。
6. **静态资源组织**：现在只有 3 个文件（`app.css` / `app.js` / `vendor/htmx.min.js`），无构建。若 v2 引入多文件或编译产物，"v1 vendored、v2 换构建产物时路径由配置注入"是旧契约 §8 已声明的方向，但**当前代码里没有配置化的静态路径** —— 引入构建步骤会新增一条 v2 依赖，需产品方案确认。
7. **回归测试**：`tests/test_webui.py`（800 行）+ `tests/test_app.py`（421 行）覆盖现有路由/契约；改版时这些测试是必须保持绿的基线（若确需改语义，要同步改测试并说明）。
8. **无 i18n**、**无图标集**、**无图表库**：三者都是"从零引入"的决策点，各有成本。

### 4.2 建议（分专题，均为"建议"非现状）

**A. htmx 局部刷新 vs 全页**
- 建议**继续以服务端整页渲染为主干**，局部刷新只在三处使用：① 长列表翻页/筛选（已有 `cursor_pager` + `/ui/fragments/paper-rows` 范式）；② 长轮询型就地更新（conformance、运行事件）；③ 对话式提交（会话消息）。**不要**把整站改成 SPA 式 htmx 拼装。
- 理由：现有片段层与首屏共用同一套模板（`include`），改版风险最低；服务端渲染让空态/错误态/权限态天然一致（本次实测 6 页全 200 且空态可读就是证据）。
- 具体做法：新增片段统一放 `/ui/fragments/*` 并 `import _macros.html`；首屏用 `{% include %}` 复用同一片段，避免"首屏 HTML 与 htmx HTML"两份漂移。
- 代价提示：htmx 局部替换后需要重新初始化 JS（现有 `app.js` 已用 `htmx:afterSwap` 处理 SSE 重绑），新交互要沿用该模式。

**B. 图表库取舍**
- 现状零图表库、CSP 禁外链。建议 **v2 第一版仍不引图表库**：mockup 里 KPI/趋势/分布都是 CSS 条 + 数字，信息密度够；真实数据主要是计数（papers/concepts/edges/events/claims），CSS 条 + 内联 SVG sparkline 即可覆盖。
- 若产品方案确实要图（如知识图谱可视化、运行时间线），再单项决策：候选都要 vendored 本地（CSP 约束），体积与许可证要在方案里写明；图谱类可视化还要考虑数据量（`/api` 现在没有返回子图结构，需要新增 service 方法 + 路由，属于后端工作量）。
- 明确不建议：为仪表盘引入整套图表框架（体积/许可证/离线 vendored 维护成本）。

**C. 静态资源组织**
- 保持"无构建"是当前最低风险路径：新增样式继续进 `app.css`（或按页面拆分多个 `<link>`），新增 JS 继续用原生 + `data-*` 绑定（CSP 下不能内联）。
- 必须补：**cache-busting**（P1）。最小值做法是给 `base.html` 里的 `/static/app.css`、`/static/app.js` 加 `?v={{ 版本或 mtime }}`，或把 `max-age` 降到 `no-cache`。
- 若要引入构建产物：先按旧契约 §8 把静态目录/文件名做成配置项（当前没有），并保证 `drbrain webui` 在"未构建"状态下仍可运行（否则破坏 `uv run drbrain webui` 开箱即用）。

**D. SSE 是否复用**
- 建议**复用现有 `/api/runs/{run_id}/stream` 机制与前端 `app.js` 的 SSE 生命周期代码**，不要在 v2 另起一套。理由：契约 §6.2 的四个硬点（seq id、续传去重、心跳、终态/登录过期）都已实现并有测试；重写等于重踩一遍。
- 改造点按优先级：① 只在状态/计数变化时发帧（P2）；② 心跳改成独立计时器（现在是轮询计数）；③ 若 v2 出现"非 run 的实时视图"（例如语料构建进度），把这个 handler 抽成通用"轮询式 SSE"工具（当前是单用途实现）——但只有真的需要时再做。
- 前端注意：`EventSource` 不能带自定义头（CSRF 靠 cookie + 同源），所以新的实时端点也必须走 cookie 认证。

**E. 其他**
- 侧栏/顶栏是"整站壳层"，改版先改 `base.html` 一处即可覆盖 10 个模板；`login.html` 是独立壳，别漏。
- 状态色/状态文案统一走 `labels.status_of()` + `_macros.status_badge()`（`STATUS_LABELS` 23 项：运行 7 / conformance 8 / claim 4 / 论文 4，每项映射到 5 类 tone：ok/run/warn/bad/muted），改版时**不要**在模板里硬编码颜色词，保持"状态→色"单一映射。（同一个字符串表也被 SSE `status` 帧复用，服务和前端不会各存一份。）
- 空态统一用 `_macros.empty_state(title, hint)`，文案里保留"下一步怎么做"（现有空态都带 CLI 提示，是很好的模式）。

### 4.3 待产品方案（02-product-plan.md）确认的问题

1. v2 是否引入构建步骤/新前端依赖？（决定 C 的走向，也决定"开箱即用"是否被破坏）
2. 是否需要图表 / 图谱可视化？（决定 B 是否升级为"新增 vendored 库 + 新增后端数据接口"）
3. 是否统一 v1 与 mockup 的视觉 token（配色/字体/圆角）？（当前实现与设计稿两套 token 并存）
4. 顶栏"全局检索（Ctrl+K）"是否落到 v2？现有 `/api/search` 是 BM25 书目检索（papers / concepts / arguments，等价 `drbrain library search` 的引擎），**不覆盖** runs / sessions / plugins，也没有跨实体聚合；要做得先定义搜索范围与后端新增接口。
5. 是否补 i18n 层？（决定 P5 是否纳入本次范围）
