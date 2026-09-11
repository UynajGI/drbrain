# WebUI 设计：单人版 → 产品

> 状态：v1 已实现（分支 `feat/webui-m1`）——M0 作用域与迁移、M1a 服务迁移与认证、
> M1b 文献路径、M2a 会话与发起、M2b SSE/裁决/导出、M3 插件与设置均已落地；
> 浏览器实机视觉验收与 release-time provider 联调仍待执行 · 更新日期：2026-09-11
> 路线图定位见 [platform-roadmap.md](platform-roadmap.md)；本文件是 webui 主线的设计契约。

## 0. 现状核对与改进优先级

当前已有 stdlib HTTP + 静态单页原型，下面的 FastAPI/htmx 方案是迁移目标。
本轮做了代码、计划和接口测试核对，未做浏览器实机视觉验收，也未实现新界面。

| 优先级 | 已核实的缺口 | 开发指导与证据 |
|---|---|---|
| P0 | 会话页已列入 v1，原 M1–M3 没有安排会话/记忆交付；项目作用域尚未进入 WebUI 接口 | M0 先定义项目、会话、运行关联与旧数据归属；M2a 单独交付会话。现有 [service.py](../src/drbrain/app/service.py) 接受全局配置，研究会话已有 [Database](../src/drbrain/storage/database.py) 的 `agent_sessions`/`agent_messages` 可复用 |
| P0 | 计划用 `loop_id` 指一次运行，代码另有计算实验 `experiment_id` | API 统一用 `run_id` 表示研究运行；计算实验继续用 `experiment_id`，见 §4.1。现有 `run_claims()`、`experiments()` 已依此关联 |
| P0 | 原计划把现有 JSON `/events` 直接改成 SSE，会破坏现有消费者；页面/API 两种鉴权也未覆盖 SSE | 保留 JSON 历史接口，新增 `/stream`；浏览器页面、htmx、SSE 共用同源登录 cookie，见 §4、§6.2。旧 `app/server.py`（M1a 删除）与 [test_app.py](../tests/test_app.py) 已有 JSON 事件契约 |
| P0 | 发起接口返回 topic/starting，没有持久化 `run_id`；运行管理器按 topic 维护内存线程和错误 | 在调度前持久化运行身份、作用域、请求幂等记录，再返回 202；不得靠轮询同名 topic 找 run。依据 `RunManager.start()`/`status()` |
| P1 | 路线图要求浏览器导出结论，计划没有导出 API；插件“启停/健康”也超出当前列表能力 | M2b 补运行报告下载；M3 交付发现信息与符合性报告。启停延至生命周期契约就绪，发现状态与健康状态分开。现有 `assets()` 仅提供 CLI 导出命令，`plugins()` 仅返回描述信息 |
| P1 | 单页原型没有媒体查询和显式输入标签；每次事件更新强制滚到底部；网络异常可能跳过按钮复位 | 将响应式、键盘操作、错误恢复、保留阅读位置加入每个里程碑。依据旧单页原型（`app/static/index.html`，M1a 迁移后随 `server.py` 一并删除）的 `addEvents()`、`doSearch()`、`doAsk()` 与启动处理器 |

实现记录：M0–M3 在 `feat/webui-m1` 落地；聚焦测试
`.venv/bin/python -m pytest tests/test_project_scope.py tests/test_app.py tests/test_webui.py -q` 全部通过，
数据库均为真实临时 SQLite。轮子打包已验证包含模板/静态资源/vendored htmx 与许可证
（`uv build --wheel` 后检查 whl 内容）。浏览器实机视觉验收与真实 provider 联调不在本轮范围内，
验收清单 §7.1 中相应条目保持未勾选。

## 1. 目标与阶段

| 阶段 | 用户 | 范围 |
|---|---|---|
| **v1 单人** | 本机研究者 | 认证、项目切换、文献检索与详情、会话与记忆、运行观察与结论导出、插件诊断、设置查看 |
| **v2 产品** | 课题组 / 外部用户 | 多用户、容器化部署、前端框架迁移、配额与审计 |

v1 的每一层都按"会被 v2 复用"设计：API 契约与前端解耦，认证抽象成依赖注入，
不把单人假设焊进业务层。

v1 的主任务路径是：**选择项目 → 找到文献/证据 → 进入会话 → 发起研究运行 →
查看裁决与证据 → 下载结论**。Dashboard 服务于继续这条任务路径。
首次启动、导入语料和配置模型仍沿用现有 CLI；配置完成后的这条研究路径在浏览器内完成。
论文删除、插件启停、任意配置编辑放入后续小版本，避免在首版引入尚无业务契约的写入口。

## 2. 技术选型

| 决策 | 选择 | 理由 |
|---|---|---|
| 服务框架 | **FastAPI + uvicorn** | 自动 OpenAPI 文档与依赖注入；M1 显式声明并锁定实际运行依赖，不依赖其他包顺带安装 Web 服务组件 |
| 前端 v1 | 服务端渲染（Jinja2）+ **htmx**（静态 vendored，无 CDN/构建链） | htmx 处理表单、分页和 HTML 局部刷新；少量独立 JS 管理 EventSource、游标及页面切换时的清理 |
| 前端 v2 | 迁移 SPA（Vue/React 由届时定） | 仅消费 v1 已固化的 JSON API，迁移不触碰服务层 |
| 实时 | **SSE**（`text/event-stream`） | ledger 事件流是单向推送，SSE 比 WebSocket 简单且断线自动重连 |
| 持久化 | 复用 `drbrain.db` 与现有 `ledger.sqlite3` | 研究会话复用核心表；`webui_` 前缀仅用于登录会话/界面审计，项目与运行绑定属于核心业务数据 |

依赖核对：[pyproject.toml](../pyproject.toml) 已直接依赖 `openai`，未直接声明
`fastapi`、`uvicorn`、`jinja2`。原“依赖 litellm 的传递依赖、仅增加一个包”依据已不成立。
M1 同时验证 wheel 包含模板、CSS、JS、vendored htmx 及其许可证，安装后离开源码目录也能启动。

## 3. 分层约束（生产化的关键）

```
app/
├── web/            # FastAPI 路由 + Jinja2 模板 + SSE —— 只做协议转换，零业务
│   ├── routes/     # papers / runs / plugins / settings / auth
│   └── static/
├── service.py      # 业务编排（复用现有门面，路由层唯一允许的业务调用对象）
└── auth.py         # token 签发/校验，作为 FastAPI Depends 注入
```

- 现有 `server.py` 在 FastAPI 对等实现与迁移测试通过后**删除**；`service.py` 业务逻辑
  保留并下沉，路由层不得出现 SQL、不得直接 import loop/workflow 内部。
- 数据库写操作经 service/Database 门面；ledger 操作复用既有持久化接口，Web 层不得自行写表。
- `/api/*` 返回 JSON，`/ui/fragments/*` 返回 Jinja2 HTML；两类路由消费相同的 service 返回模型。
  HTML 片段复用页面组件，不通过内部 HTTP 再调用自己的 API。SSE 使用专属 `/stream` 路径。
- 同步检索和阻塞调用不得阻塞 ASGI 事件循环。研究任务由应用生命周期管理的执行器调度；
  重启后从 ledger/检查点识别中断状态，不能把进程内 `_threads` 当作唯一运行状态。
  v1 先用单 worker，扩多 worker 前验证调度幂等与租约。FastAPI 的初始化和关闭接入
  [lifespan](https://fastapi.tiangolo.com/advanced/events/)；任务可恢复性仍由业务持久化保证。

## 4. 认证模型

| | v1 单人 | v2 产品 |
|---|---|---|
| 模式 | 本地 token：首次 `drbrain webui` 生成随机 token，打印到终端并存 `<root>/config/webui_token`（0600） | 多用户账号 + session |
| 传输 | 浏览器页面、htmx、JSON 请求与 SSE 统一同源 cookie；非浏览器 API 支持 `Authorization: Bearer` | 注册/登录/角色 |
| 绑定 | **默认 `127.0.0.1`**；`--host` 显式暴露时警告未配置 HTTPS | 强制 HTTPS（反代终止） |

引导 token 校验成功后签发独立登录会话；登录态存 `webui_sessions`
（凭据哈希、创建/最后活跃、过期/撤销时间、来源地址）。登录会话 ID 与研究 `session_id` 分开。

- cookie 使用 `HttpOnly`、`SameSite=Strict`；HTTPS 启用 `Secure`，本地 HTTP 单独配置。
  Cookie 认证的写请求校验 CSRF token 和来源；令牌不放 URL、localStorage 或事件 payload。
  htmx 可经 `hx-headers` 传 CSRF 值，校验在后端完成，见 [htmx CSRF 文档](https://htmx.org/docs/#csrf-prevention)。
- 原生 `EventSource` 构造选项不提供任意请求头；因此浏览器流使用同源 cookie，
  不要求 JS 为流附加 Bearer，见 [EventSource 参数](https://developer.mozilla.org/en-US/docs/Web/API/EventSource/EventSource#parameters)。
- 未登录 HTML 跳转登录页，API 返回 JSON 401；htmx 错误处理转到登录并保留安全的站内返回路径。
  登录会话设明确有效期；过期、退出或 token 重置后旧会话失效，已建立的流也须结束并要求重新登录。
- 登录信息和设置页禁用 htmx 历史快照，退出时清理敏感页面状态；
  [htmx 历史缓存](https://htmx.org/docs/#disabling-history-snapshots)可能将页面内容存入 localStorage。
- v2 单独增加用户/角色与审计模型；登录会话表不兼作所有业务操作的审计日志。

## 4.1 层级模型：项目 → 会话 → 研究运行

UI 与数据按三层组织，上层为下层的命名空间：

| 层 | 标识 | 内容 | 记忆绑定 |
|---|---|---|---|
| **项目** | `project_id` | 领域包 + 语料引用 + 一组会话 | 项目级长期记忆（跨会话继承） |
| **会话** | `session_id` | 长期对话 + 记忆 + 若干研究运行 | 会话级记忆（可标注继承自项目） |
| **研究运行**（原计划称“实验”） | `run_id` | 一次 autoresearch 运行（ledger/claims） | 运行级记忆（结论回写会话） |

运行内的真实计算作业继续使用 `experiment_id`，与 `run_id` 不是同一个对象。
v1 API 不新增 `loop_id` 别名；页面用“研究运行”和“计算实验”区分。

- 侧栏顶部为**项目切换器**；切换项目后，文献库/会话/运行按项目筛选；插件显示当前项目
  配置可用的集合及来源。全局安装信息另作标记，不能把进程级插件发现伪装成项目隔离。
- **RAG 双形态**，记忆绑定层级可选（项目 / 会话 / 研究运行，默认会话、继承项目）：
  - **面向检索的 RAG**：文献库问答，消费项目语料与会话记忆；
  - **面向 autoresearch 的 RAG**：会话对话与运行上下文，长期记忆随会话存留、运行裁决回写。
- 研究运行页显示项目、所属会话和运行 ID；计算实验与证据链接回该运行。

**M0 必须先落地的边界：**

1. 明确 `project_id` 与现有 workspace、runtime root 的映射；项目不能仅是侧栏中的名字。
   复用语料引用，避免复制一套文献库；身份保持稳定，重命名不改变 ID。
2. 复用现有研究会话；持久化项目→会话与会话→运行关系。旧数据归入明确的默认项目/
   历史会话并记录迁移依据；不按 topic 文本猜归属，不自动伪造历史对话。
3. service 接收已解析的请求作用域；详情、检索/RAG、事件、证据、导出都验证归属。
   请求级选项目不得修改进程级配置；后台任务捕获作用域快照。测试覆盖两个项目同名 topic。
4. 若项目绑定与 ledger 创建分属两个 SQLite 文件，定义失败恢复与幂等对账流程；
   返回 202 前先形成可追踪的持久化发起记录，不能仅靠两个先后写操作假设跨库原子性。

记忆语义在 M0 定义、M2a 实现：默认消费会话记忆并继承项目；运行使用启动时的版本快照。
每条记忆展示来源层级、来源引用与更新时间，模型摘要不能覆盖原始证据。
运行裁决回写会话时保留 `run_id`、证据引用和去重标识，重放不得重复写入；
向项目长期记忆提升须是显式动作。切换项目后旧请求/旧流的结果不得写入新页面。

## 4.2 LLM 客户端

WebUI/核心统一走 **OpenAI SDK**；保留现有 OpenAI 兼容端点配置。
模型回退链、超时与错误归一化由 `llm_client` 负责，WebUI 不再实现一套 provider 客户端。
设置页读取归一化配置和可用性状态。迁移完成度由 LLM 专项测试确认；UI 骨架与检索开发
不等待该专项全量结束，M2a 的真实会话问答联调以客户端契约通过为前置。

## 5. 信息架构（v1 六页）

| 页面 | 路径 | 功能 | 数据源 |
|---|---|---|---|
| Dashboard | `/` | 当前项目概况、继续最近会话/运行、首次使用引导 | service.dashboard() 扩展项目作用域 |
| 文献库 | `/papers`、`/papers/{id}` | 关键词检索、分页、元数据/章节详情、证据定位、带文献引用进入会话 | query + Database；当前 `service.search()` 是 BM25，问答另走 RAG |
| 会话 | `/sessions`、`/sessions/{id}` | 创建/继续会话、对话与来源、记忆来源层级、所属运行列表、发起研究 | 复用 sessions + 新增作用域/记忆编排 |
| 研究运行 | `/runs`、`/runs/{id}` | 阶段与运行状态、SSE/历史、claims/证据、计算实验、报告下载 | loop/ledger 的 service 门面 |
| 插件 | `/plugins` | 发现列表、manifest/ABI、符合性检查与报告；无健康探测时显示“未检测” | plugins.registry + conformance |
| 设置 | `/settings` | 有效配置与来源、模型/检索启用状态、密钥引用、登录 token 重置 | config + runtime + auth |

六页是一级导航；项目切换和登录是公共组件/辅助页。原“研究问答”并入会话，原“计算任务”
并入运行详情，原“数据与模型”分到插件/设置；迁移时保留现有 API 与有效数据入口。
不把“已发现插件”显示成“在线”，不把已终止/失败的运行显示成所有节点完成。

## 5.1 页面状态与视觉验收

| 关注点 | v1 可执行要求 |
|---|---|
| 主次与文案 | 保留浅色科研工作台方向，统一 surface/text/border/action/status 语义 token；每页一个主要动作。面向用户说明任务结果，RRF、表名和内部路径放诊断详情 |
| 布局 | 大屏可用列表+详情；中屏折叠侧栏、详情改页内标签；窄屏单列。运行详情先显示摘要/裁决，再展开事件与原始 JSON |
| 阅读与键盘 | 正文以 16px、约 1.6 行高为基准；输入有可见 label，链接/按钮使用原生元素，焦点可见。详情支持 URL 直达、刷新与浏览器返回，搜索条件保留在 URL |
| 空/加载/失败 | 区分未导入、无匹配、功能未启用、请求失败；分别给导入说明、改词/清筛选、配置指引、重试。请求失败不显示“0 条”，表单错误保留输入，按钮在成功/失败时都恢复 |
| 表单 | 重复提交禁用与后端幂等同时具备；错误在对应字段旁展示，多字段失败提供可聚焦摘要。动态替换不丢失输入、焦点和当前项目 |
| 事件流 | 用户在底部时才自动跟随；向上阅读时显示“新增 N 条/回到最新”。仅播报阶段摘要，避免屏幕阅读器逐条朗读高速日志；阶段/裁决采用文字+颜色 |
| 长列表 | 搜索默认 20 条；新列表接口分页且设服务端上限，页面不拉取整个语料库。事件按游标分批加载并限制 DOM 数量，仍可回看更早历史 |
| 证据 | 结论可进入来源文献/章节或计算产物；无权限、缺文件和证据不足都有明确状态。摘要、推测、通过裁决的结论分开标注 |

Web 验收覆盖 1440px、1024px、768px 与 320 CSS px 宽度，正文无需横向滚动；
表格/代码等必要二维区域仅在自身容器滚动。普通文字对比度至少 4.5:1，
有意义的控件/图形至少 3:1；键盘焦点不被固定区域遮挡，尊重减少动态效果设置。
这些条目参考 [WCAG 2.2](https://www.w3.org/WAI/WCAG22/quickref/)，不等于已完成全站合规认证。

技能检索中，“error summary validation”命中 Web 的可聚焦错误摘要规则；
实时滚动检索及一次改写未命中具体规则，上表的自动跟随策略是针对现有 `addEvents()` 的产品建议。

## 6. API 面（v1，JSON + SSE；v2 产品直接复用）

以下是目标接口清单，未实现项按里程碑新增；不是当前可调用能力表。

| 接口组 | 路径与用途 | 交付 |
|---|---|---|
| 登录 | `POST /api/auth/verify`、`POST /api/auth/logout`；页面 `/login` | M1a |
| 项目 | `GET /api/projects`；M0 定义默认项目/现有 workspace 映射，v1 不新增项目管理工作流 | M0/M1a |
| 检索 | `GET /api/dashboard`、`GET /api/papers`、`GET /api/papers/{id}`、`GET /api/search`；项目级查询显式带 `project_id` | M1b |
| 会话 | `GET/POST /api/projects/{pid}/sessions`、`GET /api/sessions/{sid}`、`GET /api/sessions/{sid}/memory`、`POST /api/sessions/{sid}/chat` | M2a |
| 记忆提升 | `POST /api/sessions/{sid}/memory/promote`：显式提升指定记忆到项目，保留来源与幂等标识 | M2a |
| 运行 | `GET/POST /api/runs`、`GET /api/runs/{rid}`；发起绑定 `session_id`，查询验证项目归属 | M2a/M2b |
| 事件 | `GET /api/runs/{rid}/events?after=&limit=` 保留 JSON；`GET /api/runs/{rid}/stream` 新增 SSE | M2b |
| 裁决与产物 | `GET /api/runs/{rid}/claims`、`GET /api/experiments?run_id=`、`GET /api/runs/{rid}/evidence/{eid}`、`GET /api/runs/{rid}/artifacts/{aid}` | M2b |
| 结论导出 | `GET /api/runs/{rid}/report?format=markdown\|json`：可下载的运行摘要、裁决与证据引用 | M2b |
| 插件 | `GET /api/plugins`、`POST /api/plugins/{name}/conformance`、`GET /api/plugins/{name}/conformance/{check_id}` | M3 |
| 设置 | `GET /api/settings`（脱敏有效配置）、`POST /api/auth/rotate`（确认后重置 token 并撤销旧登录态） | M3 |

**迁移兼容：**M1a 覆盖现有 `/api/ask`、`/api/assets`、`/api/run-status?topic=`、
`/api/runs`、`/api/experiments`、`/api/plugins` 等所有已有路由。认证是明确新增的访问要求；
已认证后的旧成功响应形状保留，不悄悄把数组改成对象。旧客户端未传项目时绑定默认项目；
新前端必须显式携带作用域。同 topic 的旧状态查询若存在歧义，返回明确错误并引导按 run_id 查询。

## 6.1 JSON、发起与契约校验

- 新列表响应采用 `items`、`next_cursor`；旧列表保持兼容并另行记录迁移方式。固定字段类型、
  ID/时间格式、分页顺序与参数上限；“取回 N 条”和“总量 N 条”不得混用。
- 错误保留 `error: string`，可增加 `code`、`detail`、`request_id`；详情必须脱敏。
  区分 401、403、404、409、422 与 503；框架校验异常也转换为同一契约。
  现有 400 行为在兼容阶段保留，新接口的校验状态码写入契约测试。
- `POST /api/runs` 接受 `session_id`、`topic`、`max_cycles` 和请求幂等键 `client_request_id`；
  返回 202 时包含持久化 `run_id`、实际状态、详情/流地址。同键重试返回同一运行；
  同名 topic 的新建/恢复语义由 M0 与核心统一，不允许跨会话错误复用运行。
- 发起请求与浏览器连接生命周期分开：关闭页面不取消任务；断线、后端中断、任务失败是三种状态。
  若运行需要人工审核，页面明确显示待处理原因；v1 不承诺尚未接入的暂停/恢复按钮。
- 插件 conformance 是有副作用的显式任务：返回 202 + `check_id`，页面查询结果；
  打开列表页不自动执行插件。报告绑定插件版本/内容标识，旧报告显示“已过期”。
- OpenAPI 是接口说明的生成结果；稳定性靠请求/响应模型、契约回归测试和变更审查保证。
  M0 固化样例，M1 起在 CI 验证，不能把“自动生成文档”等同于“契约已验证”。

## 6.2 SSE 与运行视图

1. `/stream` 使用 `text/event-stream`，事件 `id` 对应该 run 的 `event_seq`；数据保留
   `seq/type/actor/payload/created_at`。JSON `/events` 用于历史加载与回放。
2. 首次连接支持 `after`；重连优先使用 `Last-Event-ID`，从 ledger 按序补齐已提交事件。
   采用至少一次传输，前端按 `(run_id, seq)` 去重。快照附带对应游标，避免快照与订阅之间漏事件。
3. 无业务事件时发送心跳；关闭代理缓冲，按配置管理重试与超时。网络断开显示“连接中断/重连中”，
   保留已有内容；连接失败不改变任务本身状态。
4. 切换运行、离开页面或退出时主动关闭旧 EventSource，清理定时器并取消旧请求。
   运行终止且事件已消费完毕后结束订阅；历史回放只读，不重新执行计算。
5. 自动重连、事件 ID 与主动关闭遵循浏览器 [SSE 行为](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)；
   断点补齐、去重、终态处理与登录过期处理属于本项目必须实现的逻辑。

## 7. 里程碑与验收

每行可作为一个工作包，较大行再按接口/页面拆 PR。先写行为验收与有意义的失败测试，
再实现；测试从 M0/M1 开始随功能交付。负责人在开工时填写，不用假定人日代替验收。

| 里程碑 | 交付与主要位置 | 前置 | 完成标准 |
|---|---|---|---|
| M0 作用域与契约 | 核心存储映射/迁移、service 作用域模型、API 样例与旧接口兼容表 | 无 | 两项目同名 topic、跨项目 ID、历史数据归属、迁移重复执行均有真实 SQLite 测试；不存在跨库绑定失败后不可追踪的运行 |
| M1a 服务迁移 | `app/web/`、`auth.py`、CLI 启动/打包；已有 API 对等迁移 | M0 的作用域契约 | 登录/退出、Cookie/Bearer、CSRF、过期路径通过；旧 service/HTTP 行为在已认证条件下通过；干净安装启动可读取静态资源后移除 stdlib server |
| M1b 文献路径 | Dashboard、项目切换、检索/详情、通用表单/错误组件 | M1a + 项目语料映射 | 浏览器完成检索→详情→返回；刷新保留筛选；空库/无结果/503 可区分；窄屏与键盘操作通过 |
| M2a 会话与发起 | 会话列表/创建/对话、记忆来源/继承、run_id 发起契约 | M1b + LLM 客户端契约 | 刷新/重启后继续会话；记忆作用域可追溯；一次真实配置的问答可定位来源；重复提交只创建一次运行 |
| M2b 观察与导出 | SSE、运行状态、claims/证据/计算产物、报告下载 | M2a | 断网补齐不重复；切项目无旧流串入；服务重启后正确呈现持久化状态；浏览器可下载带 run_id/裁决/来源的报告 |
| M3 插件与设置 | 发现/符合性任务与报告、脱敏配置、token 重置、跨页验收 | UI 部分依赖 M1a；符合性功能依赖插件 v2 | “发现/未检测/检查中/通过/失败/过期”真实区分；重置后旧登录和流失效；六页主路径与回归测试通过 |

落地状态（2026-09-11，`feat/webui-m1`）：M0、M1a、M1b、M2a、M2b、M3 的代码与聚焦测试均已完成——
作用域迁移（DB v21 + ledger v9）、FastAPI + token 认证、五页主路径 + 设置页、SSE、
报告下载、插件符合性任务与 token 重置。M2a 的“真实配置问答定位来源”与 M3 的浏览器跨页验收
需要在已配置（llm.models / llamaindex / 浏览器）环境实机执行，见 §7.1。

推进顺序：先做 M0 的作用域/兼容清单与失败测试，然后 M1a→M1b→M2a→M2b；
M3 的页面壳与只读信息可在 M1a 后同步推进，符合性任务等待插件 v2。
RAG 三层记忆完整实现属于 M2a；M1 的前置是作用域与数据关联，避免把整套长期记忆工程都挡在首屏之前。

## 7.1 发布验收清单

- [x] 保留并迁移 `tests/test_app.py`，补认证、作用域、分页、错误形状、幂等发起、跨库失败恢复测试；数据库使用真实临时 SQLite。
      （认证/CSRF/过期/轮换、项目归属、`next_cursor` 分页与无效游标 422、幂等发起与重启后中断态均在 `tests/test_webui.py`；
      跨库一致性问题通过把项目/会话/幂等键并入 ledger 单次写入消解，不存在两库先后写的窗口）
- [ ] SSE 覆盖中途断线、重复/未知事件、历史超过单页、登录过期、页面切换、任务终止；可控测试事件只能作为测试夹具，不在产品空状态预加载。
      （回放/终态/Last-Event-ID/未知 run 已有测试；断线重连、页面切换清理与登录过期需浏览器实机联调）
- [ ] 浏览器自动化覆盖登录→选项目→检索详情→会话→运行→证据→报告下载，以及键盘、输入保留和断网重试；由 M1 起逐步进入 CI。
- [ ] 以长中文标题、长公式/代码、空数据、多会话、上万条 ledger 事件验证布局和分页；记录固定机器/数据规模下的检索耗时与 DOM 上限，作为后续性能比较基线。
      （分页/事件游标与 DOM 上限已实现：事件列表保留最近 500 条，历史按游标回填；尚未做真实规模测量）
- [x] 模板/静态文件随 wheel 分发，断开 CDN 访问后页面正常；CLI、JSON API 与当前已有计算任务入口完成迁移回归。
      （`uv build --wheel` 已验证模板/CSS/JS/vendored htmx 与许可证随包分发；htmx 本地 vendored，无 CDN 依赖）
- [ ] 发布前完成一次已配置环境下的浏览器主路径联调，保存真实 run_id、报告和失败说明；CI 夹具通过不替代真实 provider/插件可用性验证。

UI 的运行观察/导出验收允许如实展示失败或证据不足；获得科研上有价值的实算结论，
仍按平台路线图由研究侧单独验证。两种验收分别记录结果。

## 8. v2 产品化迁移注意（v1 设计时就遵守）

- API 契约以模型和回归测试固化，前端替换不破坏 service 返回语义
- 认证以 `Depends` 注入，业务层接收可信作用域/主体以执行归属检查，不传输原始 token 或 HTTP 对象
- 所有状态进 SQLite/文件，进程内存不保存不可重建状态（容器化前提）
- 静态资源 v1 vendored，v2 换构建产物时路径由配置注入
