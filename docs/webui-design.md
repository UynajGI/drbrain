# WebUI 设计：单人版 → 产品

> 状态：v1 设计定稿，待实现 · 基线：2026-09-11
> 路线图定位见 [platform-roadmap.md](platform-roadmap.md)；本文件是 webui 主线的设计契约。

## 1. 目标与阶段

| 阶段 | 用户 | 范围 |
|---|---|---|
| **v1 单人** | 本机研究者 | 认证、文献库检索、autoresearch 运行实时视图、插件管理、设置 |
| **v2 产品** | 课题组 / 外部用户 | 多用户、容器化部署、前端框架迁移、配额与审计 |

v1 的每一层都按"会被 v2 复用"设计：API 契约与前端解耦，认证抽象成依赖注入，
不把单人假设焊进业务层。

## 2. 技术选型

| 决策 | 选择 | 理由 |
|---|---|---|
| 服务框架 | **FastAPI + uvicorn** | uvicorn/starlette/pydantic 已在依赖树（litellm 传递依赖），显式新增仅 fastapi 一个包；产品路径（自动 OpenAPI 文档、依赖注入、后台任务），避免 stdlib 路线的二次重写 |
| 前端 v1 | 服务端渲染（Jinja2）+ **htmx**（静态 vendored，无 CDN/构建链） | 单人版零前端工程化；htmx 的局部刷新足够支撑事件流与表格交互 |
| 前端 v2 | 迁移 SPA（Vue/React 由届时定） | 仅消费 v1 已固化的 JSON API，迁移不触碰服务层 |
| 实时 | **SSE**（`text/event-stream`） | ledger 事件流是单向推送，SSE 比 WebSocket 简单且断线自动重连 |
| 持久化 | 复用现有 SQLite（drbrain.db + `webui` 前缀表） | 不引入第二存储；会话/审计有独立表 |

## 3. 分层约束（生产化的关键）

```
app/
├── web/            # FastAPI 路由 + Jinja2 模板 + SSE —— 只做协议转换，零业务
│   ├── routes/     # papers / runs / plugins / settings / auth
│   └── static/
├── service.py      # 业务编排（现有 540 行演进于此，路由层唯一允许的调用对象）
└── auth.py         # token 签发/校验，作为 FastAPI Depends 注入
```

- 现有 `server.py`（stdlib, 206 行）在 FastAPI 对等实现后**删除**；`service.py` 业务逻辑
  保留并下沉，路由层不得出现 SQL、不得直接 import loop/workflow 内部。
- 所有写操作经现有 service/Database 门面——webui 与 CLI 是同一业务层的两个皮。

## 4. 认证模型

| | v1 单人 | v2 产品 |
|---|---|---|
| 模式 | 本地 token：首次 `drbrain webui` 生成随机 token，打印到终端并存 `<root>/config/webui_token`（0600） | 多用户账号 + session |
| 传输 | `Authorization: Bearer`（API）+ cookie（页面），登录页一次性输入 | 注册/登录/角色 |
| 绑定 | **默认 `127.0.0.1`**；`--host` 显式暴露时警告未配置 HTTPS | 强制 HTTPS（反代终止） |

登录态与会话审计存 `webui_sessions` 表（token 哈希、创建/最后活跃、来源地址）——
v2 迁多用户时该表直接演化为审计日志。

## 4.1 层级模型：项目 → 会话 → 实验

UI 与数据按三层组织，上层为下层的命名空间：

| 层 | 标识 | 内容 | 记忆绑定 |
|---|---|---|---|
| **项目** | `project_id` | 领域包 + 语料 + 一组会话 | 项目级长期记忆（跨会话继承） |
| **会话** | `session_id` | 长期对话 + 记忆 + 若干实验 | 会话级记忆（可标注继承自项目） |
| **实验** | `loop_id` | 一次 autoresearch 运行（ledger/claims） | 实验级记忆（结论回写会话） |

- 侧栏顶部为**项目切换器**；切换项目后，文献库/会话/运行/插件视图全部随之切换。
- **RAG 双形态**，记忆绑定层级可选（项目 / 会话 / 实验，默认会话、继承项目）：
  - **面向检索的 RAG**：文献库问答，消费项目语料与会话记忆；
  - **面向 autoresearch 的 RAG**：会话对话与实验上下文，长期记忆随会话存留、实验结论回写。
- 研究运行页每个 run 显示其 `会话 · 实验` 绑定。

## 4.2 LLM 客户端

WebUI/核心统一走 **OpenAI SDK**（openai 包，OpenAI 兼容端点均可直连）。
**不使用 litellm**——其缓存命中行为不可控。模型回退链由 `llm_client` 自身的
fallback 逻辑承担（迁移项见 [platform-roadmap.md](platform-roadmap.md)）。

## 5. 信息架构（v1 六页）

| 页面 | 路径 | 功能 | 数据源 |
|---|---|---|---|
| Dashboard | `/` | 库概况（论文数/最近 ingest/运行状态）、快捷入口 | service.dashboard() |
| 文献库 | `/papers` | 检索 RAG：BM25 融合检索、论文详情、删除 | query 层 |
| 会话 | `/sessions` | 项目下会话列表；会话详情 = 长期记忆 + 实验 loop 列表 + 面向会话的 RAG 对话 | sessions + memory 层 |
| 研究运行 | `/runs` | 发起 autoresearch、**ledger 事件流（SSE）**、claims 视图、历史与复放；每个 run 绑定会话/实验 | loop/ledger（只读 + 受控发起） |
| 插件 | `/plugins` | 已发现插件列表（manifest/ABI/符合性报告）、健康与调用统计 | plugins.registry + conformance |
| 设置 | `/settings` | 配置查看（密钥只显引用）、webui token 重置 | config + runtime |

## 6. API 面（v1，JSON + SSE；v2 产品直接复用）

```
GET  /api/projects                  GET  /api/projects/{pid}/sessions
GET  /api/sessions/{sid}            GET  /api/sessions/{sid}/memory
POST /api/sessions/{sid}/chat       # 面向会话的 RAG 对话（记忆自动绑定）
GET  /api/papers?q=&limit=          GET  /api/papers/{id}
GET  /api/search?q=                 # BM25 融合结果（检索 RAG）
POST /api/runs                      # 发起 autoresearch（绑定 session_id）
GET  /api/runs/{id}                 GET  /api/runs/{id}/events   # SSE
GET  /api/plugins                   POST /api/plugins/{name}/conformance
GET  /api/dashboard                 POST /api/auth/verify
```

错误契约统一 `{"error": string, "detail"?: any}`；鉴权失败 401，越权 403。

## 7. 里程碑与验收

| 里程碑 | 内容 | 验收 |
|---|---|---|
| M1 | FastAPI 骨架 + token 认证 + Dashboard/文献库两页（复用 service 层） | 浏览器完成"检索 → 看论文详情"全程；stdlib server 删除 |
| M2 | 研究运行页：发起 + SSE 事件流 + claims | 一次运行全程在浏览器内观察，无需终端 |
| M3 | 插件管理页（对接 conformance 报告）+ 设置页 + webui 测试套件 | v1 验收表全部打勾；CI 含 webui 测试 |

## 8. v2 产品化迁移注意（v1 设计时就遵守）

- API 契约先行固化（OpenAPI 自动生成即契约），前端替换不破坏服务层
- 认证以 `Depends` 注入，业务函数签名不出现 token/用户概念
- 所有状态进 SQLite/文件，进程内存不保存不可重建状态（容器化前提）
- 静态资源 v1 vendored，v2 换构建产物时路径由配置注入
