# 插件系统规范（Plugin ABI）

> 状态：**v2（现行）** · 宿主 ABI：`HOST_ABI_VERSION = 1` · 支持集合：`{1}`
>
> 插件系统是本平台的**领域化入口**：四层核心（数据管线 / 检索 RAG / 能力面 / 研究回路）
> 保持领域无关，一切领域知识——评测集、语料、实算工具、领域模型——都经由本协议进入。
> 材料学（A 线）是第一个领域插件包，不是核心的一部分。

## 1. 插件形态

一个插件 = 一个 `.py` 文件，放在插件目录（如 `research/plugins/`），定义：

```python
from drbrain.plugins import Plugin

def register(registry):
    registry.register(
        Plugin(
            name="predict_flatband_score",       # 全局唯一
            description="给定成分与空间群，预测平带度",
            input_schema={"type": "object", ...},  # JSON Schema，宿主据此生成工具签名
            plugin_type="model",                  # model | software | data | formula | other
            version="flatness_prod_v2",           # 插件自身版本
            abi_version=1,                        # 声明针对的宿主协议版本（缺省 = 1）
            side_effect="read",                   # pure | read | write | irreversible | unspecified
            resource="models/flatness_prod_v2.joblib",
            summary_fields=("S_bandwidth",),      # 随原始 JSON 摘要给 LLM 的关键字段
        ),
        handler,          # Callable[[dict], Any]
        jobs=JobMethods(submit, poll, cancel),   # 可选：小时级长作业
    )
```

宿主在 `discover(plugins_dir)` 时逐文件导入并调用 `register(registry)`；
任一模块失败只跳过自身，不阻断其他插件。

## 1.1 Manifest 声明（v2，元数据与代码分离）

v2 起支持**数据优先**的声明风格：模块级 `PLUGIN_MANIFEST` 字典 + `HANDLER`
可调用对象（可选 `JOB_METHODS`），元数据与代码分离，声明即注册：

```python
from types import SimpleNamespace

PLUGIN_MANIFEST = {
    "name": "predict_flatband_score",       # 必填，全局唯一
    "description": "给定成分与空间群，预测平带度",  # 必填
    "input_schema": {"type": "object", ...},  # 必填，JSON Schema
    "plugin_type": "model",                 # 可选，缺省 "other"
    "version": "flatness_prod_v2",
    "abi_version": 1,                       # 缺省 1；协商规则与 §3 相同（fail-closed）
    "side_effect": "read",
    "timeout_s": 60.0,
    "summary_fields": ["S_bandwidth"],
    "resource": "models/flatness_prod_v2.joblib",
    "metadata": {"family": "gbdt"},
    # ... 其余 Plugin 数据类字段均可直接写进 manifest
}


def HANDLER(arguments):                     # 必须是模块级可调用对象
    ...


JOB_METHODS = SimpleNamespace(submit=..., poll=..., cancel=...)  # 可选
```

- `discover()` 发现 `PLUGIN_MANIFEST` 时直接由 manifest 构建 `Plugin` 并注册
  `HANDLER` / `JOB_METHODS`；缺 `name` / `description` / `input_schema` 或
  `HANDLER` 不可调用的模块 → 告警跳过（与 inline 风格的失败语义一致）。
- ABI 协商同样适用：manifest 里声明了不支持的 `abi_version` → fail-closed 跳过。
- manifest 同时声明了 `register()` 的模块按 manifest 注册，`register()` 不被调用。
- 未知 manifest 键由 `Plugin` 构造器静默丢弃（与 inline 风格同一条前向兼容
  契约）；符合性自检（§7）会把未知键报出来，避免拼写错误无声失效。
- 两种风格可共存于同一目录；inline `register(registry)` 风格保持原样，完全兼容。

## 2. 描述符契约（`Plugin` 字段分组）

| 分组 | 字段 | 说明 |
|---|---|---|
| 身份与模式 | `name` `description` `input_schema` `plugin_type` `version` `abi_version` | 工具签名由 `input_schema` 生成；`abi_version` 见 §3 |
| 行为分类 | `backend` `side_effect` `timeout_s` `summary_fields` | `side_effect` 是研究回路的实算门/结算依据（`write`/`irreversible` 触发更严的门） |
| 安全信封 | `resource` `resource_scope` `code_digest` `secret_refs` `max_output_bytes` `sandbox_profile` `approval_policy` | 秘钥只经 `secret_refs` 引用，永不进描述符；输出超过 `max_output_bytes` 被截断 |
| 长作业 | `JobMethods(submit, poll, cancel)` | 结果只认 `jobs/<job_id>.json`（+ `.log`），`Artifact(path, sha256)` 支持逐字节复核 |
| 能力声明 | `required_capabilities` `supports_idempotency` `supports_reconcile` `supports_cancel` `cost_hint` | 研究回路的结算幂等 / CAS 依赖这些声明 |

## 3. ABI 版本协商

- 插件声明 `abi_version`（缺省 = 1，覆盖所有早期插件，向后兼容）。
- `abi_version ∉ SUPPORTED_ABI_VERSIONS` → `register()` 抛 `ValueError`（fail-closed），
  `discover()` 跳过该插件并告警——**绝不带病加载**。
- 升级策略：`HOST_ABI_VERSION` 只增不减；被弃用的版本随主版本（major）从支持集合移除。
- 未知关键字参数目前被静默丢弃（向后兼容旧插件）；声明了 `abi_version` 的插件视为
  已知契约，未来对声明版本的插件收紧此行为。

## 4. 运行信封

默认每次调用独立子进程（超时可 SIGKILL 真回收）；不可 pickle 的 handler 自动回退共享线程。
超时取 `timeout_s`；输出超过 `max_output_bytes` 被截断并在结果中标注。

## 5. 插件作者符合性清单

- [ ] `register(registry)` 只注册一次；确需替换时显式使用 `replace=True`
- [ ] `input_schema` 是合法 JSON Schema（宿主据此生成 LLM 工具签名）
- [ ] 长作业：`submit` 只入队快速返回；结果写 `jobs/<job_id>.json` + `.log`；`Artifact` 带 sha256
- [ ] `side_effect` 如实声明（研究回路的门按此分级）
- [ ] 秘钥经 `secret_refs` 引用，不硬编码
- [ ] `abi_version` 显式声明（省略 = 1，仅限兼容期）
- [ ] 不 import 宿主内部模块（`drbrain.plugins` 除外）
- [ ] manifest 声明的键名与 `Plugin` 字段一致（未知键会被加载器静默丢弃）
- [ ] 交付前跑一遍符合性自检（§7）全绿

## 6. 参考实现

- 协议与注册表：`src/drbrain/plugins/`（protocol / registry / backends / manifest / conformance）
- 最小示例：`tests/fixtures/plugins/`（model + software 各一，经 `test_plugin_discovery.py` 验证）
- 真实领域插件：`research/plugins/`（材料学 A 线：GBDT 预测、GPAW 实算、physics/topology 重算）

## 7. 符合性自检（插件作者可自跑）

交付前对插件目录跑一遍符合性检查（**绝不执行 handler**）：

```bash
python -m drbrain.plugins.conformance <plugin_dir>
```

逐条打印 `[PASS]/[FAIL] 检查名 — 说明`，全部通过退出码 0，任一失败退出码 1。
检查项（每个 `*.py` 模块，跳过 `_` 前缀文件）：

| 检查 | 内容 |
|---|---|
| `imports` | 按 `discover()` 同样的规则可干净导入 |
| `entrypoint` | 声明了 `PLUGIN_MANIFEST`（manifest 风格）或 `register()`（inline 风格） |
| `manifest` / `manifest_fields` | 必填键齐全、类型正确；未知键直接报 FAIL（加载器会静默丢弃它们） |
| `input_schema` | 是 `type: "object"` 的 JSON Schema，且 `properties` 为字典 |
| `timeout_s` / `side_effect` / `abi_version` | `> 0` / 已知字面量 / 宿主支持集合内 |
| `code_digest` | 声明了就必须等于模块文件的 sha256——把声明的 digest 值本身置空后再哈希（自指声明：manifest 里填 `"code_digest": ""` 哈希一次，再回填值，校验时同样置空重算比对；可带 `sha256:` 前缀） |
| `secrets` | 源码不含硬编码密钥形态字符串（`sk-…` / `AKIA…` / `ghp_…` / `xox…`），秘钥只经 `secret_refs` 引用 |
| `job_methods` | 声明了 `JOB_METHODS` 就必须暴露可调用的 `submit`/`poll`/`cancel` |

inline 风格模块的 `register()` 会在一次性临时 registry 上执行一次（注册本身
是文档化契约的一部分，handler 永不执行），使描述符级检查对两种风格统一生效。

## 8. 统一能力入口（Plugin / MCP / Skill / API / CLI / Model）

`drbrain.capabilities` 提供与执行技术无关的两个对象：

- `CapabilityDescriptor`：稳定的 `id`、显示名、JSON Schema 2020-12 输入/输出、
  安全 annotations、权限、执行/作业能力和 provenance。
- `InvocationResult`：保留结构化内容、content blocks、`isError`、job、产物和
  截断标记的统一结果信封。旧的 `PluginResult` 仍可通过
  `to_invocation_result()` 适配，`ok` 语义保持兼容；`completed` 表示包括
  `NO_RESULT` 在内的正常完成。

能力 kind 目前包括 `plugin`、`mcp_tool`、`skill`、`api`、`cli`、`model`，但协议
故意接受任意非空 kind；新工具协议只需实现 `CapabilityAdapter`，无需修改 Agent
核心。长作业统一走 catalog 的 `submit_job` / `poll_job` / `cancel_job`，job ID
限制为路径安全 token；提供 `state_dir` 时，幂等键和状态以原子 JSON 文件保存；状态会保留
活动作业和近期终态，过期终态及超出上限的旧记录会自动清理，避免长时间运行的宿主无限增长。

Python 插件可通过 `PluginRegistry.list_capabilities()` 获取中立描述符。注册时会
校验 handler、schema、超时、side effect 和重复 ID；重复注册必须显式传
`replace=True`。直接 `PluginRegistry.call()` 会在 handler 运行前执行 JSON Schema
校验，错误以 `INVALID_INPUT` 返回。

`CapabilityCatalog` 是 Agent 侧的统一入口：`register_plugin_registry()`、
`register_mcp_servers()` 和 `register_skills()` 接入现有来源；
`register_adapter()` 接入任何实现 `CapabilityAdapter`（`descriptor()` +
`invoke()`）的新协议。内置 `APIAdapter`、`CLIAdapter`、`ModelAdapter` 分别覆盖
固定 HTTP API、固定 argv 的本地命令和已训练/托管模型 callable。Agent 只需要对
catalog 做 `recommend(query)` 和 `invoke(capability_id, arguments)`，不需要知道
能力来自哪个协议；所有调用都经过同一层输入校验、状态归一化和 input digest。
需要接入 LlamaIndex 时调用 `catalog.to_llamaindex_tools()`。canonical capability ID
仍由闭包保留并用于调用，暴露给函数调用供应商的名称会转换为合法且同一工具面内不冲突的
函数名。

同步代码可调用 `catalog.invoke(...)`；异步代码应调用 `await catalog.ainvoke(...)`，避免
同步兼容桥在事件循环线程上等待线程结果。

MCP 工具的 canonical ID 是 `mcp:<server_id>:<tool_name>`，发现结果保留
`outputSchema`、`annotations`、`_meta` 和分页 cursor；`call_mcp_tool_result()`
保留结构化输出与 `isError`，`call_mcp_tool()` 继续提供旧的文本结果。服务器可
选择 `stdio` 或 `streamable_http` transport，二者共用相同的信任、allowlist 和超时
策略。

Skill 只作为指令/资源包发现：`discover_skills()` 解析 `SKILL.md` frontmatter、
校验名称与描述约束并列出资源，绝不 import 或执行 Skill 目录中的脚本。Skill 的
执行仍由宿主决定，避免把说明文档误当成可调用 handler。

符合性检查分为两步：

```bash
python -m drbrain.plugins.conformance --mode lint <plugin_dir>   # AST-only, safe for untrusted source
python -m drbrain.plugins.conformance --mode probe <plugin_dir>  # controlled discovery-style registration
```

`lint` 不会 import 模块或运行 `register()`；`probe` 保留现有发现语义，便于在受控
环境中验证实际注册结果。
