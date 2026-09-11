# 插件系统规范（Plugin ABI）

> 状态：**v1（现行）** · 宿主 ABI：`HOST_ABI_VERSION = 1` · 支持集合：`{1}`
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

- [ ] `register(registry)` 幂等（重复调用不产生副作用堆积）
- [ ] `input_schema` 是合法 JSON Schema（宿主据此生成 LLM 工具签名）
- [ ] 长作业：`submit` 只入队快速返回；结果写 `jobs/<job_id>.json` + `.log`；`Artifact` 带 sha256
- [ ] `side_effect` 如实声明（研究回路的门按此分级）
- [ ] 秘钥经 `secret_refs` 引用，不硬编码
- [ ] `abi_version` 显式声明（省略 = 1，仅限兼容期）
- [ ] 不 import 宿主内部模块（`drbrain.plugins` 除外）

## 6. 参考实现

- 协议与注册表：`src/drbrain/plugins/`（protocol / registry / backends）
- 最小示例：`tests/fixtures/plugins/`（model + software 各一，经 `test_plugin_discovery.py` 验证）
- 真实领域插件：`research/plugins/`（材料学 A 线：GBDT 预测、GPAW 实算、physics/topology 重算）
