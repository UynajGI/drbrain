"""Translation strings for the interactive setup wizard (EN / ZH)."""

from __future__ import annotations

# fmt: off
TRANSLATIONS: dict[str, dict[str, str]] = {
    # Header
    "header_banner": {
        "en": "DrBrain Setup",
        "zh": "DrBrain 安装向导",
    },
    "header_note": {
        "en": "Settings in config.yaml (db, dirs, bm25, queue) are already configured.",
        "zh": "config.yaml 中的基本设置（db, dirs, bm25, queue）已配置好。",
    },
    "header_local_only": {
        "en": "This wizard only writes secrets to config.local.yaml.",
        "zh": "此向导仅将密钥写入 config.local.yaml。",
    },

    # Language
    "lang_prompt": {
        "en": "Language / 语言 (en/zh)",
        "zh": "Language / 语言 (en/zh)",
    },

    # LLM section
    "llm_heading": {
        "en": "── LLM ──",
        "zh": "── LLM 大模型 ──",
    },
    "llm_provider": {
        "en": "Provider (openai/anthropic/ollama)",
        "zh": "提供商 (openai/anthropic/ollama)",
    },
    "llm_model": {
        "en": "Model",
        "zh": "模型",
    },
    "llm_api_key": {
        "en": "API key (leave empty for ollama)",
        "zh": "API 密钥（ollama 可留空）",
    },
    "llm_base_url": {
        "en": "Base URL (empty for default)",
        "zh": "Base URL（默认则留空）",
    },
    "llm_add_fallback": {
        "en": "Add fallback model?",
        "zh": "添加备用模型？",
    },
    "llm_fb_provider": {
        "en": "Fallback provider",
        "zh": "备用提供商",
    },
    "llm_fb_model": {
        "en": "Fallback model",
        "zh": "备用模型",
    },
    "llm_fb_api_key": {
        "en": "Fallback API key",
        "zh": "备用 API 密钥",
    },
    "llm_fb_base_url": {
        "en": "Fallback base URL",
        "zh": "备用 base URL",
    },

    # MinerU section
    "mineru_heading": {
        "en": "── MinerU (PDF Parser) ──",
        "zh": "── MinerU（PDF 解析器）──",
    },
    "mineru_note": {
        "en": "Settings in config.yaml. Only the token is secret.",
        "zh": "基本设置在 config.yaml 中。仅 token 是密钥。",
    },
    "mineru_mode": {
        "en": "Mode (flash/token)",
        "zh": "模式 (flash/token)",
    },
    "mineru_token_prompt": {
        "en": "Get token: https://mineru.net/apiManage/token",
        "zh": "获取 token：https://mineru.net/apiManage/token",
    },
    "mineru_token": {
        "en": "Token",
        "zh": "Token",
    },
    "mineru_advanced": {
        "en": "Configure advanced MinerU settings?",
        "zh": "配置高级 MinerU 设置？",
    },
    "mineru_adv_model": {
        "en": "Model (vlm/pipeline/MinerU-HTML)",
        "zh": "模型 (vlm/pipeline/MinerU-HTML)",
    },
    "mineru_adv_ocr": {
        "en": "Enable OCR?",
        "zh": "启用 OCR？",
    },
    "mineru_adv_formula": {
        "en": "Enable formula parsing?",
        "zh": "启用公式解析？",
    },
    "mineru_adv_table": {
        "en": "Enable table parsing?",
        "zh": "启用表格解析？",
    },

    # API keys section
    "api_heading": {
        "en": "── API Keys ──",
        "zh": "── API 密钥 ──",
    },
    "api_deepxiv": {
        "en": "DeepXiv token (https://data.rag.ac.cn/register)",
        "zh": "DeepXiv token（https://data.rag.ac.cn/register）",
    },
    "api_s2": {
        "en": "Semantic Scholar API key (https://semanticscholar.org/product/api)",
        "zh": "Semantic Scholar API 密钥（https://semanticscholar.org/product/api）",
    },
    "api_crossref": {
        "en": "CrossRef email (optional)",
        "zh": "CrossRef 邮箱（可选）",
    },
    "api_openalex": {
        "en": "OpenAlex token (optional)",
        "zh": "OpenAlex token（可选）",
    },

    # Embedding section
    "embed_heading": {
        "en": "── Embedding (tree search + RAPTOR) ──",
        "zh": "── 嵌入模型（树搜索 + RAPTOR）──",
    },
    "embed_desc": {
        "en": "Used by 'drbrain embed --tree' for section-level retrieval.",
        "zh": "用于 'drbrain embed --tree' 章节级检索。",
    },
    "embed_provider": {
        "en": "Provider (local/openai-compat/none)",
        "zh": "提供商 (local/openai-compat/none)",
    },
    "embed_oc_model": {
        "en": "Model",
        "zh": "模型",
    },
    "embed_oc_api_base": {
        "en": "API base URL",
        "zh": "API base URL",
    },
    "embed_oc_api_key": {
        "en": "API key",
        "zh": "API 密钥",
    },
    "embed_local_model": {
        "en": "Model (sentence-transformers)",
        "zh": "模型 (sentence-transformers)",
    },

    # Admin password
    "admin_heading": {
        "en": "── Admin Password (optional) ──",
        "zh": "── 管理员密码（可选）──",
    },
    "admin_desc": {
        "en": "Protects destructive commands like 'drbrain clean --force'.",
        "zh": "保护 'drbrain clean --force' 等破坏性命令。",
    },
    "admin_set": {
        "en": "Set an admin password?",
        "zh": "设置管理员密码？",
    },
    "admin_pw": {
        "en": "Password",
        "zh": "密码",
    },
    "admin_confirm": {
        "en": "Confirm password",
        "zh": "确认密码",
    },
    "admin_mismatch": {
        "en": "Passwords don't match. Skipping.",
        "zh": "密码不匹配，跳过。",
    },

    # Review section
    "review_heading": {
        "en": "── Review ──",
        "zh": "── 确认 ──",
    },
    "review_llm": {
        "en": "LLM",
        "zh": "LLM",
    },
    "review_mineru": {
        "en": "MinerU",
        "zh": "MinerU",
    },
    "review_apis": {
        "en": "APIs",
        "zh": "API",
    },
    "review_embed": {
        "en": "Embed",
        "zh": "嵌入",
    },
    "review_write": {
        "en": "Write config.local.yaml?",
        "zh": "写入 config.local.yaml？",
    },
    "review_free_tier": {
        "en": "free tier",
        "zh": "免费版",
    },
    "review_token_set": {
        "en": "token set",
        "zh": "已设置",
    },
    "review_key_set": {
        "en": "key set",
        "zh": "已设置",
    },
    "review_not_set": {
        "en": "not set",
        "zh": "未设置",
    },
    "review_anonymous": {
        "en": "anonymous",
        "zh": "匿名",
    },
    "review_disabled": {
        "en": "disabled",
        "zh": "已禁用",
    },
    "review_none": {
        "en": "none",
        "zh": "无",
    },
    "review_config_written": {
        "en": "Config written to",
        "zh": "配置已写入",
    },
    "review_cancelled": {
        "en": "Cancelled.",
        "zh": "已取消。",
    },

    # Environment section
    "env_heading": {
        "en": "── Environment ──",
        "zh": "── 环境 ──",
    },
    "env_dirs_present": {
        "en": "Directories: all present",
        "zh": "目录：全部就绪",
    },
    "env_dirs_created": {
        "en": "Created {count} director{plural}",
        "zh": "已创建 {count} 个目录",
    },

    # Agent section
    "agent_none_detected": {
        "en": "No AI platforms detected. Install Claude Code or another AI coding tool, then re-run `drbrain setup`.",
        "zh": "未检测到 AI 平台。请先安装 Claude Code 或其他 AI 编程工具，然后重新运行 `drbrain setup`。",
    },
    "agent_detected_title": {
        "en": "Detected AI platforms:",
        "zh": "检测到的 AI 平台：",
    },
    "agent_all": {
        "en": "All detected",
        "zh": "全部",
    },
    "agent_none": {
        "en": "None (skip)",
        "zh": "跳过",
    },
    "agent_select": {
        "en": "Select platforms (comma-separated numbers)",
        "zh": "选择平台（用逗号分隔数字）",
    },
    "agent_skipped": {
        "en": "Skipped agent entry injection.",
        "zh": "已跳过 Agent 入口注入。",
    },
    "agent_no_selection": {
        "en": "No platforms selected. Skipping agent entry injection.",
        "zh": "未选择平台，跳过 Agent 入口注入。",
    },
    "agent_template_missing": {
        "en": "Template not found",
        "zh": "模板未找到",
    },
    "agent_file_exists": {
        "en": "already exists, skipping",
        "zh": "已存在，跳过",
    },

    # Already-exists flow
    "exists_title": {
        "en": "config.local.yaml already exists.",
        "zh": "config.local.yaml 已存在。",
    },
    "exists_prompt": {
        "en": "[r]e-run setup  [v]alidate environment only  [q]uit",
        "zh": "[r] 重新配置  [v] 仅验证环境  [q] 退出",
    },
    "exists_cancelled": {
        "en": "Cancelled.",
        "zh": "已取消。",
    },

    # Final messages
    "ready": {
        "en": "Ready.",
        "zh": "已就绪。",
    },
    "next_steps": {
        "en": "Next:    drbrain ingest   (see paper-ingest skill)\n  Then:    drbrain build    (see kg-build skill)\n  Then:    drbrain embed --tree && drbrain closure\n  Explore: drbrain query / drbrain ask / drbrain reason\n  Edit config.local.yaml to adjust settings.",
        "zh": "下一步： drbrain ingest   （参见 paper-ingest 技能）\n  然后：   drbrain build    （参见 kg-build 技能）\n  然后：   drbrain embed --tree && drbrain closure\n  探索：   drbrain query / drbrain ask / drbrain reason\n  编辑 config.local.yaml 来调整设置。",
    },
    "setup_warnings": {
        "en": "Setup complete with {count} warning(s).",
        "zh": "安装完成，有 {count} 个警告。",
    },
    "setup_check_hint": {
        "en": "Run `drbrain check` for detailed diagnostics.",
        "zh": "运行 `drbrain check` 查看详细诊断。",
    },
    "env_ready": {
        "en": "Environment ready. Next: drbrain ingest",
        "zh": "环境已就绪。下一步：drbrain ingest",
    },
    "quick_config_written": {
        "en": "Config written to {path} (env vars + defaults)",
        "zh": "配置已写入 {path}（环境变量 + 默认值）",
    },
}
# fmt: on


def t(key: str, lang: str, **fmt) -> str:
    """Return translated string for the given key and language.

    Args:
        key: Translation key.
        lang: Language code (``"en"`` or ``"zh"``).
        **fmt: Optional format kwargs for string interpolation.

    Returns:
        Translated string, with ``**fmt`` applied via ``str.format_map``.
    """
    entry = TRANSLATIONS.get(key, {})
    text = entry.get(lang) or entry.get("en") or key
    if fmt:
        return text.format_map(fmt)
    return text
