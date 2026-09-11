"""Single source of truth for user-facing labels and redirect error codes.

The server renders labels here and reuses them for SSE payloads, so the live
JavaScript never grows a second copy of a string table (HTML and the stream
must not drift).  Redirect handlers pass *codes*, never raw messages: a
crafted ``?error=`` link cannot spoof arbitrary text into an alert.
"""

from __future__ import annotations

from typing import Any

STATUS_LABELS: dict[str, tuple[str, str]] = {
    "created": ("待运行", "muted"),
    "running": ("运行中", "run"),
    "paused": ("暂停", "warn"),
    "succeeded": ("成功", "ok"),
    "failed": ("失败", "bad"),
    "cancelled": ("已取消", "muted"),
    "interrupted": ("已中断", "warn"),
    "keep": ("保留", "ok"),
    "discard": ("废弃", "bad"),
    "insufficient": ("证据不足", "warn"),
    "pending": ("待处理", "muted"),
    "checking": ("检查中", "run"),
    "passed": ("通过", "ok"),
    "stale": ("已过期", "warn"),
    "not_tested": ("未检测", "muted"),
    "proposed": ("已提出", "muted"),
    "critiqued": ("已评审", "run"),
    "discarded": ("已舍弃", "bad"),
    "discussion_pending": ("讨论中", "warn"),
    "uploaded": ("已上传", "run"),
    "extracted": ("已抽取", "ok"),
    "placeholder": ("占位", "muted"),
    "merged": ("已合并", "muted"),
}

ROLE_LABELS = {"user": "我", "assistant": "助手", "system": "系统", "tool": "工具"}
LAYER_LABELS = {"project": "项目记忆", "session": "会话记忆", "run": "运行记忆"}

#: Redirect codes -> fixed user-facing messages (no free-form text in URLs).
ERROR_MESSAGES = {
    "autoresearch_disabled": "Autoresearch 未启用：请在 config.yaml 中设置 autoresearch.enabled: true。",
    "empty_topic": "研究目标不能为空。",
    "run_start_failed": "启动运行失败，请检查配置后重试。",
    "invalid_max_cycles": "最大轮数需在 1–100 之间。",
    "empty_question": "请输入问题。",
    "memory_promote_failed": "记忆提升失败：条目不存在或已失效。",
    "generic": "操作未完成，请检查输入后重试。",
}

#: Error-page headings by code (the body carries the detailed message).
ERROR_TITLES = {
    "not_found": "找不到内容",
    "invalid_cursor": "分页参数无效",
    "validation_error": "请求参数无效",
    "forbidden": "没有权限",
    "http_error": "请求未完成",
    "project_not_found": "找不到项目",
    "session_not_found": "找不到会话",
    "run_not_found": "找不到研究运行",
    "paper_not_found": "找不到文献",
    "evidence_not_found": "找不到证据",
    "artifact_not_found": "找不到计算产物",
    "internal": "服务内部错误",
}


def status_of(value: Any) -> dict[str, str]:
    text = str(value or "")
    label, tone = STATUS_LABELS.get(text, (text or "未知", "muted"))
    return {"label": label, "tone": tone, "raw": text}


def error_text(code: str | None) -> str:
    if not code:
        return ""
    return ERROR_MESSAGES.get(str(code), ERROR_MESSAGES["generic"])


def error_title(code: str | None, status_code: int = 500) -> str:
    if code and code in ERROR_TITLES:
        return ERROR_TITLES[str(code)]
    if status_code == 404:
        return ERROR_TITLES["not_found"]
    if status_code == 422:
        return ERROR_TITLES["validation_error"]
    return ERROR_TITLES["http_error"]


__all__ = [
    "ERROR_MESSAGES",
    "ERROR_TITLES",
    "LAYER_LABELS",
    "ROLE_LABELS",
    "STATUS_LABELS",
    "error_text",
    "error_title",
    "status_of",
]
