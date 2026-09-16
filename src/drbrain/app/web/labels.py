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
    "interrupted": ("已中断（可恢复）", "warn"),
    # compute experiments (research_experiments.status)
    "planned": ("已计划", "muted"),
    "computed": ("已计算", "run"),
    "settled": ("已结算", "ok"),
    "keep": ("保留", "ok"),
    "discard": ("废弃", "bad"),
    "insufficient": ("证据不足", "warn"),
    "pending": ("待处理", "muted"),
    "checking": ("检查中", "run"),
    # index build job (tree_build_jobs.state)
    "done": ("已完成", "ok"),
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
    "index_build_refused": "没能启动构建：请检查配置（尤其是 embedding 模型）后重试。",
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


#: Answer-path statuses ("要答案"): never a bare 500, always a sentence plus the
#: next step.  The full six abstain states (FR-S6) land with the index façade;
#: these are the ones the current service contract can already distinguish.
ANSWER_STATUSES: dict[str, tuple[str, str, str]] = {
    "ok": ("答案", "ok", ""),
    "unavailable": (
        "要答案不可用",
        "warn",
        "当前配置没有启用要答案所需的引擎：配置 llamaindex.enabled: true 后重建索引。",
    ),
    "index_not_prepared": (
        "还没有索引",
        "warn",
        "这个引擎不会在提问时临时建索引：先建一次索引，再回到这里提问。",
    ),
    "source_unavailable": (
        "来源不可用",
        "warn",
        "这次提问需要的来源当前不可用：看下面的说明，先检查索引与引擎配置。",
    ),
    "search_failed": (
        "检索失败",
        "bad",
        "这不是“没找到”，而是检索本身没有跑完；请检查索引与服务配置后重试。",
    ),
    "empty_question": ("请输入问题", "warn", "问题不能为空。"),
}


def answer_status(result: dict[str, Any] | None) -> dict[str, str]:
    """Classify a ``service.ask`` result into one user-facing status.

    The service already says *why* (``unavailable_reason``) and *what to do*
    (``hint``, 04-arch A3); this only turns that into a label + tone, and keeps
    working for the older ``{"unavailable": True}`` shape.
    """
    payload = result or {}
    reason = str(payload.get("unavailable_reason") or "")
    status = str(payload.get("status") or "")
    if reason == "index_not_prepared":
        key = "index_not_prepared"
    elif payload.get("unavailable"):
        key = "unavailable"
    elif status in ANSWER_STATUSES:
        key = status
    elif str(payload.get("answer") or "").strip():
        key = "ok"
    elif payload.get("error"):
        key = "search_failed"
    else:
        key = "empty_question"
    label, tone, default_hint = ANSWER_STATUSES[key]
    return {
        "key": key,
        "label": label,
        "tone": tone,
        "hint": str(payload.get("hint") or "") or default_hint,
    }


#: Evidence-search outcomes (FR-S6/S8): "not found" and "broken" are different
#: states, and a display cap is not a total.
EVIDENCE_STATUSES: dict[str, tuple[str, str, str]] = {
    "ok": ("找到证据", "ok", ""),
    "degraded": ("部分来源不可用", "warn", "外部来源这次没跑通，下面是本地证据。"),
    "empty": ("没找到", "warn", "换关键词，或先在索引页确认索引是否就绪。"),
    "source_unavailable": (
        "检索不可用",
        "bad",
        "检索本身没跑起来（不是“没找到”）：先看索引页的状态。",
    ),
    "empty_question": ("请输入要检索的内容", "warn", ""),
}


def evidence_status(result: dict[str, Any] | None) -> dict[str, str]:
    """Classify an ``evidence_search`` payload into one user-facing status."""
    payload = result or {}
    key = str(payload.get("status") or "ok")
    if key not in EVIDENCE_STATUSES:
        key = "ok"
    label, tone, default_hint = EVIDENCE_STATUSES[key]
    return {
        "key": key,
        "label": label,
        "tone": tone,
        "hint": str(payload.get("hint") or "") or default_hint,
    }


#: The three states a corpus can be in — three different things, not one number.
INDEX_STATE_LABELS: dict[str, tuple[str, str]] = {
    "ingested": ("已入库", "正文已登记"),
    "indexed": ("已建索引", "已进入检索结构"),
    "retrievable": ("可检索", "当前索引版本能查到"),
}

#: The three ways of finding something (§1.3), plus the index-internal legs the
#: CLI report names separately (those only show up in the diagnostic fold).
LEG_LABELS: dict[str, tuple[str, str]] = {
    "bm25": ("关键词匹配", "术语、公式、行话最准"),
    "vector": ("语义相似", "换个说法也能找到"),
    "tree": ("按结构导航", "先看目录与主题，再读原文"),
    "lexical": ("词法索引", "关键词匹配背后的索引文件"),
    "fts": ("正文直查", "正文里的原句能被直接搜到"),
}

#: Capability/index reason codes → what it means + the next step.  The index
#: report and ``availability()`` share this vocabulary (an unavailable finder
#: and an unavailable answer path must be explained the same way).
INDEX_REASONS: dict[str, tuple[str, str]] = {
    "no_published_generation": ("还没有发布过索引版本", "运行 drbrain index build 建一次索引。"),
    "no_ready_nodes": ("索引里还没有可用节点", "运行 drbrain index build。"),
    "embedding_profile_changed": (
        "索引是用旧配置建的",
        "换过 embedding 模型/维度后建议重建：drbrain index build。",
    ),
    "embedding_profile_unavailable": ("读不到 embedding 配置", "检查 config.yaml 的 embed 段。"),
    "no_ready_vectors": ("还有片段没算向量", "运行 drbrain index build 补齐向量。"),
    "content_fts_inconsistent": ("正文索引与正文不一致", "重新运行 drbrain index build。"),
    "lexical_index_stale": ("词法索引落后于语料", "运行 drbrain index build 重建。"),
    "tree_unavailable": ("按结构导航这一路不可用", "看下面逐条找法的状态。"),
    "no_published_index": ("还没有可用的检索版本", "运行 drbrain index build。"),
    "no_published_sql_snapshot": (
        "没有已发布的 SQL 快照",
        "统一存储部署不需要它；需要时运行 rag index。",
    ),
    "sql_snapshot_unavailable": ("SQL 快照读不出来", "需要时运行 rag index 重新发布。"),
    "llamaindex_disabled": ("LlamaIndex 引擎未启用", "配置 llamaindex.enabled: true 后重建。"),
    "llamaindex_generation_not_ready": ("LlamaIndex 索引版本未就绪", "运行 rag index。"),
    "generation_unreadable": ("索引版本读不出来", "可能被删或损坏：重新运行 drbrain index build。"),
    "last_build_failed": ("上次构建有阶段失败", "看失败阶段后重新运行 drbrain index build。"),
    "documents_failed": ("有文献处理失败", "运行 drbrain index status 看明细，或重跑 ingest。"),
    "documents_stale": ("有文献在索引之后又变了", "重新运行 drbrain index build。"),
    "vectors_pending": ("还有向量没算完", "运行 drbrain index build。"),
    "storage_audit": ("存储自检发现问题", "见下方自检结果。"),
    "engine_disabled": ("要答案所需的引擎未启用", "配置 llamaindex.enabled: true 后重建索引。"),
    "llm_not_configured": ("没有配置 LLM 模型", "在 config.yaml 的 llm.models 里配置模型。"),
}

#: Verification checks (``index verify``) in user language.
INDEX_CHECKS: dict[str, tuple[str, str]] = {
    "tree_generation": ("索引版本可用", "没有已发布的索引版本：运行 drbrain index build。"),
    "content_fts": ("正文能被直接搜到", "正文索引与正文不一致：重新运行 drbrain index build。"),
    "node_vectors": ("向量与片段一致", "有片段缺向量或向量过期：运行 drbrain index build。"),
    "leaf_reachability": (
        "片段都归到父主题",
        "未归属父主题的片段是合法的多根，不算故障：下次 index build 会重试。",
    ),
    "engine_generation": (
        "引擎版本固定",
        "统一存储下由「按结构导航」腿提供服务；需要固定版本时运行 rag index。",
    ),
    "storage_audit": ("存储自检", "看 storage audit 的发现项。"),
}

#: ``severity`` from the core report → badge tone.
SEVERITY_TONES = {"error": "bad", "warning": "warn", "info": "muted"}


def index_reason(code: Any) -> dict[str, str]:
    """Translate one reason code (``code``, ``code: detail`` or ``state.x: code``).

    The core aggregates both leg reasons and state reasons into one list, so a
    state entry arrives as ``state.vector: no_ready_vectors``; that inner code
    is what the reader needs translated.
    """
    text = str(code or "").strip()
    head, _, detail = text.partition(":")
    if head.startswith("state.") and detail:
        return index_reason(detail)
    label, hint = INDEX_REASONS.get(head, (text, ""))
    return {"code": head, "detail": detail.strip(), "label": label, "hint": hint}


def index_check(name: Any) -> dict[str, str]:
    """Human label + next step for one verification check."""
    key = str(name or "")
    label, hint = INDEX_CHECKS.get(key, (key or "检查项", ""))
    return {"name": key, "label": label, "hint": hint}


def index_leg(key: Any) -> dict[str, str]:
    """Leg key → user-facing name + what it is good at."""
    k = str(key or "")
    label, note = LEG_LABELS.get(k, (k or "找法", ""))
    return {"key": k, "label": label, "note": note}


def error_text(code: str | None) -> str:
    if not code:
        return ""
    return ERROR_MESSAGES.get(str(code), ERROR_MESSAGES["generic"])


#: Index-build stages, in the order the job runs them.
JOB_STAGE_LABELS: dict[str, str] = {
    "lexical": "词法索引（BM25）",
    "fts": "正文索引（FTS）",
    "vectors": "语义向量",
    "hierarchy": "层次结构（主题摘要）",
    "publication": "发布索引版本",
}

#: Job states → one sentence the reader can act on.
JOB_STATE_NOTES: dict[str, str] = {
    "pending": "已在队列里，等待开始。",
    "running": "正在跑；可以离开本页，完成后回来就能看到结果。",
    "paused": "暂停（可继续）。",
    "done": "已完成。",
    "failed": "有阶段失败：结果没有被当成成功；修好后可以再发起一次。",
}


def job_stage(name: Any) -> dict[str, str]:
    """Stage key → user-facing name."""
    key = str(name or "")
    return {"key": key, "label": JOB_STAGE_LABELS.get(key, key or "阶段")}


def job_note(state: Any, *, stale: bool = False) -> str:
    """One honest sentence for a job state (``stale`` = the worker died)."""
    key = str(state or "")
    if stale:
        return "这次运行没有留下活跃的执行者（可能被中断或服务重启过）：可以再点一次构建接着跑。"
    return JOB_STATE_NOTES.get(key, "")


def error_title(code: str | None, status_code: int = 500) -> str:
    if code and code in ERROR_TITLES:
        return ERROR_TITLES[str(code)]
    if status_code == 404:
        return ERROR_TITLES["not_found"]
    if status_code == 422:
        return ERROR_TITLES["validation_error"]
    return ERROR_TITLES["http_error"]


__all__ = [
    "ANSWER_STATUSES",
    "ERROR_MESSAGES",
    "ERROR_TITLES",
    "EVIDENCE_STATUSES",
    "INDEX_CHECKS",
    "INDEX_REASONS",
    "INDEX_STATE_LABELS",
    "JOB_STAGE_LABELS",
    "JOB_STATE_NOTES",
    "LAYER_LABELS",
    "LEG_LABELS",
    "ROLE_LABELS",
    "SEVERITY_TONES",
    "STATUS_LABELS",
    "answer_status",
    "error_text",
    "error_title",
    "evidence_status",
    "index_check",
    "index_leg",
    "index_reason",
    "job_note",
    "job_stage",
    "status_of",
]
