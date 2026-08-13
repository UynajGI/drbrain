"""Retrieval failure semantics: distinguish "no result" from "service broke".

生产级 RAG 的两大坑之一:检索失败如果被当成"没搜到",agent 就会靠预训练知识
硬答,把失败变成幻觉。这里把一次检索显式建模为带状态的结果,让上层能在生成
答案前决定是回答、还是拒绝回答(abstain):

* ``OK`` — 检索成功且有命中,可用;
* ``NO_RESULTS`` — 检索成功但 0 命中(知识库里确实没有),不是故障;
* ``RETRIEVAL_FAILURE`` — 检索器抛异常(服务坏了),绝不能硬答;
* ``TIMEOUT`` — 向量库/检索器超时,同样是故障,不能硬答;
* ``PERMISSION_DENIED`` — 权限拒绝(ACL 层判定);
* ``SOURCE_UNAVAILABLE`` — 数据源不可用(如索引未构建/文件缺失)。

与 ACL 的配合见 :mod:`drbrain.rag.fusion`:权限过滤在 retrieval 层强制注入
(``FusionRetriever.acl_filter`` + ``with_acl``),绝不交给 LLM 事后"别泄密"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class RetrievalStatus(StrEnum):
    """Machine-readable outcome of one retrieval pass."""

    OK = "ok"
    NO_RESULTS = "no_results"
    RETRIEVAL_FAILURE = "retrieval_failure"
    PERMISSION_DENIED = "permission_denied"
    TIMEOUT = "timeout"
    SOURCE_UNAVAILABLE = "source_unavailable"


@dataclass
class RetrievalOutcome:
    """A retrieval pass packaged with its status and a human-readable reason.

    ``status`` is the single source of truth for whether the caller may
    synthesize an answer; ``nodes`` is only meaningful when ``status`` is
    ``OK``. ``message`` explains the outcome for humans (e.g. "检索成功但无结果"
    vs "向量库连接超时").
    """

    status: RetrievalStatus
    nodes: list[Any] = field(default_factory=list)
    message: str = ""

    def is_usable(self) -> bool:
        """True only for a successful retrieval that actually hit something."""
        return self.status is RetrievalStatus.OK and bool(self.nodes)

    def should_abstain(self) -> bool:
        """True when the caller must refuse to answer rather than guess."""
        return not self.is_usable()


class RetrievalError(Exception):
    """Every fusion leg failed; the caller must abstain, never hallucinate.

    Raised by :class:`~drbrain.rag.fusion.FusionRetriever` only when *all*
    legs raise (not when they all return empty — that is ``NO_RESULTS``). A
    single failing leg is still degraded away silently inside the fusion
    layer, so this error signals a systemic outage, not a sparse index.
    """

    def __init__(
        self,
        message: str = "retrieval failed",
        failures: list[tuple[str, RetrievalStatus]] | None = None,
    ) -> None:
        super().__init__(message)
        self.failures: list[tuple[str, RetrievalStatus]] = list(failures or [])


def classify_failure(exc: BaseException) -> RetrievalStatus:
    """Map a leg exception to a failure status.

    ``TimeoutError`` (and ``asyncio.TimeoutError``, aliased since 3.11) →
    ``TIMEOUT``; everything else → ``RETRIEVAL_FAILURE``. ``PERMISSION_DENIED``
    and ``SOURCE_UNAVAILABLE`` are reserved for callers that can detect them
    more specifically than a bare exception type allows.
    """
    if isinstance(exc, TimeoutError):
        return RetrievalStatus.TIMEOUT
    return RetrievalStatus.RETRIEVAL_FAILURE
