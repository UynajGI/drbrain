"""讨论层基础设施：消息板 + 研究队列（进程内多 agent 讨论）。

对齐 AutoScientists 的 ClawInstitute ``workshop`` / ``workspace`` 语义，但在
drbrain 进程内实现（纯 Python，不引入外部服务）：

- **消息板（workshop）**：agent 异步发帖（``post``）与评论（``comment``），
  8 种 post 类型（``[PROPOSAL]`` / ``[RESULT]`` / ``[DISCUSSION]`` /
  ``[NEAR-MISS]`` / ``[AUDIT]`` / ``[DISCUSSION-TRIGGER]`` /
  ``[DISCUSS-DONE]`` / ``[TEAM-REFORMED]``）。
- **队列（queue.md）**：``pending`` / ``claims`` 两段 + 乐观锁 claim，对齐
  ``If-Match`` 语义（单进程内用锁保证原子，``version`` 计数对齐版本号）。

Discussion-Before-Queuing 门：一个 proposal 只有在消息板上收到 **非作者**
评论后才能被 claim（``discussion_pending`` 为 True 的条目被 claim 拒绝），
对齐 ROLE-ANALYST Step 5 与 ROLE-GPU Step 3。

本模块零依赖：只 import 标准库，不 import llama-index / pydantic / 竞赛代码。
``QueueItem.hypothesis`` 存运行时对象（workflow 桥接时传入 ``Hypothesis``），
本模块不 import 它，保持可独立测试。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── 8 种 post 类型（对齐 AutoScientists workshop） ────────────────────────────

POST_PROPOSAL = "PROPOSAL"
POST_RESULT = "RESULT"
POST_DISCUSSION = "DISCUSSION"
POST_NEAR_MISS = "NEAR-MISS"
POST_AUDIT = "AUDIT"
POST_DISCUSSION_TRIGGER = "DISCUSSION-TRIGGER"
POST_DISCUSS_DONE = "DISCUSS-DONE"
POST_TEAM_REFORMED = "TEAM-REFORMED"

POST_TYPES = (
    POST_PROPOSAL,
    POST_RESULT,
    POST_DISCUSSION,
    POST_NEAR_MISS,
    POST_AUDIT,
    POST_DISCUSSION_TRIGGER,
    POST_DISCUSS_DONE,
    POST_TEAM_REFORMED,
)


@dataclass
class Comment:
    """A single review/remark on a post.

    ``author`` is the commenter's agent name; ``content`` is the free-text
    counter-argument/flaw. ``score`` (0~1) and ``verdict`` (KEEP/DISCARD) are
    optional and only meaningful for critic comments on ``[PROPOSAL]`` posts.
    """

    author: str
    content: str = ""
    score: float | None = None
    verdict: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "author": self.author,
            "content": self.content,
            "score": self.score,
            "verdict": self.verdict,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Comment:
        return cls(
            author=str(d.get("author", "")),
            content=str(d.get("content", "")),
            score=d.get("score"),
            verdict=d.get("verdict"),
        )


@dataclass
class Post:
    """A message-board post (e.g. a ``[PROPOSAL]`` or ``[RESULT]``)."""

    id: str
    post_type: str
    author: str
    content: str
    comments: list[Comment] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "post_type": self.post_type,
            "author": self.author,
            "content": self.content,
            "comments": [c.to_dict() for c in self.comments],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Post:
        return cls(
            id=str(d.get("id", "")),
            post_type=str(d.get("post_type", "")),
            author=str(d.get("author", "")),
            content=str(d.get("content", "")),
            comments=[Comment.from_dict(c) for c in d.get("comments", [])],
            created_at=float(d.get("created_at", 0.0)),
        )


class MessageBoard:
    """In-process message board (AutoScientists workshop equivalent).

    Thread-safe: a single lock guards both ``post`` and ``comment`` so concurrent
    agent sessions (async critics) never interleave a read-modify-write.
    """

    def __init__(self) -> None:
        self._posts: dict[str, Post] = {}
        self._order: list[str] = []  # insertion order for list_posts
        self._lock = threading.Lock()

    def post(self, post_type: str, author: str, content: str) -> str:
        """Create a post and return its id."""
        if post_type not in POST_TYPES:
            raise ValueError(f"unknown post type {post_type!r}; expected one of {POST_TYPES}")
        pid = uuid.uuid4().hex[:12]
        with self._lock:
            self._posts[pid] = Post(id=pid, post_type=post_type, author=author, content=content)
            self._order.append(pid)
        return pid

    def comment(
        self,
        post_id: str,
        author: str,
        content: str,
        score: float | None = None,
        verdict: str | None = None,
    ) -> None:
        """Append a comment to ``post_id`` (no-op if the post is unknown)."""
        with self._lock:
            post = self._posts.get(post_id)
            if post is None:
                return
            post.comments.append(
                Comment(author=author, content=content, score=score, verdict=verdict)
            )

    def get_post(self, post_id: str) -> Post | None:
        with self._lock:
            return self._posts.get(post_id)

    def list_posts(self, post_type: str | None = None) -> list[Post]:
        with self._lock:
            posts = [self._posts[pid] for pid in self._order if pid in self._posts]
        if post_type is None:
            return posts
        return [p for p in posts if p.post_type == post_type]

    def non_author_comments(self, post_id: str, author: str) -> list[Comment]:
        """Comments on ``post_id`` written by anyone except ``author``.

        The Discussion-Before-Queuing gate: a proposal needs ≥1 of these before
        it may be claimed. A comment from the proposer themself never counts.
        """
        post = self.get_post(post_id)
        if post is None:
            return []
        return [c for c in post.comments if c.author != author]

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            posts = [self._posts[pid] for pid in self._order if pid in self._posts]
        return {"posts": [p.to_dict() for p in posts]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MessageBoard:
        board = cls()
        for p in d.get("posts", []):
            post = Post.from_dict(p)
            board._posts[post.id] = post  # noqa: SLF001 — same-class private access
            board._order.append(post.id)
        return board

    def save(self, path: str | Path) -> None:
        """Persist the whole board to a JSON file (for director audit)."""
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> MessageBoard:
        p = Path(path)
        if not p.is_file():
            return cls()
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))


@dataclass
class QueueItem:
    """A queued experiment awaiting a claim (AutoScientists ``queue.md`` row)."""

    id: str
    statement: str
    proposed_by: str
    discussion_pending: bool = True
    claimed_by: str | None = None
    claimed_at: float | None = None
    score: float = 0.0
    hypothesis: Any = None  # runtime Hypothesis object (workflow bridges it in)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "proposed_by": self.proposed_by,
            "discussion_pending": self.discussion_pending,
            "claimed_by": self.claimed_by,
            "claimed_at": self.claimed_at,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QueueItem:
        return cls(
            id=str(d.get("id", "")),
            statement=str(d.get("statement", "")),
            proposed_by=str(d.get("proposed_by", "")),
            discussion_pending=bool(d.get("discussion_pending", True)),
            claimed_by=d.get("claimed_by"),
            claimed_at=d.get("claimed_at"),
            score=float(d.get("score", 0.0)),
        )


class ResearchQueue:
    """In-process research queue with optimistic-lock claim semantics.

    Mirrors AutoScientists ``queue.md``: a ``pending`` list plus a ``claims`` map.
    ``claim`` refuses ``discussion_pending`` items (Discussion-Before-Queuing) and
    is guarded by a lock so two async agents can never claim the same item. The
    monotonically increasing ``version`` aligns with the ``If-Match`` header the
    real workspace uses — every mutation bumps it.
    """

    def __init__(self) -> None:
        self._pending: list[QueueItem] = []
        self._claims: dict[str, QueueItem] = {}
        self._version = 0
        self._lock = threading.Lock()

    @property
    def version(self) -> int:
        return self._version

    def add(self, item: QueueItem) -> None:
        with self._lock:
            self._pending.append(item)
            self._version += 1

    def claim(self, agent_name: str) -> QueueItem | None:
        """Claim the first non-``discussion_pending`` item, or ``None``.

        ``discussion_pending`` items are skipped (never claimed) — mirroring
        ROLE-GPU Step 3's refusal to claim a proposal that lacks a non-author
        comment. The read-modify-write is atomic under the lock (the single-
        process equivalent of a 409-conflict-free ``If-Match`` PUT).
        """
        with self._lock:
            for item in self._pending:
                if item.discussion_pending:
                    continue
                item.claimed_by = agent_name
                item.claimed_at = time.time()
                self._pending.remove(item)
                self._claims[item.id] = item
                self._version += 1
                return item
        return None

    def release(self, item_id: str) -> None:
        """Release a claimed item back to pending (e.g. failed/stale claim)."""
        with self._lock:
            item = self._claims.pop(item_id, None)
            if item is not None:
                item.claimed_by = None
                item.claimed_at = None
                self._pending.append(item)
                self._version += 1

    def list_pending(self) -> list[QueueItem]:
        with self._lock:
            return list(self._pending)

    def list_claimed(self) -> list[QueueItem]:
        with self._lock:
            return list(self._claims.values())

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "pending": [i.to_dict() for i in self._pending],
                "claims": [i.to_dict() for i in self._claims.values()],
                "version": self._version,
            }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResearchQueue:
        q = cls()
        q._pending = [QueueItem.from_dict(i) for i in d.get("pending", [])]  # noqa: SLF001
        q._claims = {i.id: i for i in (QueueItem.from_dict(x) for x in d.get("claims", []))}  # noqa: SLF001
        q._version = int(d.get("version", 0))  # noqa: SLF001
        return q
