"""讨论层单元测试：消息板 + 队列（进程内多 agent 讨论基础设施）。

验证 Discussion-Before-Queuing 的两个核心不变量：
  1. 消息板只把「非作者」评论算作讨论（proposer 自己的评论不计入门）。
  2. 队列的 ``claim`` 拒绝 ``discussion_pending`` 条目，且并发 claim 不会
     把同一个条目分给两个 agent（乐观锁语义）。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from drbrain.loop.discussion import (
    POST_PROPOSAL,
    POST_RESULT,
    MessageBoard,
    QueueItem,
    ResearchQueue,
)


def test_post_creates_typed_post():
    board = MessageBoard()
    pid = board.post(POST_PROPOSAL, author="analyst", content="h1 假设")
    post = board.get_post(pid)
    assert post is not None
    assert post.post_type == POST_PROPOSAL
    assert post.author == "analyst"
    assert post.content == "h1 假设"


def test_post_and_comment_accept_stable_ids_without_replaying_them():
    board = MessageBoard()
    pid = board.post(POST_PROPOSAL, author="analyst", content="h1", post_id="prp-stable")
    assert board.post(POST_PROPOSAL, author="analyst", content="h1", post_id="prp-stable") == pid

    board.comment(pid, author="critic", content="review", comment_id="rev-stable")
    board.comment(pid, author="critic", content="review", comment_id="rev-stable")
    assert len(board.get_post(pid).comments) == 1
    try:
        board.post(POST_PROPOSAL, author="analyst", content="changed", post_id="prp-stable")
        raise AssertionError("expected conflicting stable post to fail")
    except ValueError:
        pass


def test_post_rejects_unknown_type():
    board = MessageBoard()
    try:
        board.post("NOT-A-TYPE", author="x", content="y")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_comment_and_non_author_gate():
    """Discussion-Before-Queuing: proposer 的评论不算「非作者评论」。"""
    board = MessageBoard()
    pid = board.post(POST_PROPOSAL, author="analyst", content="h1")

    # proposer 自己的评论
    board.comment(pid, author="analyst", content="我再补充一句")
    assert board.non_author_comments(pid, author="analyst") == []

    # 非作者评论
    board.comment(pid, author="critic-1", content="有个漏洞", score=0.8, verdict="KEEP")
    non_author = board.non_author_comments(pid, author="analyst")
    assert len(non_author) == 1
    assert non_author[0].author == "critic-1"
    assert non_author[0].score == 0.8


def test_list_posts_filters_by_type():
    board = MessageBoard()
    board.post(POST_PROPOSAL, author="analyst", content="p1")
    board.post(POST_RESULT, author="compute", content="r1")
    board.post(POST_PROPOSAL, author="analyst", content="p2")
    assert len(board.list_posts()) == 3
    assert len(board.list_posts(POST_PROPOSAL)) == 2
    assert len(board.list_posts(POST_RESULT)) == 1


def test_board_save_load_roundtrip(tmp_path):
    board = MessageBoard()
    pid = board.post(POST_PROPOSAL, author="analyst", content="h1")
    board.comment(pid, author="critic-1", content="漏洞", score=0.7)
    path = tmp_path / "board.json"
    board.save(path)

    loaded = MessageBoard.load(path)
    assert len(loaded.list_posts()) == 1
    post = loaded.list_posts()[0]
    assert post.content == "h1"
    assert len(loaded.non_author_comments(pid, author="analyst")) == 1


def test_queue_claim_skips_discussion_pending():
    """discussion_pending 的条目不可 claim（对齐 ROLE-GPU Step 3）。"""
    q = ResearchQueue()
    q.add(QueueItem(id="a", statement="h1", proposed_by="analyst", discussion_pending=True))
    q.add(QueueItem(id="b", statement="h2", proposed_by="analyst", discussion_pending=False))
    item = q.claim("compute")
    assert item is not None
    assert item.id == "b"  # 跳过 pending 的 a
    assert item.claimed_by == "compute"


def test_queue_claim_returns_none_when_all_pending():
    q = ResearchQueue()
    q.add(QueueItem(id="a", statement="h1", proposed_by="analyst", discussion_pending=True))
    assert q.claim("compute") is None


def test_queue_concurrent_claim_no_double_claim():
    """乐观锁：并发 claim 不会把同一 item 分给两个 agent。"""
    q = ResearchQueue()
    for i in range(20):
        q.add(
            QueueItem(
                id=f"i{i}", statement=f"h{i}", proposed_by="analyst", discussion_pending=False
            )
        )

    claimed_ids: list[str] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(q.claim, f"gpu-{i}") for i in range(8)]
        for fut in futures:
            item = fut.result()
            if item is not None:
                claimed_ids.append(item.id)

    # 每个 id 至多被 claim 一次
    assert len(claimed_ids) == len(set(claimed_ids))
    # 8 个 claim 请求，全部 20 个 pending 都被至少一个请求拿走（8 次并发抢 20 个，
    # 每次 claim 只拿一个，所以拿走了 8 个不同的）。
    assert len(claimed_ids) == 8


def test_queue_release_returns_to_pending():
    q = ResearchQueue()
    q.add(QueueItem(id="a", statement="h1", proposed_by="analyst", discussion_pending=False))
    item = q.claim("compute")
    assert item is not None
    assert len(q.list_claimed()) == 1
    assert q.claim("compute") is None  # 没有可 claim 的了

    q.release("a")
    assert len(q.list_claimed()) == 0
    assert len(q.list_pending()) == 1
    assert q.list_pending()[0].claimed_by is None


def test_queue_version_bumps_on_mutation():
    q = ResearchQueue()
    v0 = q.version
    q.add(QueueItem(id="a", statement="h1", proposed_by="analyst", discussion_pending=False))
    assert q.version > v0
    v1 = q.version
    q.claim("compute")
    assert q.version > v1


def test_queue_ignores_a_replayed_stable_item_id():
    q = ResearchQueue()
    item = QueueItem(id="que-stable", statement="h1", proposed_by="analyst")
    q.add(item)
    q.add(item)

    assert [queued.id for queued in q.list_pending()] == ["que-stable"]


def test_queue_refreshes_a_replayed_stable_item_from_canonical_state():
    q = ResearchQueue()
    q.add(
        QueueItem(id="que-stable", statement="h1", proposed_by="analyst", discussion_pending=True)
    )
    q.add(
        QueueItem(
            id="que-stable",
            statement="h1",
            proposed_by="analyst",
            discussion_pending=False,
            score=0.9,
        )
    )

    claimed = q.claim("compute")
    assert claimed is not None
    assert claimed.score == 0.9


def test_queue_removes_a_pending_item_that_becomes_ineligible():
    q = ResearchQueue()
    q.add(
        QueueItem(id="que-stable", statement="h1", proposed_by="analyst", discussion_pending=False)
    )
    q.remove_pending("que-stable")

    assert q.claim("compute") is None


def test_queue_roundtrip(tmp_path):
    import json

    q = ResearchQueue()
    q.add(QueueItem(id="a", statement="h1", proposed_by="analyst", discussion_pending=False))
    q.claim("compute")
    data = q.to_dict()
    path = tmp_path / "queue.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = ResearchQueue.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert loaded.version == q.version
    assert len(loaded.list_claimed()) == 1
    assert loaded.list_claimed()[0].claimed_by == "compute"
