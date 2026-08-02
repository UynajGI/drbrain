"""Tests for the Sciverse HTTP transport (rate limiter + client)."""

from __future__ import annotations

from typing import Any

import pytest

from drbrain.concept_graph.sources._http import (
    SciverseAPIError,
    SciverseClient,
    TokenBucketRateLimiter,
)


class FakeClock:
    """Deterministic monotonic clock with controllable sleep."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_rate_limiter_allows_burst_up_to_limit() -> None:
    fake = FakeClock()
    limiter = TokenBucketRateLimiter(3, clock=fake.clock, sleep=fake.sleep)
    for _ in range(3):
        limiter.acquire()
    assert fake.slept == []  # no waiting within capacity


def test_rate_limiter_blocks_beyond_limit() -> None:
    fake = FakeClock()
    limiter = TokenBucketRateLimiter(2, clock=fake.clock, sleep=fake.sleep)
    limiter.acquire()
    limiter.acquire()
    limiter.acquire()  # third call must wait for the window to slide
    assert len(fake.slept) >= 1
    assert fake.slept[0] > 0


def test_rate_limiter_disabled_when_nonpositive() -> None:
    fake = FakeClock()
    limiter = TokenBucketRateLimiter(0, clock=fake.clock, sleep=fake.sleep)
    for _ in range(10):
        limiter.acquire()
    assert fake.slept == []


class FakeResponse:
    def __init__(self, status_code: int, payload: Any):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]):
        self.headers: dict[str, str] = {}
        self._responses = responses
        self.calls: list[tuple[str, str]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((method, url))
        return self._responses.pop(0)


def _make_client(responses: list[FakeResponse]) -> tuple[SciverseClient, FakeSession]:
    client = SciverseClient("tok", rate_limit=0)  # disable limiter for these tests
    session = FakeSession(responses)
    client._session = session  # type: ignore[assignment]
    return client, session


def test_client_sets_bearer_header() -> None:
    client = SciverseClient("secret-token")
    assert client._session.headers["Authorization"] == "Bearer secret-token"


def test_client_post_success() -> None:
    client, session = _make_client(
        [FakeResponse(200, {"results": [], "code": "SUCCESS", "biz_code": 0})]
    )
    out = client.post("meta-search", json={"query": "x"})
    assert out["code"] == "SUCCESS"
    assert session.calls[0][0] == "POST"
    assert session.calls[0][1].endswith("/meta-search")


def test_client_raises_on_http_error() -> None:
    client, _ = _make_client([FakeResponse(403, {"message": "forbidden"})])
    with pytest.raises(SciverseAPIError) as exc:
        client.get("meta-catalog")
    assert exc.value.status == 403


def test_client_raises_on_non_success_envelope() -> None:
    client, _ = _make_client(
        [FakeResponse(200, {"code": "ERROR", "biz_code": 1, "message": "bad"})]
    )
    with pytest.raises(SciverseAPIError):
        client.post("meta-search", json={})
