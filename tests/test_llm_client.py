"""Tests for LLM client with logging and metrics tracking."""

from unittest import mock

import pytest

from drbrain.extractor.llm_client import KeyRotator


def test_call_with_fallback_records_metrics():
    """Successful LLM call records metrics."""
    mock_response = mock.Mock()
    mock_response.choices = [mock.Mock()]
    mock_response.choices[0].message.content = '{"result": "ok"}'
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50

    with (
        mock.patch("drbrain.extractor.llm_client.litellm.completion", return_value=mock_response),
        mock.patch("drbrain.extractor.llm_client._record_llm") as mock_record,
    ):
        from drbrain.extractor.llm_client import call_with_fallback

        result = call_with_fallback(
            "test",
            [{"provider": "openai", "model": "gpt-4", "api_key": "sk-test"}],
        )
        assert result == {"result": "ok"}
        mock_record.assert_called_once()


def test_call_with_fallback_tries_next_model_on_failure():
    """When first model fails, second model is tried."""
    mock_fail = mock.Mock(side_effect=Exception("API error"))
    mock_success = mock.Mock()
    mock_success.choices = [mock.Mock()]
    mock_success.choices[0].message.content = '{"ok": true}'

    with (
        mock.patch(
            "drbrain.extractor.llm_client.litellm.completion",
            side_effect=[mock_fail, mock_success],
        ),
        mock.patch("drbrain.extractor.llm_client._record_llm"),
    ):
        from drbrain.extractor.llm_client import call_with_fallback

        result = call_with_fallback(
            "test",
            [
                {"provider": "openai", "model": "broken", "api_key": "x"},
                {"provider": "openai", "model": "working", "api_key": "x"},
            ],
        )
        assert result == {"ok": True}


def test_call_with_fallback_all_fail():
    """When all models fail, returns None."""
    with mock.patch(
        "drbrain.extractor.llm_client.litellm.completion",
        side_effect=Exception("All dead"),
    ):
        from drbrain.extractor.llm_client import call_with_fallback

        result = call_with_fallback(
            "test",
            [{"provider": "openai", "model": "broken", "api_key": "x"}],
        )
        assert result is None


def test_call_with_fallback_can_disable_thinking_for_one_request():
    """A structured request can opt into the Qwen/Zhipu thinking switch."""
    mock_response = mock.Mock()
    mock_response.choices = [mock.Mock()]
    mock_response.choices[0].message.content = '{"ok": true}'
    mock_response.usage.prompt_tokens = 1
    mock_response.usage.completion_tokens = 1

    with (
        mock.patch(
            "drbrain.extractor.llm_client.litellm.completion", return_value=mock_response
        ) as completion,
        mock.patch("drbrain.extractor.llm_client._record_llm"),
        mock.patch("drbrain.extractor.llm_client._log_llm_call"),
    ):
        from drbrain.extractor.llm_client import call_with_fallback

        result = call_with_fallback(
            "test",
            [
                {
                    "provider": "openai",
                    "model": "gpt-4",
                    "api_key": "sk-test",
                    "thinking_param": "enable_thinking",
                }
            ],
            disable_thinking=True,
        )

    assert result == {"ok": True}
    assert completion.call_args.kwargs["extra_body"] == {"enable_thinking": False}


class TestKeyRotator:
    """Key rotation strategies: round_robin cycling and hash-bound mapping."""

    def test_round_robin_cycles_in_order(self):
        rotator = KeyRotator(["k1", "k2", "k3"])
        assert [rotator.next() for _ in range(6)] == ["k1", "k2", "k3", "k1", "k2", "k3"]

    def test_round_robin_single_key_always_same(self):
        rotator = KeyRotator(["only"])
        assert [rotator.next() for _ in range(4)] == ["only"] * 4

    def test_round_robin_ignores_key_hint(self):
        rotator = KeyRotator(["k1", "k2"])
        assert rotator.next(key_hint="whatever") == "k1"

    def test_hash_strategy_stable_for_same_hint(self):
        rotator = KeyRotator(["k1", "k2", "k3"], strategy="hash")
        first = rotator.next("material-A")
        for _ in range(5):
            assert rotator.next("material-A") == first

    def test_hash_strategy_mapping_within_range(self):
        rotator = KeyRotator(["k1", "k2", "k3"], strategy="hash")
        for entity in ["a", "bb", "ccc", "dddd"]:
            assert rotator.next(entity) in {"k1", "k2", "k3"}

    def test_hash_strategy_requires_key_hint(self):
        rotator = KeyRotator(["k1", "k2"], strategy="hash")
        with pytest.raises(ValueError):
            rotator.next()

    def test_empty_keys_raises(self):
        with pytest.raises(ValueError):
            KeyRotator([])

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            KeyRotator(["k1"], strategy="random")


class TestResolveApiKey:
    """``_resolve_api_key``: ``api_keys`` (list) rotates, bare ``api_key`` passes through."""

    def test_api_keys_round_robin(self):
        from drbrain.extractor.llm_client import _resolve_api_key

        cfg = {"provider": "openai", "model": "m", "api_keys": ["k1", "k2", "k3"]}
        picks = [_resolve_api_key(cfg) for _ in range(6)]
        assert picks == ["k1", "k2", "k3", "k1", "k2", "k3"]

    def test_bare_api_key_passes_through(self):
        from drbrain.extractor.llm_client import _resolve_api_key

        assert _resolve_api_key({"api_key": "sk-x"}) == "sk-x"

    def test_no_key_returns_none(self):
        from drbrain.extractor.llm_client import _resolve_api_key

        assert _resolve_api_key({}) is None

    def test_api_keys_with_empties_skipped(self):
        from drbrain.extractor.llm_client import _resolve_api_key

        cfg = {"api_keys": ["k1", "", None, "k2"]}
        picks = [_resolve_api_key(cfg) for _ in range(4)]
        assert picks == ["k1", "k2", "k1", "k2"]


class TestResolveAgentKey:
    """``resolve_agent_key``: one agent = one fixed key; different agents rotate."""

    def test_pins_single_key_and_drops_api_keys(self):
        from drbrain.extractor.llm_client import resolve_agent_key

        cfg = {"provider": "openai", "model": "m", "api_keys": ["k1", "k2", "k3"]}
        out = resolve_agent_key(cfg)
        assert out["api_key"] in {"k1", "k2", "k3"}
        assert "api_keys" not in out

    def test_agents_rotate_across_instances(self):
        from drbrain.extractor.llm_client import resolve_agent_key

        cfg = {"api_keys": ["k1", "k2", "k3"]}
        picked = [resolve_agent_key(cfg)["api_key"] for _ in range(6)]
        # round-robin across agents: k1,k2,k3,k1,k2,k3 (modulo starting counter)
        assert picked == picked[:3] * 2

    def test_does_not_mutate_input_dict(self):
        from drbrain.extractor.llm_client import resolve_agent_key

        cfg = {"api_keys": ["k1", "k2"]}
        resolve_agent_key(cfg)
        assert cfg == {"api_keys": ["k1", "k2"]}  # original untouched

    def test_bare_api_key_passthrough(self):
        from drbrain.extractor.llm_client import resolve_agent_key

        assert resolve_agent_key({"api_key": "sk-x"}) == {"api_key": "sk-x"}


class TestRpmThrottle:
    """Per-model RPM throttle: pace calls to stay within a configured rpm."""

    def test_interval_secs(self):
        from drbrain.extractor.llm_client import _rpm_interval_secs

        assert _rpm_interval_secs({"rpm": 30}) == 2.0
        assert _rpm_interval_secs({"rpm": 60}) == 1.0
        assert _rpm_interval_secs({"rpm": 0}) == 0.0
        assert _rpm_interval_secs({}) == 0.0
        assert _rpm_interval_secs({"rpm": "oops"}) == 0.0

    def test_throttle_sleeps_on_second_call(self):
        from drbrain.extractor.llm_client import _rpm_last_call, _throttle

        cfg = {"provider": "openai", "model": "throttle-test", "rpm": 30}
        _rpm_last_call.pop("openai/throttle-test", None)  # 清残留状态
        with mock.patch("drbrain.extractor.llm_client.time.sleep") as m_sleep:
            _throttle(cfg)  # 第一次：不 sleep
            m_sleep.assert_not_called()
            _throttle(cfg)  # 第二次：等 ~2s 保持 30 rpm
            m_sleep.assert_called_once()
            wait = m_sleep.call_args[0][0]
            assert 1.5 <= wait <= 2.1

    def test_throttle_skips_without_rpm(self):
        from drbrain.extractor.llm_client import _throttle

        with mock.patch("drbrain.extractor.llm_client.time.sleep") as m_sleep:
            _throttle({"provider": "openai", "model": "no-rpm"})
            m_sleep.assert_not_called()

    def test_athrottle_awaits_interval(self):
        import asyncio

        from drbrain.extractor.llm_client import _athrottle, _rpm_async_last_call

        cfg = {"provider": "openai", "model": "throttle-async-test", "rpm": 30}
        _rpm_async_last_call.pop("openai/throttle-async-test", None)

        async def _go():
            with mock.patch("drbrain.extractor.llm_client.asyncio.sleep") as m_sleep:
                await _athrottle(cfg)  # 第一次：不等
                m_sleep.assert_not_called()
                await _athrottle(cfg)  # 第二次：等 ~2s
                m_sleep.assert_called_once()
                wait = m_sleep.call_args[0][0]
                assert 1.5 <= wait <= 2.1

        asyncio.run(_go())


class TestRateLimitStateMachine:
    """自动路由状态机：限流 → 冷却跳过 → 到期恢复 / 成功复位（per-key）。"""

    def _cfg(self, name="rl-test", **kw):
        return {"provider": "openai", "model": name, **kw}

    def test_is_rate_limit_detection(self):
        from drbrain.extractor.llm_client import _is_rate_limit

        assert _is_rate_limit(Exception("RateLimitError: free model capacity is limited"))
        assert _is_rate_limit(Exception("429 Too Many Requests"))
        assert _is_rate_limit(Exception("Out of credits: balance $0.0000"))
        assert _is_rate_limit(Exception("quota exceeded"))
        assert _is_rate_limit(Exception("Monthly usage limit reached"))
        # 非限流错误不误判
        assert not _is_rate_limit(Exception("Expecting property name enclosed in double quotes"))
        assert not _is_rate_limit(Exception("connection reset"))

    def test_cooldown_skip_and_recovery(self):
        from drbrain.extractor.llm_client import RateLimitStateMachine

        sm = RateLimitStateMachine()
        cfg = self._cfg(cooldown_secs=60)
        key = "sk-a"
        assert sm.is_key_available(cfg, key)  # 初始 NORMAL
        wait = sm.on_rate_limit(cfg, key)
        assert wait == 60.0
        assert not sm.is_key_available(cfg, key)  # 冷却中 → 跳过
        assert sm.remaining(cfg, key) > 0
        # 冷却到期 → 自动恢复 NORMAL
        with mock.patch("drbrain.extractor.llm_client.time.monotonic", return_value=1e9 + 100):
            assert sm.is_key_available(cfg, key)

    def test_success_resets_cooldown(self):
        from drbrain.extractor.llm_client import RateLimitStateMachine

        sm = RateLimitStateMachine()
        cfg = self._cfg(cooldown_secs=60)
        key = "sk-a"
        sm.on_rate_limit(cfg, key)
        assert not sm.is_key_available(cfg, key)
        sm.on_success(cfg, key)
        assert sm.is_key_available(cfg, key)
        assert sm.remaining(cfg, key) == 0.0

    def test_exponential_backoff_capped(self):
        from drbrain.extractor.llm_client import RateLimitStateMachine

        sm = RateLimitStateMachine()
        cfg = self._cfg(cooldown_secs=10, max_cooldown_secs=40)
        key = "sk-a"
        waits = [sm.on_rate_limit(cfg, key) for _ in range(5)]
        assert waits == [10.0, 20.0, 40.0, 40.0, 40.0]  # 10→20→40 封顶

    def test_zero_cooldown_disables_skip(self):
        from drbrain.extractor.llm_client import RateLimitStateMachine

        sm = RateLimitStateMachine()
        cfg = self._cfg(cooldown_secs=0)
        key = "sk-a"
        assert sm.on_rate_limit(cfg, key) == 0.0
        assert sm.is_key_available(cfg, key)  # 冷却 0s → 立即可用

    def test_per_key_isolation(self):
        """一个 key 冷却不影响同模型的其他 key。"""
        from drbrain.extractor.llm_client import RateLimitStateMachine

        sm = RateLimitStateMachine()
        cfg = self._cfg(cooldown_secs=60)
        sm.on_rate_limit(cfg, "sk-bad")
        assert not sm.is_key_available(cfg, "sk-bad")
        assert sm.is_key_available(cfg, "sk-good")  # 其他 key 不受影响

    def test_resolve_api_key_skips_cooldown_key(self):
        """round-robin 跳过冷却中的 key，用下一个可用 key。"""
        from drbrain.extractor.llm_client import _RATE_LIMIT_SM, _resolve_api_key

        cfg = {
            "provider": "openai",
            "model": "rl-keys",
            "api_keys": ["sk-bad", "sk-good"],
            "cooldown_secs": 60,
        }
        _RATE_LIMIT_SM.on_success(cfg, "sk-bad")
        _RATE_LIMIT_SM.on_success(cfg, "sk-good")
        try:
            _RATE_LIMIT_SM.on_rate_limit(cfg, "sk-bad")  # 把 sk-bad 打入冷却
            # 轮换应跳过 sk-bad，返回 sk-good
            got = {_resolve_api_key(cfg) for _ in range(4)}
            assert got == {"sk-good"}
        finally:
            _RATE_LIMIT_SM.on_success(cfg, "sk-bad")
            _RATE_LIMIT_SM.on_success(cfg, "sk-good")

    def test_call_with_fallback_skips_exhausted_key(self):
        """一个 key 限流 → 只跳过该 key，用池里下一个 key 继续（不跳过整个模型）。"""
        from drbrain.extractor.llm_client import _RATE_LIMIT_SM, call_with_fallback

        mock_response = mock.Mock()
        mock_response.choices = [mock.Mock()]
        mock_response.choices[0].message.content = '{"ok": true}'
        mock_response.usage.prompt_tokens = 1
        mock_response.usage.completion_tokens = 1

        cfg = {
            "provider": "openai",
            "model": "rl-keys2",
            "api_keys": ["sk-bad", "sk-good"],
            "cooldown_secs": 60,
        }
        _RATE_LIMIT_SM.on_success(cfg, "sk-bad")
        _RATE_LIMIT_SM.on_success(cfg, "sk-good")

        def _side_effect(**kwargs):
            if kwargs.get("api_key") == "sk-bad":
                raise Exception("Monthly usage limit reached. Resets in 19 days.")
            return mock_response

        try:
            with (
                mock.patch(
                    "drbrain.extractor.llm_client.litellm.completion", side_effect=_side_effect
                ) as m_comp,
                mock.patch("drbrain.extractor.llm_client._record_llm"),
                mock.patch("drbrain.extractor.llm_client._log_llm_call"),
            ):
                result = call_with_fallback("p", [cfg])
            assert result == {"ok": True}
            # sk-bad 被跳过，sk-good 成功
            used_keys = [c.kwargs.get("api_key") for c in m_comp.call_args_list]
            assert "sk-good" in used_keys
            # sk-bad 已进入冷却
            assert not _RATE_LIMIT_SM.is_key_available(cfg, "sk-bad")
            assert _RATE_LIMIT_SM.is_key_available(cfg, "sk-good")
        finally:
            _RATE_LIMIT_SM.on_success(cfg, "sk-bad")
            _RATE_LIMIT_SM.on_success(cfg, "sk-good")

    def test_call_with_fallback_all_keys_exhausted_skips_model(self):
        """所有 key 都冷却 → 跳过该模型，fallback 链走下一个模型。"""
        from drbrain.extractor.llm_client import _RATE_LIMIT_SM, call_with_fallback

        mock_response = mock.Mock()
        mock_response.choices = [mock.Mock()]
        mock_response.choices[0].message.content = '{"ok": true}'
        mock_response.usage.prompt_tokens = 1
        mock_response.usage.completion_tokens = 1

        rl_cfg = {
            "provider": "openai",
            "model": "rl-all",
            "api_keys": ["sk-a", "sk-b"],
            "cooldown_secs": 60,
        }
        ok_cfg = {"provider": "openai", "model": "rl-ok3"}
        _RATE_LIMIT_SM.on_success(rl_cfg, "sk-a")
        _RATE_LIMIT_SM.on_success(rl_cfg, "sk-b")
        # 两个 key 都打入冷却
        _RATE_LIMIT_SM.on_rate_limit(rl_cfg, "sk-a")
        _RATE_LIMIT_SM.on_rate_limit(rl_cfg, "sk-b")
        try:
            with (
                mock.patch(
                    "drbrain.extractor.llm_client.litellm.completion", return_value=mock_response
                ) as m_comp,
                mock.patch("drbrain.extractor.llm_client._record_llm"),
                mock.patch("drbrain.extractor.llm_client._log_llm_call"),
            ):
                result = call_with_fallback("p", [rl_cfg, ok_cfg])
            assert result == {"ok": True}
            # 只调用了 ok_cfg（rl-all 全 key 冷却被跳过）
            called_models = [c.kwargs["model"] for c in m_comp.call_args_list]
            assert called_models == ["openai/rl-ok3"]
        finally:
            _RATE_LIMIT_SM.on_success(rl_cfg, "sk-a")
            _RATE_LIMIT_SM.on_success(rl_cfg, "sk-b")
