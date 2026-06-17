"""Tests for http_retry decorator."""

import pytest

from drbrain.utils.http_retry import _is_retryable_exception, http_retry


class TestIsRetryable:
    def test_connection_error_is_retryable(self):
        assert _is_retryable_exception(ConnectionError("refused"))

    def test_timeout_is_retryable(self):
        assert _is_retryable_exception(TimeoutError("timed out"))

    def test_os_error_is_retryable(self):
        assert _is_retryable_exception(OSError("network unreachable"))

    def test_value_error_not_retryable(self):
        assert not _is_retryable_exception(ValueError("bad input"))

    def test_type_error_not_retryable(self):
        assert not _is_retryable_exception(TypeError("wrong type"))

    def test_http_429_is_retryable(self):
        err = Exception()
        err.code = 429
        assert _is_retryable_exception(err)

    def test_http_404_not_retryable(self):
        err = Exception()
        err.code = 404
        assert not _is_retryable_exception(err)


class TestHttpRetry:
    def test_success_no_retry(self):
        call_count = [0]

        @http_retry(max_retries=3, base_delay=0.01)
        def fetch():
            call_count[0] += 1
            return "ok"

        assert fetch() == "ok"
        assert call_count[0] == 1

    def test_retries_on_connection_error(self):
        call_count = [0]

        @http_retry(max_retries=2, base_delay=0.01)
        def fetch():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError("refused")
            return "ok"

        assert fetch() == "ok"
        assert call_count[0] == 2

    def test_raises_after_max_retries(self):
        call_count = [0]

        @http_retry(max_retries=2, base_delay=0.01)
        def fetch():
            call_count[0] += 1
            raise ConnectionError("always fails")

        with pytest.raises(ConnectionError):
            fetch()
        assert call_count[0] == 3  # initial + 2 retries

    def test_non_retryable_error_raises_immediately(self):
        call_count = [0]

        @http_retry(max_retries=3, base_delay=0.01)
        def fetch():
            call_count[0] += 1
            raise ValueError("not a network error")

        with pytest.raises(ValueError):
            fetch()
        assert call_count[0] == 1  # no retries

    def test_preserves_function_metadata(self):
        @http_retry(max_retries=1, base_delay=0.01)
        def my_fetch(url: str) -> str:
            """My docstring."""
            return url

        assert my_fetch.__name__ == "my_fetch"
        assert my_fetch.__doc__ == "My docstring."
