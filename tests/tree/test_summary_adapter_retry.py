"""IndexModelSummary: one retry on a transient endpoint hiccup (still fail-closed)."""

from __future__ import annotations

import pytest

from drbrain.tree.prepare import IndexModelSummary


def test_transient_model_error_is_retried_once() -> None:
    class FlakyModel:
        def __init__(self) -> None:
            self.calls = 0

        def call_text(self, prompt, *, max_tokens):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("HTTP 500")
            return {"text": "ok"}

    model = FlakyModel()
    adapter = IndexModelSummary(model)
    assert adapter.complete("prompt", max_tokens=16) == {"text": "ok"}
    assert model.calls == 2
    assert adapter.calls == 2


def test_persistent_model_error_propagates() -> None:
    class AlwaysFails:
        def __init__(self) -> None:
            self.calls = 0

        def call_text(self, prompt, *, max_tokens):
            self.calls += 1
            raise RuntimeError("HTTP 500")

    model = AlwaysFails()
    adapter = IndexModelSummary(model)
    with pytest.raises(RuntimeError):
        adapter.complete("prompt", max_tokens=16)
    assert model.calls == 2
