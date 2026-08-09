from __future__ import annotations

import pytest

from src.monitoring.langsmith_integration import is_configured, log_feedback, traceable_rag_call


def test_is_configured_false_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert is_configured() is False


def test_is_configured_true_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "fake-key-for-test")
    assert is_configured() is True


def test_traceable_decorator_is_a_real_working_decorator() -> None:
    """Verifies the decorator imports and wraps correctly against the real
    installed langsmith SDK -- without a configured project, langsmith's own
    `traceable` degrades to passing calls through unchanged (real SDK
    behavior), so the wrapped function must still work normally.
    """

    @traceable_rag_call("test-stage")
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5


def test_log_feedback_raises_without_configured_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="LANGSMITH_API_KEY"):
        log_feedback("run-123", "quality", 1.0)
