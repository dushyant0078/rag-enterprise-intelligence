from __future__ import annotations

import pytest

from frontend.api_client import DEFAULT_BASE_URL, RAGAPIClient, resolve_default_base_url


def test_resolve_default_base_url_uses_env_var_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_API_BASE_URL", "http://api:8000")
    assert resolve_default_base_url() == "http://api:8000"


def test_resolve_default_base_url_falls_back_to_constant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_API_BASE_URL", raising=False)
    assert resolve_default_base_url() == DEFAULT_BASE_URL


def test_client_env_var_fallback_only_applies_when_base_url_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for a real deployment-breaking bug: frontend/state.py
    and app.py used to seed st.session_state's default with the bare
    DEFAULT_BASE_URL constant, then always passed that non-None value
    explicitly into RAGAPIClient(base_url=...) -- which short-circuits
    RAGAPIClient's own env-var fallback (`base_url or os.environ.get(...)`),
    since a non-None base_url always wins the `or`. In the real
    docker-compose.yml / k8s deployment, RAG_API_BASE_URL=http://api:8000
    would have had zero effect and the frontend container would have tried
    to reach itself on localhost:8000 instead of the api service. Caught by
    reasoning through the real deployment path (Milestone 12), not by any
    existing frontend test, since every one of those explicitly sets
    session_state before running -- bypassing this default resolution
    entirely. The actual fix lives in frontend/state.py and app.py (using
    resolve_default_base_url() instead of the bare constant); this test
    locks down the underlying contract those call sites depend on.
    """
    monkeypatch.setenv("RAG_API_BASE_URL", "http://api:8000")
    client_using_default = RAGAPIClient(base_url=None)
    assert client_using_default.base_url == "http://api:8000"

    client_with_explicit_url = RAGAPIClient(base_url="http://localhost:8000")
    assert client_with_explicit_url.base_url == "http://localhost:8000"
