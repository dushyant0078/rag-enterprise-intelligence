from __future__ import annotations

import anthropic
import pytest

from src.llm.claude_llm import ClaudeLLM, _split_system_message


def test_split_system_message_extracts_system_content() -> None:
    messages = [
        {"role": "system", "content": "sys part 1"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "follow up"},
    ]
    system, chat = _split_system_message(messages)
    assert system == "sys part 1"
    assert chat == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "follow up"},
    ]
    assert all(m["role"] != "system" for m in chat)


def test_split_system_message_joins_multiple_system_messages() -> None:
    messages = [
        {"role": "system", "content": "part one"},
        {"role": "system", "content": "part two"},
        {"role": "user", "content": "hi"},
    ]
    system, chat = _split_system_message(messages)
    assert system == "part one\n\npart two"
    assert len(chat) == 1


def test_split_system_message_no_system_message() -> None:
    messages = [{"role": "user", "content": "hi"}]
    system, chat = _split_system_message(messages)
    assert system == ""
    assert chat == messages


def test_claude_request_reaches_the_real_api_and_fails_only_on_auth() -> None:
    """A real, live connectivity check (not mocked): confirms this backend's
    request actually reaches api.anthropic.com and is rejected only for
    missing/invalid credentials, not for a malformed request -- see
    docs/07_conversational_rag_pipeline.md for why this sandbox can verify
    that much for Claude specifically, unlike the other cloud LLM backends.
    Skips (rather than fails) if the network genuinely isn't reachable, so
    this test doesn't break CI in an environment without that access.
    """
    llm = ClaudeLLM(api_key="sk-ant-invalid-key-for-connectivity-test")
    try:
        with pytest.raises(RuntimeError, match="authentication failed"):
            llm.generate([{"role": "user", "content": "hello"}])
    except anthropic.APIConnectionError:
        pytest.skip("api.anthropic.com not reachable from this environment")
