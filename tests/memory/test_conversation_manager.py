from __future__ import annotations

from src.memory.context_window_manager import ContextWindowManager
from src.memory.conversation_manager import ConversationManager
from src.memory.session_memory import SessionAccessError


def test_start_session_creates_session_with_user() -> None:
    cm = ConversationManager()
    session = cm.start_session(user_id="alice")
    assert session.user_id == "alice"


def test_prepare_turn_records_user_message() -> None:
    cm = ConversationManager()
    session = cm.start_session(user_id="alice")
    cm.prepare_turn(session.session_id, "What is the notice period?")
    stored = cm.session_store.get_session(session.session_id)
    assert len(stored.turns) == 1
    assert stored.turns[0].content == "What is the notice period?"


def test_record_response_appends_assistant_turn() -> None:
    cm = ConversationManager()
    session = cm.start_session(user_id="alice")
    cm.prepare_turn(session.session_id, "question")
    cm.record_response(session.session_id, "answer")
    stored = cm.session_store.get_session(session.session_id)
    assert len(stored.turns) == 2
    assert stored.turns[1].role == "assistant"
    assert stored.turns[1].content == "answer"


def test_prepare_turn_returns_rewritten_query() -> None:
    cm = ConversationManager()
    session = cm.start_session(user_id="alice")
    cm.prepare_turn(session.session_id, "What is the wire transfer threshold?")
    cm.record_response(session.session_id, "It is $10,000.")
    rewritten = cm.prepare_turn(session.session_id, "What about it internationally?")
    assert rewritten != "What about it internationally?"  # pronoun triggered a rewrite


def test_prepare_turn_compresses_old_turns_into_summary() -> None:
    # Tiny budget forces compression almost immediately.
    cm = ConversationManager(context_window_manager=ContextWindowManager(max_history_tokens=5))
    session = cm.start_session(user_id="alice")
    cm.prepare_turn(session.session_id, "first question")
    cm.record_response(session.session_id, "first answer")
    cm.prepare_turn(session.session_id, "second question")

    stored = cm.session_store.get_session(session.session_id)
    assert stored.summary is not None
    assert "first" in stored.summary.lower()


def test_isolation_enforced_through_conversation_manager() -> None:
    import pytest

    cm = ConversationManager()
    session = cm.start_session(user_id="alice")
    with pytest.raises(SessionAccessError):
        cm.prepare_turn(session.session_id, "question", requesting_user_id="bob")
