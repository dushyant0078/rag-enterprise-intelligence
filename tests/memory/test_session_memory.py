from __future__ import annotations

from pathlib import Path

import pytest

from src.memory.session_memory import SessionAccessError, SessionMemoryStore


def test_create_and_get_session() -> None:
    store = SessionMemoryStore()
    session = store.create_session(user_id="alice")
    assert (
        store.get_session(session.session_id) is session
        or store.get_session(session.session_id).session_id == session.session_id
    )


def test_get_nonexistent_session_returns_none() -> None:
    store = SessionMemoryStore()
    assert store.get_session("does-not-exist") is None


def test_add_turn_appends_and_updates_last_active() -> None:
    store = SessionMemoryStore()
    session = store.create_session(user_id="alice")
    first_active = session.last_active_at
    updated = store.add_turn(session.session_id, "user", "hello")
    assert len(updated.turns) == 1
    assert updated.turns[0].role == "user"
    assert updated.turns[0].content == "hello"
    assert updated.last_active_at >= first_active


def test_add_turn_to_nonexistent_session_raises() -> None:
    store = SessionMemoryStore()
    with pytest.raises(KeyError):
        store.add_turn("does-not-exist", "user", "hello")


def test_isolation_blocks_cross_user_access() -> None:
    store = SessionMemoryStore()
    session = store.create_session(user_id="alice")
    with pytest.raises(SessionAccessError):
        store.get_session(session.session_id, requesting_user_id="bob")


def test_isolation_allows_matching_user() -> None:
    store = SessionMemoryStore()
    session = store.create_session(user_id="alice")
    result = store.get_session(session.session_id, requesting_user_id="alice")
    assert result is not None


def test_isolation_enforced_on_add_turn_too() -> None:
    store = SessionMemoryStore()
    session = store.create_session(user_id="alice")
    with pytest.raises(SessionAccessError):
        store.add_turn(session.session_id, "user", "hi", requesting_user_id="bob")


def test_list_sessions_for_user() -> None:
    store = SessionMemoryStore()
    store.create_session(user_id="alice")
    store.create_session(user_id="alice")
    store.create_session(user_id="bob")
    assert len(store.list_sessions_for_user("alice")) == 2
    assert len(store.list_sessions_for_user("bob")) == 1


def test_persists_across_instances(tmp_path: Path) -> None:
    store1 = SessionMemoryStore(persist_dir=tmp_path)
    session = store1.create_session(user_id="alice")
    store1.add_turn(session.session_id, "user", "hello")

    store2 = SessionMemoryStore(persist_dir=tmp_path)
    reloaded = store2.get_session(session.session_id)
    assert reloaded is not None
    assert reloaded.user_id == "alice"
    assert len(reloaded.turns) == 1
