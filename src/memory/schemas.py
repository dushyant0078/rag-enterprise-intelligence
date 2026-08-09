"""Data model for src/memory/: a conversation session is a user-scoped,
ordered list of turns, with an optional rolling summary standing in for
turns compressed out of the active context window (see
context_window_manager.py and summarization.py).
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ConversationSession(BaseModel):
    session_id: str
    user_id: str
    turns: list[ConversationTurn] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_active_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    summary: str | None = None  # rolling summary of turns compressed out of the active window
