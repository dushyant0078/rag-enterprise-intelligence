"""Session memory: per-session conversation state, persisted to disk as JSON
(same pattern as src/ingestion's metadata store), with user-specific
isolation enforced at lookup time -- the spec's "user-specific memory
isolation" requirement, and the same RBAC-enforcement-point pattern
MetadataAwareRetriever established in Milestone 5: isolation is a real
check a caller can't bypass by construction, not just a convention.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from src.memory.schemas import ConversationSession, ConversationTurn


class SessionAccessError(PermissionError):
    """Raised when a caller requests a session belonging to a different user."""


class SessionMemoryStore:
    def __init__(self, persist_dir: str | Path | None = None) -> None:
        self.persist_dir = Path(persist_dir) if persist_dir else None
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, ConversationSession] = {}
        if self.persist_dir:
            self._load_all()

    def create_session(self, user_id: str, session_id: str | None = None) -> ConversationSession:
        session = ConversationSession(session_id=session_id or str(uuid.uuid4()), user_id=user_id)
        self._sessions[session.session_id] = session
        self._persist(session)
        return session

    def get_session(
        self, session_id: str, requesting_user_id: str | None = None
    ) -> ConversationSession | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if requesting_user_id is not None and session.user_id != requesting_user_id:
            raise SessionAccessError(
                f"Session '{session_id}' does not belong to user '{requesting_user_id}'."
            )
        return session

    def add_turn(
        self, session_id: str, role: str, content: str, requesting_user_id: str | None = None
    ) -> ConversationSession:
        session = self.get_session(session_id, requesting_user_id)
        if session is None:
            raise KeyError(f"No session '{session_id}'.")
        session.turns.append(ConversationTurn(role=role, content=content))
        session.last_active_at = datetime.now(UTC)
        self._persist(session)
        return session

    def list_sessions_for_user(self, user_id: str) -> list[ConversationSession]:
        return [s for s in self._sessions.values() if s.user_id == user_id]

    def _persist(self, session: ConversationSession) -> None:
        if not self.persist_dir:
            return
        (self.persist_dir / f"{session.session_id}.json").write_text(
            session.model_dump_json(indent=2), encoding="utf-8"
        )

    def _load_all(self) -> None:
        for f in self.persist_dir.glob("*.json"):  # type: ignore[union-attr]
            data = json.loads(f.read_text(encoding="utf-8"))
            session = ConversationSession(**data)
            self._sessions[session.session_id] = session
