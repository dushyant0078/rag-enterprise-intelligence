"""Long-term memory: unlike SessionMemoryStore (one session's active turn
history), this persists a rolling list of session summaries *per user*,
retrievable across sessions -- "long-term memory" as distinct from "session
memory" in the spec. A new session can pull in what a returning user's past
sessions were about without replaying their full transcripts.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel


class LongTermMemoryEntry(BaseModel):
    session_id: str
    summary: str
    recorded_at: datetime


class LongTermMemoryStore:
    def __init__(self, persist_dir: str | Path) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, user_id: str) -> Path:
        return self.persist_dir / f"{user_id}.json"

    def append_summary(self, user_id: str, session_id: str, summary: str) -> None:
        entries = self._load(user_id)
        entries.append(
            LongTermMemoryEntry(
                session_id=session_id, summary=summary, recorded_at=datetime.now(UTC)
            )
        )
        self._save(user_id, entries)

    def get_recent_summaries(self, user_id: str, limit: int = 5) -> list[LongTermMemoryEntry]:
        return self._load(user_id)[-limit:]

    def _load(self, user_id: str) -> list[LongTermMemoryEntry]:
        path = self._path_for(user_id)
        if not path.exists():
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [LongTermMemoryEntry(**e) for e in raw]

    def _save(self, user_id: str, entries: list[LongTermMemoryEntry]) -> None:
        self._path_for(user_id).write_text(
            json.dumps([json.loads(e.model_dump_json()) for e in entries], indent=2, default=str),
            encoding="utf-8",
        )
