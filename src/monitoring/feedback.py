"""User feedback analytics: persists thumbs up/down (+ optional comment) per
chat response, and aggregates it. Real, disk-persisted storage using the
same JSON-per-record pattern as src/memory/session_memory.py, not an
in-memory-only placeholder that loses data on restart.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from src.monitoring.schemas import FeedbackRecord


class FeedbackStore:
    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        session_id: str,
        user_id: str,
        rating: str,
        comment: str | None = None,
        query: str | None = None,
        answer_excerpt: str | None = None,
    ) -> FeedbackRecord:
        if rating not in ("up", "down"):
            raise ValueError(f"rating must be 'up' or 'down', got {rating!r}")

        record = FeedbackRecord(
            feedback_id=str(uuid.uuid4()),
            session_id=session_id,
            user_id=user_id,
            rating=rating,
            comment=comment,
            query=query,
            answer_excerpt=(answer_excerpt or "")[:280] or None,
        )
        path = self.storage_dir / f"{record.feedback_id}.json"
        path.write_text(record.model_dump_json(), encoding="utf-8")
        return record

    def all_records(self) -> list[FeedbackRecord]:
        records = []
        for path in sorted(self.storage_dir.glob("*.json")):
            records.append(FeedbackRecord(**json.loads(path.read_text(encoding="utf-8"))))
        return records

    def summary(self) -> dict[str, float | int]:
        records = self.all_records()
        if not records:
            return {"total": 0, "up": 0, "down": 0, "satisfaction_rate": 0.0}
        up = sum(1 for r in records if r.rating == "up")
        down = sum(1 for r in records if r.rating == "down")
        return {
            "total": len(records),
            "up": up,
            "down": down,
            "satisfaction_rate": round(up / len(records), 4),
        }
