"""Version tracking for logical documents.

A doc_id identifies a *logical* document (e.g. "the HR handbook"); its content
hash changes each time the underlying file is revised and re-ingested. This
module is the append-only history of that: which hash was version N, when,
and from which source path — so retrieval can prefer the current version and
compliance review can reconstruct what changed and when (see the compliance
risk about serving superseded policy versions in docs/01_business_understanding.md).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel


class VersionRecord(BaseModel):
    version: int
    sha256: str
    source_path: str
    ingested_at: datetime


class VersionTracker:
    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self._history: dict[str, list[VersionRecord]] = {}
        self._load()

    def _load(self) -> None:
        if self.manifest_path.exists():
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self._history = {
                doc_id: [VersionRecord(**record) for record in records]
                for doc_id, records in raw.items()
            }

    def _save(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            doc_id: [json.loads(record.model_dump_json()) for record in records]
            for doc_id, records in self._history.items()
        }
        self.manifest_path.write_text(
            json.dumps(serializable, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )

    def latest(self, doc_id: str) -> VersionRecord | None:
        records = self._history.get(doc_id)
        return records[-1] if records else None

    def is_unchanged(self, doc_id: str, sha256: str) -> bool:
        current = self.latest(doc_id)
        return current is not None and current.sha256 == sha256

    def record_new_version(self, doc_id: str, sha256: str, source_path: str) -> int:
        """Append a new version for doc_id and return its version number.
        Caller is responsible for having already checked ``is_unchanged``.
        """
        records = self._history.setdefault(doc_id, [])
        next_version = (records[-1].version + 1) if records else 1
        records.append(
            VersionRecord(
                version=next_version,
                sha256=sha256,
                source_path=source_path,
                ingested_at=datetime.now(UTC),
            )
        )
        self._save()
        return next_version

    def history(self, doc_id: str) -> list[VersionRecord]:
        return list(self._history.get(doc_id, []))
