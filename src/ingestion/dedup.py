"""Deduplication index: sha256(file bytes) -> the doc_id that first ingested
that exact content. Two source files with identical bytes (a policy PDF
copied into two folders, an emailed duplicate) resolve to one logical
document instead of being embedded and indexed twice.
"""

from __future__ import annotations

import json
from pathlib import Path


class DeduplicationIndex:
    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path)
        self._index: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self.index_path.exists():
            self._index = json.loads(self.index_path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(self._index, indent=2, sort_keys=True), encoding="utf-8"
        )

    def lookup(self, sha256: str) -> str | None:
        """Return the doc_id that already owns this content hash, if any."""
        return self._index.get(sha256)

    def register(self, sha256: str, doc_id: str) -> None:
        """Record that this content hash belongs to doc_id (first-write-wins;
        callers should only call this the first time a hash is seen).
        """
        if sha256 not in self._index:
            self._index[sha256] = doc_id
            self._save()

    def __len__(self) -> int:
        return len(self._index)
