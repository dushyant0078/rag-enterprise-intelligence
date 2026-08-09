"""Embedding cache: sha256(model_name + text) -> vector, persisted as one
.npy file per entry under cache_dir/<model_name>/. Re-embedding an unchanged
chunk (the common case on every incremental ingestion run) becomes a disk
read instead of a model call — the same "don't redo unchanged work" principle
as src/ingestion/dedup.py and version_tracking.py, applied one stage later.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from src.utils.hashing import sha256_text

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(value: str) -> str:
    return _SLUG_RE.sub("-", value.lower()).strip("-") or "model"


class EmbeddingCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)

    def _path_for(self, text: str, model_name: str) -> Path:
        key = sha256_text(f"{model_name}::{text}")
        return self.cache_dir / _slugify(model_name) / f"{key}.npy"

    def get(self, text: str, model_name: str) -> np.ndarray | None:
        path = self._path_for(text, model_name)
        if not path.exists():
            return None
        return np.load(path)

    def set(self, text: str, model_name: str, vector: np.ndarray) -> None:
        path = self._path_for(text, model_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vector)

    def get_many(self, texts: list[str], model_name: str) -> list[np.ndarray | None]:
        return [self.get(t, model_name) for t in texts]
