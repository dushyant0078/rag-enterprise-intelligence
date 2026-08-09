"""File and content hashing used by ingestion for deduplication and version tracking."""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024  # 1 MiB streaming read, so large PDFs don't load fully into memory


def sha256_file(path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a file's raw bytes on disk.

    Used as the *content identity* of a source document: two files with the
    same hash are byte-identical regardless of filename, so this is what
    deduplication keys off (see ``src/ingestion/dedup.py``).
    """
    path = Path(path)
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK_SIZE):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_text(text: str) -> str:
    """Hash extracted text content (post-parsing). Useful for detecting when
    two files with different bytes (e.g. re-exported PDF) parsed to identical
    text, which sha256_file alone would miss.
    """
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def short_id(value: str, length: int = 12) -> str:
    """Deterministic short identifier derived from a hash, used to build
    human-scannable doc_ids (e.g. ``hr-policy-2025-<short_id>``).
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]
