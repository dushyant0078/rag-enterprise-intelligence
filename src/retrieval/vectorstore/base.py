"""Base vector store interface. Every backend in src/retrieval/vectorstore/
implements this, so src/retrieval/dense.py (and everything built on top of
it) works identically regardless of which store is configured
(configs/vectorstore/*.yaml).

Namespace = one isolated collection, matching the "namespace-isolated vector
collections per domain" decision in docs/02_architecture.md #8 — RBAC and
cross-domain term-collision concerns are handled by never searching across
namespaces implicitly, not by post-hoc filtering.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.retrieval.vectorstore.schemas import SearchResult, VectorRecord

DEFAULT_NAMESPACE = "default"


class BaseVectorStore(ABC):
    #: Set by subclasses that persist to a local directory (FAISS, Chroma,
    #: local-mode Qdrant) — enables the generic backup() below. Cloud-only
    #: backends (Pinecone) leave this None and override backup() themselves.
    persist_dir: Path | None = None

    @abstractmethod
    def upsert(self, records: list[VectorRecord], namespace: str = DEFAULT_NAMESPACE) -> None:
        """Insert or overwrite records by id."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        namespace: str = DEFAULT_NAMESPACE,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return the top_k nearest records to query_vector, optionally
        restricted to records whose metadata matches every key/value in
        `filters` (equality filtering — e.g. {"domain": "hr"}).
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: list[str], namespace: str = DEFAULT_NAMESPACE) -> None:
        raise NotImplementedError

    @abstractmethod
    def count(self, namespace: str = DEFAULT_NAMESPACE) -> int:
        raise NotImplementedError

    def persist(self) -> None:
        """Flush any in-memory state to disk. No-op for backends that write
        through on every call (Chroma's PersistentClient, Qdrant local mode).
        """
        return None

    def backup(self, dest_dir: str | Path) -> Path:
        """Copy this store's persistence directory to dest_dir. Generic
        implementation for local-disk-backed stores; a real server-mode
        deployment would instead use each store's native snapshot API
        (Qdrant snapshots, a managed Chroma export, etc.) — this is the
        directory-copy equivalent, appropriate for the local/offline backends
        this project actually runs (see docs/05_vectorstore_and_retrieval.md).
        """
        if self.persist_dir is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no local persist_dir to back up "
                "(cloud-only backend — use its native backup/export tooling instead)."
            )
        self.persist()
        dest = Path(dest_dir)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(self.persist_dir, dest)
        return dest
