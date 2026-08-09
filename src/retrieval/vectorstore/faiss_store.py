"""FAISS vector store backend. Fully local — no server, no network call at
any point — so this is what's actually exercised end-to-end in this sandbox
(see the other backends' docstrings for which of *them* need a reachable
service and why that couldn't be verified here).

Design notes:
- One `faiss.IndexIDMap(IndexFlatIP)` per namespace, so namespaces really are
  isolated indexes, not a shared index with a metadata tag (matches the
  "namespace-isolated vector collections" decision in docs/02_architecture.md).
  IndexFlatIP (inner product) assumes vectors are already L2-normalized
  (true for both embedding backends in Milestone 4), so inner product ==
  cosine similarity without a separate normalization step here.
- FAISS has no native metadata filtering. Filtered search here over-fetches
  candidates and filters in Python — correct and fine at the corpus sizes
  this project actually runs, but a documented scaling limitation: a
  production deployment with millions of vectors per namespace would need
  either per-filter-value sub-indexes or a store with native filtered ANN
  search (Qdrant/Chroma, both implemented alongside this one).
- IndexIDMap supports int64 ids only, so this store maintains its own
  str_id <-> int64_id mapping rather than exposing that constraint upward.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import faiss
import numpy as np

from src.retrieval.vectorstore.base import DEFAULT_NAMESPACE, BaseVectorStore
from src.retrieval.vectorstore.schemas import SearchResult, VectorRecord


def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        actual = metadata.get(key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


class _Namespace:
    def __init__(self, dimensions: int) -> None:
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dimensions))
        self.str_to_int: dict[str, int] = {}
        self.int_to_str: dict[int, str] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        self.next_int_id = 0


class FaissVectorStore(BaseVectorStore):
    def __init__(self, persist_dir: str | Path, dimensions: int) -> None:
        self.persist_dir: Path = Path(persist_dir)  # narrows base class's Path | None for this
        # always-local-disk-backed backend, so mypy knows every use below is safe.
        self.dimensions = dimensions
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._namespaces: dict[str, _Namespace] = {}
        self._load_all()

    def _ns(self, namespace: str) -> _Namespace:
        if namespace not in self._namespaces:
            self._namespaces[namespace] = _Namespace(self.dimensions)
        return self._namespaces[namespace]

    def upsert(self, records: list[VectorRecord], namespace: str = DEFAULT_NAMESPACE) -> None:
        if not records:
            return
        ns = self._ns(namespace)
        new_int_ids: list[int] = []
        new_vectors: list[list[float]] = []
        stale_int_ids: list[int] = []

        for record in records:
            if record.id in ns.str_to_int:
                stale_int_ids.append(ns.str_to_int[record.id])
            int_id = ns.next_int_id
            ns.next_int_id += 1
            ns.str_to_int[record.id] = int_id
            ns.int_to_str[int_id] = record.id
            ns.metadata[record.id] = record.metadata
            new_int_ids.append(int_id)
            new_vectors.append(record.vector)

        if stale_int_ids:
            # faiss-cpu's type stubs claim remove_ids wants an IDSelector, but the real runtime
            # API accepts a plain int64 ndarray directly (verified in this project's smoke
            # testing) -- a stub/reality mismatch, not an actual type error.
            ns.index.remove_ids(np.array(stale_int_ids, dtype="int64"))  # type: ignore[arg-type]
        vectors = np.array(new_vectors, dtype="float32")
        ns.index.add_with_ids(vectors, np.array(new_int_ids, dtype="int64"))

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        namespace: str = DEFAULT_NAMESPACE,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if namespace not in self._namespaces or self._namespaces[namespace].index.ntotal == 0:
            return []
        ns = self._namespaces[namespace]

        fetch_k = min(top_k * 10 if filters else top_k, ns.index.ntotal)
        query = np.array([query_vector], dtype="float32")
        scores, int_ids = ns.index.search(query, fetch_k)

        results: list[SearchResult] = []
        for score, int_id in zip(scores[0], int_ids[0], strict=True):
            if int_id == -1:
                continue
            str_id = ns.int_to_str.get(int(int_id))
            if str_id is None:
                continue
            metadata = ns.metadata.get(str_id, {})
            if filters and not _matches_filters(metadata, filters):
                continue
            results.append(SearchResult(id=str_id, score=float(score), metadata=metadata))
            if len(results) >= top_k:
                break
        return results

    def delete(self, ids: list[str], namespace: str = DEFAULT_NAMESPACE) -> None:
        if namespace not in self._namespaces:
            return
        ns = self._namespaces[namespace]
        int_ids = [ns.str_to_int.pop(i) for i in ids if i in ns.str_to_int]
        for i in ids:
            ns.metadata.pop(i, None)
        for int_id in int_ids:
            ns.int_to_str.pop(int_id, None)
        if int_ids:
            ns.index.remove_ids(np.array(int_ids, dtype="int64"))  # type: ignore[arg-type]  # see upsert()'s note

    def count(self, namespace: str = DEFAULT_NAMESPACE) -> int:
        if namespace not in self._namespaces:
            return 0
        return int(self._namespaces[namespace].index.ntotal)

    # -- persistence ----------------------------------------------------

    def persist(self) -> None:
        for namespace, ns in self._namespaces.items():
            ns_dir = self.persist_dir / namespace
            ns_dir.mkdir(parents=True, exist_ok=True)
            faiss.write_index(ns.index, str(ns_dir / "index.faiss"))
            (ns_dir / "state.json").write_text(
                json.dumps(
                    {
                        "str_to_int": ns.str_to_int,
                        "metadata": ns.metadata,
                        "next_int_id": ns.next_int_id,
                    }
                ),
                encoding="utf-8",
            )

    def _load_all(self) -> None:
        if not self.persist_dir.exists():
            return
        for ns_dir in self.persist_dir.iterdir():
            index_path = ns_dir / "index.faiss"
            state_path = ns_dir / "state.json"
            if not (index_path.exists() and state_path.exists()):
                continue
            ns = _Namespace(self.dimensions)
            ns.index = cast(faiss.IndexIDMap, faiss.read_index(str(index_path)))
            state = json.loads(state_path.read_text(encoding="utf-8"))
            ns.str_to_int = state["str_to_int"]
            ns.int_to_str = {v: k for k, v in ns.str_to_int.items()}
            ns.metadata = state["metadata"]
            ns.next_int_id = state["next_int_id"]
            self._namespaces[ns_dir.name] = ns
