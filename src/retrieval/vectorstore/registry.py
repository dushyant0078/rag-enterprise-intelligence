"""Maps a configured vector store name (configs/vectorstore/*.yaml) to a
concrete BaseVectorStore instance.
"""

from __future__ import annotations

from typing import Any

from src.retrieval.vectorstore.base import BaseVectorStore


class UnknownVectorStoreError(ValueError):
    pass


def available_stores() -> tuple[str, ...]:
    return ("faiss", "chroma", "qdrant", "weaviate", "pinecone")


def get_vector_store(name: str, **kwargs: Any) -> BaseVectorStore:
    """Instantiate a vector store backend by name.

    - "faiss", "chroma", "qdrant": fully local, no external service required
      — real, tested backends (see docs/05_vectorstore_and_retrieval.md).
    - "weaviate": needs a reachable Weaviate instance (kwargs: url, api_key,
      dimensions).
    - "pinecone": needs a Pinecone account/API key and network access to
      Pinecone's API (kwargs: api_key, index_name, dimensions).
    """
    if name == "faiss":
        from src.retrieval.vectorstore.faiss_store import FaissVectorStore

        return FaissVectorStore(**kwargs)
    if name == "chroma":
        from src.retrieval.vectorstore.chroma_store import ChromaVectorStore

        return ChromaVectorStore(**kwargs)
    if name == "qdrant":
        from src.retrieval.vectorstore.qdrant_store import QdrantVectorStore

        return QdrantVectorStore(**kwargs)
    if name == "weaviate":
        from src.retrieval.vectorstore.weaviate_store import WeaviateVectorStore

        return WeaviateVectorStore(**kwargs)
    if name == "pinecone":
        from src.retrieval.vectorstore.pinecone_store import PineconeVectorStore

        return PineconeVectorStore(**kwargs)
    raise UnknownVectorStoreError(f"Unknown vector store '{name}'. Available: {available_stores()}")
