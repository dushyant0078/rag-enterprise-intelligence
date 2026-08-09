"""Maps a configured embedding name (configs/embedding/*.yaml) to a concrete
EmbeddingModel instance — the single place backends are registered, mirroring
src/parsing/registry.py and src/chunking/registry.py.
"""

from __future__ import annotations

from typing import Any

from src.embeddings.base import EmbeddingModel
from src.embeddings.sentence_transformers_backend import MODEL_CATALOG
from src.embeddings.tfidf_backend import TfidfEmbedding


class UnknownEmbeddingModelError(ValueError):
    pass


def available_models() -> tuple[str, ...]:
    return (*MODEL_CATALOG.keys(), "tfidf_offline")


def get_embedding_model(name: str, **kwargs: Any) -> EmbeddingModel:
    """Instantiate an embedding backend by short name.

    - Any name in sentence_transformers_backend.MODEL_CATALOG (bge_large,
      e5_large, minilm, instructor_xl, gte_large) returns a
      SentenceTransformerEmbedding — requires network access to the Hugging
      Face Hub on first use.
    - "tfidf_offline" returns an unfitted TfidfEmbedding — call .fit(corpus)
      before use. No network access required at any point.
    """
    if name == "tfidf_offline":
        return TfidfEmbedding(**kwargs)
    if name in MODEL_CATALOG:
        from src.embeddings.sentence_transformers_backend import from_short_name

        return from_short_name(name, **kwargs)
    raise UnknownEmbeddingModelError(
        f"Unknown embedding model '{name}'. Available: {available_models()}"
    )
