"""Base embedding model interface. Every backend in src/embeddings/
implements this, so the rest of the pipeline (chunking's SemanticChunker,
the EmbeddingPipeline, and eventually retrieval) works identically regardless
of which model is configured (configs/embedding/*.yaml).
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class EmbeddingModel(ABC):
    model_name: str

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Vector dimensionality this model produces. May only be known
        after fit() for corpus-fitted backends (e.g. TF-IDF) — such backends
        should raise a clear error if queried before fitting.
        """
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of texts. Returns an array of shape (len(texts), dimensions)."""
        raise NotImplementedError

    async def embed_async(self, texts: Sequence[str]) -> np.ndarray:
        """Default async wrapper: runs the sync embed() in a worker thread so
        callers can `await` regardless of whether a given backend has a truly
        async-native client underneath. Backends with a real async HTTP
        client (a future hosted-embedding-API backend) should override this.
        """
        return await asyncio.to_thread(self.embed, texts)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization, shared by every backend that opts into it,
    so cosine similarity reduces to a plain dot product downstream in
    retrieval (Milestone 5).
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return vectors / norms
