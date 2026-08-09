"""Offline embedding backend: TF-IDF (scikit-learn), optionally dimensionality
-reduced with Truncated SVD to a fixed dense size.

This is not a drop-in quality replacement for the transformer models in
sentence_transformers_backend.py — it's a genuinely different, purely lexical
technique. It earns its place in the architecture for two real reasons, not
just as a sandbox workaround:

1. Some regulated enterprise deployments (see docs/01_business_understanding.md
   security constraints) prohibit calling out to an external model hub
   entirely. This backend needs no network access at any point — it fits
   locally on the ingested corpus — so it's a legitimate "air-gapped" config
   option (configs/embedding/tfidf_offline.yaml), not a toy.
2. It's what actually runs in this development sandbox, whose network
   allowlist doesn't reach huggingface.co — so it's what backs the real,
   verified chunking-strategy comparison in docs/04_chunking_and_embeddings.md,
   rather than that comparison being unverifiable here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.embeddings.base import EmbeddingModel, l2_normalize


class NotFittedError(RuntimeError):
    """Raised when embed() or dimensions is called before fit()."""


class TfidfEmbedding(EmbeddingModel):
    def __init__(
        self,
        *,
        max_features: int = 4096,
        svd_dimensions: int | None = 256,
        normalize: bool = True,
        model_name: str = "tfidf_offline",
    ) -> None:
        self.model_name = model_name
        self.normalize = normalize
        self._svd_dimensions = svd_dimensions
        self._vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self._svd = (
            TruncatedSVD(n_components=svd_dimensions, random_state=42) if svd_dimensions else None
        )
        self._fitted = False

    def fit(self, corpus: Sequence[str]) -> TfidfEmbedding:
        """Fit the vocabulary (and SVD projection, if configured) on a
        corpus. Must be called once before embed(). Fitting is corpus-wide by
        design — TF-IDF's IDF term is only meaningful relative to a document
        collection, unlike a pretrained transformer's fixed weights.
        """
        matrix = self._vectorizer.fit_transform(list(corpus))
        if self._svd is not None:
            # TruncatedSVD needs n_components < n_features; clamp for tiny corpora/vocab.
            effective_k = min(self._svd_dimensions, matrix.shape[1] - 1, matrix.shape[0] - 1)
            if effective_k < 1:
                self._svd = (
                    None  # corpus too small to reduce meaningfully — fall back to raw TF-IDF
                )
            else:
                self._svd.n_components = effective_k
                self._svd.fit(matrix)
        self._fitted = True
        return self

    @property
    def dimensions(self) -> int:
        if not self._fitted:
            raise NotFittedError("Call fit(corpus) before reading dimensions.")
        if self._svd is not None:
            return self._svd.n_components
        return len(self._vectorizer.vocabulary_)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise NotFittedError("Call fit(corpus) before embed().")
        if not texts:
            return np.empty((0, self.dimensions))

        matrix = self._vectorizer.transform(list(texts))
        vectors = self._svd.transform(matrix) if self._svd is not None else matrix.toarray()
        vectors = np.asarray(vectors, dtype=float)
        return l2_normalize(vectors) if self.normalize else vectors
