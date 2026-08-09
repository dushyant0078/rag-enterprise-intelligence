"""Production embedding backend: sentence-transformers, covering every model
named in the spec (BGE Large, E5 Large, all-MiniLM-L6-v2, Instructor XL, GTE
Large) behind one class, selected by model_id via configs/embedding/*.yaml.

Instantiating this class downloads model weights from the Hugging Face Hub on
first use — this development sandbox's network allowlist doesn't reach
huggingface.co, so this class is correct and ready to run, but untested
*here*; src/embeddings/tfidf_backend.py is what's actually exercised in this
sandbox (see docs/04_chunking_and_embeddings.md for the full explanation —
the same constraint that applied to SEC/ArXiv downloads in Milestone 3 and
OCR-language packs applies to HF model downloads here).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.embeddings.base import EmbeddingModel, l2_normalize
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Catalog of the models named in the spec — short name -> HF model id.
# configs/embedding/*.yaml select one of these by `model_id`.
MODEL_CATALOG: dict[str, str] = {
    "bge_large": "BAAI/bge-large-en-v1.5",
    "e5_large": "intfloat/e5-large-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "instructor_xl": "hkunlp/instructor-xl",
    "gte_large": "thenlper/gte-large",
}


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class SentenceTransformerEmbedding(EmbeddingModel):
    def __init__(
        self,
        model_id: str,
        *,
        device: str = "auto",
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_id
        self.batch_size = batch_size
        self.normalize = normalize
        self._resolved_device = _resolve_device(device)
        self._model = self._load_model(model_id, self._resolved_device)
        self._dimensions = self._model.get_sentence_embedding_dimension()

    @staticmethod
    def _load_model(model_id: str, device: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedding "
                "(poetry install --with ...; it's a core dependency)."
            ) from exc

        try:
            return SentenceTransformer(model_id, device=device)
        except Exception as exc:  # noqa: BLE001 — surface a clear, actionable message
            raise RuntimeError(
                f"Failed to load embedding model '{model_id}'. This usually means either the "
                "model weights couldn't be downloaded from the Hugging Face Hub (check network "
                "access / HF_HOME cache) or the model id is wrong. Available short names in "
                f"MODEL_CATALOG: {list(MODEL_CATALOG.keys())}."
            ) from exc

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimensions))
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return l2_normalize(vectors) if self.normalize else vectors


def from_short_name(name: str, **kwargs) -> SentenceTransformerEmbedding:
    """Convenience constructor: SentenceTransformerEmbedding.from_short_name('bge_large')."""
    model_id = MODEL_CATALOG.get(name)
    if model_id is None:
        raise ValueError(f"Unknown model short name '{name}'. Available: {list(MODEL_CATALOG)}")
    return SentenceTransformerEmbedding(model_id=model_id, **kwargs)
