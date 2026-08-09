"""Production reranking backend: sentence-transformers' CrossEncoder, which
jointly encodes (query, candidate) pairs — the actual "cross-encoder"
technique, as distinct from comparing two independently-computed embeddings.
Covers both `bge_reranker` and `cross_encoder_ms_marco` from the spec behind
one class + a model catalog, same pattern as
src/embeddings/sentence_transformers_backend.py.

Like that module, instantiating this downloads model weights from the
Hugging Face Hub on first use — this sandbox's network allowlist doesn't
reach huggingface.co, so this class is correct and ready to run, but
untested *here*. src/reranking/lexical_overlap.py is what's actually
exercised end-to-end in this sandbox.
"""

from __future__ import annotations

from src.reranking.base import BaseReranker
from src.retrieval.schemas import RetrievedChunk

MODEL_CATALOG: dict[str, str] = {
    "bge_reranker": "BAAI/bge-reranker-large",
    "cross_encoder_ms_marco": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_id: str, *, device: str = "auto", batch_size: int = 32) -> None:
        self.reranker_name = model_id
        self.batch_size = batch_size
        self._model = self._load_model(model_id, device)

    @staticmethod
    def _load_model(model_id: str, device: str):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker "
                "(it's a core dependency; poetry install should have it)."
            ) from exc

        resolved_device = device
        if device == "auto":
            try:
                import torch

                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                resolved_device = "cpu"

        try:
            return CrossEncoder(model_id, device=resolved_device)
        except Exception as exc:  # noqa: BLE001 — surface a clear, actionable message
            raise RuntimeError(
                f"Failed to load reranker model '{model_id}'. This usually means the model "
                "weights couldn't be downloaded from the Hugging Face Hub (check network "
                f"access). Available short names in MODEL_CATALOG: {list(MODEL_CATALOG.keys())}."
            ) from exc

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int | None = None
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._model.predict(pairs, batch_size=self.batch_size)

        reranked = [
            self._annotate(chunk, float(score), rank)
            for rank, (chunk, score) in enumerate(zip(chunks, scores, strict=True))
        ]
        reranked.sort(key=lambda c: -c.score)
        return reranked[:top_k] if top_k is not None else reranked


def from_short_name(name: str, **kwargs) -> CrossEncoderReranker:
    """Convenience constructor: CrossEncoderReranker.from_short_name('bge_reranker')."""
    model_id = MODEL_CATALOG.get(name)
    if model_id is None:
        raise ValueError(f"Unknown reranker short name '{name}'. Available: {list(MODEL_CATALOG)}")
    return CrossEncoderReranker(model_id=model_id, **kwargs)
