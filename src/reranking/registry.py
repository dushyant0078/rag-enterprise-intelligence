"""Reranking strategy registry."""

from __future__ import annotations

from typing import Any

from src.reranking.base import BaseReranker


class UnknownRerankerError(ValueError):
    pass


def available_rerankers() -> tuple[str, ...]:
    return ("bge_reranker", "cross_encoder_ms_marco", "cohere_rerank", "lexical_overlap")


def get_reranker(name: str, **kwargs: Any) -> BaseReranker:
    """Instantiate a reranker by name.

    - "bge_reranker", "cross_encoder_ms_marco": CrossEncoderReranker — needs
      network access to the Hugging Face Hub on first use.
    - "cohere_rerank": CohereReranker(api_key=...) — needs a Cohere API key
      and network access to Cohere's API.
    - "lexical_overlap": LexicalOverlapReranker — fully offline, no
      dependencies beyond this repo (see src/reranking/lexical_overlap.py).
    """
    if name == "lexical_overlap":
        from src.reranking.lexical_overlap import LexicalOverlapReranker

        return LexicalOverlapReranker()
    if name in ("bge_reranker", "cross_encoder_ms_marco"):
        from src.reranking.cross_encoder_reranker import from_short_name

        return from_short_name(name, **kwargs)
    if name == "cohere_rerank":
        from src.reranking.cohere_reranker import CohereReranker

        return CohereReranker(**kwargs)
    raise UnknownRerankerError(f"Unknown reranker '{name}'. Available: {available_rerankers()}")
