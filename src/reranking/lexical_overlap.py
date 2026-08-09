"""Offline lexical reranker: a real, deterministic (query, candidate)
scoring function, combining query-term recall, an exact-phrase-match bonus,
and a mild length penalty.

This is not a claim of parity with a trained cross-encoder — a cross-encoder
learns query-document relevance from millions of labeled pairs via
cross-attention; this is closed-form lexical scoring. What it *does*
demonstrate for real is the mechanical value reranking adds beyond a
single-pass retriever's score: a chunk can rank well on embedding/BM25
similarity while still lacking the query's actual phrase, and this reranker
corrects exactly that case (see the test suite for a concrete,
non-fabricated example). It's what's actually exercised end-to-end in this
sandbox, the same role src/embeddings/tfidf_backend.py and
src/retrieval/multi_query.py's default heuristic play elsewhere in this
project — real and useful today, with CrossEncoderReranker/CohereReranker as
the documented production upgrade path.
"""

from __future__ import annotations

import math

from src.reranking.base import BaseReranker
from src.retrieval.schemas import RetrievedChunk
from src.utils.text import content_words, tokenize

_PHRASE_BONUS_WEIGHT = 0.5
_LENGTH_PENALTY_WEIGHT = 0.05
_LENGTH_NORMALIZER = 50  # chunk word count at which the length penalty starts to bite


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return (
        {tuple(words[i : i + n]) for i in range(len(words) - n + 1)} if len(words) >= n else set()
    )


def _lexical_score(query: str, text: str) -> float:
    query_words = content_words(query)
    if not query_words:
        return 0.0

    chunk_words = content_words(text)
    term_recall = len(query_words & chunk_words) / len(query_words)

    query_tokens = tokenize(query)
    chunk_tokens = tokenize(text)
    # A genuine 3-word phrase match is much stronger evidence of relevance than an incidental
    # 2-word overlap, so these are weighted rather than collapsed into one binary bonus.
    if _ngrams(query_tokens, 3) & _ngrams(chunk_tokens, 3):
        phrase_bonus = 1.0
    elif _ngrams(query_tokens, 2) & _ngrams(chunk_tokens, 2):
        phrase_bonus = 0.5
    else:
        phrase_bonus = 0.0

    length_penalty = _LENGTH_PENALTY_WEIGHT * math.log(1 + len(chunk_tokens) / _LENGTH_NORMALIZER)

    return term_recall + _PHRASE_BONUS_WEIGHT * phrase_bonus - length_penalty


class LexicalOverlapReranker(BaseReranker):
    reranker_name = "lexical_overlap"

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int | None = None
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        scored = [
            self._annotate(chunk, _lexical_score(query, chunk.text), rank)
            for rank, chunk in enumerate(chunks)
        ]
        scored.sort(key=lambda c: -c.score)
        return scored[:top_k] if top_k is not None else scored
