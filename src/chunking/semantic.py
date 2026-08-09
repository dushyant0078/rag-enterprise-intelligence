"""Semantic chunking: embed each sentence, walk through consecutive
sentences, and start a new chunk wherever the topic visibly shifts (cosine
similarity between neighbors drops below a threshold) or the size cap would
be exceeded. This is the one strategy whose boundaries come from meaning
rather than syntax or formatting.

Takes a bare `embed_fn: Callable[[Sequence[str]], np.ndarray]` rather than a
concrete embedding class, so src/chunking (an earlier pipeline stage) doesn't
import src/embeddings (a later one) — callers pass `some_embedding_model.embed`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from src.chunking.base import BaseChunker
from src.chunking.schemas import Chunk
from src.chunking.text_utils import split_sentences
from src.parsing.schemas import ParsedDocument

EmbedFn = Callable[[Sequence[str]], np.ndarray]


def _cosine_similarities(vectors: np.ndarray) -> np.ndarray:
    """Cosine similarity between each consecutive pair of rows in `vectors`.
    Returns an array of length len(vectors) - 1.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    normalized = vectors / norms
    return np.sum(normalized[:-1] * normalized[1:], axis=1)


class SemanticChunker(BaseChunker):
    strategy_name = "semantic"

    def __init__(
        self,
        embed_fn: EmbedFn,
        similarity_threshold: float = 0.5,
        max_chunk_size: int = 1500,
        min_sentences_per_chunk: int = 1,
    ) -> None:
        self.embed_fn = embed_fn
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_sentences_per_chunk = min_sentences_per_chunk

    def chunk(self, parsed_document: ParsedDocument, doc_id: str) -> list[Chunk]:
        sentences = split_sentences(parsed_document.full_text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [self._make_chunk(doc_id=doc_id, text=sentences[0], index=0)]

        vectors = np.asarray(self.embed_fn(sentences))
        similarities = _cosine_similarities(vectors)

        chunks: list[Chunk] = []
        current: list[str] = [sentences[0]]
        current_len = len(sentences[0])
        index = 0

        for i, similarity in enumerate(similarities, start=1):
            sentence = sentences[i]
            topic_shift = similarity < self.similarity_threshold
            too_big = current_len + len(sentence) + 1 > self.max_chunk_size
            enough_sentences = len(current) >= self.min_sentences_per_chunk

            if enough_sentences and (topic_shift or too_big):
                chunks.append(
                    self._make_chunk(
                        doc_id=doc_id,
                        text=" ".join(current),
                        index=index,
                        metadata={"break_reason": "topic_shift" if topic_shift else "max_size"},
                    )
                )
                index += 1
                current = [sentence]
                current_len = len(sentence)
            else:
                current.append(sentence)
                current_len += len(sentence) + 1

        if current:
            chunks.append(self._make_chunk(doc_id=doc_id, text=" ".join(current), index=index))

        return chunks
