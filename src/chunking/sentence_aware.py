"""Sentence-aware chunking: pack whole sentences up to a size cap, never
splitting a sentence across two chunks. One sentence of overlap carries
forward for continuity across a chunk boundary.
"""

from __future__ import annotations

from src.chunking.base import BaseChunker
from src.chunking.schemas import Chunk
from src.chunking.text_utils import split_sentences
from src.parsing.schemas import ParsedDocument


class SentenceAwareChunker(BaseChunker):
    strategy_name = "sentence_aware"

    def __init__(self, max_chunk_size: int = 800, sentence_overlap: int = 1) -> None:
        self.max_chunk_size = max_chunk_size
        self.sentence_overlap = sentence_overlap

    def chunk(self, parsed_document: ParsedDocument, doc_id: str) -> list[Chunk]:
        sentences = split_sentences(parsed_document.full_text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        current: list[str] = []
        current_len = 0
        index = 0

        for sentence in sentences:
            projected_len = current_len + len(sentence) + (1 if current else 0)
            if current and projected_len > self.max_chunk_size:
                chunks.append(self._make_chunk(doc_id=doc_id, text=" ".join(current), index=index))
                index += 1
                carry = current[-self.sentence_overlap :] if self.sentence_overlap else []
                current = list(carry)
                current_len = sum(len(s) for s in current) + max(0, len(current) - 1)

            current.append(sentence)
            current_len += len(sentence) + (1 if len(current) > 1 else 0)

        if current:
            chunks.append(self._make_chunk(doc_id=doc_id, text=" ".join(current), index=index))

        return chunks
