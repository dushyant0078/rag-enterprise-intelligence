"""Fixed-size chunking: a plain sliding character window.

This is the deliberately naive baseline every other strategy is compared
against — it will cut mid-word and mid-sentence, because that's exactly the
failure mode the smarter strategies exist to fix (see
docs/04_chunking_and_embeddings.md for the comparison).
"""

from __future__ import annotations

from src.chunking.base import BaseChunker
from src.chunking.schemas import Chunk
from src.parsing.schemas import ParsedDocument


class FixedSizeChunker(BaseChunker):
    strategy_name = "fixed_size"

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, parsed_document: ParsedDocument, doc_id: str) -> list[Chunk]:
        text = parsed_document.full_text.strip()
        if not text:
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks: list[Chunk] = []
        index = 0
        start = 0
        while start < len(text):
            piece = text[start : start + self.chunk_size]
            chunks.append(self._make_chunk(doc_id=doc_id, text=piece, index=index))
            index += 1
            start += step

        return chunks
