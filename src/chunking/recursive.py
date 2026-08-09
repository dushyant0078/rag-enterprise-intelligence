"""Recursive chunking: try the largest-granularity separator first (paragraph
breaks), and only fall back to a finer one (sentence, then word) for pieces
that are still too big — so most chunks land on a natural boundary, unlike
fixed-size chunking, while still guaranteeing every chunk fits the size cap.
"""

from __future__ import annotations

from src.chunking.base import BaseChunker
from src.chunking.schemas import Chunk
from src.parsing.schemas import ParsedDocument

_DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ")


class RecursiveChunker(BaseChunker):
    strategy_name = "recursive"

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        separators: tuple[str, ...] = _DEFAULT_SEPARATORS,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def chunk(self, parsed_document: ParsedDocument, doc_id: str) -> list[Chunk]:
        text = parsed_document.full_text.strip()
        if not text:
            return []

        pieces = self._split(text, list(self.separators))
        merged = self._merge_with_overlap(pieces)
        return [
            self._make_chunk(doc_id=doc_id, text=piece, index=i) for i, piece in enumerate(merged)
        ]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split `text` until every piece fits chunk_size, trying
        separators from coarsest to finest.
        """
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if not separators:
            # No separator left to try — hard character-split as a last resort.
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        sep, rest = separators[0], separators[1:]
        raw_parts = text.split(sep) if sep else list(text)
        result: list[str] = []
        for part in raw_parts:
            if not part.strip():
                continue
            if len(part) > self.chunk_size:
                result.extend(self._split(part, rest))
            else:
                result.append(part)
        return result

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """Pack small pieces back together up to chunk_size, carrying a
        tail-overlap from the previous chunk into the next for context
        continuity across a boundary.
        """
        if not pieces:
            return []

        merged: list[str] = []
        current = ""
        for piece in pieces:
            candidate = f"{current} {piece}".strip() if current else piece
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                    overlap_tail = current[-self.chunk_overlap :] if self.chunk_overlap else ""
                    current = f"{overlap_tail} {piece}".strip() if overlap_tail else piece
                else:
                    current = piece
        if current:
            merged.append(current)
        return merged
