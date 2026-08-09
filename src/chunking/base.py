"""Base chunker interface. Every strategy in src/chunking/ implements this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.chunking.schemas import Chunk
from src.chunking.text_utils import estimate_tokens
from src.parsing.schemas import ParsedDocument
from src.utils.hashing import sha256_text


class BaseChunker(ABC):
    strategy_name: str = "base"

    @abstractmethod
    def chunk(self, parsed_document: ParsedDocument, doc_id: str) -> list[Chunk]:
        """Split a parsed document into retrievable Chunks. Implementations
        should never raise on an empty document — return an empty list.
        """
        raise NotImplementedError

    def _make_chunk(
        self,
        *,
        doc_id: str,
        text: str,
        index: int,
        page_numbers: list[int] | None = None,
        is_table: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Chunk:
        """Shared constructor so every strategy produces consistently-shaped
        Chunks (deterministic ID, hash, size accounting) without repeating
        the bookkeeping in each subclass.
        """
        return Chunk(
            chunk_id=f"{doc_id}__{self.strategy_name}__{index:04d}",
            doc_id=doc_id,
            strategy=self.strategy_name,
            chunk_index=index,
            text=text,
            content_sha256=sha256_text(text),
            char_count=len(text),
            token_count_estimate=estimate_tokens(text),
            page_numbers=page_numbers or [],
            is_table=is_table,
            metadata=metadata or {},
        )
