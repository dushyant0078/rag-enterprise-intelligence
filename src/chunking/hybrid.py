"""Hybrid chunking: the practical production default (see
docs/02_architecture.md and configs/retrieval/hybrid.yaml's sibling
philosophy — swap-in components combined rather than one strategy winning
outright). Tables are always pulled out as atomic chunks first; prose is
then chunked with MarkdownAwareChunker where structure is available (falls
back to RecursiveChunker automatically otherwise, via that chunker's own
fallback).
"""

from __future__ import annotations

from src.chunking.base import BaseChunker
from src.chunking.markdown_aware import MarkdownAwareChunker
from src.chunking.schemas import Chunk
from src.chunking.table_aware import serialize_table
from src.parsing.schemas import ParsedDocument

_TABLE_LINE_PREFIXES = ("|", "+-", "---")


def _is_mostly_markdown_table(text: str) -> bool:
    """True when a prose section is (almost) entirely raw markdown table
    syntax. Used to drop sections that would otherwise duplicate a table
    already captured cleanly and separately as an atomic table chunk.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    table_like = sum(1 for line in lines if line.startswith(_TABLE_LINE_PREFIXES))
    return (table_like / len(lines)) > 0.6


class HybridChunker(BaseChunker):
    strategy_name = "hybrid"

    def __init__(self, max_chunk_size: int = 800, chunk_overlap: int = 100) -> None:
        self._prose_chunker = MarkdownAwareChunker(
            max_chunk_size=max_chunk_size, chunk_overlap=chunk_overlap
        )

    def chunk(self, parsed_document: ParsedDocument, doc_id: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0

        for table in parsed_document.tables:
            serialized = serialize_table(table)
            if not serialized.strip():
                continue
            chunks.append(
                self._make_chunk(
                    doc_id=doc_id,
                    text=serialized,
                    index=index,
                    page_numbers=[table.page_number] if table.page_number else [],
                    is_table=True,
                )
            )
            index += 1

        for prose_chunk in self._prose_chunker.chunk(parsed_document, doc_id):
            if _is_mostly_markdown_table(prose_chunk.text):
                # Already captured cleanly above via serialize_table() — skip the
                # raw-markdown duplicate instead of indexing the same table twice.
                continue
            chunks.append(
                self._make_chunk(
                    doc_id=doc_id,
                    text=prose_chunk.text,
                    index=index,
                    is_table=False,
                    metadata=prose_chunk.metadata,
                )
            )
            index += 1

        return chunks
