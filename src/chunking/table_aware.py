"""Table-aware chunking: every table becomes its own atomic chunk (serialized
as a markdown pipe table, which reads cleanly to both a human and an LLM),
tagged is_table=True. Surrounding prose is chunked separately with
RecursiveChunker. This is what stops a table's rows/columns from being
diced into meaningless fragments by a size-based splitter that has no idea
it's looking at tabular data.

Known limitation (see docs/04_chunking_and_embeddings.md): for native-text
PDFs, table cell text is typically *also* present in the page's plain-text
stream (PyMuPDF's text extraction doesn't exclude table regions), so a table
may appear twice — once cleanly here, once jumbled into a prose chunk. A
clean fix needs bounding-box-aware text extraction that masks out detected
table regions before prose extraction; deferred rather than done partially.
"""

from __future__ import annotations

from src.chunking.base import BaseChunker
from src.chunking.recursive import RecursiveChunker
from src.chunking.schemas import Chunk
from src.parsing.schemas import ParsedDocument, TableContent


def serialize_table(table: TableContent) -> str:
    """Render a TableContent as a markdown pipe table. First row is treated
    as the header — a reasonable default for the tables this pipeline
    encounters (policy thresholds, deductible schedules, eligibility grids).
    """
    if not table.rows:
        return ""

    lines: list[str] = []
    if table.caption:
        lines.append(f"Table: {table.caption}")

    header, *body = table.rows
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in body:
        # Pad/truncate ragged rows to the header width so the markdown table stays well-formed.
        cells = (row + [""] * len(header))[: len(header)]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


class TableAwareChunker(BaseChunker):
    strategy_name = "table_aware"

    def __init__(self, prose_chunk_size: int = 800, prose_chunk_overlap: int = 100) -> None:
        self._prose_chunker = RecursiveChunker(
            chunk_size=prose_chunk_size, chunk_overlap=prose_chunk_overlap
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
            chunks.append(
                self._make_chunk(doc_id=doc_id, text=prose_chunk.text, index=index, is_table=False)
            )
            index += 1

        return chunks
