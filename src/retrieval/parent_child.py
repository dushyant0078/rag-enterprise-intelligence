"""Parent-child retrieval: index small "child" chunks (precise enough for
good similarity matching) but return their larger "parent" chunk as context
(enough surrounding text for the LLM to actually answer from). Solves the
tension between "small chunks retrieve more precisely" and "small chunks
don't carry enough context to generate a good answer" without compromising
on either.

`build_parent_child_chunks` constructs the pairing from any two chunkers —
each parent is chunked once at a large size, then each parent's own text is
independently re-chunked at a small size to produce its children, so every
child belongs to exactly one parent by construction (no fuzzy offset
matching needed).
"""

from __future__ import annotations

from typing import Any

from src.chunking.base import BaseChunker
from src.chunking.recursive import RecursiveChunker
from src.chunking.schemas import Chunk
from src.parsing.schemas import ParsedDocument
from src.retrieval.base import BaseRetriever
from src.retrieval.schemas import RetrievedChunk

_PARENT_CHILD_KEY = "parent_chunk_id"


def build_parent_child_chunks(
    parsed_document: ParsedDocument,
    doc_id: str,
    parent_chunker: BaseChunker | None = None,
    child_chunker: BaseChunker | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """Returns (parent_chunks, child_chunks). Only child_chunks should be
    embedded/indexed for search; parent_chunks are looked up by id at
    retrieval time (see ParentChildRetriever's `parent_lookup`).
    """
    parent_chunker = parent_chunker or RecursiveChunker(chunk_size=2000, chunk_overlap=200)
    child_chunker = child_chunker or RecursiveChunker(chunk_size=400, chunk_overlap=50)

    parents = parent_chunker.chunk(parsed_document, doc_id)
    children: list[Chunk] = []
    for parent in parents:
        sub_doc = parsed_document.model_copy(update={"full_text": parent.text})
        for child in child_chunker.chunk(sub_doc, doc_id):
            child.metadata[_PARENT_CHILD_KEY] = parent.chunk_id
            children.append(child)

    return parents, children


class ParentChildRetriever(BaseRetriever):
    retriever_name = "parent_child"

    def __init__(self, child_retriever: BaseRetriever, parent_lookup: dict[str, Chunk]) -> None:
        self.child_retriever = child_retriever
        self.parent_lookup = parent_lookup

    def retrieve(
        self, query: str, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        # Over-fetch children: several children commonly share one parent, and only the
        # first (highest-scoring) hit per parent is kept.
        child_results = self.child_retriever.retrieve(query, top_k=top_k * 5, filters=filters)

        seen_parents: dict[str, RetrievedChunk] = {}
        for child in child_results:
            parent_chunk_id = child.metadata.get(_PARENT_CHILD_KEY)
            if not parent_chunk_id or parent_chunk_id in seen_parents:
                continue
            parent = self.parent_lookup.get(parent_chunk_id)
            if parent is None:
                continue
            seen_parents[parent_chunk_id] = RetrievedChunk(
                chunk_id=parent.chunk_id,
                doc_id=parent.doc_id,
                text=parent.text,
                score=child.score,
                domain=child.domain,
                page_numbers=parent.page_numbers,
                is_table=parent.is_table,
                metadata={"matched_child_chunk_id": child.chunk_id},
                source_retriever=self.retriever_name,
            )
            if len(seen_parents) >= top_k:
                break

        return list(seen_parents.values())
