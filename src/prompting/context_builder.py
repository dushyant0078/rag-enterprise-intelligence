"""Context Builder (docs/02_architecture.md's pipeline stage between
Reranker and LLM Inference Engine): assembles a list of RetrievedChunk into
one formatted context string, tagged with citations, within a token budget.

Default budget (6000) matches `context_builder.max_context_tokens` in
configs/config.yaml (Milestone 2) — this is that config value made real.
"""

from __future__ import annotations

from src.chunking.text_utils import estimate_tokens
from src.retrieval.schemas import RetrievedChunk

DEFAULT_MAX_CONTEXT_TOKENS = 6000


def _citation_tag(chunk: RetrievedChunk) -> str:
    page_part = f", page {chunk.page_numbers[0]}" if chunk.page_numbers else ""
    return f"[Source: {chunk.doc_id}{page_part}]"


class ContextBuilder:
    def __init__(self, max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS) -> None:
        self.max_context_tokens = max_context_tokens

    def build(self, chunks: list[RetrievedChunk]) -> str:
        """Assemble `chunks` (expected pre-sorted by relevance — the
        retriever/reranker's job, not this one's) into one context string.
        Stops adding chunks once the token budget would be exceeded, but
        always includes at least the first chunk even if it alone exceeds
        the budget — an empty context is a worse failure mode than a
        slightly over-budget one.
        """
        if not chunks:
            return ""

        blocks: list[str] = []
        used_tokens = 0
        for chunk in chunks:
            block = f"{_citation_tag(chunk)}\n{chunk.text}"
            block_tokens = estimate_tokens(block)
            if blocks and used_tokens + block_tokens > self.max_context_tokens:
                break
            blocks.append(block)
            used_tokens += block_tokens

        return "\n\n---\n\n".join(blocks)

    def included_chunk_count(self, chunks: list[RetrievedChunk]) -> int:
        """How many of `chunks` build() would actually include — useful for
        callers/tests that want to know without re-parsing the joined string.
        """
        if not chunks:
            return 0
        count = 0
        used_tokens = 0
        for chunk in chunks:
            block_tokens = estimate_tokens(f"{_citation_tag(chunk)}\n{chunk.text}")
            if count and used_tokens + block_tokens > self.max_context_tokens:
                break
            count += 1
            used_tokens += block_tokens
        return count
