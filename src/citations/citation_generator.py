"""Citation Generator (docs/02_architecture.md's pipeline stage between LLM
Inference and Response Validator): parses `[Source: doc_id, page N]` tags
out of an LLM's answer text (the format every prompt template in
src/prompting instructs the model to use) and resolves each one against the
chunks actually provided as context.

A tag that doesn't match any provided chunk is a fabricated citation --
exactly the failure mode "prevent unsupported claims" is about, and exactly
what ResponseValidator checks for.
"""

from __future__ import annotations

import re

from src.citations.schemas import Citation
from src.retrieval.schemas import RetrievedChunk

_CITATION_TAG_RE = re.compile(r"\[Source:\s*([^,\]]+?)(?:,\s*page\s*(\d+))?\]")
_EXCERPT_LENGTH = 200


class CitationGenerator:
    def generate(self, answer_text: str, chunks: list[RetrievedChunk]) -> list[Citation]:
        citations = []
        for match in _CITATION_TAG_RE.finditer(answer_text):
            doc_id = match.group(1).strip()
            page_number = int(match.group(2)) if match.group(2) else None
            citations.append(self._resolve(doc_id, page_number, chunks))
        return citations

    @staticmethod
    def _resolve(doc_id: str, page_number: int | None, chunks: list[RetrievedChunk]) -> Citation:
        candidates = [c for c in chunks if c.doc_id == doc_id]
        if not candidates:
            return Citation(doc_id=doc_id, page_number=page_number)  # chunk_id=None -> fabricated

        # Prefer an exact page match; fall back to any chunk from the right document if the
        # page wasn't specified or didn't match one of the provided chunks exactly.
        exact_page_matches = (
            [c for c in candidates if page_number in c.page_numbers]
            if page_number is not None
            else []
        )
        matched = exact_page_matches[0] if exact_page_matches else candidates[0]

        return Citation(
            doc_id=doc_id,
            page_number=page_number,
            chunk_id=matched.chunk_id,
            confidence=matched.score,
            excerpt=matched.text[:_EXCERPT_LENGTH],
        )
