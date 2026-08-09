"""Markdown-aware chunking: split on header boundaries first, so a chunk
never straddles two unrelated sections, then sub-chunk any section still
over the size cap with RecursiveChunker. Each chunk carries its heading path
(e.g. "AML Policy > Enhanced Due Diligence Thresholds") in metadata, which is
exactly the structural cue a reader uses to know what a passage is *about*
before reading it — and prose chunking alone throws it away.

Requires `parsed_document.raw_source` (the original markdown text) since
header markers don't survive HTML/plain-text extraction. Falls back to
RecursiveChunker, with a warning recorded in the resulting chunks' metadata,
when raw_source isn't available (i.e. any non-markdown document).
"""

from __future__ import annotations

import re

from src.chunking.base import BaseChunker
from src.chunking.recursive import RecursiveChunker
from src.chunking.schemas import Chunk
from src.parsing.schemas import ParsedDocument

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


class _Section:
    __slots__ = ("heading_path", "body")

    def __init__(self, heading_path: str, body: str) -> None:
        self.heading_path = heading_path
        self.body = body


def _split_into_sections(raw_markdown: str) -> list[_Section]:
    matches = list(_HEADER_RE.finditer(raw_markdown))
    if not matches:
        return [_Section(heading_path="", body=raw_markdown)]

    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []  # (level, title) ancestry stack

    preamble = raw_markdown[: matches[0].start()].strip()
    if preamble:
        sections.append(_Section(heading_path="", body=preamble))

    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_markdown)
        body = raw_markdown[body_start:body_end].strip()

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        heading_path = " > ".join(t for _, t in stack)

        if body:
            sections.append(_Section(heading_path=heading_path, body=body))

    return sections


class MarkdownAwareChunker(BaseChunker):
    strategy_name = "markdown_aware"

    def __init__(self, max_chunk_size: int = 800, chunk_overlap: int = 100) -> None:
        self.max_chunk_size = max_chunk_size
        # Clamp so a small max_chunk_size (e.g. in tests, or a deliberately tight config)
        # can never collide with RecursiveChunker's chunk_overlap < chunk_size requirement.
        safe_overlap = min(chunk_overlap, max(0, max_chunk_size // 5))
        self._sub_chunker = RecursiveChunker(chunk_size=max_chunk_size, chunk_overlap=safe_overlap)

    def chunk(self, parsed_document: ParsedDocument, doc_id: str) -> list[Chunk]:
        if not parsed_document.raw_source:
            fallback = self._sub_chunker.chunk(parsed_document, doc_id)
            for c in fallback:
                c.strategy = self.strategy_name
                c.metadata["fallback_reason"] = "no raw_source available; used recursive chunking"
            return fallback

        sections = _split_into_sections(parsed_document.raw_source)
        chunks: list[Chunk] = []
        index = 0
        for section in sections:
            if not section.body.strip():
                continue
            if len(section.body) <= self.max_chunk_size:
                chunks.append(
                    self._make_chunk(
                        doc_id=doc_id,
                        text=section.body,
                        index=index,
                        metadata={"heading_path": section.heading_path},
                    )
                )
                index += 1
            else:
                # Section still too big -- sub-chunk with the recursive strategy, keeping
                # the same heading path on each piece.
                sub_doc = parsed_document.model_copy(update={"full_text": section.body})
                for sub_chunk in self._sub_chunker.chunk(sub_doc, doc_id):
                    chunks.append(
                        self._make_chunk(
                            doc_id=doc_id,
                            text=sub_chunk.text,
                            index=index,
                            metadata={"heading_path": section.heading_path},
                        )
                    )
                    index += 1

        return chunks
