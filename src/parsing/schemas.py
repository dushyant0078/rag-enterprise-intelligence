"""Data model produced by every parser in src/parsing/.

One shared shape means everything downstream (chunking, metadata extraction,
ingestion bookkeeping) works identically regardless of source format.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TableContent(BaseModel):
    """A table extracted from a document, kept as rows of cell strings so
    table-aware chunking (Milestone 4) can decide how to serialize it later
    rather than losing row/column structure at parse time.
    """

    page_number: int | None = None
    rows: list[list[str]]
    caption: str | None = None


class PageContent(BaseModel):
    """One page/slide/section of a document."""

    page_number: int
    text: str
    tables: list[TableContent] = Field(default_factory=list)
    used_ocr: bool = False
    ocr_confidence: float | None = None  # mean Tesseract word confidence, 0-100, when used_ocr


class ParsedDocument(BaseModel):
    """Full output of parsing one source file."""

    source_path: str
    format: str  # pdf | docx | pptx | html | markdown | txt | csv
    pages: list[PageContent]
    full_text: str
    tables: list[TableContent] = Field(
        default_factory=list
    )  # documents without page structure (csv, txt)
    page_count: int
    used_ocr: bool = False
    parser_warnings: list[str] = Field(default_factory=list)
    raw_source: str | None = None  # original source text, preserved where structure (e.g.
    # markdown headers) would otherwise be lost by full_text extraction — see
    # src/chunking/markdown_aware.py, which is the only current consumer.

    @property
    def is_empty(self) -> bool:
        return not self.full_text.strip()
