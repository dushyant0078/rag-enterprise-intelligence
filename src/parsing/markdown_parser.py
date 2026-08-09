"""Markdown parser. Renders to HTML (with the tables extension) and reuses the
HTML extraction logic so headers/tables are structurally recognized instead of
treating markdown as opaque text — the same structural cues that
markdown-aware chunking (Milestone 4) will key off.
"""

from __future__ import annotations

from pathlib import Path

import markdown as markdown_lib

from src.parsing.base import BaseParser
from src.parsing.html_parser import extract_text_and_tables
from src.parsing.schemas import PageContent, ParsedDocument


class MarkdownParser(BaseParser):
    supported_extensions = (".md", ".markdown")

    def parse(self, path: str | Path) -> ParsedDocument:
        p = self._read_path(path)
        raw_markdown = p.read_text(encoding="utf-8", errors="replace")

        html = markdown_lib.markdown(raw_markdown, extensions=["tables", "fenced_code"])
        text, tables = extract_text_and_tables(html)

        return ParsedDocument(
            source_path=str(p),
            format="markdown",
            pages=[PageContent(page_number=1, text=text, tables=tables)],
            full_text=text,
            tables=tables,
            page_count=1,
            raw_source=raw_markdown,
        )
