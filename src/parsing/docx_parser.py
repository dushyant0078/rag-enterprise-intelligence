"""DOCX parser (python-docx).

DOCX has no fixed pagination in the file format itself (real page breaks are
a rendering-time concept), so "page_number" here is a pseudo-page counter
that increments on explicit page-break runs — good enough for citation
purposes ("around page N") while being honest that it's not authoritative
for a Word document the way it is for a PDF.
"""

from __future__ import annotations

from pathlib import Path

import docx
from docx.table import Table

from src.parsing.base import BaseParser
from src.parsing.schemas import PageContent, ParsedDocument, TableContent


def _table_to_rows(table: Table) -> list[list[str]]:
    return [[cell.text.strip() for cell in row.cells] for row in table.rows]


def _has_page_break(paragraph) -> bool:
    # A hard page break shows up as a <w:br w:type="page"/> run property in the XML.
    xml = paragraph._p.xml
    return 'w:type="page"' in xml


class DocxParser(BaseParser):
    supported_extensions = (".docx",)

    def parse(self, path: str | Path) -> ParsedDocument:
        p = self._read_path(path)
        document = docx.Document(str(p))

        pages: list[PageContent] = []
        current_lines: list[str] = []
        current_tables: list[TableContent] = []
        page_number = 1

        def flush_page() -> None:
            nonlocal current_lines, current_tables, page_number
            pages.append(
                PageContent(
                    page_number=page_number,
                    text="\n".join(current_lines).strip(),
                    tables=current_tables,
                )
            )
            current_lines = []
            current_tables = []
            page_number += 1

        # Walk the document body in order so paragraphs and tables interleave correctly.
        body = document.element.body
        table_index = 0
        tables_in_order = document.tables

        for child in body.iterchildren():
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "p":
                paragraph = next(pp for pp in document.paragraphs if pp._p is child)
                if paragraph.text.strip():
                    current_lines.append(paragraph.text.strip())
                if _has_page_break(paragraph):
                    flush_page()
            elif tag == "tbl":
                if table_index < len(tables_in_order):
                    rows = _table_to_rows(tables_in_order[table_index])
                    current_tables.append(TableContent(page_number=page_number, rows=rows))
                    table_index += 1

        if current_lines or current_tables or not pages:
            flush_page()

        full_text = "\n\n".join(pg.text for pg in pages if pg.text)
        all_tables = [t for pg in pages for t in pg.tables]

        return ParsedDocument(
            source_path=str(p),
            format="docx",
            pages=pages,
            full_text=full_text,
            tables=all_tables,
            page_count=len(pages),
        )
