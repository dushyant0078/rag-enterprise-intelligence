"""PPTX parser (python-pptx). Each slide maps to one PageContent — slides are
already the natural pagination unit for a deck, unlike DOCX."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Emu  # noqa: F401  (kept for future geometry-aware chunking use)

from src.parsing.base import BaseParser
from src.parsing.schemas import PageContent, ParsedDocument, TableContent


class PptxParser(BaseParser):
    supported_extensions = (".pptx",)

    def parse(self, path: str | Path) -> ParsedDocument:
        p = self._read_path(path)
        presentation = Presentation(str(p))

        pages: list[PageContent] = []
        for slide_number, slide in enumerate(presentation.slides, start=1):
            lines: list[str] = []
            tables: list[TableContent] = []

            for shape in slide.shapes:
                if shape.has_text_frame and shape.text_frame.text.strip():
                    lines.append(shape.text_frame.text.strip())
                if shape.has_table:
                    rows = [[cell.text.strip() for cell in row.cells] for row in shape.table.rows]
                    tables.append(TableContent(page_number=slide_number, rows=rows))

            if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
                notes = slide.notes_slide.notes_text_frame.text.strip()
                lines.append(f"[Speaker notes] {notes}")

            pages.append(
                PageContent(
                    page_number=slide_number,
                    text="\n".join(lines),
                    tables=tables,
                )
            )

        full_text = "\n\n".join(
            f"--- Slide {pg.page_number} ---\n{pg.text}" for pg in pages if pg.text
        )
        all_tables = [t for pg in pages for t in pg.tables]

        return ParsedDocument(
            source_path=str(p),
            format="pptx",
            pages=pages,
            full_text=full_text,
            tables=all_tables,
            page_count=len(pages),
        )
