"""PDF parser (PyMuPDF).

Per page: try native text extraction first (cheap, exact). If a page has no
meaningful extractable text — the signature of a scanned/image-only page —
fall back to rasterizing the page and running OCR (see parsing/ocr.py). Table
extraction uses PyMuPDF's built-in table finder. One bad page logs a warning
and contributes empty content rather than failing the whole document.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from src.parsing.base import BaseParser
from src.parsing.ocr import OcrUnavailableError, ocr_image
from src.parsing.schemas import PageContent, ParsedDocument, TableContent
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# A page with fewer than this many non-whitespace characters of native text
# is treated as scanned/image-only and routed to OCR.
_MIN_NATIVE_TEXT_CHARS = 20
_OCR_RASTER_DPI = 300


class PdfParser(BaseParser):
    supported_extensions = (".pdf",)

    def __init__(self, *, ocr_language: str = "eng", ocr_raster_dpi: int = _OCR_RASTER_DPI) -> None:
        self.ocr_language = ocr_language
        self.ocr_raster_dpi = ocr_raster_dpi

    def parse(self, path: str | Path) -> ParsedDocument:
        p = self._read_path(path)
        warnings: list[str] = []
        pages: list[PageContent] = []
        any_ocr = False

        try:
            doc = pymupdf.open(str(p))
        except Exception as exc:  # noqa: BLE001 — malformed PDF must not crash a batch run
            warnings.append(f"failed to open PDF: {exc}")
            return ParsedDocument(
                source_path=str(p),
                format="pdf",
                pages=[],
                full_text="",
                page_count=0,
                parser_warnings=warnings,
            )

        try:
            for page_index in range(doc.page_count):
                page_number = page_index + 1
                try:
                    page_content = self._parse_page(doc, page_index, page_number)
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"page {page_number}: parse error — {exc}")
                    page_content = PageContent(page_number=page_number, text="")
                pages.append(page_content)
                any_ocr = any_ocr or page_content.used_ocr
        finally:
            doc.close()

        full_text = "\n\n".join(pg.text for pg in pages if pg.text)
        all_tables = [t for pg in pages for t in pg.tables]

        return ParsedDocument(
            source_path=str(p),
            format="pdf",
            pages=pages,
            full_text=full_text,
            tables=all_tables,
            page_count=len(pages),
            used_ocr=any_ocr,
            parser_warnings=warnings,
        )

    def _parse_page(self, doc: pymupdf.Document, page_index: int, page_number: int) -> PageContent:
        page = doc[page_index]

        native_text = page.get_text("text").strip()
        tables = self._extract_tables(page, page_number)

        if len(native_text) >= _MIN_NATIVE_TEXT_CHARS:
            return PageContent(
                page_number=page_number, text=native_text, tables=tables, used_ocr=False
            )

        # Likely a scanned page — rasterize and OCR it.
        try:
            pixmap = page.get_pixmap(dpi=self.ocr_raster_dpi)
            image = pixmap.pil_image() if hasattr(pixmap, "pil_image") else _pixmap_to_pil(pixmap)
            ocr_text, confidence = ocr_image(image, lang=self.ocr_language)
            return PageContent(
                page_number=page_number,
                text=ocr_text.strip(),
                tables=tables,
                used_ocr=True,
                ocr_confidence=confidence,
            )
        except OcrUnavailableError:
            logger.warning(
                "OCR unavailable; returning empty text for scanned page",
                extra={"page": page_number},
            )
            return PageContent(
                page_number=page_number, text=native_text, tables=tables, used_ocr=False
            )

    @staticmethod
    def _extract_tables(page: pymupdf.Page, page_number: int) -> list[TableContent]:
        tables: list[TableContent] = []
        try:
            finder = page.find_tables()
            for table in finder.tables:
                rows = table.extract()
                cleaned = [[(cell or "").strip() for cell in row] for row in rows]
                if cleaned:
                    tables.append(TableContent(page_number=page_number, rows=cleaned))
        except Exception as exc:  # noqa: BLE001 — table-finding is best-effort
            logger.debug("table extraction skipped", extra={"page": page_number, "error": str(exc)})
        return tables


def _pixmap_to_pil(pixmap: pymupdf.Pixmap):
    """Fallback conversion for older PyMuPDF versions without Pixmap.pil_image()."""
    from PIL import Image

    mode = "RGB" if pixmap.n < 4 else "RGBA"
    return Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples)
