from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.parsing.pdf_parser import PdfParser

_TESSERACT_AVAILABLE = shutil.which("tesseract") is not None


def test_pdf_parser_native_text_two_pages(text_pdf_file: Path) -> None:
    doc = PdfParser().parse(text_pdf_file)
    assert doc.format == "pdf"
    assert doc.page_count == 2
    assert doc.used_ocr is False
    assert "Page one native text." in doc.pages[0].text
    assert "Page two native text." in doc.pages[1].text


def test_pdf_parser_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PdfParser().parse(tmp_path / "does_not_exist.pdf")


def test_pdf_parser_malformed_file_does_not_raise(tmp_path: Path) -> None:
    bad_pdf = tmp_path / "not_really_a_pdf.pdf"
    bad_pdf.write_bytes(b"this is not a valid PDF")
    doc = PdfParser().parse(bad_pdf)
    assert doc.page_count == 0
    assert doc.parser_warnings  # failure is recorded, not raised


@pytest.mark.skipif(not _TESSERACT_AVAILABLE, reason="tesseract-ocr binary not installed")
def test_pdf_parser_ocr_fallback_on_scanned_page(scanned_pdf_file: Path) -> None:
    doc = PdfParser().parse(scanned_pdf_file)
    assert doc.used_ocr is True
    assert doc.pages[0].used_ocr is True
    assert doc.pages[0].ocr_confidence is not None
    # OCR of rendered text isn't pixel-perfect, but the key words should survive.
    assert "SCANNED" in doc.full_text.upper()
    assert "SAMPLE" in doc.full_text.upper()
