from __future__ import annotations

from pathlib import Path

from src.parsing.docx_parser import DocxParser
from src.parsing.pptx_parser import PptxParser


def test_docx_parser_extracts_text_and_table(docx_file: Path) -> None:
    doc = DocxParser().parse(docx_file)
    assert doc.format == "docx"
    assert doc.page_count == 1
    assert "Title" in doc.full_text
    assert "First paragraph." in doc.full_text
    assert len(doc.tables) == 1
    assert doc.tables[0].rows == [["A", "B"], ["1", "2"]]


def test_pptx_parser_extracts_slide_text_and_notes(pptx_file: Path) -> None:
    doc = PptxParser().parse(pptx_file)
    assert doc.format == "pptx"
    assert doc.page_count == 1
    assert doc.pages[0].page_number == 1
    assert "Slide Title" in doc.full_text
    assert "Body bullet text" in doc.full_text
    assert "Speaker note text" in doc.full_text
