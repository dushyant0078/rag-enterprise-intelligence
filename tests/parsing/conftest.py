"""Fixture files for src/parsing tests. Each fixture writes a real, minimal
file of its format using the same libraries the parsers use, so tests
exercise real parsing rather than mocked data.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest
from docx import Document
from PIL import Image, ImageDraw
from pptx import Presentation


@pytest.fixture()
def txt_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.txt"
    p.write_text("Line one.\nLine two.\n")
    return p


@pytest.fixture()
def csv_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.csv"
    p.write_text("name,age\nAlice,30\nBob,25\n")
    return p


@pytest.fixture()
def html_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.html"
    p.write_text(
        "<html><body><h1>Title</h1><p>Some paragraph text.</p>"
        "<table><caption>Nums</caption><tr><th>A</th><th>B</th></tr>"
        "<tr><td>1</td><td>2</td></tr></table></body></html>"
    )
    return p


@pytest.fixture()
def markdown_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.md"
    p.write_text("# Title\n\nSome paragraph text.\n\n| A | B |\n|---|---|\n| 1 | 2 |\n")
    return p


@pytest.fixture()
def docx_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.docx"
    doc = Document()
    doc.add_heading("Title", level=1)
    doc.add_paragraph("First paragraph.")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "A"
    table.rows[0].cells[1].text = "B"
    row = table.add_row()
    row.cells[0].text = "1"
    row.cells[1].text = "2"
    doc.save(str(p))
    return p


@pytest.fixture()
def pptx_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.pptx"
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Slide Title"
    slide.placeholders[1].text_frame.text = "Body bullet text"
    slide.notes_slide.notes_text_frame.text = "Speaker note text"
    prs.save(str(p))
    return p


@pytest.fixture()
def text_pdf_file(tmp_path: Path) -> Path:
    """A native-text (non-scanned) two-page PDF."""
    p = tmp_path / "sample_text.pdf"
    pdf = pymupdf.open()
    for text in ("Page one native text.", "Page two native text."):
        page = pdf.new_page(width=595, height=842)
        page.insert_text((72, 72), text, fontsize=12)
    pdf.save(str(p))
    pdf.close()
    return p


@pytest.fixture()
def scanned_pdf_file(tmp_path: Path) -> Path:
    """A genuinely image-only PDF (no text layer) for OCR-path tests."""
    p = tmp_path / "sample_scanned.pdf"
    img = Image.new("RGB", (600, 300), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "SCANNED SAMPLE TEXT", fill="black")
    img_path = tmp_path / "sample_scanned.png"
    img.save(img_path)

    pdf = pymupdf.open()
    page = pdf.new_page(width=img.width, height=img.height)
    page.insert_image(pymupdf.Rect(0, 0, img.width, img.height), filename=str(img_path))
    pdf.save(str(p))
    pdf.close()
    return p
