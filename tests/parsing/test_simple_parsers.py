from __future__ import annotations

from pathlib import Path

from src.parsing.csv_parser import CsvParser
from src.parsing.html_parser import HtmlParser
from src.parsing.markdown_parser import MarkdownParser
from src.parsing.txt_parser import TxtParser


def test_txt_parser(txt_file: Path) -> None:
    doc = TxtParser().parse(txt_file)
    assert doc.format == "txt"
    assert doc.page_count == 1
    assert "Line one." in doc.full_text
    assert "Line two." in doc.full_text
    assert not doc.is_empty


def test_csv_parser_preserves_rows(csv_file: Path) -> None:
    doc = CsvParser().parse(csv_file)
    assert doc.format == "csv"
    assert len(doc.tables) == 1
    rows = doc.tables[0].rows
    assert rows[0] == ["name", "age"]
    assert rows[1] == ["Alice", "30"]
    assert rows[2] == ["Bob", "25"]
    assert "Alice" in doc.full_text


def test_html_parser_extracts_text_and_table(html_file: Path) -> None:
    doc = HtmlParser().parse(html_file)
    assert doc.format == "html"
    assert "Title" in doc.full_text
    assert "Some paragraph text." in doc.full_text
    assert len(doc.tables) == 1
    assert doc.tables[0].caption == "Nums"
    assert doc.tables[0].rows == [["A", "B"], ["1", "2"]]
    # table content is extracted structurally, not duplicated into the plain-text stream
    assert doc.full_text == "Title\nSome paragraph text."


def test_markdown_parser_extracts_headers_and_table(markdown_file: Path) -> None:
    doc = MarkdownParser().parse(markdown_file)
    assert doc.format == "markdown"
    assert "Title" in doc.full_text
    assert "Some paragraph text." in doc.full_text
    assert len(doc.tables) == 1
    assert doc.tables[0].rows == [["A", "B"], ["1", "2"]]
