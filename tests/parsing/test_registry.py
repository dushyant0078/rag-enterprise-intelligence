from __future__ import annotations

from pathlib import Path

import pytest

from src.parsing.pdf_parser import PdfParser
from src.parsing.registry import UnsupportedFormatError, get_parser_for, supported_extensions
from src.parsing.txt_parser import TxtParser


def test_supported_extensions_includes_all_spec_formats() -> None:
    exts = supported_extensions()
    for ext in (".pdf", ".docx", ".pptx", ".html", ".md", ".csv", ".txt"):
        assert ext in exts


def test_get_parser_for_selects_correct_class(txt_file: Path, text_pdf_file: Path) -> None:
    assert isinstance(get_parser_for(txt_file), TxtParser)
    assert isinstance(get_parser_for(text_pdf_file), PdfParser)


def test_get_parser_for_unsupported_extension_raises(tmp_path: Path) -> None:
    f = tmp_path / "file.xyz"
    f.write_text("data")
    with pytest.raises(UnsupportedFormatError):
        get_parser_for(f)
