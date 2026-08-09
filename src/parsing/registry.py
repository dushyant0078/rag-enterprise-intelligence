"""Maps a file extension to the parser that handles it. This is the single
place format support is registered — adding a new format means adding one
parser class and one line here.
"""

from __future__ import annotations

from pathlib import Path

from src.parsing.base import BaseParser
from src.parsing.csv_parser import CsvParser
from src.parsing.docx_parser import DocxParser
from src.parsing.html_parser import HtmlParser
from src.parsing.markdown_parser import MarkdownParser
from src.parsing.pdf_parser import PdfParser
from src.parsing.pptx_parser import PptxParser
from src.parsing.txt_parser import TxtParser


class UnsupportedFormatError(ValueError):
    """Raised when no registered parser handles a file's extension."""


_PARSER_CLASSES: tuple[type[BaseParser], ...] = (
    PdfParser,
    DocxParser,
    PptxParser,
    HtmlParser,
    MarkdownParser,
    CsvParser,
    TxtParser,
)

_EXTENSION_MAP: dict[str, type[BaseParser]] = {
    ext: cls for cls in _PARSER_CLASSES for ext in cls.supported_extensions
}


def supported_extensions() -> tuple[str, ...]:
    return tuple(sorted(_EXTENSION_MAP.keys()))


def get_parser_for(path: str | Path) -> BaseParser:
    """Return an instantiated parser for the given file, based on its extension."""
    ext = Path(path).suffix.lower()
    parser_cls = _EXTENSION_MAP.get(ext)
    if parser_cls is None:
        raise UnsupportedFormatError(
            f"No parser registered for extension '{ext}'. Supported: {supported_extensions()}"
        )
    return parser_cls()
