"""Plain-text (.txt) parser."""

from __future__ import annotations

from pathlib import Path

from src.parsing.base import BaseParser
from src.parsing.schemas import PageContent, ParsedDocument

_ENCODINGS_TO_TRY = ("utf-8", "utf-8-sig", "latin-1")


class TxtParser(BaseParser):
    supported_extensions = (".txt",)

    def parse(self, path: str | Path) -> ParsedDocument:
        p = self._read_path(path)
        warnings: list[str] = []

        text = None
        for encoding in _ENCODINGS_TO_TRY:
            try:
                text = p.read_text(encoding=encoding)
                if encoding != "utf-8":
                    warnings.append(f"decoded using fallback encoding '{encoding}'")
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            text = p.read_text(encoding="utf-8", errors="replace")
            warnings.append(
                "could not decode cleanly with any known encoding; used utf-8 with replacement"
            )

        return ParsedDocument(
            source_path=str(p),
            format="txt",
            pages=[PageContent(page_number=1, text=text)],
            full_text=text,
            page_count=1,
            parser_warnings=warnings,
        )
