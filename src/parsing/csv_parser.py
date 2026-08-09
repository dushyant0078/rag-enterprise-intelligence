"""CSV parser. Kept as structured table data (not flattened text) so
table-aware chunking (Milestone 4) has real rows/columns to work with."""

from __future__ import annotations

import csv
from pathlib import Path

from src.parsing.base import BaseParser
from src.parsing.schemas import PageContent, ParsedDocument, TableContent


class CsvParser(BaseParser):
    supported_extensions = (".csv",)

    def parse(self, path: str | Path) -> ParsedDocument:
        p = self._read_path(path)
        warnings: list[str] = []

        try:
            with p.open("r", newline="", encoding="utf-8") as f:
                sample = f.read(4096)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample)
                except csv.Error:
                    dialect = csv.excel
                    warnings.append("could not sniff CSV dialect; assumed standard comma-delimited")
                reader = csv.reader(f, dialect)
                rows = [row for row in reader]
        except UnicodeDecodeError:
            with p.open("r", newline="", encoding="latin-1") as f:
                reader = csv.reader(f)
                rows = [row for row in reader]
            warnings.append("decoded using fallback encoding 'latin-1'")

        table = TableContent(rows=rows)
        # Text form kept too, so this document is still searchable/embeddable like any other.
        full_text = "\n".join(", ".join(cell for cell in row) for row in rows)

        return ParsedDocument(
            source_path=str(p),
            format="csv",
            pages=[PageContent(page_number=1, text=full_text, tables=[table])],
            full_text=full_text,
            tables=[table],
            page_count=1,
            parser_warnings=warnings,
        )
