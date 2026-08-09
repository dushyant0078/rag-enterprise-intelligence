"""HTML parser (BeautifulSoup). Also exposes ``extract_text_and_tables``,
reused by the Markdown parser after it renders markdown -> HTML.
"""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from src.parsing.base import BaseParser
from src.parsing.schemas import PageContent, ParsedDocument, TableContent

_NOISE_TAGS = ("script", "style", "nav", "footer", "noscript")


def extract_text_and_tables(html: str) -> tuple[str, list[TableContent]]:
    """Shared HTML -> (readable text, structured tables) extraction.

    Tables are pulled out and returned separately (not inlined verbatim into
    the text) so downstream chunking can treat them as tables rather than as
    prose with lots of short lines.
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    tables: list[TableContent] = []
    for table_tag in soup.find_all("table"):
        rows: list[list[str]] = []
        for tr in table_tag.find_all("tr"):
            cells = [cell.get_text(strip=True) for cell in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if rows:
            caption_tag = table_tag.find("caption")
            tables.append(
                TableContent(
                    rows=rows, caption=caption_tag.get_text(strip=True) if caption_tag else None
                )
            )
        table_tag.decompose()  # remove so it isn't duplicated in the plain-text extraction below

    text = soup.get_text(separator="\n")
    # Collapse the run of blank lines BeautifulSoup tends to leave behind.
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    return text, tables


class HtmlParser(BaseParser):
    supported_extensions = (".html", ".htm")

    def parse(self, path: str | Path) -> ParsedDocument:
        p = self._read_path(path)
        raw_html = p.read_text(encoding="utf-8", errors="replace")

        text, tables = extract_text_and_tables(raw_html)

        return ParsedDocument(
            source_path=str(p),
            format="html",
            pages=[PageContent(page_number=1, text=text, tables=tables)],
            full_text=text,
            tables=tables,
            page_count=1,
        )
