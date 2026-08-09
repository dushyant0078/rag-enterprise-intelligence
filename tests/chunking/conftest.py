from __future__ import annotations

import numpy as np
import pytest

from src.parsing.schemas import PageContent, ParsedDocument, TableContent


@pytest.fixture()
def prose_document() -> ParsedDocument:
    paragraphs = [
        "The Remote Work Policy governs hybrid arrangements for all employees. "
        "It applies from an employee's first day, with no minimum tenure requirement.",
        "Individual contributors may work remotely up to five days per week with "
        "manager approval. Team leads may work remotely up to four days per week.",
        "Employees returning from parental leave retain their prior remote-days "
        "eligibility for ninety days before standard eligibility rules resume.",
    ]
    text = "\n\n".join(paragraphs)
    return ParsedDocument(
        source_path="fixture.txt",
        format="txt",
        pages=[PageContent(page_number=1, text=text)],
        full_text=text,
        page_count=1,
    )


@pytest.fixture()
def table_document() -> ParsedDocument:
    table = TableContent(
        page_number=1,
        caption="Remote Days by Level",
        rows=[
            ["Job Level", "Remote Days"],
            ["Individual Contributor", "5"],
            ["Team Lead", "4"],
        ],
    )
    text = "Eligibility Overview\n\nSee the table below for remote-day limits by level."
    return ParsedDocument(
        source_path="fixture.html",
        format="html",
        pages=[PageContent(page_number=1, text=text, tables=[table])],
        full_text=text,
        tables=[table],
        page_count=1,
    )


@pytest.fixture()
def markdown_document() -> ParsedDocument:
    raw_markdown = (
        "# AML Policy\n\n"
        "**Owner:** Compliance\n\n"
        "## Enhanced Due Diligence\n\n"
        "Wire transfers over $10,000 require enhanced due diligence review.\n\n"
        "## Monitoring\n\n"
        "Flagged accounts are reviewed quarterly by the compliance team.\n"
    )
    # full_text/pages intentionally omit markdown syntax, as the real MarkdownParser would.
    plain_text = (
        "AML Policy\nOwner: Compliance\nEnhanced Due Diligence\n"
        "Wire transfers over $10,000 require enhanced due diligence review.\n"
        "Monitoring\nFlagged accounts are reviewed quarterly by the compliance team."
    )
    return ParsedDocument(
        source_path="fixture.md",
        format="markdown",
        pages=[PageContent(page_number=1, text=plain_text)],
        full_text=plain_text,
        page_count=1,
        raw_source=raw_markdown,
    )


@pytest.fixture()
def empty_document() -> ParsedDocument:
    return ParsedDocument(
        source_path="empty.txt", format="txt", pages=[], full_text="", page_count=0
    )


@pytest.fixture()
def fake_embed_fn():
    """Deterministic embed_fn for SemanticChunker tests: sentences are mapped
    to one of two well-separated clusters based on whether they contain a
    'topic marker' word, so a real topic-shift boundary is genuinely
    detectable without needing a real trained embedding model.
    """

    def _embed(texts):
        vectors = []
        for text in texts:
            cluster = 1.0 if "weather" in text.lower() else 0.0
            base = np.array([cluster, 1 - cluster, 0.1])
            vectors.append(base)
        return np.array(vectors)

    return _embed
