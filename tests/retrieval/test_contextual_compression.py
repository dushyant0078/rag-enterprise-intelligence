from __future__ import annotations

from src.retrieval.contextual_compression import ContextualCompressionRetriever, default_compressor
from tests.retrieval.conftest import FakeRetriever, make_chunk


def test_default_compressor_keeps_only_relevant_sentences() -> None:
    text = (
        "Wire transfers over ten thousand dollars require enhanced due diligence. "
        "The office is closed on public holidays. "
        "Enhanced due diligence review happens within five business days."
    )
    compressed = default_compressor("enhanced due diligence wire transfer", text)
    assert "office is closed" not in compressed
    assert "enhanced due diligence" in compressed.lower()


def test_default_compressor_never_returns_empty_when_no_sentence_matches() -> None:
    text = "This sentence is about something else entirely."
    compressed = default_compressor("completely unrelated topic zzz", text)
    assert compressed == text  # falls back to the original rather than compressing to nothing


def test_default_compressor_empty_query_returns_original() -> None:
    text = "Some chunk text here."
    assert default_compressor("the a an", text) == text  # query is all stopwords


def test_contextual_compression_retriever_compresses_each_result() -> None:
    base = FakeRetriever(
        [
            make_chunk(
                "a",
                0.9,
                text=(
                    "Enhanced due diligence applies to large wire transfers. "
                    "Unrelated sentence about the office kitchen."
                ),
            )
        ]
    )
    retriever = ContextualCompressionRetriever(base)
    results = retriever.retrieve("enhanced due diligence wire transfers", top_k=1)

    assert "kitchen" not in results[0].text
    assert "diligence" in results[0].text.lower()
    assert results[0].source_retriever == "contextual_compression"


def test_contextual_compression_uses_custom_compressor() -> None:
    base = FakeRetriever([make_chunk("a", 0.9, text="original text")])
    retriever = ContextualCompressionRetriever(base, compression_fn=lambda q, t: "COMPRESSED")
    results = retriever.retrieve("query", top_k=1)
    assert results[0].text == "COMPRESSED"
