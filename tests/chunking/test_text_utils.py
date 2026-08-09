from __future__ import annotations

from src.chunking.text_utils import estimate_tokens, split_sentences


def test_split_sentences_basic() -> None:
    text = "This is one sentence. This is another! Is this a third?"
    sentences = split_sentences(text)
    assert sentences == ["This is one sentence.", "This is another!", "Is this a third?"]


def test_split_sentences_empty() -> None:
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_split_sentences_handles_abbreviations() -> None:
    text = "Dr. Smith reviewed the policy. It was approved."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert sentences[0].startswith("Dr. Smith")


def test_estimate_tokens_roughly_scales_with_length() -> None:
    short = estimate_tokens("hello")
    long = estimate_tokens("hello " * 100)
    assert short >= 1
    assert long > short * 10
