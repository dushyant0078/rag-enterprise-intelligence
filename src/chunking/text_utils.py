"""Lightweight text utilities shared across chunkers.

No heavy NLP dependency (spaCy/NLTK) is pulled in deliberately — both need a
model/data download on first use, which this sandbox's network allowlist
can't reach (mirrors the OCR-language and embedding-model constraints noted
elsewhere). The regex-based sentence splitter below is a reasonable
approximation, not a full sentence-boundary-detection model; it's documented
as such rather than presented as more rigorous than it is.
"""

from __future__ import annotations

import re

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")

# Common abbreviations/initials that a naive ". " split would incorrectly treat as a sentence
# end. Not exhaustive — a best-effort heuristic, not a substitute for a trained sentence
# segmenter.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "inc", "ltd",
    "co", "no", "fig", "sec", "vol", "eg", "ie", "approx", "u.s", "u.k",
}  # fmt: skip


def _ends_with_abbreviation(sentence: str) -> bool:
    words = sentence.rstrip(".!?").split()
    if not words:
        return False
    last_word = words[-1].lower().rstrip(".")
    # Also catches single-letter initials, e.g. "J. K. Rowling".
    return last_word in _ABBREVIATIONS or (len(last_word) == 1 and words[-1][-1].isalpha())


def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Deliberately conservative: when in doubt it
    merges rather than incorrectly splits, since an over-long "sentence" is a
    much smaller downstream problem for chunking than a mid-thought split.
    """
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []

    pieces = _SENTENCE_BOUNDARY_RE.split(normalized)
    sentences: list[str] = []
    for piece in pieces:
        if sentences and _ends_with_abbreviation(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return [s.strip() for s in sentences if s.strip()]


def estimate_tokens(text: str) -> int:
    """Rough token-count estimate (~4 chars/token for English) used for
    chunk-size accounting before a specific LLM tokenizer is wired in
    (Milestone 6-7). Not exact — treat as an order-of-magnitude figure.
    """
    return max(1, len(text) // 4)
