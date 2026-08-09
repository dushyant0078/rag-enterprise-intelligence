"""Shared lightweight text helpers used by BM25 tokenization, multi-query
expansion, contextual compression, and lexical reranking. Consolidated here
after the same stopword list and tokenization pattern showed up
independently in three places (src/retrieval/multi_query.py,
contextual_compression.py, sparse_bm25.py) — one definition, reused, rather
than four copies quietly drifting apart.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"\w+")

STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "what", "which", "who",
        "how", "does", "do", "did", "can", "could", "should", "would", "of",
        "in", "on", "for", "to", "and", "or", "with", "at", "by", "from",
    }
)  # fmt: skip


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens, punctuation stripped. Order-preserving."""
    return _TOKEN_RE.findall(text.lower())


def content_words(text: str) -> set[str]:
    """Tokenized words with stopwords removed, as a set (for overlap checks)."""
    return {w for w in tokenize(text) if w not in STOPWORDS}
