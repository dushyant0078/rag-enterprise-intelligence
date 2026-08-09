"""Maps a strategy name to its chunker class — the single place chunking
strategies are registered, mirroring src/parsing/registry.py's pattern.
"""

from __future__ import annotations

from typing import Any

from src.chunking.base import BaseChunker
from src.chunking.fixed_size import FixedSizeChunker
from src.chunking.hybrid import HybridChunker
from src.chunking.markdown_aware import MarkdownAwareChunker
from src.chunking.recursive import RecursiveChunker
from src.chunking.semantic import SemanticChunker
from src.chunking.sentence_aware import SentenceAwareChunker
from src.chunking.table_aware import TableAwareChunker


class UnknownChunkingStrategyError(ValueError):
    pass


_STRATEGY_CLASSES: dict[str, type[BaseChunker]] = {
    "fixed_size": FixedSizeChunker,
    "recursive": RecursiveChunker,
    "sentence_aware": SentenceAwareChunker,
    "markdown_aware": MarkdownAwareChunker,
    "table_aware": TableAwareChunker,
    "semantic": SemanticChunker,  # requires embed_fn kwarg — see get_chunker docstring
    "hybrid": HybridChunker,
}

# Strategies that need no extra construction arguments — the set a comparison
# run can instantiate with defaults alone. "semantic" is deliberately excluded
# here (it needs an embed_fn) and included explicitly by callers that have one.
STRATEGIES_WITHOUT_DEPENDENCIES: tuple[str, ...] = tuple(
    name for name in _STRATEGY_CLASSES if name != "semantic"
)


def available_strategies() -> tuple[str, ...]:
    return tuple(_STRATEGY_CLASSES.keys())


def get_chunker(strategy: str, **kwargs: Any) -> BaseChunker:
    """Instantiate a chunker by strategy name.

    `semantic` requires `embed_fn=<callable>` in kwargs (see
    src/chunking/semantic.py) — every other strategy works with defaults or
    with any subset of its constructor's tuning parameters (chunk_size, etc.)
    passed through kwargs.
    """
    cls = _STRATEGY_CLASSES.get(strategy)
    if cls is None:
        raise UnknownChunkingStrategyError(
            f"Unknown chunking strategy '{strategy}'. Available: {available_strategies()}"
        )
    return cls(**kwargs)
