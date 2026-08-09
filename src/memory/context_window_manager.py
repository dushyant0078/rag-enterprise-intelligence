"""Context window management: given a session's full turn history, decide
how many of the most recent turns fit verbatim within a token budget. Older
turns that don't fit are candidates for compression (see summarization.py) --
this module only does the budget accounting, not the compression itself.
"""

from __future__ import annotations

from src.chunking.text_utils import estimate_tokens
from src.memory.schemas import ConversationTurn

DEFAULT_MAX_HISTORY_TOKENS = 2000


class ContextWindowManager:
    def __init__(self, max_history_tokens: int = DEFAULT_MAX_HISTORY_TOKENS) -> None:
        self.max_history_tokens = max_history_tokens

    def split(
        self, turns: list[ConversationTurn]
    ) -> tuple[list[ConversationTurn], list[ConversationTurn]]:
        """Returns (included, excluded) -- included is the longest
        most-recent suffix of `turns` that fits the token budget; excluded
        is everything older than that, in original order.
        """
        if not turns:
            return [], []

        included_reversed: list[ConversationTurn] = []
        used_tokens = 0
        cutoff_index = len(turns)  # index into `turns` where "included" starts

        for i in range(len(turns) - 1, -1, -1):
            turn_tokens = estimate_tokens(turns[i].content)
            if included_reversed and used_tokens + turn_tokens > self.max_history_tokens:
                break
            included_reversed.append(turns[i])
            used_tokens += turn_tokens
            cutoff_index = i

        included = list(reversed(included_reversed))
        excluded = turns[:cutoff_index]
        return included, excluded
