"""Query rewriter (docs/02_architecture.md's pipeline stage between
Conversation Manager and Embedding Service): resolves a follow-up query that
depends on conversation context ("what about the second one?") into a
self-contained query the retriever can act on without seeing the history.

Same optional-LLM pattern as summarization.py: pass an `llm` for real
LLM-based rewriting, or omit it for a deterministic heuristic fallback that
detects a pronoun-led follow-up and prepends the previous turn's content
words -- real and useful, not a claim of resolving arbitrary coreference.
"""

from __future__ import annotations

import re

from src.llm.base import BaseLLM
from src.memory.schemas import ConversationTurn
from src.utils.text import content_words

_REWRITE_SYSTEM_PROMPT = (
    "You rewrite a follow-up question into a standalone question that doesn't require the "
    "prior conversation to understand, using the conversation history to resolve pronouns and "
    "implicit references. Preserve the user's intent exactly. Output only the rewritten "
    "question, nothing else."
)

_LEADING_PRONOUN_RE = re.compile(
    r"^\s*(?:what about|and|but|so|also)?\s*\b(it|that|this|they|those|these)\b", re.IGNORECASE
)


class QueryRewriter:
    def __init__(self, llm: BaseLLM | None = None) -> None:
        self.llm = llm

    def rewrite(self, query: str, history: list[ConversationTurn]) -> str:
        if not history:
            return query

        if self.llm is not None:
            return self._llm_rewrite(query, history)
        return self._heuristic_rewrite(query, history)

    def _llm_rewrite(self, query: str, history: list[ConversationTurn]) -> str:
        assert self.llm is not None  # guarded by the caller in rewrite()
        transcript = "\n".join(f"{t.role}: {t.content}" for t in history)
        messages = [
            {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Conversation history:\n{transcript}\n\nFollow-up question: {query}\n\n"
                "Standalone question:",
            },
        ]
        response = self.llm.generate(messages, max_tokens=150)
        return response.content.strip() or query

    @staticmethod
    def _heuristic_rewrite(query: str, history: list[ConversationTurn]) -> str:
        """Only acts when the query visibly leans on context (starts with a
        pronoun / continuation word) -- otherwise returns the query
        unchanged rather than guessing. Not general coreference resolution.
        """
        if not _LEADING_PRONOUN_RE.match(query):
            return query

        previous_user_turns = [t for t in reversed(history) if t.role == "user"]
        if not previous_user_turns:
            return query

        previous_topic_words = content_words(previous_user_turns[0].content)
        if not previous_topic_words:
            return query

        topic_phrase = " ".join(sorted(previous_topic_words))
        return f"Regarding {topic_phrase}: {query}"
