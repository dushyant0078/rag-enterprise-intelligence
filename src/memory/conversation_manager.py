"""Conversation Manager (docs/02_architecture.md's first pipeline stage
after Auth): the session-state coordinator every conversational turn goes
through. Ties SessionMemoryStore (per-session history), ContextWindowManager
(what fits in-budget), ConversationSummarizer (compresses what doesn't), and
QueryRewriter (resolves context-dependent follow-ups) into one per-turn call.
"""

from __future__ import annotations

from src.llm.base import BaseLLM
from src.memory.context_window_manager import ContextWindowManager
from src.memory.query_rewriter import QueryRewriter
from src.memory.schemas import ConversationSession
from src.memory.session_memory import SessionMemoryStore
from src.memory.summarization import ConversationSummarizer


class ConversationManager:
    def __init__(
        self,
        session_store: SessionMemoryStore | None = None,
        context_window_manager: ContextWindowManager | None = None,
        summarizer: ConversationSummarizer | None = None,
        query_rewriter: QueryRewriter | None = None,
        llm: BaseLLM | None = None,
    ) -> None:
        """`llm`, if given, is used for both summarization and query
        rewriting (unless a caller passes its own `summarizer`/
        `query_rewriter` instances) -- one place to wire in a real model for
        both, rather than configuring each separately for the common case.
        """
        self.session_store = session_store or SessionMemoryStore()
        self.context_window_manager = context_window_manager or ContextWindowManager()
        self.summarizer = summarizer or ConversationSummarizer(llm=llm)
        self.query_rewriter = query_rewriter or QueryRewriter(llm=llm)

    def start_session(self, user_id: str, session_id: str | None = None) -> ConversationSession:
        return self.session_store.create_session(user_id, session_id=session_id)

    def prepare_turn(
        self, session_id: str, user_query: str, requesting_user_id: str | None = None
    ) -> str:
        """Called at the start of handling a new user message: records it,
        compresses any history that's fallen out of the context-window
        budget into the session's rolling summary, and returns the
        query rewritten to be self-contained (ready for retrieval).
        """
        session = self.session_store.add_turn(
            session_id, "user", user_query, requesting_user_id=requesting_user_id
        )

        included, excluded = self.context_window_manager.split(
            session.turns[:-1]
        )  # history before this turn
        if excluded:
            session.summary = self.summarizer.summarize(excluded, previous_summary=session.summary)

        return self.query_rewriter.rewrite(user_query, included)

    def record_response(
        self, session_id: str, response_text: str, requesting_user_id: str | None = None
    ) -> None:
        self.session_store.add_turn(
            session_id, "assistant", response_text, requesting_user_id=requesting_user_id
        )
