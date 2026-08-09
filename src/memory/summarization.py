"""Conversation summarization: compress turns pushed out of the active
context window (see context_window_manager.py) into a rolling summary,
rather than dropping them outright -- "memory compression" from the spec.

Same optional-LLM pattern as src/retrieval's heuristic retrievers: pass an
`llm` for real LLM-generated summaries, or omit it for a deterministic
extractive fallback (first sentence of each turn) that needs no model call.
"""

from __future__ import annotations

from src.chunking.text_utils import split_sentences
from src.llm.base import BaseLLM
from src.memory.schemas import ConversationTurn

_SUMMARIZE_SYSTEM_PROMPT = (
    "You summarize conversation history concisely, preserving specific facts, decisions, and "
    "named entities the user mentioned. Do not add information not present in the conversation."
)


class ConversationSummarizer:
    def __init__(self, llm: BaseLLM | None = None) -> None:
        self.llm = llm

    def summarize(
        self, turns: list[ConversationTurn], *, previous_summary: str | None = None
    ) -> str:
        if not turns:
            return previous_summary or ""

        if self.llm is not None:
            return self._llm_summarize(turns, previous_summary)
        return self._heuristic_summarize(turns, previous_summary)

    def _llm_summarize(self, turns: list[ConversationTurn], previous_summary: str | None) -> str:
        assert self.llm is not None  # guarded by the caller in summarize()
        transcript = "\n".join(f"{t.role}: {t.content}" for t in turns)
        prior = f"Prior summary: {previous_summary}\n\n" if previous_summary else ""
        messages = [
            {"role": "system", "content": _SUMMARIZE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{prior}Conversation to summarize:\n{transcript}\n\nSummary:",
            },
        ]
        response = self.llm.generate(messages, max_tokens=300)
        return response.content.strip()

    @staticmethod
    def _heuristic_summarize(turns: list[ConversationTurn], previous_summary: str | None) -> str:
        """Extractive fallback: the first sentence of each turn, prefixed by
        role. Not as coherent as an LLM summary, but a real, deterministic
        compression -- shorter than the original, and needs no model call.
        """
        lines = []
        if previous_summary:
            lines.append(previous_summary)
        for turn in turns:
            sentences = split_sentences(turn.content)
            first_sentence = sentences[0] if sentences else turn.content
            lines.append(f"{turn.role}: {first_sentence}")
        return " ".join(lines)
