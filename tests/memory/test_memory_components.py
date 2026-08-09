from __future__ import annotations

from pathlib import Path

from src.memory.context_window_manager import ContextWindowManager
from src.memory.long_term_memory import LongTermMemoryStore
from src.memory.query_rewriter import QueryRewriter
from src.memory.schemas import ConversationTurn
from src.memory.summarization import ConversationSummarizer

# -- ContextWindowManager --------------------------------------------------


def test_context_window_splits_by_recency() -> None:
    turns = [ConversationTurn(role="user", content="word " * 100) for _ in range(5)]
    included, excluded = ContextWindowManager(max_history_tokens=150).split(turns)
    assert included == turns[-len(included) :]
    assert excluded == turns[: len(turns) - len(included)]
    assert len(included) + len(excluded) == 5


def test_context_window_empty_turns() -> None:
    assert ContextWindowManager().split([]) == ([], [])


def test_context_window_always_includes_most_recent_even_if_over_budget() -> None:
    huge = [ConversationTurn(role="user", content="word " * 10000)]
    included, excluded = ContextWindowManager(max_history_tokens=10).split(huge)
    assert included == huge
    assert excluded == []


# -- ConversationSummarizer -------------------------------------------------


def test_heuristic_summarize_is_shorter_than_original() -> None:
    turns = [
        ConversationTurn(role="user", content="This is a long question about policy details. " * 5),
        ConversationTurn(
            role="assistant", content="This is a long detailed answer explaining it. " * 5
        ),
    ]
    summary = ConversationSummarizer().summarize(turns)
    original_length = sum(len(t.content) for t in turns)
    assert len(summary) < original_length
    assert "user:" in summary and "assistant:" in summary


def test_heuristic_summarize_empty_turns_returns_previous_summary() -> None:
    result = ConversationSummarizer().summarize([], previous_summary="prior summary")
    assert result == "prior summary"


def test_heuristic_summarize_carries_forward_previous_summary() -> None:
    turns = [ConversationTurn(role="user", content="New question.")]
    summary = ConversationSummarizer().summarize(turns, previous_summary="Earlier context.")
    assert "Earlier context." in summary
    assert "New question." in summary


def test_llm_summarize_uses_llm_when_provided() -> None:
    class FakeLLM:
        model_name = "fake"

        def generate(self, messages, **kwargs):
            from src.llm.schemas import LLMResponse

            return LLMResponse(content="LLM-generated summary.", model="fake")

    turns = [ConversationTurn(role="user", content="Some question.")]
    summary = ConversationSummarizer(llm=FakeLLM()).summarize(turns)
    assert summary == "LLM-generated summary."


# -- LongTermMemoryStore -----------------------------------------------------


def test_long_term_memory_append_and_retrieve(tmp_path: Path) -> None:
    store = LongTermMemoryStore(persist_dir=tmp_path)
    store.append_summary("alice", "session1", "Discussed parental leave policy.")
    store.append_summary("alice", "session2", "Discussed remote work policy.")
    recent = store.get_recent_summaries("alice")
    assert len(recent) == 2
    assert recent[0].session_id == "session1"


def test_long_term_memory_respects_limit(tmp_path: Path) -> None:
    store = LongTermMemoryStore(persist_dir=tmp_path)
    for i in range(5):
        store.append_summary("alice", f"session{i}", f"summary {i}")
    recent = store.get_recent_summaries("alice", limit=2)
    assert len(recent) == 2
    assert recent[-1].session_id == "session4"


def test_long_term_memory_isolated_per_user(tmp_path: Path) -> None:
    store = LongTermMemoryStore(persist_dir=tmp_path)
    store.append_summary("alice", "s1", "alice's summary")
    assert store.get_recent_summaries("bob") == []


def test_long_term_memory_persists_across_instances(tmp_path: Path) -> None:
    LongTermMemoryStore(persist_dir=tmp_path).append_summary("alice", "s1", "summary text")
    reloaded = LongTermMemoryStore(persist_dir=tmp_path)
    assert len(reloaded.get_recent_summaries("alice")) == 1


# -- QueryRewriter ------------------------------------------------------------


def test_heuristic_rewrite_no_history_returns_unchanged() -> None:
    assert QueryRewriter().rewrite("what about it?", []) == "what about it?"


def test_heuristic_rewrite_no_pronoun_returns_unchanged() -> None:
    history = [ConversationTurn(role="user", content="What is the notice period?")]
    query = "What is the parental leave policy?"
    assert QueryRewriter().rewrite(query, history) == query


def test_heuristic_rewrite_pronoun_prepends_previous_topic() -> None:
    history = [
        ConversationTurn(role="user", content="What is the wire transfer threshold?"),
        ConversationTurn(role="assistant", content="It is $10,000."),
    ]
    result = QueryRewriter().rewrite("What about it internationally?", history)
    assert result != "What about it internationally?"
    assert "wire" in result.lower() or "transfer" in result.lower() or "threshold" in result.lower()


def test_llm_rewrite_uses_llm_when_provided() -> None:
    class FakeLLM:
        model_name = "fake"

        def generate(self, messages, **kwargs):
            from src.llm.schemas import LLMResponse

            return LLMResponse(content="Standalone rewritten query.", model="fake")

    history = [ConversationTurn(role="user", content="Some prior question.")]
    result = QueryRewriter(llm=FakeLLM()).rewrite("follow up", history)
    assert result == "Standalone rewritten query."
