from __future__ import annotations

from src.llm.deterministic_test_llm import DeterministicTestLLM
from src.memory.rag_pipeline import ConversationalRAGPipeline
from src.prompting.guardrails import INSUFFICIENT_EVIDENCE_PHRASE
from src.prompting.registry import get_template
from src.retrieval.base import BaseRetriever
from src.retrieval.schemas import RetrievedChunk


class _FixedRetriever(BaseRetriever):
    retriever_name = "fixed"

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.last_query: str | None = None

    def retrieve(self, query, top_k=10, filters=None):
        self.last_query = query
        return self._chunks[:top_k]


def _banking_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        doc_id="banking-aml-policy",
        text="Wire transfers over $10,000 require enhanced due diligence review.",
        score=0.9,
        page_numbers=[1],
    )


def test_pipeline_produces_grounded_answer_for_relevant_query() -> None:
    retriever = _FixedRetriever([_banking_chunk()])
    pipeline = ConversationalRAGPipeline(
        retriever=retriever, llm=DeterministicTestLLM(), prompt_template=get_template("qa")
    )
    session_id = pipeline.start_session(user_id="alice")
    result = pipeline.ask(session_id, "What triggers enhanced due diligence for wire transfers?")

    assert result.is_grounded
    assert "banking-aml-policy" in result.answer_text
    assert len(result.citations) >= 1
    assert result.retrieved_passages == [_banking_chunk()]


def test_pipeline_falls_back_for_off_topic_query() -> None:
    retriever = _FixedRetriever([_banking_chunk()])
    pipeline = ConversationalRAGPipeline(
        retriever=retriever, llm=DeterministicTestLLM(), prompt_template=get_template("qa")
    )
    session_id = pipeline.start_session(user_id="alice")
    result = pipeline.ask(session_id, "What is the capital of France?")

    assert not result.is_grounded
    assert result.answer_text == INSUFFICIENT_EVIDENCE_PHRASE


def test_pipeline_records_conversation_turns() -> None:
    retriever = _FixedRetriever([_banking_chunk()])
    pipeline = ConversationalRAGPipeline(
        retriever=retriever, llm=DeterministicTestLLM(), prompt_template=get_template("qa")
    )
    session_id = pipeline.start_session(user_id="alice")
    pipeline.ask(session_id, "What triggers enhanced due diligence?")

    session = pipeline.conversation_manager.session_store.get_session(session_id)
    assert len(session.turns) == 2
    assert session.turns[0].role == "user"
    assert session.turns[1].role == "assistant"


def test_pipeline_query_rewriting_flows_into_retrieval() -> None:
    retriever = _FixedRetriever([_banking_chunk()])
    pipeline = ConversationalRAGPipeline(
        retriever=retriever, llm=DeterministicTestLLM(), prompt_template=get_template("qa")
    )
    session_id = pipeline.start_session(user_id="alice")
    pipeline.ask(session_id, "What triggers enhanced due diligence for wire transfers?")
    pipeline.ask(session_id, "What about it for international transfers?")

    # The second query contained a pronoun following prior history -- the retriever should
    # have received a rewritten (not verbatim) query.
    assert retriever.last_query != "What about it for international transfers?"


def test_pipeline_metadata_includes_rewritten_query_and_model() -> None:
    retriever = _FixedRetriever([_banking_chunk()])
    pipeline = ConversationalRAGPipeline(
        retriever=retriever, llm=DeterministicTestLLM(), prompt_template=get_template("qa")
    )
    session_id = pipeline.start_session(user_id="alice")
    result = pipeline.ask(session_id, "What triggers enhanced due diligence?")
    assert "rewritten_query" in result.metadata
    assert result.metadata["model"] == "deterministic-test-llm"


def test_pipeline_reranker_is_applied_when_provided() -> None:
    from src.reranking.lexical_overlap import LexicalOverlapReranker

    retriever = _FixedRetriever([_banking_chunk()])
    pipeline = ConversationalRAGPipeline(
        retriever=retriever,
        llm=DeterministicTestLLM(),
        prompt_template=get_template("qa"),
        reranker=LexicalOverlapReranker(),
    )
    session_id = pipeline.start_session(user_id="alice")
    result = pipeline.ask(session_id, "What triggers enhanced due diligence?")
    assert result.is_grounded  # reranking didn't break the pipeline; still resolves correctly
