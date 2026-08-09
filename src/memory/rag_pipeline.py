"""ConversationalRAGPipeline: the full pipeline from docs/02_architecture.md,
wired together end to end --

    Conversation Manager -> Query Rewriter -> Retriever -> Reranker (optional)
    -> Context Builder -> LLM Inference -> Citation Generator ->
    Response Validator

-- as one per-turn call. Lives in src/memory/ alongside ConversationManager
rather than a dedicated top-level package: the spec's src/ folder list has
no slot for pipeline orchestration itself, and every turn through this
pipeline is fundamentally conversation-turn-driven, so it sits with the
component that already anchors that concept. Milestone 9's FastAPI `/chat`
endpoint will be a thin wrapper around this class, not a reimplementation.

Each stage is wrapped in a real OpenTelemetry span (Milestone 11) via an
injected tracer -- defaults to whatever global tracer provider is
configured (a harmless no-op if none is), so this class works unchanged
whether or not tracing is wired up. Real per-stage latency (not just an
end-to-end total, which was this class's own documented limitation through
Milestones 8-9) is recorded in `ValidatedAnswer.metadata["stage_latencies_ms"]`.
"""

from __future__ import annotations

import time
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Tracer

from src.citations.response_validator import ResponseValidator
from src.citations.schemas import ValidatedAnswer
from src.llm.base import BaseLLM
from src.memory.conversation_manager import ConversationManager
from src.prompting.context_builder import ContextBuilder
from src.prompting.schemas import PromptTemplate
from src.reranking.base import BaseReranker
from src.retrieval.base import BaseRetriever

_TRACER_NAME = "rag_enterprise_intelligence.pipeline"


class ConversationalRAGPipeline:
    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLM,
        prompt_template: PromptTemplate,
        conversation_manager: ConversationManager | None = None,
        reranker: BaseReranker | None = None,
        context_builder: ContextBuilder | None = None,
        response_validator: ResponseValidator | None = None,
        retrieval_top_k: int = 10,
        rerank_top_k: int = 5,
        tracer: Tracer | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template
        self.conversation_manager = conversation_manager or ConversationManager()
        self.reranker = reranker
        self.context_builder = context_builder or ContextBuilder()
        self.response_validator = response_validator or ResponseValidator()
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k
        self._tracer = tracer or trace.get_tracer(_TRACER_NAME)

    def start_session(self, user_id: str, session_id: str | None = None) -> str:
        return self.conversation_manager.start_session(user_id, session_id=session_id).session_id

    def ask(
        self,
        session_id: str,
        query: str,
        *,
        requesting_user_id: str | None = None,
        filters: dict[str, Any] | None = None,
        template_kwargs: dict[str, Any] | None = None,
        prompt_template: PromptTemplate | None = None,
    ) -> ValidatedAnswer:
        """Run one full conversational turn. Returns a ValidatedAnswer
        carrying the final answer text, resolved citations, confidence, and
        the retrieved passages -- everything the spec's citation-aware
        answering section requires, in one object.

        `prompt_template`, if given, overrides `self.prompt_template` for
        this call only -- lets one pipeline instance (sharing the same
        retriever/LLM/conversation state) answer with any of Milestone 6's
        task-specific templates (compliance_checking, risk_extraction, ...)
        instead of constructing a separate pipeline per template. Added in
        Milestone 10 when the Compliance Analyzer page needed the real
        compliance_checking template, not a relabeled QA call.
        """
        template = prompt_template or self.prompt_template
        stage_latencies_ms: dict[str, float] = {}

        with self._tracer.start_as_current_span("conversation_manager.prepare_turn"):
            start = time.monotonic()
            rewritten_query = self.conversation_manager.prepare_turn(
                session_id, query, requesting_user_id=requesting_user_id
            )
            stage_latencies_ms["query_rewrite_ms"] = (time.monotonic() - start) * 1000

        with self._tracer.start_as_current_span("retriever.retrieve") as span:
            start = time.monotonic()
            chunks = self.retriever.retrieve(
                rewritten_query, top_k=self.retrieval_top_k, filters=filters
            )
            stage_latencies_ms["retrieval_ms"] = (time.monotonic() - start) * 1000
            span.set_attribute("chunks_retrieved", len(chunks))

        if self.reranker is not None and chunks:
            with self._tracer.start_as_current_span("reranker.rerank"):
                start = time.monotonic()
                chunks = self.reranker.rerank(rewritten_query, chunks, top_k=self.rerank_top_k)
                stage_latencies_ms["rerank_ms"] = (time.monotonic() - start) * 1000
        elif self.rerank_top_k < len(chunks):
            chunks = chunks[: self.rerank_top_k]

        with self._tracer.start_as_current_span("context_builder.build"):
            start = time.monotonic()
            context = self.context_builder.build(chunks)
            stage_latencies_ms["context_build_ms"] = (time.monotonic() - start) * 1000

        messages = template.render(
            query=rewritten_query, context=context, **(template_kwargs or {})
        )

        with self._tracer.start_as_current_span("llm.generate") as span:
            llm_response = self.llm.generate(messages)
            stage_latencies_ms["generation_ms"] = llm_response.latency_ms or 0.0
            span.set_attribute("model", llm_response.model)
            if llm_response.input_tokens is not None:
                span.set_attribute("input_tokens", llm_response.input_tokens)
            if llm_response.output_tokens is not None:
                span.set_attribute("output_tokens", llm_response.output_tokens)

        with self._tracer.start_as_current_span("response_validator.validate") as span:
            start = time.monotonic()
            validated = self.response_validator.validate(llm_response.content, chunks)
            stage_latencies_ms["validation_ms"] = (time.monotonic() - start) * 1000
            span.set_attribute("is_grounded", validated.is_grounded)

        stage_latencies_ms["total_ms"] = sum(stage_latencies_ms.values())

        validated.metadata = {
            "rewritten_query": rewritten_query,
            "model": llm_response.model,
            "latency_ms": llm_response.latency_ms,
            "input_tokens": llm_response.input_tokens,
            "output_tokens": llm_response.output_tokens,
            "stage_latencies_ms": stage_latencies_ms,
        }

        self.conversation_manager.record_response(
            session_id, validated.answer_text, requesting_user_id=requesting_user_id
        )
        return validated
