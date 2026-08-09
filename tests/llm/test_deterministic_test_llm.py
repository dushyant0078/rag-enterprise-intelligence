from __future__ import annotations

from src.llm.deterministic_test_llm import (
    DeterministicTestLLM,
    _extract_question,
    _parse_source_blocks,
)
from src.prompting.context_builder import ContextBuilder
from src.prompting.guardrails import INSUFFICIENT_EVIDENCE_PHRASE
from src.prompting.registry import get_template
from src.retrieval.schemas import RetrievedChunk


def _build_qa_messages(query: str) -> list[dict[str, str]]:
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            doc_id="banking-aml-policy",
            text="Wire transfers over $10,000 require enhanced due diligence review by the "
            "compliance team.",
            score=0.9,
            page_numbers=[1],
        ),
        RetrievedChunk(
            chunk_id="c2",
            doc_id="hr-remote-policy",
            text="Employees may work remotely up to five days per week with manager approval.",
            score=0.3,
            page_numbers=[2],
        ),
    ]
    context = ContextBuilder().build(chunks)
    return get_template("qa").render(query=query, context=context)


def test_extract_question_finds_the_real_question_not_a_few_shot_examples() -> None:
    # The QA template's own few-shot examples contain "Question:" labels earlier in the
    # message -- this is the exact bug caught during development: a naive first-match
    # regex silently grabbed an example's question instead of the real one.
    messages = _build_qa_messages("What triggers enhanced due diligence for wire transfers?")
    question = _extract_question(messages[1]["content"])
    assert question == "What triggers enhanced due diligence for wire transfers?"


def test_parse_source_blocks_finds_real_context_despite_few_shot_noise() -> None:
    messages = _build_qa_messages("some question")
    blocks = _parse_source_blocks(messages[1]["content"])
    doc_ids_found = {tag for tag, _ in blocks}
    assert any("banking-aml-policy" in tag for tag in doc_ids_found)
    assert any("hr-remote-policy" in tag for tag in doc_ids_found)


def test_generate_answers_from_the_correct_relevant_block() -> None:
    messages = _build_qa_messages("What triggers enhanced due diligence for wire transfers?")
    response = DeterministicTestLLM().generate(messages)
    assert "banking-aml-policy" in response.content
    assert "hr-remote-policy" not in response.content


def test_generate_falls_back_on_off_topic_question() -> None:
    messages = _build_qa_messages("What is the capital of France?")
    response = DeterministicTestLLM().generate(messages)
    assert response.content == INSUFFICIENT_EVIDENCE_PHRASE


def test_generate_falls_back_when_no_context_blocks_present() -> None:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Question: anything at all\n\nAnswer:"},
    ]
    response = DeterministicTestLLM().generate(messages)
    assert response.content == INSUFFICIENT_EVIDENCE_PHRASE


def test_generate_ignores_few_shot_example_content_even_when_lexically_similar() -> None:
    """Regression test for a real bug: the QA template's own few-shot example
    is about parental leave, phrased similarly to this query -- an earlier
    version of DeterministicTestLLM picked the few-shot example's content
    over the actual retrieved context because it scored higher on lexical
    term-recall. Caught by running the real end-to-end demo against the
    sample corpus, not by a hand-picked unit test.
    """
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            doc_id="real-hr-doc",
            text="Employees in their first year of employment are eligible for 12 weeks of "
            "paid parental leave, per the 2025 handbook revision.",
            score=1.0,
            page_numbers=[1],
        )
    ]
    context = ContextBuilder().build(chunks)
    messages = get_template("qa").render(
        query="how many weeks of paid parental leave in the first year", context=context
    )
    response = DeterministicTestLLM().generate(messages)
    assert "real-hr-doc" in response.content
    assert "hr-policy-2025" not in response.content  # the few-shot example's doc_id


def test_generate_sets_model_name_and_latency() -> None:
    messages = _build_qa_messages("What triggers enhanced due diligence?")
    response = DeterministicTestLLM().generate(messages)
    assert response.model == "deterministic-test-llm"
    assert response.latency_ms is not None
    assert response.latency_ms >= 0


def test_generate_handles_summarization_template_without_trailing_instructions() -> None:
    """Regression test: DeterministicTestLLM's block-boundary parser was
    originally scoped only to the QA template's exact closing markers.
    Running it against the summarization template (via the API's
    /summarize endpoint, Milestone 9) revealed the block capture didn't
    stop before that template's trailing instruction sentence
    ("Summarize the above in..."), so it ended up folded into the
    "answer." Fixed with a general short-label boundary pattern plus one
    named exception for summarization's full-sentence instruction --
    caught by exercising the real endpoint, not by QA-only unit tests.
    """
    from src.prompting.templates.summarization import SUMMARIZATION_TEMPLATE

    messages = SUMMARIZATION_TEMPLATE.render(
        context="[Source: text]\nWire transfers over $10,000 require enhanced due diligence "
        "review by compliance.",
        length_hint="3-4 sentences",
    )
    response = DeterministicTestLLM().generate(messages)
    assert "Summarize the above" not in response.content
    assert "Preserve all numbers" not in response.content
    assert "enhanced due diligence" in response.content.lower()


def test_generate_qa_template_unaffected_by_summarization_boundary_fix() -> None:
    """The boundary-pattern fix above must not regress the QA template's
    inline "Question: {query}" format, which has content on the same line
    as the label rather than the label appearing alone -- an earlier
    version of the fix broke exactly this case.
    """
    messages = _build_qa_messages("What triggers enhanced due diligence for wire transfers?")
    response = DeterministicTestLLM().generate(messages)
    assert "Question:" not in response.content
    assert "banking-aml-policy" in response.content
