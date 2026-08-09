from __future__ import annotations

import pytest

from src.prompting.registry import UnknownPromptTemplateError, available_templates, get_template

# Every template's render() needs different kwargs depending on its user_template's
# placeholders -- this mirrors how a caller would actually invoke each one.
_RENDER_KWARGS: dict[str, dict[str, str]] = {
    "qa": {"query": "What is the notice period?", "context": "[Source: doc1]\nSome policy text."},
    "summarization": {
        "context": "[Source: doc1]\nSome policy text.",
        "length_hint": "2-3 sentences",
    },
    "contract_analysis": {
        "query": "Is there an auto-renewal clause?",
        "context": "[Source: doc1]\nContract text.",
    },
    "compliance_checking": {
        "query": "A $12,000 transfer was processed.",
        "context": "[Source: doc1]\nPolicy text.",
    },
    "risk_extraction": {"query": "quarterly risks", "context": "[Source: doc1]\nRisk factor text."},
    "financial_analysis": {
        "query": "What was revenue growth?",
        "context": "[Source: doc1]\nRevenue figures.",
    },
    "technical_troubleshooting": {
        "query": "What are the LOTO steps?",
        "context": "[Source: doc1]\nSOP text.",
    },
    "research_assistance": {
        "query": "What methodology was used?",
        "context": "[Source: doc1]\nAbstract text.",
    },
}


def test_available_templates_lists_all_eight() -> None:
    templates = available_templates()
    assert set(templates) == set(_RENDER_KWARGS.keys())


@pytest.mark.parametrize("name", list(_RENDER_KWARGS.keys()))
def test_every_template_renders_successfully(name: str) -> None:
    template = get_template(name)
    messages = template.render(**_RENDER_KWARGS[name])
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert len(messages[0]["content"]) > 0
    assert len(messages[1]["content"]) > 0


@pytest.mark.parametrize("name", list(_RENDER_KWARGS.keys()))
def test_every_template_has_grounding_and_citation_guardrails(name: str) -> None:
    template = get_template(name)
    combined = " ".join(template.guardrails).lower()
    assert "context" in combined  # grounding-style instruction present
    assert "citation" in combined or "[source" in combined.lower()


@pytest.mark.parametrize("name", list(_RENDER_KWARGS.keys()))
def test_every_template_has_at_least_one_few_shot_example(name: str) -> None:
    template = get_template(name)
    assert len(template.few_shot_examples) >= 1


def test_qa_template_demonstrates_insufficient_evidence_fallback() -> None:
    template = get_template("qa")
    outputs = [ex.output for ex in template.few_shot_examples]
    assert any("don't have enough information" in o for o in outputs)


def test_get_template_unknown_raises() -> None:
    with pytest.raises(UnknownPromptTemplateError):
        get_template("not_a_real_template")
