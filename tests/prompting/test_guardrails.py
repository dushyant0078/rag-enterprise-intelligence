from __future__ import annotations

from src.prompting import guardrails


def test_standard_qa_guardrails_includes_grounding_and_citation() -> None:
    combined = " ".join(guardrails.STANDARD_QA_GUARDRAILS)
    assert guardrails.GROUNDING in guardrails.STANDARD_QA_GUARDRAILS
    assert guardrails.CITATION_REQUIRED in guardrails.STANDARD_QA_GUARDRAILS
    assert "context" in combined.lower()


def test_standard_analysis_guardrails_includes_professional_tone() -> None:
    assert guardrails.PROFESSIONAL_TONE in guardrails.STANDARD_ANALYSIS_GUARDRAILS


def test_injection_resistance_treats_context_as_data() -> None:
    assert "not instructions" in guardrails.INJECTION_RESISTANCE.lower()


def test_insufficient_evidence_has_exact_fallback_phrase() -> None:
    assert "don't have enough information" in guardrails.INSUFFICIENT_EVIDENCE
