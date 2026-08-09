"""Response Validator (docs/02_architecture.md's principle #3: "Grounding is
enforced, not requested" -- a real pipeline stage, not just a system-prompt
instruction). Checks an LLM's answer against the chunks it was actually
given: any fabricated citation, or an answer with no resolvable citation at
all, is replaced with the standard insufficient-evidence fallback rather
than shown to the user.

Default `min_confidence` (0.7) matches `response_validator.min_faithfulness
_score` in configs/config.yaml (Milestone 2) -- this milestone is what makes
that config value real, the same way Milestone 6's ContextBuilder made
`max_context_tokens` real.
"""

from __future__ import annotations

from src.citations.citation_generator import CitationGenerator
from src.citations.schemas import Citation, ValidatedAnswer
from src.prompting.guardrails import INSUFFICIENT_EVIDENCE_PHRASE
from src.retrieval.schemas import RetrievedChunk

DEFAULT_MIN_CONFIDENCE = 0.7


class ResponseValidator:
    def __init__(
        self,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        citation_generator: CitationGenerator | None = None,
    ) -> None:
        self.min_confidence = min_confidence
        self.citation_generator = citation_generator or CitationGenerator()

    def validate(self, answer_text: str, chunks: list[RetrievedChunk]) -> ValidatedAnswer:
        citations = self.citation_generator.generate(answer_text, chunks)
        resolved = [c for c in citations if not c.is_fabricated]
        has_fabricated = any(c.is_fabricated for c in citations)

        confidence_score = self._confidence(resolved)
        # Strict by design: even one fabricated citation invalidates the whole answer, and an
        # answer with zero resolvable citations is never "grounded" regardless of how
        # confident its prose sounds -- both are exactly the failure modes this stage exists
        # to catch, and a partially-good answer isn't worth the risk of an unflagged bad claim.
        is_grounded = (
            bool(resolved) and not has_fabricated and confidence_score >= self.min_confidence
        )

        final_text = answer_text if is_grounded else INSUFFICIENT_EVIDENCE_PHRASE

        return ValidatedAnswer(
            answer_text=final_text,
            original_answer_text=answer_text,
            citations=citations,
            confidence_score=confidence_score,
            is_grounded=is_grounded,
            retrieved_passages=chunks,
        )

    @staticmethod
    def _confidence(resolved_citations: list[Citation]) -> float:
        if not resolved_citations:
            return 0.0
        return sum(c.confidence for c in resolved_citations) / len(resolved_citations)
