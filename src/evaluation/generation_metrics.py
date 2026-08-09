"""Generation quality metrics.

Citation accuracy needs no LLM or heuristic at all -- it's a direct count
against CitationGenerator's real resolution logic (Milestone 7). The other
three -- hallucination rate, faithfulness, answer correctness -- are
RAGAS/DeepEval concepts that, in their standard form, use an LLM judge for
entailment/semantic-similarity checking. No LLM is reachable end-to-end in
this sandbox (see docs/08_evaluation_framework.md), so these three are
implemented as real, deterministic *lexical* proxies -- genuinely
informative at the scale this project runs, explicitly not a claim of
parity with an LLM judge. `src/evaluation/deepeval_integration.py` and
`ragas_integration.py` are the documented upgrade path once a judge model
is available.
"""

from __future__ import annotations

from src.chunking.text_utils import split_sentences
from src.citations.schemas import Citation
from src.retrieval.schemas import RetrievedChunk
from src.utils.text import content_words

_MIN_SENTENCE_SUPPORT = 0.3  # fraction of a sentence's content words that must appear in
# *some* chunk for that sentence to count as "supported" rather than hallucinated


def citation_accuracy(citations: list[Citation]) -> float:
    """Fraction of citations in an answer that resolved to a real provided
    chunk. An answer with zero citations returns 0.0 -- no citations to be
    accurate is not the same as perfect accuracy.
    """
    if not citations:
        return 0.0
    resolved = sum(not c.is_fabricated for c in citations)
    return resolved / len(citations)


def _sentence_is_supported(sentence: str, chunks: list[RetrievedChunk]) -> bool:
    sentence_words = content_words(sentence)
    if not sentence_words:
        return True  # nothing to hallucinate in a sentence with no content words
    for chunk in chunks:
        chunk_words = content_words(chunk.text)
        overlap = len(sentence_words & chunk_words) / len(sentence_words)
        if overlap >= _MIN_SENTENCE_SUPPORT:
            return True
    return False


def hallucination_rate(answer_text: str, chunks: list[RetrievedChunk]) -> float:
    """Fraction of the answer's sentences with no lexical support in any
    provided chunk. Lexical overlap is a real, computable signal but not
    entailment -- a sentence can share words with a chunk without the chunk
    actually supporting its claim, and conversely a paraphrase with no word
    overlap would be wrongly flagged. That gap is exactly what an LLM-judge
    metric (RAGAS/DeepEval) closes; this is the offline stand-in.
    """
    sentences = split_sentences(answer_text)
    if not sentences:
        return 0.0
    unsupported = sum(not _sentence_is_supported(s, chunks) for s in sentences)
    return unsupported / len(sentences)


def faithfulness_lexical(answer_text: str, chunks: list[RetrievedChunk]) -> float:
    """1 - hallucination_rate, framed as the complementary "how much of the
    answer is grounded" view RAGAS's Faithfulness metric reports.
    """
    return 1.0 - hallucination_rate(answer_text, chunks)


def answer_correctness_lexical(generated_answer: str, expected_answer: str) -> float:
    """Token-overlap F1 between the generated and expected answers -- a
    real, deterministic proxy for RAGAS/DeepEval's LLM-judge-based semantic
    Answer Correctness metric. Rewards lexical overlap, not meaning, so a
    correct paraphrase scores lower than a correct near-verbatim match --
    a known, stated limitation of this proxy, not a claim it doesn't have one.
    """
    generated_words = content_words(generated_answer)
    expected_words = content_words(expected_answer)
    if not generated_words or not expected_words:
        return 0.0

    overlap = len(generated_words & expected_words)
    precision = overlap / len(generated_words)
    recall = overlap / len(expected_words)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
