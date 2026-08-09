"""Deterministic test LLM.

**Not a production model -- deliberately excluded from configs/llm/.** A real
generative model can't be meaningfully replicated offline (unlike
TfidfEmbedding or LexicalOverlapReranker, which are legitimate lexical
techniques in their own right, "generation" has no closed-form offline
equivalent). What this class *is*: a deterministic, fully offline stand-in
specifically for QA-shaped prompts built by this project's own
`src/prompting` templates + `ContextBuilder`, used to exercise the full
conversational RAG pipeline (retrieval -> context building -> "generation"
-> citation parsing -> validation) end-to-end in this sandbox and in this
repo's own tests, without API cost or network access.

Behavior: parses the `[Source: doc_id, page N]\\ntext` blocks
`ContextBuilder.build()` produces out of the last user message, scores each
block's relevance to the question with the same term-recall signal as
`LexicalOverlapReranker`, and returns the most relevant sentence(s) from the
best-matching block with its citation tag attached. If no block clears a
minimum relevance bar, returns the exact insufficient-evidence phrase from
`src/prompting/guardrails.py`'s plain fallback phrase -- so the "answer only what's grounded, refuse
otherwise" contract this project's prompts specify is actually exercised by
this stand-in, not merely assumed.
"""

from __future__ import annotations

import re
import time

from src.chunking.text_utils import split_sentences
from src.llm.base import BaseLLM
from src.llm.schemas import LLMResponse
from src.prompting.guardrails import INSUFFICIENT_EVIDENCE_PHRASE
from src.utils.text import content_words

_BLOCK_RE = re.compile(
    r"\[Source: ([^\]]+)\]\s*\n(.*?)"
    r"(?=\n\n\[Source:"  # the next context block ContextBuilder produced
    r"|\n\n[A-Z][A-Za-z ,'()/-]{1,60}:"  # a short trailing "Label:" line (with or without
    # inline content after the colon, e.g. "Question: {query}") -- how every one of this
    # project's own templates (schemas.py's PromptTemplate) closes its rendered prompt
    # ("Question:", "Answer:", "Verdict and reasoning:", ...) and how few-shot examples
    # introduce their sections ("Example input:", "Example output:")
    r"|\n\nSummarize the above"  # summarization.py's one template whose trailing instruction
    # is a full sentence, not a short label -- named explicitly rather than guessed at, since
    # this class is already tightly coupled to this project's specific, finite set of templates
    r"|\Z)",
    re.DOTALL,
)
_MIN_RELEVANCE = 0.15  # minimum query-term-recall a block needs to be considered "grounded"
_LAST_EXAMPLE_OUTPUT_RE = re.compile(r"Example output:")


def _strip_few_shot_examples(user_content: str) -> str:
    """Cut everything up through the last few-shot example's output, so
    `_parse_source_blocks` only ever sees the real retrieved context.

    This was *not* originally here -- the first version scanned the whole
    message on the theory that few-shot blocks would just be harmless
    low-relevance noise. That assumption was wrong: this project's own QA
    template's few-shot example is about parental leave, phrased very
    similarly to a real eval query about parental leave, so the example
    content scored *higher* on term-recall than the real retrieved context
    and got picked as the "answer" instead -- caught by running the actual
    demo end-to-end (scripts/run_conversational_pipeline_demo.py) against
    the real sample corpus, not by a unit test with hand-picked wording.
    A real LLM doesn't confuse a few-shot demonstration with the actual
    context to draw facts from; this stand-in has to be told where that
    boundary is explicitly.
    """
    matches = list(_LAST_EXAMPLE_OUTPUT_RE.finditer(user_content))
    return user_content[matches[-1].end() :] if matches else user_content


def _parse_source_blocks(text: str) -> list[tuple[str, str]]:
    """Returns [(citation_tag, block_text), ...] found anywhere in `text`
    (expected to already have few-shot examples stripped -- see
    `_strip_few_shot_examples`). Scans the whole remaining string rather
    than assuming a fixed position, since ContextBuilder-joined chunks can
    appear with a "Context:\\n" prefix on the first one and arbitrary
    template text trailing the last one.
    """
    return [(f"[Source: {m.group(1)}]", m.group(2).strip()) for m in _BLOCK_RE.finditer(text)]


_QUESTION_LABEL_RE = re.compile(r"(?:Question|Situation):")


def _extract_question(user_content: str) -> str:
    """This project's templates render the question as "Question: ..." or
    "Situation: ..." right before a trailing label ("Answer:",
    "Verdict and reasoning:", etc., which varies per template) at the very
    end -- but few-shot examples earlier in the same message can contain
    their own "Question:" labels too (this project's own QA template does).

    A single regex anchored to end-of-string can't find the *last*
    occurrence via `finditer` the way it seems like it should: the first
    (leftmost) match's own `.+?$` greedily spans all the way to the true
    end of the string to satisfy `$`, consuming every later "Question:"
    inside that one match, so only one match is ever found -- and it's the
    wrong one. (Caught by testing against a real rendered template with
    few-shot examples, not a hand-simplified one.) Finding the label's
    *position* first, independently of any trailing content, and only then
    slicing from the last position found, avoids that trap.
    """
    label_matches = list(_QUESTION_LABEL_RE.finditer(user_content))
    if not label_matches:
        return user_content
    after_label = user_content[label_matches[-1].end() :].strip()
    return after_label.split("\n\n")[0].strip()


class DeterministicTestLLM(BaseLLM):
    def __init__(self) -> None:
        self.model_name = "deterministic-test-llm"

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        start = time.monotonic()

        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        user_content = user_messages[-1] if user_messages else ""
        question = _extract_question(user_content)
        blocks = _parse_source_blocks(_strip_few_shot_examples(user_content))

        content = self._answer(question, blocks)
        latency_ms = (time.monotonic() - start) * 1000

        return LLMResponse(
            content=content,
            model=self.model_name,
            input_tokens=None,
            output_tokens=None,
            finish_reason="stop",
            latency_ms=latency_ms,
        )

    @staticmethod
    def _answer(question: str, blocks: list[tuple[str, str]]) -> str:
        query_words = content_words(question)
        if not query_words or not blocks:
            return INSUFFICIENT_EVIDENCE_PHRASE

        scored = []
        for citation_tag, block_text in blocks:
            block_words = content_words(block_text)
            recall = len(query_words & block_words) / len(query_words)
            scored.append((recall, citation_tag, block_text))
        scored.sort(key=lambda t: -t[0])

        best_recall, best_tag, best_text = scored[0]
        if best_recall < _MIN_RELEVANCE:
            return INSUFFICIENT_EVIDENCE_PHRASE

        sentences = split_sentences(best_text)
        relevant_sentences = [s for s in sentences if content_words(s) & query_words] or sentences[
            :1
        ]
        return f"{' '.join(relevant_sentences)} {best_tag}"
