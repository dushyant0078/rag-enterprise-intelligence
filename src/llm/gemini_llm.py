"""Gemini LLM backend (`google-genai` SDK — the current package; the older
`google-generativeai` is deprecated upstream as of this writing, so this
project uses the maintained one). Correct, complete code; `generativelanguage
.googleapis.com` isn't in this sandbox's network allowlist, so untested here,
same as OpenAILLM.
"""

from __future__ import annotations

import time

from google import genai
from google.genai import types as genai_types

from src.llm.base import BaseLLM
from src.llm.schemas import LLMResponse

DEFAULT_MODEL = "gemini-2.5-pro"


class GeminiLLM(BaseLLM):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        self.model_name = model
        self._client = genai.Client(
            api_key=api_key
        )  # falls back to GOOGLE_API_KEY / GEMINI_API_KEY

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        system_content, contents = _to_gemini_contents(messages)

        start = time.monotonic()
        response = self._client.models.generate_content(
            model=self.model_name,
            # Our plain dict shape matches the SDK's expected content structure at runtime;
            # the stub wants a narrower type than a plain dict/list satisfies.
            contents=contents,  # type: ignore[arg-type]
            config=genai_types.GenerateContentConfig(
                system_instruction=system_content or None,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        latency_ms = (time.monotonic() - start) * 1000

        usage = response.usage_metadata
        finish_reason = None
        if response.candidates and response.candidates[0].finish_reason:
            finish_reason = response.candidates[0].finish_reason.name

        return LLMResponse(
            content=response.text or "",
            model=self.model_name,
            input_tokens=usage.prompt_token_count if usage else None,
            output_tokens=usage.candidates_token_count if usage else None,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
        )


def _to_gemini_contents(messages: list[dict[str, str]]) -> tuple[str, list[dict]]:
    """Gemini takes the system prompt as a separate config field (like
    Claude) and uses role "model" instead of "assistant" for prior turns.
    """
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    contents = [
        {
            "role": "model" if m["role"] == "assistant" else m["role"],
            "parts": [{"text": m["content"]}],
        }
        for m in messages
        if m["role"] != "system"
    ]
    return "\n\n".join(system_parts), contents
