"""GPT-family LLM backend (OpenAI SDK) — covers GPT-4.1/GPT-5-compatible
APIs per the spec. Correct, complete code against the real `openai` v2 SDK.
Unlike ClaudeLLM, `api.openai.com` isn't in this sandbox's network
allowlist at all, so this backend's request path couldn't even be
connectivity-checked the way Claude's was — untested here for a stronger
reason (no network path) than a missing key alone.
"""

from __future__ import annotations

import time

import openai

from src.llm.base import BaseLLM
from src.llm.schemas import LLMResponse

DEFAULT_MODEL = "gpt-4.1"


class OpenAILLM(BaseLLM):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        self.model_name = model
        self._client = openai.OpenAI(api_key=api_key)  # falls back to OPENAI_API_KEY env var

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        start = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                # Our plain dict shape matches the SDK's expected structure at runtime; the
                # stub wants a narrower TypedDict union than a plain dict satisfies.
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except openai.AuthenticationError as exc:
            raise RuntimeError(
                "OpenAI API authentication failed -- set OPENAI_API_KEY (see .env.example)."
            ) from exc
        latency_ms = (time.monotonic() - start) * 1000

        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
            finish_reason=choice.finish_reason,
            latency_ms=latency_ms,
        )
