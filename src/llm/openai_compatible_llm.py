"""OpenAI-compatible LLM backend, covering Llama 3, Mistral Large, Qwen, and
DeepSeek.

These four are open-weight model families, not single hosted products — in
production they're reached through an inference provider (Together AI, Groq,
Fireworks, Mistral's own API, DeepSeek's own API, a self-hosted vLLM/Ollama
endpoint, ...), nearly all of which expose an OpenAI-compatible
`/chat/completions` endpoint. Rather than write four near-identical classes
that only differ in base_url and model_id, one configurable backend covers
all four — matching how most real deployments actually integrate them.
`configs/llm/llama3.yaml` / `mistral_large.yaml` / `qwen.yaml` /
`deepseek.yaml` each set the base_url + model_id for their provider.

Correct, complete code (reuses the `openai` SDK's client against a custom
`base_url`, which is the standard way to talk to any OpenAI-compatible
endpoint) — untested here, since every plausible provider host is outside
this sandbox's network allowlist.
"""

from __future__ import annotations

import time

import openai

from src.llm.base import BaseLLM
from src.llm.schemas import LLMResponse


class OpenAICompatibleLLM(BaseLLM):
    def __init__(self, model: str, base_url: str, api_key: str | None = None) -> None:
        self.model_name = model
        self.base_url = base_url
        self._client = openai.OpenAI(api_key=api_key or "not-required", base_url=base_url)

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
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except openai.AuthenticationError as exc:
            raise RuntimeError(
                f"Authentication failed against {self.base_url} -- check the API key for this "
                "provider (see .env.example)."
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
