"""Claude LLM backend (Anthropic SDK).

Unlike most other cloud-dependent backends in this project, `api.anthropic.com`
*is* reachable from this development sandbox's network allowlist — what's
missing here is an API key, not network access. That's a meaningfully
different, stronger verification story than "never contacted": this
backend's request-building and error-handling were smoke-tested against the
real live API (see docs/07_conversational_rag_pipeline.md), which confirmed
the request reaches Anthropic and is rejected only for missing
authentication, not for a malformed request shape.
"""

from __future__ import annotations

import time

import anthropic

from src.llm.base import BaseLLM
from src.llm.schemas import LLMResponse

DEFAULT_MODEL = "claude-sonnet-4-5"


class ClaudeLLM(BaseLLM):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        self.model_name = model
        self._client = anthropic.Anthropic(
            api_key=api_key
        )  # falls back to ANTHROPIC_API_KEY env var

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        system_content, chat_messages = _split_system_message(messages)

        start = time.monotonic()
        try:
            response = self._client.messages.create(
                model=self.model_name,
                system=system_content,
                # Our plain dict shape matches the SDK's MessageParam structure at runtime
                # (verified live against api.anthropic.com); the stub wants a narrower
                # TypedDict than a plain dict satisfies.
                messages=chat_messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except anthropic.AuthenticationError as exc:
            raise RuntimeError(
                "Anthropic API authentication failed -- set ANTHROPIC_API_KEY (see "
                ".env.example). The request itself reached the API correctly; only "
                "credentials are missing."
            ) from exc
        latency_ms = (time.monotonic() - start) * 1000

        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(
            content=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            finish_reason=response.stop_reason,
            latency_ms=latency_ms,
        )


def _split_system_message(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    """Anthropic's API takes the system prompt as a separate top-level
    argument, not a message with role="system" — split it out here so
    callers can pass the same standard message list every other backend uses.
    """
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    chat_messages = [dict(m) for m in messages if m["role"] != "system"]
    return "\n\n".join(system_parts), chat_messages
