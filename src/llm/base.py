"""Base LLM interface. Every backend in src/llm/ implements this — the
"abstraction layer so models can be swapped via configuration" the spec
calls for. src/prompting's PromptTemplate.render() already produces the
message shape every backend here consumes, so the seam between prompting
and generation is just this interface.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from src.llm.schemas import LLMResponse


class BaseLLM(ABC):
    model_name: str

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generate a response to a chat-format message list
        (`[{"role": "system"|"user"|"assistant", "content": str}, ...]`).
        Low default temperature — grounded QA, not creative generation
        (matches configs/llm/*.yaml).
        """
        raise NotImplementedError

    async def agenerate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Default async wrapper: runs the sync generate() in a worker
        thread, same pattern as EmbeddingModel.embed_async. Backends with a
        real async-native client should override this.
        """
        return await asyncio.to_thread(
            self.generate, messages, temperature=temperature, max_tokens=max_tokens
        )
