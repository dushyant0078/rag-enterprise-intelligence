"""Maps a configured LLM name (configs/llm/*.yaml) to a concrete BaseLLM
instance -- the "abstraction layer so models can be swapped via
configuration" the spec calls for.
"""

from __future__ import annotations

from typing import Any

from src.llm.base import BaseLLM

# Provider -> (module, class_name, requires base_url). Llama3/Mistral/Qwen/DeepSeek all route
# through the same OpenAICompatibleLLM class with a provider-specific base_url -- see
# src/llm/openai_compatible_llm.py for why one class covers all four.
_OPENAI_COMPATIBLE_NAMES = {"llama3", "mistral_large", "qwen", "deepseek"}


class UnknownLLMError(ValueError):
    pass


def available_llms() -> tuple[str, ...]:
    return ("claude", "gpt4", "gemini", *sorted(_OPENAI_COMPATIBLE_NAMES), "deterministic_test")


def get_llm(name: str, **kwargs: Any) -> BaseLLM:
    """Instantiate an LLM backend by name.

    - "claude": ClaudeLLM -- network-reachable in this sandbox, needs an API key.
    - "gpt4": OpenAILLM -- needs network access to api.openai.com (not reachable here).
    - "gemini": GeminiLLM -- needs network access to Google's API (not reachable here).
    - "llama3" / "mistral_large" / "qwen" / "deepseek": OpenAICompatibleLLM --
      pass model= and base_url= for the chosen inference provider.
    - "deterministic_test": DeterministicTestLLM -- fully offline, QA-prompt-shaped
      stand-in for pipeline testing. Not a production option (see its docstring).
    """
    if name == "claude":
        from src.llm.claude_llm import ClaudeLLM

        return ClaudeLLM(**kwargs)
    if name == "gpt4":
        from src.llm.openai_llm import OpenAILLM

        return OpenAILLM(**kwargs)
    if name == "gemini":
        from src.llm.gemini_llm import GeminiLLM

        return GeminiLLM(**kwargs)
    if name in _OPENAI_COMPATIBLE_NAMES:
        from src.llm.openai_compatible_llm import OpenAICompatibleLLM

        return OpenAICompatibleLLM(**kwargs)
    if name == "deterministic_test":
        from src.llm.deterministic_test_llm import DeterministicTestLLM

        return DeterministicTestLLM()
    raise UnknownLLMError(f"Unknown LLM '{name}'. Available: {available_llms()}")
