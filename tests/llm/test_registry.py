from __future__ import annotations

import pytest

from src.llm.deterministic_test_llm import DeterministicTestLLM
from src.llm.registry import UnknownLLMError, available_llms, get_llm


def test_available_llms_lists_expected_names() -> None:
    llms = available_llms()
    for name in (
        "claude",
        "gpt4",
        "gemini",
        "llama3",
        "mistral_large",
        "qwen",
        "deepseek",
        "deterministic_test",
    ):
        assert name in llms


def test_get_llm_deterministic_test() -> None:
    llm = get_llm("deterministic_test")
    assert isinstance(llm, DeterministicTestLLM)


def test_get_llm_unknown_raises() -> None:
    with pytest.raises(UnknownLLMError):
        get_llm("not_a_real_llm")
