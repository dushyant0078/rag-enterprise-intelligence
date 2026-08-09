from __future__ import annotations

import pytest

from src.evaluation.ragas_integration import build_evaluation_dataset


def test_ragas_import_fails_with_the_diagnosed_dependency_conflict() -> None:
    """This is not a test that ragas integration *works* -- it's a
    documented, reproducible confirmation of why it can't be verified in
    this sandbox: a real, current upstream dependency conflict between
    ragas, langchain-community, and langchain-openai (see
    src/evaluation/ragas_integration.py's module docstring for the full
    diagnosis). If a future ragas/langchain release fixes this, this test
    will start failing -- at which point it should be replaced with a real
    integration test, not patched to keep passing.
    """
    with pytest.raises(ModuleNotFoundError, match="vertexai"):
        build_evaluation_dataset([])
