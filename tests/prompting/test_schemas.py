from __future__ import annotations

from src.prompting.schemas import FewShotExample, PromptTemplate


def test_render_produces_system_and_user_messages() -> None:
    template = PromptTemplate(
        name="t",
        description="d",
        system_prompt="You are a helpful assistant.",
        user_template="Context: {context}\n\nQuestion: {query}",
    )
    messages = template.render(context="some context", query="some question")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "some context" in messages[1]["content"]
    assert "some question" in messages[1]["content"]


def test_guardrails_appended_to_system_prompt() -> None:
    template = PromptTemplate(
        name="t",
        description="d",
        system_prompt="Base prompt.",
        user_template="{query}",
        guardrails=["Rule one.", "Rule two."],
    )
    messages = template.render(query="q")
    assert "Rule one." in messages[0]["content"]
    assert "Rule two." in messages[0]["content"]
    assert "Base prompt." in messages[0]["content"]


def test_no_guardrails_leaves_system_prompt_unchanged() -> None:
    template = PromptTemplate(
        name="t", description="d", system_prompt="Base prompt.", user_template="{query}"
    )
    messages = template.render(query="q")
    assert messages[0]["content"] == "Base prompt."


def test_few_shot_examples_included_before_rendered_template() -> None:
    template = PromptTemplate(
        name="t",
        description="d",
        system_prompt="Base.",
        user_template="Actual question: {query}",
        few_shot_examples=[FewShotExample(input="example in", output="example out")],
    )
    messages = template.render(query="real question")
    content = messages[1]["content"]
    assert "example in" in content
    assert "example out" in content
    assert content.index("example out") < content.index("Actual question: real question")


def test_no_few_shot_examples_renders_only_user_template() -> None:
    template = PromptTemplate(
        name="t", description="d", system_prompt="Base.", user_template="Q: {query}"
    )
    messages = template.render(query="hello")
    assert messages[1]["content"] == "Q: hello"


def test_missing_placeholder_raises_key_error() -> None:
    import pytest

    template = PromptTemplate(
        name="t", description="d", system_prompt="Base.", user_template="{query} {context}"
    )
    with pytest.raises(KeyError):
        template.render(query="only query given")
