"""Prompt template data model. Every template in src/prompting/templates/ is
a PromptTemplate instance; `.render()` produces a chat-message list
(`[{"role": "system", ...}, {"role": "user", ...}]`) in the shape
Milestone 6-7's LLM abstraction will expect — this module has no dependency
on that abstraction existing yet, only on the message shape being standard.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FewShotExample(BaseModel):
    input: str
    output: str


class PromptTemplate(BaseModel):
    name: str
    description: str
    system_prompt: str
    user_template: str  # Python str.format() placeholders, e.g. "{query}" / "{context}"
    guardrails: list[str] = Field(default_factory=list)
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)

    def render(self, **kwargs: str) -> list[dict[str, str]]:
        system_content = self.system_prompt
        if self.guardrails:
            system_content += "\n\nRules you must follow:\n" + "\n".join(
                f"- {rule}" for rule in self.guardrails
            )

        user_content = ""
        if self.few_shot_examples:
            examples_block = "\n\n".join(
                f"Example input:\n{ex.input}\n\nExample output:\n{ex.output}"
                for ex in self.few_shot_examples
            )
            user_content += f"{examples_block}\n\n---\n\n"
        user_content += self.user_template.format(**kwargs)

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
