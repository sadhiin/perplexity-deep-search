from backend.models.prompt_model import PromptTemplate


class PromptRegistry:
    def __init__(self):
        self._prompts = {}

    def register(self, prompt_template: PromptTemplate):
        self._prompts[prompt_template.name] = prompt_template

    def get(self, name: str) -> PromptTemplate:
        if name not in self._prompts:
            raise ValueError(f"Prompt '{name}' not found")
        return self._prompts[name]

    def list_prompts(self) -> list[str]:
        return list(self._prompts.keys())
