import yaml
import json
from pathlib import Path

class PromptLoader:
    def __init__(self, prompts_dir: str):
        self.prompts_dir = Path(prompts_dir)
        self.registry = PromptRegistry()

    def load_all(self):
        for file_path in self.prompts_dir.glob("*.yaml"):
            self._load_yaml_file(file_path)
        for file_path in self.prompts_dir.glob("*.json"):
            self._load_json_file(file_path)

    def _load_yaml_file(self, file_path: Path):
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
            for name, config in data.items():
                prompt = PromptTemplate(
                    name=name,
                    template=config['template'],
                    description=config.get('description', ''),
                    input_variables=config.get('input_variables', []),
                    version=config.get('version', '1.0')
                )
                self.registry.register(prompt)
