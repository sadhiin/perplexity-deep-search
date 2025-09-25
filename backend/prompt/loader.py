import yaml
import json
from pathlib import Path
from backend.models.prompt_model import PromptTemplate
from backend.prompt.prompt_registry import PromptRegistry
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
                # Check if this is a versioned prompt structure
                if isinstance(config, dict) and any(key.startswith('v') or key[0].isdigit() for key in config.keys()):
                    # Versioned structure: {prompt_name: {version: config}}
                    for version_key, version_config in config.items():
                        if isinstance(version_config, dict) and 'template' in version_config:
                            # Extract version from key (remove 'v' prefix if present)
                            version = version_key.lstrip('v')
                            prompt = PromptTemplate(
                                name=name,
                                template=version_config['template'],
                                description=version_config.get('description', ''),
                                input_variables=version_config.get('input_variables', []),
                                version=version,
                                metadata=version_config.get('metadata')
                            )
                            self.registry.register(prompt)
                else:
                    # Legacy flat structure: {prompt_name: config}
                    if isinstance(config, dict) and 'template' in config:
                        prompt = PromptTemplate(
                            name=name,
                            template=config['template'],
                            description=config.get('description', ''),
                            input_variables=config.get('input_variables', []),
                            version=config.get('version', '1.0'),
                            metadata=config.get('metadata')
                        )
                        self.registry.register(prompt)

    def _load_json_file(self, file_path: Path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            for name, config in data.items():
                # Check if this is a versioned prompt structure
                if isinstance(config, dict) and any(key.startswith('v') or key[0].isdigit() for key in config.keys()):
                    # Versioned structure: {prompt_name: {version: config}}
                    for version_key, version_config in config.items():
                        if isinstance(version_config, dict) and 'template' in version_config:
                            # Extract version from key (remove 'v' prefix if present)
                            version = version_key.lstrip('v')
                            prompt = PromptTemplate(
                                name=name,
                                template=version_config['template'],
                                description=version_config.get('description', ''),
                                input_variables=version_config.get('input_variables', []),
                                version=version,
                                metadata=version_config.get('metadata')
                            )
                            self.registry.register(prompt)
                else:
                    # Legacy flat structure: {prompt_name: config}
                    if isinstance(config, dict) and 'template' in config:
                        prompt = PromptTemplate(
                            name=name,
                            template=config['template'],
                            description=config.get('description', ''),
                            input_variables=config.get('input_variables', []),
                            version=config.get('version', '1.0'),
                            metadata=config.get('metadata')
                        )
                        self.registry.register(prompt)
