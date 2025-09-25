from backend.models.prompt_model import PromptTemplate
from typing import Optional, Dict


class PromptRegistry:
    def __init__(self):
        self._prompts: Dict[str, Dict[str, PromptTemplate]] = {}

    def register(self, prompt_template: PromptTemplate):
        """Register a prompt template. Supports multiple versions of the same prompt name."""
        name = prompt_template.name
        version = prompt_template.version

        if name not in self._prompts:
            self._prompts[name] = {}

        self._prompts[name][version] = prompt_template

    def get(self, name: str, version: Optional[str] = None) -> PromptTemplate:
        """
        Get a prompt template by name and optional version.

        Args:
            name: Name of the prompt template
            version: Specific version to retrieve. If None, returns the latest version.

        Returns:
            PromptTemplate instance

        Raises:
            ValueError: If prompt name or version is not found
        """
        if name not in self._prompts:
            raise ValueError(f"Prompt '{name}' not found")

        versions = self._prompts[name]

        if version is None:
            # Return the latest version (sorted by version string)
            if not versions:
                raise ValueError(f"No versions found for prompt '{name}'")
            latest_version = max(versions.keys(), key=lambda v: [int(x) for x in v.split('.')])
            return versions[latest_version]
        else:
            if version not in versions:
                available_versions = list(versions.keys())
                raise ValueError(f"Version '{version}' not found for prompt '{name}'. Available versions: {available_versions}")
            return versions[version]

    def list_prompts(self) -> list[str]:
        """List all unique prompt names."""
        return list(self._prompts.keys())

    def list_versions(self, name: str) -> list[str]:
        """
        List all available versions for a specific prompt name.

        Args:
            name: Name of the prompt template

        Returns:
            List of available versions

        Raises:
            ValueError: If prompt name is not found
        """
        if name not in self._prompts:
            raise ValueError(f"Prompt '{name}' not found")
        return list(self._prompts[name].keys())

    def get_all_versions(self, name: str) -> Dict[str, PromptTemplate]:
        """
        Get all versions of a prompt template.

        Args:
            name: Name of the prompt template

        Returns:
            Dictionary mapping version to PromptTemplate

        Raises:
            ValueError: If prompt name is not found
        """
        if name not in self._prompts:
            raise ValueError(f"Prompt '{name}' not found")
        return self._prompts[name].copy()
