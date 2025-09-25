from pathlib import Path
from typing import Optional, Dict, Any, List
from backend.prompt.loader import PromptLoader
from backend.models.prompt_model import PromptTemplate


class PromptManager:
    """
    High-level manager for prompt templates.

    Provides centralized access to prompt templates loaded from YAML/JSON files.
    Handles loading, caching, and formatting of prompts.
    Supports multiple versions of the same prompt template.
    """

    def __init__(self, prompts_dir: str = "./prompts"):
        """
        Initialize the PromptManager with a prompts directory.

        Args:
            prompts_dir: Path to the directory containing prompt YAML/JSON files
        """
        self.prompts_dir = Path(prompts_dir)
        self.loader = PromptLoader(str(self.prompts_dir))
        self._loaded = False

    def load_prompts(self) -> None:
        """
        Load all prompt templates from the configured directory.
        This should be called before accessing prompts.
        """
        if not self._loaded:
            self.loader.load_all()
            self._loaded = True

    def get_prompt(self, name: str, version: Optional[str] = None) -> PromptTemplate:
        """
        Get a prompt template by name and optional version.

        Args:
            name: Name of the prompt template
            version: Specific version to retrieve. If None, returns the latest version.

        Returns:
            PromptTemplate instance

        Raises:
            ValueError: If prompt is not found
        """
        self.load_prompts()  # Ensure prompts are loaded
        return self.loader.registry.get(name, version)

    def get_formatted_prompt(self, name: str, version: Optional[str] = None, **kwargs) -> str:
        """
        Get a formatted prompt string with variables substituted.

        Args:
            name: Name of the prompt template
            version: Specific version to retrieve. If None, returns the latest version.
            **kwargs: Variables to substitute in the template

        Returns:
            Formatted prompt string

        Raises:
            ValueError: If prompt is not found
            KeyError: If required variables are missing
        """
        prompt = self.get_prompt(name, version)
        return prompt.format(**kwargs)

    def list_prompts(self) -> List[str]:
        """
        List all available prompt names.

        Returns:
            List of prompt names
        """
        self.load_prompts()  # Ensure prompts are loaded
        return self.loader.registry.list_prompts()

    def list_versions(self, name: str) -> List[str]:
        """
        List all available versions for a specific prompt name.

        Args:
            name: Name of the prompt template

        Returns:
            List of available versions

        Raises:
            ValueError: If prompt name is not found
        """
        self.load_prompts()  # Ensure prompts are loaded
        return self.loader.registry.list_versions(name)

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
        self.load_prompts()  # Ensure prompts are loaded
        return self.loader.registry.get_all_versions(name)

    def has_prompt(self, name: str, version: Optional[str] = None) -> bool:
        """
        Check if a prompt exists.

        Args:
            name: Name of the prompt template
            version: Specific version to check. If None, checks if any version exists.

        Returns:
            True if prompt exists, False otherwise
        """
        try:
            self.get_prompt(name, version)
            return True
        except ValueError:
            return False

    def get_prompt_info(self, name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed information about a prompt.

        Args:
            name: Name of the prompt template
            version: Specific version to retrieve. If None, returns the latest version.

        Returns:
            Dictionary with prompt information

        Raises:
            ValueError: If prompt is not found
        """
        prompt = self.get_prompt(name, version)
        return {
            "name": prompt.name,
            "description": prompt.description,
            "input_variables": prompt.input_variables,
            "version": prompt.version,
            "template_length": len(prompt.template),
            "metadata": prompt.metadata
        }

    def reload_prompts(self) -> None:
        """
        Reload all prompts from disk. Useful for development.
        """
        self._loaded = False
        self.load_prompts()

if __name__ == "__main__":
    # from backend.prompt.prompt_manager import PromptManager

    # manager = PromptManager()
    # formatted_prompt = manager.get_formatted_prompt(
    #     "data_analysis",
    #     version="1.2",
    #     data="your data here",
    #     focus_areas="key areas to focus on",
    #     output_format="markdown"
    # )
    # print(formatted_prompt)